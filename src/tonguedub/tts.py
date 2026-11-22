"""Aşama 4b: Chatterbox ile ses klonlamalı Türkçe TTS, ortak iş havuzu, birleşik üretim ve perde yeniden üretimi."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import active_rms, seg_f0, trim_silence
from .config import Settings, work_path
from .constants import CHARS_PER_SEC, MERGE_CAP, REROLL_ST, SHORT_SEC, SR
from .proc import CHILDREN
from .speakers import stage_speakers
from .textutil import VALID, clean_desc


def load_tts(cfg: Settings):
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    dev = cfg.tts.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    orig = torch.load  # checkpoint'leri hedef cihaza map et
    torch.load = lambda *a, **k: orig(*a, **{**k, "map_location": torch.device(dev)})
    try:
        return ChatterboxMultilingualTTS.from_pretrained(device=dev)
    finally:
        torch.load = orig


def gen_wav(cfg: Settings, tts, text: str, ref: str, seed: int):
    import torch
    for temp in (0.6, 0.8, 0.5):
        try:
            torch.manual_seed(seed)  # sabit tohum: tekrarlanabilir, daha tutarlı ton
            w = tts.generate(text, language_id="tr", audio_prompt_path=ref,
                             exaggeration=cfg.tts.exaggeration, cfg_weight=cfg.tts.cfg_weight, temperature=temp)
            w = w.squeeze().cpu().numpy()
            if len(w) > 1600:
                return w
        except Exception:
            pass
    return None


def save_clip(w, tts_sr, i, segs, level, out_dir: Path):
    import librosa
    s = segs[i]
    w = librosa.resample(w, orig_sr=tts_sr, target_sr=SR)
    w = trim_silence(w, SR)
    cur = active_rms(w) + 1e-9
    w = w * min(10 ** ((level[s["spk"]] - 20 * np.log10(cur)) / 20), 8.0)  # konuşmacı seviyesine eşitle
    pk = float(np.abs(w).max())
    if pk > 0.9:
        w = w * (0.9 / pk)
    p = out_dir / f"{i:04d}.wav"
    tmp = p.with_suffix(".tmp.wav")
    sf.write(tmp, w, SR)
    os.replace(tmp, p)  # atomik


def plan_groups(segs: list[dict], merge_sec: float, min_dur: float) -> list[list[int]]:
    """KISA satırları konuşmacıya göre toplar (sıra önemsiz: her klip kendi zaman damgasına yerleşir),
    merge_sec'e kadar gruplar. Uzun satırlar tek başına. Deterministik (tüm süreçlerde aynı)."""
    groups, by_spk = [], {}
    for i, s in enumerate(segs):
        tr = clean_desc(s.get("tr", ""))
        if not tr or not VALID.search(tr) or (s["end"] - s["start"]) < min_dur:
            continue
        if merge_sec <= 0 or (s["end"] - s["start"]) >= SHORT_SEC:
            groups.append([i])
        else:
            by_spk.setdefault(s["spk"], []).append(i)
    for _spk, idxs in by_spk.items():
        cur, est = [], 0.0
        for i in idxs:
            t = len(clean_desc(segs[i]["tr"])) / CHARS_PER_SEC + 0.3
            if cur and (est + t > merge_sec or len(cur) >= MERGE_CAP):
                groups.append(cur)
                cur, est = [], 0.0
            cur.append(i)
            est += t
        if cur:
            groups.append(cur)
    groups.sort(key=lambda g: g[0])
    return groups


def split_by_silence(w, sr, texts):
    """Birleşik üretimi, satır başına karakter oranından beklenen konumların yakınındaki en sessiz
    noktalardan keser. Net bir boşluk yoksa None (güvenlik ağı: tek tek üretilecek)."""
    n = len(texts)
    hop = max(1, int(sr * 0.01))
    nf = len(w) // hop
    if nf < 10 * n:
        return None
    env = np.sqrt(np.mean(w[: nf * hop].reshape(nf, hop) ** 2, axis=1))
    env = np.convolve(env, np.ones(5) / 5, mode="same")
    p90 = np.percentile(env, 90)
    if p90 < 1e-4:
        return None
    voiced = np.where(env > 0.1 * p90)[0]
    if len(voiced) == 0:
        return None
    f0, f1 = int(voiced[0]), int(voiced[-1]) + 1
    chars = np.array([len(t) for t in texts], dtype=float)
    cum = np.cumsum(chars)[:-1] / chars.sum()
    half = max(15, int(0.4 * (f1 - f0) / n))
    cuts = []
    for c in cum:
        pos = int(f0 + c * (f1 - f0))
        lo, hi = max(f0 + 5, pos - half), min(f1 - 5, pos + half)
        if hi <= lo:
            return None
        k = lo + int(np.argmin(env[lo:hi]))
        if env[k] > 0.15 * p90:  # net sessizlik yok -> güvenme
            return None
        cuts.append(k)
    if any(b - a < 15 for a, b in zip([f0] + cuts, cuts + [f1])):  # <150 ms parça
        return None
    edges = [f0] + cuts + [f1]
    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = w[a * hop: b * hop]
        v = np.where(env[a:b] > 0.1 * p90)[0]
        if len(v):  # parçanın başındaki/sonundaki sessizliği kırp (zamanlama için)
            s0 = max(0, int(v[0]) - 3) * hop
            s1 = min(len(seg), (int(v[-1]) + 4) * hop)
            seg = seg[s0:s1]
        pieces.append(seg)
    return pieces


def group_cost(segs, g):
    return sum(len(clean_desc(segs[i]["tr"])) for i in g)


def reset_claims(W: Path) -> None:
    shutil.rmtree(W / "claims", ignore_errors=True)
    (W / "claims").mkdir(exist_ok=True)


def synth(cfg, tts, segs, refs, level, groups, out_dir: Path, redo, log=False, stats=None, claim_dir=None, total=0):
    """Grupları üretir. claim_dir verilirse ORTAK İŞ HAVUZU: her süreç sıradaki grubu atomik olarak
    sahiplenir (büyük gruplar önce) -> boşta bekleyen süreç kalmaz. Tek satırlıklar doğrudan;
    çok satırlılar birleşik üretilip sessizlikten kesilir."""
    skipped = []
    order = (sorted(range(len(groups)), key=lambda k: (-group_cost(segs, groups[k]), k))
             if claim_dir is not None else range(len(groups)))
    t0, last = time.time(), 0.0
    base = len(list(out_dir.glob("[0-9]*.wav")))

    def prog():
        nonlocal last
        if log and total and time.time() - last >= 20:
            last = time.time()
            n = len(list(out_dir.glob("[0-9]*.wav")))
            el = (time.time() - t0) / 60
            rate = (n - base) / max(el, 0.01)
            eta = (total - n) / rate if rate > 0 else 0
            print(f"  TTS {n}/{total} klip | {el:.1f} dk geçti | kalan ~{eta:.1f} dk ({time.strftime('%H:%M:%S')})", flush=True)

    def do_group(g):
        todo = [i for i in g if redo or not (out_dir / f"{i:04d}.wav").exists()]
        if not todo:
            return
        if len(g) > 1:
            texts = [clean_desc(segs[i]["tr"]) for i in g]
            ref = refs[segs[g[0]]["spk"]]
            w = gen_wav(cfg, tts, " ".join(t if t[-1] in ".?!…" else t + "." for t in texts), ref, 1000 + g[0])
            pieces = split_by_silence(w, tts.sr, texts) if w is not None else None
            ok = pieces is not None
            with open(out_dir.parent / "merge_log.txt", "a") as mf:  # canlı izlenebilsin
                mf.write(f"{len(g)} {'ok' if ok else 'fail'}\n")
            if ok:
                for i, pc in zip(g, pieces):
                    save_clip(pc, tts.sr, i, segs, level, out_dir)
                if stats is not None:
                    stats.append("ok")
                return
            if stats is not None:
                stats.append("fallback")
            mid = len(g) // 2          # ikiye böl: en kötü ihtimalle 2 çağrı, 4 değil
            do_group(g[:mid])
            do_group(g[mid:])
            return
        for i in g:  # tek satır
            if not redo and (out_dir / f"{i:04d}.wav").exists():
                continue
            w = gen_wav(cfg, tts, clean_desc(segs[i]["tr"]), refs[segs[i]["spk"]], 1000 + i)
            if w is None:
                skipped.append(i)  # orijinal ses kalır
                continue
            save_clip(w, tts.sr, i, segs, level, out_dir)

    for gid in order:
        if claim_dir is not None:
            try:
                os.close(os.open(claim_dir / str(gid), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue  # başka süreç aldı
        do_group(groups[gid])
        prog()
    return skipped


def reroll_voices(cfg, tts, segs, refs, level, voc, out_dir: Path, order="fwd"):
    """Perdesi konuşmacıdan çok sapan klipleri (örn. yüksek perde grubuna ait klip alçak perdede çıkmışsa) farklı tohumlarla
    yeniden üretir, hedefe en yakın olanı tutar."""
    import librosa
    y, _ = sf.read(voc, dtype="float32")
    y16 = librosa.resample(y.mean(axis=1) if y.ndim > 1 else y, orig_sr=SR, target_sr=16000).astype(np.float32)
    have = [i for i in range(len(segs)) if (out_dir / f"{i:04d}.wav").exists()]
    byspk = {}
    for i in have:
        a, b = int(segs[i]["start"] * 16000), int(segs[i]["end"] * 16000)
        f = seg_f0(y16[a:b]) if b - a >= 4000 else None
        if f:
            byspk.setdefault(segs[i].get("spk", 0), []).append(f)
    target = {c: float(np.median(v)) for c, v in byspk.items()}

    def dev(w, sr, c):
        f = seg_f0(librosa.resample(w, orig_sr=sr, target_sr=16000).astype(np.float32)) if len(w) > sr * 0.25 else None
        return 12 * np.log2(target[c] / f) if (f and c in target) else 0.0

    bad = []
    for i in have:
        w, sr = sf.read(out_dir / f"{i:04d}.wav", dtype="float32")
        w = w if w.ndim == 1 else w.mean(1)
        st = dev(w, sr, segs[i].get("spk", 0))
        if abs(st) > REROLL_ST:
            bad.append((i, abs(st)))
    if order == "rev":
        bad = bad[::-1]
    elif order == "mid":
        bad = bad[len(bad) // 2:] + bad[:len(bad) // 2]
    fixed = 0
    for i, st0 in bad:
        c = segs[i].get("spk", 0)
        best, bw = st0, None
        for k in range(1, 4):
            w = gen_wav(cfg, tts, clean_desc(segs[i]["tr"]), refs[c], 2000 + i * 7 + k)
            if w is None:
                continue
            st = abs(dev(w, tts.sr, c))
            if st < best:
                best, bw = st, w
            if st <= 2.0:
                break
        if bw is not None:
            save_clip(bw, tts.sr, i, segs, level, out_dir)
            fixed += 1
    print(f"  tekrar üretim: {len(bad)} sapmış klip bulundu, {fixed} tanesi daha iyi bir sürümle değiştirildi", flush=True)


def stage_tts(cfg, segs, voc, W: Path, redo, workers=1, kids=(), tts=None, spk=None):
    out_dir = W / "tts"
    out_dir.mkdir(exist_ok=True)
    pl = cfg.pipeline
    print(f"[4/5] Ses klonlama + Türkçe TTS (chatterbox, {workers} süreç)...")
    tts = tts or load_tts(cfg)
    if spk:  # çeviri sürerken hazırlanmış konuşmacı analizi
        refs, level, labels = spk
        for sg, c in zip(segs, labels):
            sg["spk"] = c
    else:
        refs, level = stage_speakers(segs, voc, tts, pl.speakers, W)
    tmp = W / "speakers.tmp"
    tmp.write_text(json.dumps({"refs": refs, "level": level, "spk": [s["spk"] for s in segs]}))
    os.replace(tmp, W / "speakers.json")  # atomik: bekleyen süreçler yarım dosya görmesin
    groups = plan_groups(segs, pl.merge, pl.min_dur)
    stats = []
    (W / "claims").mkdir(exist_ok=True)
    synth(cfg, tts, segs, refs, level, groups, out_dir, redo, log=True, stats=stats,
          claim_dir=W / "claims", total=sum(len(g) for g in groups))
    for k, kd in kids:
        rc = kd.wait()
        if rc != 0:
            print(f"  ! TTS süreci {k} beklenmedik şekilde kapandı (kod {rc}, bkz. {W.name}/worker{k}.log); eksikler tamamlanacak", flush=True)
    # eksik kalan var mı? (üretilmemiş ve atlanmamış)
    missing = [i for i, sg in enumerate(segs)
               if sg.get("tr") and VALID.search(sg["tr"]) and not (out_dir / f"{i:04d}.wav").exists()]
    if pl.merge > 0:
        ms = W / "merge_stats.txt"
        allst = stats + (ms.read_text().split() if ms.exists() else [])
        print(f"  birleşik gruplar: {allst.count('ok')} başarılı, {allst.count('fallback')} güvenlik ağına düştü", flush=True)
    if missing:
        # çocuk süreç düştüyse ana süreç tamamlasın
        synth(cfg, tts, segs, refs, level, [[i] for i in missing], out_dir, redo)
        missing = [i for i in missing if not (out_dir / f"{i:04d}.wav").exists()]
    if missing:
        print(f"  {len(missing)} satır üretilemedi, orijinal ses kalacak")
    if pl.reroll:
        reroll_voices(cfg, tts, segs, refs, level, voc, out_dir)
    return out_dir


def spawn_tts_workers(cfg, video: Path, W: Path, workers: int, redo: bool):
    """Ek TTS süreçlerini çeviri sürerken başlatır; modeli yükleyip speakers.json'u bekler."""
    (W / "speakers.json").unlink(missing_ok=True)
    (W / "merge_stats.txt").unlink(missing_ok=True)
    reset_claims(W)
    kids = []
    for k in range(1, workers):
        cmd = [sys.executable, "-m", "tonguedub", "_tts-worker", str(video), "--k", str(k), "--workers", str(workers),
               "--parent-pid", str(os.getpid()), "--merge", str(cfg.pipeline.merge),
               "--min-dur", str(cfg.pipeline.min_dur)] + (["--redo-tts"] if redo else [])
        logf = open(W / f"worker{k}.log", "w")
        kids.append((k, CHILDREN.add(subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT))))
    return kids


def tts_worker(cfg, video: Path, k: int, redo: bool, parent_pid: int) -> None:
    W = work_path(cfg, video)
    tts = load_tts(cfg)  # model yüklemesi çeviri/konuşmacı analiziyle üst üste biner
    t0 = time.time()
    while not (W / "speakers.json").exists():
        if os.getppid() != parent_pid or time.time() - t0 > 1800:
            return  # ana süreç öldü ya da çok bekledi
        time.sleep(1)
    segs = json.loads((W / "translated.json").read_text())
    sp = json.loads((W / "speakers.json").read_text())
    for sg, c in zip(segs, sp["spk"]):
        sg["spk"] = c
    stats = []
    synth(cfg, tts, segs, {int(a): b for a, b in sp["refs"].items()}, {int(a): b for a, b in sp["level"].items()},
          plan_groups(segs, cfg.pipeline.merge, cfg.pipeline.min_dur), W / "tts", redo, stats=stats, claim_dir=W / "claims")
    with open(W / "merge_stats.txt", "a") as f:
        f.write(" ".join(stats) + " ")


__all__ = ["load_tts", "gen_wav", "save_clip", "plan_groups", "split_by_silence", "synth", "reroll_voices",
           "stage_tts", "spawn_tts_workers", "tts_worker", "reset_claims"]
