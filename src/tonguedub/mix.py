"""Aşama 5: ses cilası (perde/seviye), üst üste binmeyi önleyen sığdırma ve mix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import active_rms, run, seg_f0, stretch_audio
from .config import Settings
from .constants import (EXPRESS_MAX, GAP, MAX_PUSH, MAX_RATE, PITCH_DEADBAND, PITCH_MAX_FIX, SOFT_RATE, SR,
                        TRUNC_FADE)


def polish_clips(cfg: Settings, segs, items, ws, y_voc, W: Path) -> None:
    """Klonlanan sesi konuşmacının gerçek özelliklerine yaklaştırır (yeniden üretim gerektirmez):
    1) PERDE: klibin f0'ı konuşmacının orijinal medyanından 2-4.5 yarım ton uzaksa Rubber Band ile düzelt;
       daha uzaksa (örn. yüksek perde grubuna ait klip alçak perdede çıkmışsa) düzeltme, işaretle.
    2) SEVİYE: her satırı, konuşmacı medyanına göre orijinal satırın yüksek/alçaklığının yarısı kadar (±3 dB) ayarla."""
    import librosa
    express = cfg.pipeline.expressive
    y = y_voc.mean(axis=1) if y_voc.ndim > 1 else y_voc
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000).astype(np.float32)
    f0s, dbs = {}, {}
    for _, i in items:
        a, b = int(segs[i]["start"] * 16000), int(segs[i]["end"] * 16000)
        f0s[i] = seg_f0(y16[a:b]) if b - a >= 4000 else None
        dbs[i] = 20 * np.log10(active_rms(y16[a:b]) + 1e-9) if b > a else None
    spk_f0, spk_db = {}, {}
    for c in {segs[i].get("spk", 0) for _, i in items}:
        ii = [i for _, i in items if segs[i].get("spk", 0) == c]
        fv = [f0s[i] for i in ii if f0s[i]]
        dv = [dbs[i] for i in ii if dbs[i] is not None and dbs[i] > -80]
        spk_f0[c] = float(np.median(fv)) if fv else None
        spk_db[c] = float(np.median(dv)) if dv else None
    fixed = flagged = leveled = 0
    flags, rows = [], []
    for _, i in items:
        w = ws[i]
        c = segs[i].get("spk", 0)
        w16 = librosa.resample(w, orig_sr=SR, target_sr=16000).astype(np.float32)
        f = seg_f0(w16) if len(w16) >= 4000 else None
        f_fin = f
        if f and spk_f0.get(c):
            st = 12 * np.log2(spk_f0[c] / f)
            if abs(st) > PITCH_MAX_FIX:
                flags.append({"satır": int(i), "sapma_yarım_ton": round(float(st), 1)})
                flagged += 1
            else:
                want = 0.0   # klibin konuşmacı medyanının kaç yarım ton üstünde/altında olması isteniyor
                if express > 0 and f0s.get(i):
                    want = float(np.clip(express * 12 * np.log2(f0s[i] / spk_f0[c]), -EXPRESS_MAX, EXPRESS_MAX))
                shift = st + want
                if abs(shift) > (PITCH_DEADBAND if express <= 0 else 1.0):
                    shift = float(np.clip(shift, -6.0, 6.0))
                    try:
                        import pyrubberband as pyrb
                        w = pyrb.pitch_shift(w, SR, shift).astype(np.float32)
                        fixed += 1
                        f_fin = f * 2 ** (shift / 12)
                    except Exception:
                        pass
        if dbs.get(i) is not None and spk_db.get(c) is not None and dbs[i] > -80:
            gain = 0.5 * float(np.clip(dbs[i] - spk_db[c], -6.0, 6.0))
            if abs(gain) > 0.3:
                w = w * (10 ** (gain / 20))
                leveled += 1
        pk = float(np.abs(w).max()) if len(w) else 0.0
        if pk > 0.95:
            w = w * (0.95 / pk)
        ws[i] = w
        rows.append({"i": int(i), "spk": int(c), "f_raw": f, "f_fin": f_fin, "target": spk_f0.get(c), "db_target": spk_db.get(c)})
    (W / "voice_stats.json").write_text(json.dumps(rows))
    (W / "voice_flags.json").write_text(json.dumps(flags, ensure_ascii=False))
    print(f"  ses cilası: {fixed} klibin perdesi düzeltildi, {leveled} satırın seviyesi ayarlandı, "
          f"{flagged} klip işaretlendi (perde çok sapmış: yeniden üretim önerilir)", flush=True)


def stage_mix(cfg: Settings, video: Path, segs, tts_dir: Path, voc: Path, bg: Path, W: Path, start, end, out: Path,
              stems=None) -> None:
    pl = cfg.pipeline
    print("[5/5] Mix + birleştirme...")
    if stems:  # TTS sürerken arka planda okunmuştu
        y_voc, y_bg = stems[0][0], stems[1][0]
    else:
        y_voc, _ = sf.read(voc, dtype="float32")
        y_bg, _ = sf.read(bg, dtype="float32")
    n = len(y_bg)
    dub = np.zeros(n, dtype=np.float32)
    keep = np.ones(n, dtype=np.float32)  # orijinal vokal kanalının kalma oranı (dublajlı yerlerde 0)
    fade = int(0.12 * SR)  # uzun çapraz geçiş: ortam sesi birden kesilmesin
    items = sorted((s["start"], i) for i, s in enumerate(segs)
                   if (tts_dir / f"{i:04d}.wav").exists() and (s["end"] - s["start"]) >= pl.min_dur)
    ws = {i: sf.read(tts_dir / f"{i:04d}.wav", dtype="float32")[0] for _, i in items}
    if pl.polish and items:
        polish_clips(cfg, segs, items, ws, y_voc, W)
    starts = [st for st, _ in items]
    place, fs = {}, {"hızlandırıldı": 0, "ertelendi": 0, "kesildi": 0, "doğal çakışma": 0}
    for k, (_, i) in enumerate(items):
        w, t = ws[i], starts[k]
        d = len(w) / SR
        if pl.fit and k + 1 < len(items):
            nxt = items[k + 1][1]
            window = starts[k + 1] - t - GAP
            if d > window:
                r = d / window if window > 0.05 else 99.0
                if r <= SOFT_RATE:
                    w = stretch_audio(w, r)
                    fs["hızlandırıldı"] += 1
                else:
                    push = min(MAX_PUSH, max(0.0, d / SOFT_RATE - window))
                    if push > 0:
                        starts[k + 1] += push
                        fs["ertelendi"] += 1
                    window2 = window + push
                    r2 = d / max(window2, 0.05)
                    if r2 <= MAX_RATE:
                        w = stretch_audio(w, r2)
                        fs["hızlandırıldı"] += 1
                    else:
                        w2 = stretch_audio(w, MAX_RATE)
                        keep_len = int((window2 + 0.15) * SR)
                        cut = len(w2) - keep_len
                        same = segs[nxt].get("spk", 0) == segs[i].get("spk", 0)
                        if cut > 0 and (same or cut < 0.4 * len(w2)):
                            w2 = w2[:keep_len].copy()
                            f = min(int(TRUNC_FADE * SR), len(w2))
                            w2[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)  # ani kesilme olmasın
                            fs["kesildi"] += 1
                        else:
                            fs["doğal çakışma"] += 1
                        w = w2
        place[i] = [round(t, 3), round(len(w) / SR, 3)]
        a = int(t * SR)
        b = min(n, a + len(w))
        if b > a:
            dub[a:b] += w[: b - a]
            k0, k1 = max(0, a - fade), min(n, b + fade)
            keep[k0:k1] = 0.0
    print("  sığdırma: " + ", ".join(f"{v} {k}" for k, v in fs.items()), flush=True)
    (W / "placements.json").write_text(json.dumps(place))
    from scipy.ndimage import uniform_filter1d
    keep = uniform_filter1d(keep, size=fade)
    g = 10 ** (-pl.duck / 20)                   # konuşma sırasında arka planı hafifçe kıs (anlaşılırlık)
    bgk = g + (1.0 - g) * keep
    y_bg = y_bg * (bgk[:, None] if y_bg.ndim > 1 else bgk)
    # dublajlı yerlerde orijinal vokal tamamen kapanmaz: ortam sesi sürsün diye %bleed kalır
    keep = pl.bleed + (1.0 - pl.bleed) * keep
    kept = y_voc * (keep[:, None] if y_voc.ndim > 1 else keep)
    mix = y_bg + kept + (dub[:, None] if y_bg.ndim > 1 else dub) * 1.0
    peak = np.abs(mix).max()
    if peak > 0.97:
        mix = mix * (0.97 / peak)
    mixp = W / "mix.wav"
    sf.write(mixp, mix, SR, subtype="PCM_16")
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(video), "-i", str(mixp), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    run(cmd, "Kur: brew install ffmpeg")
    print("Hazır:", out)
