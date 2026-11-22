"""Nesnel kalite raporu (ses kalitesini kulakla değerlendir):
  - çakışma: aynı anda çalan Türkçe ses süresi, en karışık anlar
  - ses seviyesi: konuşmacı başına yayılım
  - perde: konuşmacı başına dublaj perdesi (perde grubu kontrolü)
  - kapsama: referans transkriptteki konuşmanın ne kadarı dublajlanmış (isteğe bağlı)
Kullanım: python -m tonguedub.report <çalışma klasörü> [referans_segments.json]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def build_report(W: Path, ref_path: str | None = None) -> list[str]:
    import librosa
    W = str(W).rstrip("/") + "/"
    spk = json.load(open(W + "speakers.json"))["spk"]
    PLACE = json.load(open(W + "placements.json")) if os.path.exists(W + "placements.json") else {}
    segs = json.load(open(W + "translated.json"))
    iv, rms, f0 = [], {}, {}
    for p in sorted(glob.glob(W + "tts/*.wav")):
        i = int(os.path.basename(p)[:4])
        if PLACE and str(i) not in PLACE:
            continue  # mix'e girmeyen (atlanan) klipleri sayma
        x, sr = sf.read(p, dtype="float32")
        x = x if x.ndim == 1 else x.mean(1)
        pl = PLACE.get(str(i))
        a, dd = (pl if isinstance(pl, list) else (pl if pl is not None else segs[i]["start"], len(x) / sr))
        iv.append((a, a + dd, spk[i], i))
        e = np.abs(x)
        m = e > 0.1 * np.percentile(e, 95)
        rms[i] = 20 * np.log10(np.sqrt(np.mean(x[m] ** 2)) + 1e-9) if m.any() else -90
        y = librosa.resample(x, orig_sr=sr, target_sr=16000)
        if len(y) > 4000:
            f = librosa.yin(y, fmin=70, fmax=500, sr=16000, frame_length=1024)
            r = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f)]
            f = f[:len(r)][r > np.percentile(r, 50)]
            f0[i] = float(np.median(f)) if len(f) else None
    out: list[str] = []
    if not iv:
        return ["KLİP: 0 (rapor edilecek dublaj klibi yok)"]
    res = 100
    T = int(max(b for _, b, _, _ in iv)) + 2
    cnt = np.zeros(T * res, int)
    for a, b, c, i in iv:
        cnt[int(a * res):int(b * res)] += 1
    dub = (cnt >= 1).sum() / res
    two = (cnt >= 2).sum() / res
    out.append(f"KLİP: {len(iv)} | dublaj süresi {dub:.0f} sn | 2+ ses aynı anda: {two:.0f} sn (%{two / max(dub, 1) * 100:.0f}) | "
               f"3+: {(cnt >= 3).sum() / res:.0f} sn | en yoğun {cnt.max()} ses")
    same = diff = 0.0
    for k in range(len(iv)):
        for m in range(k + 1, len(iv)):
            if iv[m][0] >= iv[k][1]:
                break
            o = min(iv[k][1], iv[m][1]) - max(iv[k][0], iv[m][0])
            if o > 0.05:
                if iv[k][2] == iv[m][2]:
                    same += o
                else:
                    diff += o
    out.append(f"ÇAKIŞMA: aynı konuşmacı kendi üstüne {same:.0f} sn | farklı konuşmacılar {diff:.0f} sn")
    win = 10 * res
    sc = sorted(((cnt[k:k + win] >= 2).sum() / res, k // res) for k in range(0, len(cnt) - win, win // 2))[::-1]
    seen, line = [], "EN KARIŞIK ANLAR: "
    for s_, t in sc:
        if any(abs(t - u) < 12 for u in seen):
            continue
        seen.append(t)
        line += f"{t // 60}:{t % 60:02d} ({s_:.1f}sn)  "
        if len(seen) == 6:
            break
    out.append(line.rstrip())
    idx = sorted(rms)
    for c in sorted(set(spk)):
        v = np.array([rms[i] for i in idx if spk[i] == c])
        p = np.array([f0[i] for i in idx if spk[i] == c and f0.get(i)])
        if len(v) and len(p):
            out.append(f"SES spk{c}: {len(v)} klip | seviye medyan {np.median(v):.1f} dBFS, yayılım(%10-%90) "
                       f"{np.percentile(v, 90) - np.percentile(v, 10):.1f} dB | perde medyan {np.median(p):.0f} Hz")
    if ref_path:
        ref = json.load(open(ref_path))
        tot = cov = 0.0
        for r in ref:
            d = r["end"] - r["start"]
            tot += d
            if any(a - 0.3 <= r["start"] <= b for a, b, _, _ in iv):
                cov += d
        out.append(f"KAPSAMA: referans konuşma {tot:.0f} sn | dublajlanan {cov:.0f} sn (%{cov / tot * 100:.0f}) | İngilizce kalan ~{tot - cov:.0f} sn")
    vs = W + "voice_stats.json"
    if os.path.exists(vs):
        rows = json.load(open(vs))
        out.append("TUTARLILIK (perde: konuşmacının orijinal medyanına göre, yarım ton):")
        for c in sorted({x["spk"] for x in rows}):
            rr = [x for x in rows if x["spk"] == c and x["f_raw"] and x["target"]]
            if not rr:
                continue
            tgt = rr[0]["target"]
            st_raw = np.array([abs(12 * np.log2(tgt / x["f_raw"])) for x in rr])
            st_fin = np.array([abs(12 * np.log2(tgt / x["f_fin"])) for x in rr])
            low = tgt < 150
            wrong = sum((x["f_fin"] >= 150) if low else (x["f_fin"] < 150) for x in rr)
            out.append(f"  spk{c} ({'alçak perde' if low else 'yüksek perde'}, hedef {tgt:.0f} Hz): {len(rr)} klip | "
                       f"±3.5 yarım ton içinde: cila öncesi %{(st_raw <= 3.5).mean() * 100:.0f} -> sonrası %{(st_fin <= 3.5).mean() * 100:.0f} | "
                       f"perde grubu dışına çıkan: {wrong} | perde medyan {np.median([x['f_fin'] for x in rr]):.0f} Hz")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("\n".join(build_report(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else None)))
