"""Aşama 4a: konuşmacı ayırma (önce perde grubu, sonra ses gömmesi), referans klip ve hedef seviye."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import active_rms, seg_f0
from .constants import LOW_HZ, SR


def stage_speakers(segs: list[dict], voc: Path, tts, n_spk: int, W: Path):
    """Konuşmacıları önce perde grubuna (alçak/yüksek), sonra ses gömmesine göre ayırır.
    n_spk = yüksek perdeli konuşmacı sayısı; alçak perdeli konuşmacılar f0 ile otomatik ayrılır.
    Her konuşmacı için temiz bir referans klip ve hedef ses seviyesi çıkarır."""
    import librosa
    from sklearn.cluster import KMeans
    y, _ = sf.read(voc, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000).astype(np.float32)
    idx, wavs, f0s = [], [], []
    for i, s in enumerate(segs):
        a, b = int(s["start"] * 16000), int(s["end"] * 16000)
        if b - a >= 16000 * 1.0:
            idx.append(i)
            wavs.append(y16[a:b])
            f0s.append(seg_f0(y16[a:b]))
    emb = tts.ve.embeds_from_wavs(wavs, sample_rate=16000)
    low = np.array([(f is not None and f < LOW_HZ) for f in f0s])
    lab = np.full(len(idx), -1)
    high = np.where(~low)[0]
    k = max(1, min(n_spk, len(high)))
    if k > 1:
        lab[high] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb[high])
    else:
        lab[high] = 0
    n_high = k
    if low.sum() >= 4:           # yeterli alçak perdeli segment varsa ayrı konuşmacı
        lab[low] = n_high
        n_tot = n_high + 1
    else:
        lab[low] = 0
        n_tot = n_high
    for sg in segs:
        sg["spk"] = 0
    for i, lb in zip(idx, lab):
        segs[i]["spk"] = int(lb)
    # kısa segmentler (<1 sn): perdesine göre alçak/yüksek grup; yüksekse en son yüksek grup etiketi
    idxset = set(idx)
    low_lab = n_high if n_tot > n_high else None
    last, last_high = 0, 0
    for i, sg in enumerate(segs):
        if i not in idxset:
            a, b = int(sg["start"] * 16000), int(sg["end"] * 16000)
            f = seg_f0(y16[a:b]) if b - a >= 4000 else None
            if low_lab is not None and f is not None and f < LOW_HZ:
                sg["spk"] = low_lab
            elif f is not None:
                sg["spk"] = last_high
            else:
                sg["spk"] = last
        last = sg["spk"]
        if sg["spk"] != low_lab:
            last_high = sg["spk"]
    refs, level = {}, {}
    pos = {i: n for n, i in enumerate(idx)}
    for c in range(n_tot):
        mem = [i for i in idx if segs[i]["spk"] == c]
        fs = [f0s[pos[i]] for i in mem if f0s[pos[i]]]
        med = float(np.median(fs)) if fs else 200.0
        # referans adayları: 3-12 sn, perdesi grup medyanına yakın (±12%), temiz (yüksek RMS)
        cand = []
        for i in mem:
            d = segs[i]["end"] - segs[i]["start"]
            f = f0s[pos[i]]
            if 3 <= d <= 12 and f and abs(f - med) / med <= 0.12:
                a, b = int(segs[i]["start"] * SR), int(segs[i]["end"] * SR)
                cand.append((float(np.sqrt(np.mean(y[a:b] ** 2))), i))
        cand.sort(reverse=True)
        if not cand:  # gevşet: en uzun segmentler
            cand = [(segs[i]["end"] - segs[i]["start"], i) for i in mem]
            cand.sort(reverse=True)
        clip, tot = [], 0
        for _, i in cand:
            a, b = int(segs[i]["start"] * SR), int(segs[i]["end"] * SR)
            clip.append(y[a:b])
            tot += (b - a) / SR
            if tot >= 10:
                break
        p = W / f"ref_spk{c}.wav"
        sf.write(p, np.concatenate(clip) if clip else np.zeros(SR), SR)
        refs[c] = str(p)
        # hedef ses seviyesi: bu konuşmacının orijinal vokal seviyesinin medyanı
        rs = []
        for i in mem:
            a, b = int(segs[i]["start"] * SR), int(segs[i]["end"] * SR)
            r = active_rms(y[a:b])
            if r > 1e-4:
                rs.append(r)
        level[c] = float(20 * np.log10(np.median(rs))) if rs else -28.0
        print(f"  konuşmacı {c} ({'alçak perde' if med < LOW_HZ else 'yüksek perde'}, f0~{med:.0f} Hz): "
              f"{len(mem)} segment, referans {tot:.1f} sn, seviye {level[c]:.1f} dBFS")
    return refs, level
