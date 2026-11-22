"""Ses işleme: çıkarma, vokal ayırma, perde/seviye ölçümü, kırpma, hızlandırma."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .constants import SR, VOCAL_MIN_RMS
from .errors import TonguedubError
from .proc import CHILDREN


def tail(text: str, lines: int = 6, limit: int = 600) -> str:
    return "\n".join((text or "").strip().splitlines()[-lines:])[-limit:]


def run(cmd: list[str], install_hint: str = "") -> None:
    """Komutu çalıştırır; başarısızsa stderr'in son satırlarıyla anlaşılır hata verir."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise TonguedubError(f"'{cmd[0]}' bulunamadı. {install_hint}".strip()) from e
    if p.returncode != 0:
        raise TonguedubError(f"{Path(cmd[0]).name} başarısız oldu (kod {p.returncode}):\n{tail(p.stderr)}")


# ---------- ses çıkar + vokal ayır ----------
def extract_audio(video: Path, W: Path, start, end) -> Path:
    audio = W / "audio.wav"
    if not audio.exists():
        cmd = ["ffmpeg", "-y", "-i", str(video)]
        if start is not None:
            cmd += ["-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        run(cmd + ["-vn", "-ac", "2", "-ar", str(SR), str(audio)], "Kur: brew install ffmpeg")
    return audio


def start_demucs(audio: Path, W: Path):
    """Demucs'u arka planda başlatır (işlemci); Whisper aynı anda nöral motorda çalışır."""
    if (W / "vocals.wav").exists():
        return None
    print("[1/5] Vokal ayrıştırma (demucs) arka planda başladı...", flush=True)
    log = open(W / "demucs.log", "w")
    return CHILDREN.add(subprocess.Popen(
        [sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(W / "demucs"), str(audio)],
        stdout=subprocess.DEVNULL, stderr=log))


def finish_demucs(proc, W: Path):
    voc, bg = W / "vocals.wav", W / "no_vocals.wav"
    if proc is not None:
        if proc.wait() != 0:
            log = W / "demucs.log"
            raise TonguedubError("demucs başarısız oldu:\n" + tail(log.read_text() if log.exists() else ""))
        d = W / "demucs" / "htdemucs" / "audio"
        shutil.move(str(d / "vocals.wav"), voc)
        shutil.move(str(d / "no_vocals.wav"), bg)
    return voc, bg


def filter_by_vocals(segs: list[dict], voc: Path) -> list[dict]:
    """Whisper orijinal sesle çalıştığı için müzik/ortamdan uydurulmuş satırları,
    demucs'un vokal kanalına bakarak ele."""
    y, _ = sf.read(voc, dtype="float32")
    y = y.mean(axis=1) if y.ndim > 1 else y
    out = []
    for s in segs:
        a, b = int(s["start"] * SR), int(s["end"] * SR)
        if b > a and float(np.sqrt(np.mean(y[a:b] ** 2))) >= VOCAL_MIN_RMS:
            out.append(s)
    if len(out) != len(segs):
        print(f"  vokal kontrolü: {len(segs) - len(out)} şüpheli satır elendi", flush=True)
    return out


# ---------- ölçümler ----------
def active_rms(x: np.ndarray) -> float:
    """Sadece konuşulan (sessiz olmayan) anların RMS'i: duraklamalar seviyeyi aldatmasın."""
    e = np.abs(x)
    if len(e) == 0 or e.max() < 1e-6:
        return 1e-9
    m = e > 0.1 * np.percentile(e, 95)
    return float(np.sqrt(np.mean(x[m] ** 2))) if m.any() else float(np.sqrt(np.mean(x ** 2)))


def seg_f0(x16: np.ndarray):
    """Segmentin medyan temel frekansı (Hz); yetersizse None."""
    import librosa
    if len(x16) < 4000:
        return None
    f = librosa.yin(x16, fmin=70, fmax=500, sr=16000, frame_length=1024)
    rms = librosa.feature.rms(y=x16, frame_length=1024, hop_length=256)[0][:len(f)]
    f = f[:len(rms)][rms > np.percentile(rms, 50)]
    return float(np.median(f)) if len(f) else None


def trim_silence(w: np.ndarray, sr: int) -> np.ndarray:
    """Klibin başındaki/sonundaki sessizliği ve uzatılmış kuyruğu kırpar (kısa satırlarda dublaj
    orijinalin birkaç katı uzun çıkıyordu)."""
    hop = max(1, int(sr * 0.01))
    nf = len(w) // hop
    if nf < 5:
        return w
    env = np.sqrt(np.mean(w[: nf * hop].reshape(nf, hop) ** 2, axis=1))
    env = np.convolve(env, np.ones(3) / 3, mode="same")
    v = np.where(env > 0.08 * np.percentile(env, 95))[0]
    if len(v) == 0:
        return w
    a, b = max(0, int(v[0]) - 3) * hop, min(len(w), (int(v[-1]) + 6) * hop)
    return w[a:b]


def stretch_audio(w: np.ndarray, rate: float) -> np.ndarray:
    """Perdeyi koruyarak hızlandırır: Rubber Band varsa onunla (daha az bozulma), yoksa librosa."""
    try:
        import pyrubberband as pyrb
        return pyrb.time_stretch(w, SR, rate).astype(np.float32)
    except Exception:
        import librosa
        return librosa.effects.time_stretch(w, rate=rate).astype(np.float32)
