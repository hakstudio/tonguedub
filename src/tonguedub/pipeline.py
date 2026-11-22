"""Tek video için uçtan uca boru hattı ve ara dosya temizliği."""
from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

from .audio import extract_audio, filter_by_vocals, finish_demucs, start_demucs
from .config import Settings, private_dir, resolve_workers, work_path
from .doctor import preflight
from .errors import TonguedubError
from .mix import stage_mix
from .proc import CHILDREN
from .report import build_report
from .speakers import stage_speakers
from .transcribe import stage_transcribe
from .translate import stage_translate
from .tts import load_tts, reset_claims, spawn_tts_workers, stage_tts


class _Tee:
    """Ekrana yazılanları özet log için de biriktirir (transkript içermez: yalnızca ilerleme/süre/rapor)."""

    def __init__(self, stream):
        self.stream, self.buf = stream, []

    def write(self, t):
        self.stream.write(t)
        self.buf.append(t)

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return False


def default_output(video: Path) -> Path:
    return video.with_name(video.stem + "_TR.mp4")


def run_video(cfg: Settings, video: Path, *, start=None, end=None, redo_tts=False, until=None, out: Path | None = None,
              mix_only=False) -> None:
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise TonguedubError(f"Dosya yok: {video}")
    out = Path(out).expanduser().resolve() if out else default_output(video)
    if out == video:
        raise TonguedubError("Çıktı dosyası girdi videosuyla aynı olamaz.")
    pl = cfg.pipeline
    W = work_path(cfg, video)
    if mix_only and not (W / "translated.json").exists():
        raise TonguedubError(f"--mix-only için ara dosyalar gerekir ({W} yok). Önce aynı videoyu --test ile çalıştır.")
    preflight(cfg, need_translate=not mix_only and not (W / "translated.json").exists())

    tee = _Tee(sys.stdout)
    sys.stdout = tee
    private_dir(Path(W.parent))
    private_dir(W)
    t0 = time.time()
    ok = False
    bg_ex = ThreadPoolExecutor(3)
    try:
        if mix_only:
            _mix_only(cfg, video, W, start, end, out)
        else:
            _full(cfg, video, W, start, end, redo_tts, until, out, bg_ex, t0)
        ok = True
    finally:
        bg_ex.shutdown(wait=False, cancel_futures=True)
        CHILDREN.terminate_all()
        sys.stdout = tee.stream
        _write_log(cfg, video, tee, ok, until)
        keep_for_queue = ok and until == "translate"  # kuyruk: hazırlık bitti, TTS aşaması bu klasörü kullanacak
        if not pl.keep and not keep_for_queue and W.exists():
            shutil.rmtree(W, ignore_errors=True)
            if ok:
                print("Ara dosyalar ve cache silindi (özet log: " + cfg.paths.log_dir + "). Saklamak için --test.")
            else:
                print("İşlem tamamlanamadı; ara dosyalar (transkript/çeviri dahil) silindi. Hata ayıklamak için --test ile çalıştır.")


def _mix_only(cfg, video, W, start, end, out):
    segs = json.loads((W / "translated.json").read_text())
    for sg, c in zip(segs, json.loads((W / "speakers.json").read_text())["spk"]):
        sg["spk"] = c
    stage_mix(cfg, video, segs, W / "tts", W / "vocals.wav", W / "no_vocals.wav", W, start, end, out)


def _full(cfg, video, W, start, end, redo_tts, until, out, bg_ex, T0):
    pl = cfg.pipeline
    workers = resolve_workers(pl.workers)

    def tick(name):
        print(f"  [{name} bitti: toplam {(time.time() - T0) / 60:.1f} dk]", flush=True)

    audio = extract_audio(video, W, start, end)
    tts_fut = bg_ex.submit(load_tts, cfg) if until != "translate" else None  # model, Whisper/demucs sürerken yüklensin
    dem = start_demucs(audio, W)                                            # CPU'da, arka planda
    segs = stage_transcribe(cfg, audio, W)
    tick("transkript (demucs ile aynı anda)")                                # nöral motorda
    voc, bg = finish_demucs(dem, W)
    tick("ses ayırma")
    kids = ()
    if until != "translate" and workers <= 1:
        reset_claims(W)
    spk_fut = None
    if not (W / "translated.json").exists():
        segs = filter_by_vocals(segs, voc)
        if tts_fut is not None:  # konuşmacı analizi çeviriyle aynı anda
            seg_copy = [dict(x) for x in segs]

            def _an():
                r, lv = stage_speakers(seg_copy, voc, tts_fut.result(), pl.speakers, W)
                return r, lv, [x["spk"] for x in seg_copy]
            spk_fut = bg_ex.submit(_an)
    if until != "translate" and workers > 1:
        kids = spawn_tts_workers(cfg, video, W, workers, redo_tts)  # modeller çeviri sırasında yüklensin
    segs = stage_translate(cfg, segs, W)
    tick("çeviri")
    if until == "translate":
        print("Çeviriye kadar hazır; TTS için aynı komutu --until olmadan çalıştır.")
        return
    stems_fut = bg_ex.submit(lambda: (sf.read(voc, dtype="float32"), sf.read(bg, dtype="float32")))  # mix için önden oku
    tts_dir = stage_tts(cfg, segs, voc, W, redo_tts, workers, kids, tts=tts_fut.result(),
                        spk=(spk_fut.result() if spk_fut else None))
    tick("TTS")
    stage_mix(cfg, video, segs, tts_dir, voc, bg, W, start, end, out, stems_fut.result())
    tick("mix")
    try:
        for line in build_report(W):
            print("  " + line, flush=True)
    except Exception as e:  # rapor yardımcıdır; başarısızlığı işi bozmasın
        print(f"  (kalite raporu üretilemedi: {type(e).__name__})", flush=True)


def _write_log(cfg: Settings, video: Path, tee: _Tee, ok: bool, until) -> None:
    if not tee.buf or (ok and until == "translate"):
        return
    try:
        logd = private_dir(Path(cfg.paths.log_dir).expanduser())
        stamp = time.strftime("%Y%m%d-%H%M%S")
        header = f"{video.name} | {time.strftime('%Y-%m-%d %H:%M:%S')} | {'tamam' if ok else 'BAŞARISIZ'}\n"
        (logd / f"{video.stem}_{stamp}{'' if ok else '_FAILED'}.log").write_text(header + "".join(tee.buf))
    except OSError:
        pass  # log yazılamıyorsa iş başarısı etkilenmesin
