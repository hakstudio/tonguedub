"""Komut satırı: `tonguedub video.mp4 [seçenekler]`, `tonguedub add|status|clear|clean|doctor`."""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from . import __version__
from .config import Settings, load_settings, resolve_workers, with_pipeline
from .errors import TonguedubError

COMMANDS = {"run", "add", "status", "queue", "clear", "clean", "doctor"}

HELP = """tonguedub: yerel EN→TR video dublajı (konuşmacı sesi klonlanır)

  tonguedub video.mp4 [seçenekler]     videoyu dublajla (çıktı: video_TR.mp4)
  tonguedub add video.mp4 [seçenekler] kuyruğa ekle (hazırlık ve TTS aşamaları üst üste biner)
  tonguedub status                     kuyruk durumu
  tonguedub clear                      biten/başarısız kuyruk kayıtlarını sil
  tonguedub clean                      sahipsiz ara dosyaları ve eski logları sil
  tonguedub doctor                     ortam ve ayar kontrolü

Seçenekler için: tonguedub run --help
Ayarlar: config.toml (örnek: config.example.toml)
"""


def build_run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tonguedub run", description="Videoyu Türkçeye dublajla.")
    ap.add_argument("video")
    ap.add_argument("--config", help="ayar dosyası (varsayılan: config.toml aranır)")
    ap.add_argument("--out", help="çıktı dosyası (varsayılan: <video>_TR.mp4)")
    ap.add_argument("--start", type=float, help="deneme için başlangıç (sn)")
    ap.add_argument("--end", type=float, help="deneme için bitiş (sn)")
    ap.add_argument("--speakers", type=int, help="yüksek perdeli konuşmacı sayısı (alçak perdeliler otomatik ayrılır)")
    ap.add_argument("--workers", help="paralel TTS süreci: sayı ya da auto (RAM'e göre)")
    ap.add_argument("--merge", type=float, help="kısa satırları bu süreye (sn) kadar birleştirip tek seferde üret (0=kapalı)")
    ap.add_argument("--min-dur", type=float, help="bu süreden (sn) kısa satırları dublajlama, orijinal ses kalsın")
    ap.add_argument("--expressive", type=float, help="perde, orijinal satırın sapmasını bu oranda takip etsin (0=düz, 1=tam)")
    ap.add_argument("--bleed", type=float, help="dublaj sırasında orijinal vokalden kalan oran (0-1)")
    ap.add_argument("--duck", type=float, help="dublaj sırasında arka planın kısılacağı miktar (dB)")
    ap.add_argument("--no-reroll", action="store_true", help="perdesi sapan klipleri yeniden üretmeyi kapat")
    ap.add_argument("--no-polish", action="store_true", help="perde/seviye cilasını kapat")
    ap.add_argument("--no-fit", action="store_true", help="üst üste binmeyi önleyen sığdırmayı kapat")
    ap.add_argument("--redo-tts", action="store_true", help="çeviriyi elle düzelttikten sonra sesleri yeniden üret")
    ap.add_argument("--mix-only", action="store_true", help="TTS'i atla, hazır kliplerle yalnızca mix'i yeniden yap (--test gerekir)")
    ap.add_argument("--until", choices=["translate"], help="çeviriye kadar çalış, dur (kuyruk kullanır)")
    ap.add_argument("--keep", "--test", dest="keep", action="store_true",
                    help="TEST MODU: ara dosyaları (transkript, çeviri, ses) saklar; varsayılan: işlem bitince/başarısız olunca silinir")
    return ap


def apply_cli(cfg: Settings, a: argparse.Namespace) -> Settings:
    cfg = with_pipeline(
        cfg, speakers=a.speakers, workers=a.workers, merge=a.merge, min_dur=a.min_dur, expressive=a.expressive,
        bleed=a.bleed, duck=a.duck, fit=False if a.no_fit else None, polish=False if a.no_polish else None,
        reroll=False if a.no_reroll else None, keep=True if a.keep else None)
    pl = cfg.pipeline
    if pl.speakers < 1:
        raise TonguedubError("--speakers en az 1 olmalı")
    if not 0 <= pl.expressive <= 1:
        raise TonguedubError("--expressive 0 ile 1 arasında olmalı")
    if not 0 <= pl.bleed <= 1:
        raise TonguedubError("--bleed 0 ile 1 arasında olmalı")
    if pl.merge < 0 or pl.min_dur < 0 or pl.duck < 0:
        raise TonguedubError("--merge, --min-dur ve --duck negatif olamaz")
    resolve_workers(pl.workers)  # geçersiz değeri başlamadan yakala
    return cfg


def _terminate(signum, _frame):
    raise SystemExit(128 + signum)  # try/finally çalışsın: alt süreçler durdurulur, ara dosyalar silinir


def cmd_run(argv: list[str]) -> int:
    from .pipeline import run_video
    a = build_run_parser().parse_args(argv)
    cfg = apply_cli(load_settings(a.config), a)
    signal.signal(signal.SIGTERM, _terminate)
    run_video(cfg, Path(a.video), start=a.start, end=a.end, redo_tts=a.redo_tts, until=a.until,
              out=Path(a.out) if a.out else None, mix_only=a.mix_only)
    return 0


def cmd_worker(argv: list[str]) -> int:
    """Gizli: ana sürecin başlattığı ek TTS işçisi."""
    from .tts import tts_worker
    ap = argparse.ArgumentParser(prog="tonguedub _tts-worker")
    ap.add_argument("video")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--merge", type=float, default=0.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--redo-tts", action="store_true")
    a = ap.parse_args(argv)
    cfg = with_pipeline(load_settings(), merge=a.merge, min_dur=a.min_dur)
    signal.signal(signal.SIGTERM, _terminate)
    tts_worker(cfg, Path(a.video), a.k, a.redo_tts, a.parent_pid)
    return 0


def cmd_add(argv: list[str]) -> int:
    from .scheduler import Queue
    videos = []
    while argv and not argv[0].startswith("--"):
        videos.append(argv.pop(0))
    if not videos:
        raise TonguedubError("kullanım: tonguedub add video.mp4 [seçenekler]")
    for v in videos:  # seçenek yazım hatalarını kuyruğa eklerken yakala, işte değil
        build_run_parser().parse_args([v] + argv)
    Queue(load_settings()).add(videos, argv)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(HELP)
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"tonguedub {__version__}")
        return 0
    # `tonguedub video.mp4 ...` (alt komutsuz) = `tonguedub run video.mp4 ...`
    cmd = argv.pop(0) if (argv[0] in COMMANDS or argv[0] == "_tts-worker") else "run"
    try:
        if cmd == "run":
            return cmd_run(argv)
        if cmd == "_tts-worker":
            return cmd_worker(argv)
        if cmd == "add":
            return cmd_add(argv)
        if cmd == "doctor":
            from .doctor import cmd_doctor
            return cmd_doctor(load_settings())
        from .scheduler import Queue
        q = Queue(load_settings())
        if cmd == "queue":
            q.run_scheduler()
        elif cmd == "status":
            q.status()
        elif cmd == "clear":
            q.clear()
        elif cmd == "clean":
            q.clean()
        return 0
    except TonguedubError as e:
        print(f"Hata: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nKesildi.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
