"""Ortam kontrolü: `tonguedub doctor` ve her çalıştırmadan önce yapılan hızlı ön kontrol."""
from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Settings, env_files, expand, resolve_api_key
from .errors import TonguedubError


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    detail: str = ""


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None or Path(cmd).expanduser().is_file()


def check_ffmpeg() -> Check:
    return Check("ffmpeg", _which("ffmpeg"), True, "" if _which("ffmpeg") else "Kur: brew install ffmpeg")


def check_whisper(cfg: Settings) -> list[Check]:
    t = cfg.transcribe
    out = [Check("whisperkit-cli", _which(t.whisperkit_cli), True,
                 "" if _which(t.whisperkit_cli) else "Kur: brew install whisperkit-cli")]
    if t.model_path:
        mp = expand(t.model_path)
        out.append(Check("whisper modeli", mp.is_dir(), True,
                         str(mp) if mp.is_dir() else f"Klasör yok: {mp} (config.toml [transcribe] model_path)"))
    else:
        out.append(Check("whisper modeli", True, False, f"ilk çalıştırmada indirilecek: {t.model} -> {expand(t.model_download_dir)}"))
    return out


def check_api_key(cfg: Settings) -> Check:
    try:
        resolve_api_key(cfg)
        return Check("çeviri API anahtarı", True, True, "bulundu")
    except TonguedubError as e:
        return Check("çeviri API anahtarı", False, True, str(e))


def check_env_permissions() -> list[Check]:
    """.env başkaları tarafından okunabiliyorsa uyarır (sır dosyası yalnızca sahibine açık olmalı)."""
    out = []
    for p in env_files():
        if p.is_file():
            open_to_others = bool(p.stat().st_mode & 0o077)
            out.append(Check(f"{p.name} izinleri", not open_to_others, False,
                             f"{p} başkalarınca okunabilir: chmod 600 {p}" if open_to_others else "yalnızca sahibine açık"))
    return out


def run_checks(cfg: Settings) -> list[Check]:
    mac_arm = sys.platform == "darwin" and platform.machine() == "arm64"
    checks = [
        Check("Apple Silicon (macOS arm64)", mac_arm, True, "" if mac_arm else f"{sys.platform}/{platform.machine()}: yalnızca Apple Silicon test edildi"),
        Check("Python 3.11", sys.version_info[:2] == (3, 11), True, platform.python_version()),
        check_ffmpeg(),
        *check_whisper(cfg),
        Check("rubberband (perde/hız kalitesi)", _which("rubberband"), False, "" if _which("rubberband") else "Önerilir: brew install rubberband (yoksa librosa kullanılır)"),
        check_api_key(cfg),
        *check_env_permissions(),
        Check("ayar dosyası", True, False, str(cfg.source) if cfg.source else "yok, varsayılanlar kullanılıyor (config.example.toml'u config.toml olarak kopyala)"),
    ]
    for mod, hint in (("torch", ""), ("demucs", ""), ("chatterbox", "scripts/install.sh çalıştır"), ("librosa", ""), ("sklearn", "")):
        ok = importlib.util.find_spec(mod) is not None
        checks.append(Check(f"python paketi: {mod}", ok, True, "" if ok else (hint or "pip install -e . ile kur")))
    if importlib.util.find_spec("torch") is not None:
        import torch
        checks.append(Check("GPU (MPS)", torch.backends.mps.is_available(), False, "" if torch.backends.mps.is_available() else "yok: TTS CPU'da çok yavaş çalışır"))
    wd = expand(cfg.paths.work_dir)
    probe = wd
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 2**30
    checks.append(Check("disk alanı", free >= 10, False, f"{free:.0f} GB boş ({probe})"))
    return checks


def cmd_doctor(cfg: Settings) -> int:
    bad = 0
    for c in run_checks(cfg):
        mark = "✓" if c.ok else ("✗" if c.required else "!")
        print(f"  {mark} {c.name}" + (f": {c.detail}" if c.detail else ""))
        bad += (not c.ok and c.required)
    print("\nHer şey hazır." if not bad else f"\n{bad} zorunlu sorun var; yukarıdakileri düzeltince tekrar çalıştır.")
    return 1 if bad else 0


def preflight(cfg: Settings, need_translate: bool) -> None:
    """Uzun işe başlamadan önce eksik araç/ayarı açık mesajla bildirir."""
    checks = [check_ffmpeg(), *check_whisper(cfg)]
    if need_translate:
        checks.append(check_api_key(cfg))
    bad = [c for c in checks if not c.ok and c.required]
    if bad:
        raise TonguedubError("Ön kontrol başarısız:\n" + "\n".join(f"  - {c.name}: {c.detail}" for c in bad)
                          + "\n(Tüm kontroller için: tonguedub doctor)")
