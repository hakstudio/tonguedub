"""Ayarlar: config.toml okuma, doğrulama ve varsayılanlar; sırlar (.env) yükleme.

Ayar arama sırası: --config > $TONGUEDUB_CONFIG > <proje kökü>/config.toml > ~/.config/tonguedub/config.toml.
Hiçbiri yoksa varsayılanlarla çalışır (çeviri için API anahtarı yine gerekir).
Sırlar (API anahtarı) config.toml'da DEĞİL, ortam değişkeninde ya da .env dosyasında tutulur:
<proje kökü>/.env, sonra ~/.config/tonguedub/.env. Gerçek ortam değişkeni .env'den önceliklidir.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .errors import TonguedubError

ENV_CONFIG = "TONGUEDUB_CONFIG"
ENV_API_KEY = "DEEPSEEK_API_KEY"
USER_CONFIG = Path("~/.config/tonguedub/config.toml").expanduser()
USER_ENV = Path("~/.config/tonguedub/.env").expanduser()


@dataclass
class Paths:
    work_dir: str = "~/.cache/tonguedub/work"          # ara dosyalar (transkript, çeviri, ses); işlem bitince silinir
    log_dir: str = "~/.local/state/tonguedub/logs"     # özet loglar (transkript içermez)
    queue_dir: str = "~/.local/state/tonguedub/queue"  # kuyruk durumu


@dataclass
class Transcribe:
    whisperkit_cli: str = "whisperkit-cli"
    model_path: str = ""                            # hazır model klasörü; boşsa `model` indirilir
    model: str = "large-v3-v20240930_turbo"
    model_download_dir: str = "~/.cache/tonguedub/models"
    language: str = "en"


@dataclass
class Translate:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    fallback_model: str = "deepseek-flash"          # son denemelerde (ret/bozuk yanıt) kullanılır
    temperature: float = 0.3
    batch_size: int = 12
    max_threads: int = 12
    timeout: int = 300


@dataclass
class Tts:
    device: str = "auto"                            # auto | mps | cpu
    exaggeration: float = 0.5
    cfg_weight: float = 0.5


@dataclass
class Pipeline:
    speakers: int = 2                               # yüksek perdeli konuşmacı sayısı (alçak perdeliler otomatik ayrılır)
    workers: str = "auto"                           # paralel TTS süreci: sayı ya da auto (RAM'e göre)
    merge: float = 0.0                              # kısa satırları bu süreye (sn) kadar birleştir (0 = kapalı)
    min_dur: float = 1.0                            # bundan kısa satırlar dublajlanmaz (orijinal kalır)
    expressive: float = 0.5                         # 0 = düz perde, 1 = orijinal perdeyi tam takip
    bleed: float = 0.12                             # dublaj sırasında orijinal vokalden kalan oran
    duck: float = 3.0                               # dublaj sırasında arka plan kısma (dB)
    fit: bool = True                                # üst üste binmeyi önleyen sığdırma
    polish: bool = True                             # perde/seviye cilası
    reroll: bool = True                             # perdesi sapan klipleri yeniden üret
    keep: bool = False                              # ara dosyaları sakla (test modu)


@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    transcribe: Transcribe = field(default_factory=Transcribe)
    translate: Translate = field(default_factory=Translate)
    tts: Tts = field(default_factory=Tts)
    pipeline: Pipeline = field(default_factory=Pipeline)
    source: Path | None = None                      # okunan ayar dosyası (yoksa None)


# ---------- yükleme ----------
def project_root() -> Path | None:
    """Kaynaktan (editable) kurulumda depo kökü: pyproject.toml'un olduğu klasör."""
    for p in Path(__file__).resolve().parents:
        if (p / "pyproject.toml").exists() and (p / "src" / "tonguedub").is_dir():
            return p
    return None


def find_config(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise TonguedubError(f"Ayar dosyası yok: {p}")
        return p
    env = os.environ.get(ENV_CONFIG)
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise TonguedubError(f"${ENV_CONFIG} geçersiz, dosya yok: {p}")
        return p
    root = project_root()
    for p in ([root / "config.toml"] if root else []) + [USER_CONFIG]:
        if p.is_file():
            return p
    return None


def _merge(section_obj, data: dict, where: str, path: Path) -> None:
    known = {f.name: f for f in fields(section_obj)}
    for key, val in data.items():
        if key not in known:
            raise TonguedubError(f"{path}: [{where}] bölümünde bilinmeyen ayar: '{key}' (olası yazım hatası). "
                              f"Geçerli ayarlar: {', '.join(known)}")
        cur = getattr(section_obj, key)
        if isinstance(cur, bool):
            ok = isinstance(val, bool)
        elif isinstance(cur, float):
            ok = isinstance(val, (int, float)) and not isinstance(val, bool)
            val = float(val) if ok else val
        elif isinstance(cur, int):
            ok = isinstance(val, int) and not isinstance(val, bool)
        else:  # str (workers hem sayı hem "auto" olabilir)
            ok = isinstance(val, str) or (key == "workers" and isinstance(val, int) and not isinstance(val, bool))
            val = str(val) if ok else val
        if not ok:
            raise TonguedubError(f"{path}: [{where}] {key} için geçersiz değer {val!r} (beklenen tür: {type(cur).__name__})")
        setattr(section_obj, key, val)


def env_files() -> list[Path]:
    root = project_root()
    return ([root / ".env"] if root else []) + [USER_ENV]


def parse_dotenv(text: str) -> dict[str, str]:
    """Basit .env ayrıştırıcı: ANAHTAR=değer, '#' yorumları, isteğe bağlı 'export ' ve tırnaklar."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        if key.strip():
            out[key.strip()] = val
    return out


def load_dotenv() -> list[Path]:
    """.env dosyalarını ortama yükler (var olan ortam değişkenlerini ezmez). Yüklenen dosyaları döndürür."""
    loaded = []
    for p in env_files():
        if p.is_file():
            for k, v in parse_dotenv(p.read_text(encoding="utf-8")).items():
                os.environ.setdefault(k, v)
            loaded.append(p)
    return loaded


def load_settings(explicit: str | None = None) -> Settings:
    load_dotenv()
    cfg = Settings()
    path = find_config(explicit)
    if path is None:
        return cfg
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise TonguedubError(f"{path}: TOML sözdizimi hatası: {e}") from e
    for name, body in data.items():
        if not hasattr(cfg, name) or name == "source" or not isinstance(body, dict):
            raise TonguedubError(f"{path}: bilinmeyen bölüm: [{name}]. Geçerli bölümler: paths, transcribe, translate, tts, pipeline")
        _merge(getattr(cfg, name), body, name, path)
    cfg.source = path.resolve()
    os.environ[ENV_CONFIG] = str(cfg.source)  # alt süreçler (TTS işçileri, kuyruk) aynı ayarı bulsun
    return cfg


def with_pipeline(cfg: Settings, **overrides) -> Settings:
    """CLI bayraklarıyla (None olmayanlar) pipeline ayarlarını ezer."""
    kv = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, pipeline=replace(cfg.pipeline, **kv))


# ---------- yardımcılar ----------
def expand(p: str) -> Path:
    return Path(p).expanduser()


def private_dir(p: Path) -> Path:
    """Yalnızca kullanıcıya açık, Spotlight dışı klasör oluşturur."""
    p.mkdir(parents=True, exist_ok=True)
    os.chmod(p, 0o700)
    (p / ".metadata_never_index").touch()
    return p


def work_path(cfg: Settings, video: Path) -> Path:
    """Video başına ara dosya klasörü; farklı klasörlerdeki aynı adlı videolar çakışmasın diye yol özeti eklenir."""
    video = Path(video).expanduser().resolve()
    h = hashlib.sha1(str(video).encode()).hexdigest()[:6]
    return expand(cfg.paths.work_dir) / f"{video.stem}-{h}"


def resolve_api_key(cfg: Settings) -> str:
    key = os.environ.get(ENV_API_KEY, "").strip()
    if not key:
        raise TonguedubError(f"API anahtarı bulunamadı. {ENV_API_KEY} değerini .env dosyasına yaz (şablon: .env.example) "
                          "ya da ortam değişkeni olarak tanımla.")
    return key


def total_ram_gb() -> float:
    return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30


def max_workers() -> int:
    gb = total_ram_gb()
    return 1 if gb < 16 else (2 if gb < 24 else 3)


def resolve_workers(value: str | int) -> int:
    if str(value) == "auto":
        return max_workers()
    try:
        n = int(value)
    except ValueError as e:
        raise TonguedubError(f"workers 'auto' ya da pozitif sayı olmalı (verilen: {value!r})") from e
    return max(1, n)
