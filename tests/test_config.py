from pathlib import Path

import pytest

from tonguedub import config as config_mod
from tonguedub.config import (ENV_API_KEY, Settings, load_dotenv, load_settings, parse_dotenv,
                           resolve_api_key, resolve_workers, with_pipeline, work_path)
from tonguedub.errors import TonguedubError

ROOT = Path(__file__).resolve().parents[1]


def write(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text)
    return str(p)


def test_example_config_loads_and_matches_defaults():
    """config.example.toml her zaman geçerli olmalı ve kod varsayılanlarıyla aynı değerleri taşımalı."""
    cfg = load_settings(str(ROOT / "config.example.toml"))
    assert cfg.pipeline == Settings().pipeline
    assert cfg.translate == Settings().translate
    assert cfg.transcribe == Settings().transcribe
    assert cfg.tts == Settings().tts


def test_values_are_read(tmp_path):
    cfg = load_settings(write(tmp_path, '[pipeline]\nspeakers = 3\nmerge = 20\nfit = false\n[translate]\nmodel = "x"\n'))
    assert (cfg.pipeline.speakers, cfg.pipeline.merge, cfg.pipeline.fit, cfg.translate.model) == (3, 20.0, False, "x")


def test_unknown_key_and_section_are_rejected(tmp_path):
    with pytest.raises(TonguedubError, match="bilinmeyen ayar"):
        load_settings(write(tmp_path, "[pipeline]\nspeekers = 3\n"))
    with pytest.raises(TonguedubError, match="bilinmeyen bölüm"):
        load_settings(write(tmp_path, "[pipelin]\nspeakers = 3\n"))


def test_wrong_type_is_rejected(tmp_path):
    with pytest.raises(TonguedubError, match="geçersiz değer"):
        load_settings(write(tmp_path, '[pipeline]\nfit = "yes"\n'))


def test_broken_toml_gives_clear_error(tmp_path):
    with pytest.raises(TonguedubError, match="TOML"):
        load_settings(write(tmp_path, "[pipeline\n"))


def test_missing_explicit_config_is_an_error(tmp_path):
    with pytest.raises(TonguedubError, match="yok"):
        load_settings(str(tmp_path / "nope.toml"))


def test_api_key_comes_from_environment(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "  from-env \n")
    assert resolve_api_key(Settings()) == "from-env"


def test_missing_api_key_raises_actionable_error():
    with pytest.raises(TonguedubError, match=r"\.env"):
        resolve_api_key(Settings())


def test_secrets_are_not_accepted_in_config_toml(tmp_path):
    """API anahtarı config.toml'a yazılamaz (yanlışlıkla paylaşılmasın); .env kullanılmalı."""
    with pytest.raises(TonguedubError, match="bilinmeyen ayar"):
        load_settings(write(tmp_path, '[translate]\napi_key = "dummy-value"\n'))


def test_parse_dotenv_handles_comments_quotes_and_export():
    text = "# yorum\nA=1\nexport B=\"iki kelime\"\nC='x y'\nD=val # sonda yorum\nE=\nbozuk satır\n"
    assert parse_dotenv(text) == {"A": "1", "B": "iki kelime", "C": "x y", "D": "val", "E": ""}


def test_dotenv_loads_key_but_never_overrides_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=from-file\nOTHER_VAR=ok\n")
    monkeypatch.setattr(config_mod, "env_files", lambda: [env])
    monkeypatch.delenv("OTHER_VAR", raising=False)
    assert load_dotenv() == [env]
    assert resolve_api_key(Settings()) == "from-file"
    monkeypatch.setenv(ENV_API_KEY, "from-shell")
    load_dotenv()
    assert resolve_api_key(Settings()) == "from-shell"


def test_cli_overrides_only_apply_when_given():
    cfg = with_pipeline(Settings(), merge=None, min_dur=2.0)
    assert cfg.pipeline.merge == 0.0 and cfg.pipeline.min_dur == 2.0


def test_workers_validation():
    assert resolve_workers("2") == 2
    assert resolve_workers(0) == 1
    assert resolve_workers("auto") >= 1
    with pytest.raises(TonguedubError):
        resolve_workers("lots")


def test_work_path_differs_for_same_name_in_different_dirs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cfg = Settings()
    assert work_path(cfg, a / "v.mp4") != work_path(cfg, b / "v.mp4")
    assert work_path(cfg, a / "v.mp4") == work_path(cfg, a / "v.mp4")
