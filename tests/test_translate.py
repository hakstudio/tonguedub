import json

import pytest

from tonguedub import translate
from tonguedub.config import Settings
from tonguedub.errors import TonguedubError


def segs(n=4):
    return [{"start": i * 2.0, "end": i * 2.0 + 1.5, "en": f"line {i}"} for i in range(n)]


def reply(items):
    return json.dumps([{"i": i, "tr": t} for i, t in enumerate(items)])


def test_translate_chunk_returns_parsed_lines(monkeypatch):
    monkeypatch.setattr(translate, "ask", lambda cfg, msgs, model=None: reply(["bir", "iki"]))
    assert translate.translate_chunk(Settings(), segs(2), 0, 12) == ["bir", "iki"]


def test_refusal_in_reply_triggers_retry_with_fallback_model(monkeypatch):
    models = []

    def fake_ask(cfg, msgs, model=None):
        models.append(model)
        return reply(["Bunu çeviremem.", "iki"]) if len(models) < 4 else reply(["bir", "iki"])

    monkeypatch.setattr(translate, "ask", fake_ask)
    assert translate.translate_chunk(Settings(), segs(2), 0, 12) == ["bir", "iki"]
    assert models[:3] == [None, None, None] and models[3] == "deepseek-flash"


def test_auth_error_is_not_swallowed(monkeypatch):
    def boom(cfg, msgs, model=None):
        raise TonguedubError("401")

    monkeypatch.setattr(translate, "ask", boom)
    with pytest.raises(TonguedubError):
        translate.translate_chunk(Settings(), segs(2), 0, 12)


def test_stage_translate_aborts_when_most_lines_fail(monkeypatch, tmp_path):
    cfg = Settings()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(translate, "translate_chunk", lambda cfg, segs, k, B: [""] * len(segs[k:k + B]))
    with pytest.raises(TonguedubError, match="çevrilemedi"):
        translate.stage_translate(cfg, segs(6), tmp_path)
    assert not (tmp_path / "translated.json").exists()  # bozuk çıktı önbelleğe yazılmamalı
