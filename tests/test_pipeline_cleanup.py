from pathlib import Path

import pytest

from tonguedub import pipeline
from tonguedub.cli import apply_cli, build_run_parser
from tonguedub.config import Settings, work_path
from tonguedub.errors import TonguedubError
from tonguedub.scheduler import JOB_PROC


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = Settings()
    cfg.paths.work_dir = str(tmp_path / "work")
    cfg.paths.log_dir = str(tmp_path / "logs")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    monkeypatch.setattr(pipeline, "preflight", lambda *a, **k: None)
    return cfg, video


def _fake_full(side_effect=None):
    def full(cfg, video, W, *a, **k):
        (W / "segments.json").write_text('[{"en": "secret transcript"}]')
        if side_effect:
            raise side_effect
    return full


def test_work_dir_is_removed_after_failure(setup, monkeypatch):
    cfg, video = setup
    monkeypatch.setattr(pipeline, "_full", _fake_full(TonguedubError("boom")))
    with pytest.raises(TonguedubError):
        pipeline.run_video(cfg, video)
    assert not work_path(cfg, video).exists()


def test_work_dir_is_removed_after_interrupt(setup, monkeypatch):
    cfg, video = setup
    monkeypatch.setattr(pipeline, "_full", _fake_full(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_video(cfg, video)
    assert not work_path(cfg, video).exists()


def test_work_dir_is_removed_after_success(setup, monkeypatch):
    cfg, video = setup
    monkeypatch.setattr(pipeline, "_full", _fake_full())
    pipeline.run_video(cfg, video)
    assert not work_path(cfg, video).exists()


def test_keep_flag_preserves_work_dir_even_on_failure(setup, monkeypatch):
    cfg, video = setup
    cfg.pipeline.keep = True
    monkeypatch.setattr(pipeline, "_full", _fake_full(TonguedubError("boom")))
    with pytest.raises(TonguedubError):
        pipeline.run_video(cfg, video)
    assert (work_path(cfg, video) / "segments.json").exists()


def test_until_translate_keeps_work_dir_for_the_queue(setup, monkeypatch):
    cfg, video = setup
    monkeypatch.setattr(pipeline, "_full", _fake_full())
    pipeline.run_video(cfg, video, until="translate")
    assert (work_path(cfg, video) / "segments.json").exists()


def test_until_translate_failure_still_cleans_up(setup, monkeypatch):
    cfg, video = setup
    monkeypatch.setattr(pipeline, "_full", _fake_full(TonguedubError("boom")))
    with pytest.raises(TonguedubError):
        pipeline.run_video(cfg, video, until="translate")
    assert not work_path(cfg, video).exists()


def test_failed_run_writes_log_without_transcript(setup, monkeypatch):
    cfg, video = setup

    def full(cfg, video, W, *a, **k):
        print("  [2/5] bir şey yapıldı")
        raise TonguedubError("boom")

    monkeypatch.setattr(pipeline, "_full", full)
    with pytest.raises(TonguedubError):
        pipeline.run_video(cfg, video)
    logs = list(Path(cfg.paths.log_dir).glob("*_FAILED.log"))
    assert len(logs) == 1 and "secret transcript" not in logs[0].read_text()


def test_missing_video_and_same_output_are_clear_errors(setup):
    cfg, video = setup
    with pytest.raises(TonguedubError, match="Dosya yok"):
        pipeline.run_video(cfg, video.with_name("nope.mp4"))
    with pytest.raises(TonguedubError, match="aynı olamaz"):
        pipeline.run_video(cfg, video, out=video)


def test_job_process_pattern_separates_jobs_from_management_commands():
    job = "caffeinate -i /x/.venv/bin/python -m tonguedub run /v/a.mp4 --workers 3"
    worker = "/x/.venv/bin/python -m tonguedub _tts-worker /v/a.mp4 --k 1"
    shim = "/x/.venv/bin/python -m tonguedub /v/a.mp4"
    assert all(JOB_PROC.search(c) for c in (job, worker, shim))
    for c in ("python -m tonguedub queue", "python -m tonguedub status", "python -m tonguedub clean", "python -m tonguedub doctor",
              "python -m tonguedub add /v/a.mp4"):
        assert not JOB_PROC.search(c)


def test_cli_validates_ranges_and_applies_overrides():
    a = build_run_parser().parse_args(["v.mp4", "--merge", "20", "--no-fit", "--test"])
    cfg = apply_cli(Settings(), a)
    assert cfg.pipeline.merge == 20.0 and cfg.pipeline.fit is False and cfg.pipeline.keep is True
    with pytest.raises(TonguedubError):
        apply_cli(Settings(), build_run_parser().parse_args(["v.mp4", "--expressive", "2"]))
    with pytest.raises(TonguedubError):
        apply_cli(Settings(), build_run_parser().parse_args(["v.mp4", "--workers", "lots"]))
