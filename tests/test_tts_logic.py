import numpy as np

from tonguedub.tts import plan_groups, split_by_silence

SR = 24000
_rng = np.random.default_rng(0)


def burst(sec):
    return (_rng.standard_normal(int(SR * sec)) * 0.1).astype(np.float32)


def silence(sec):
    return (_rng.standard_normal(int(SR * sec)) * 1e-4).astype(np.float32)


def test_split_by_silence_cuts_at_the_gap():
    w = np.concatenate([silence(.1), burst(1.0), silence(.3), burst(2.0), silence(.1)])
    pieces = split_by_silence(w, SR, ["a" * 14, "b" * 28])
    assert pieces and abs(len(pieces[0]) / SR - 1.0) < 0.2 and abs(len(pieces[1]) / SR - 2.0) < 0.2


def test_split_by_silence_refuses_when_no_clear_gap():
    assert split_by_silence(burst(3.0), SR, ["aaaa", "bbbb"]) is None


def test_plan_groups_covers_every_line_once_and_never_mixes_speakers():
    segs = [{"start": i, "end": i + 0.6, "tr": "Evet bu", "spk": i % 2} for i in range(10)]
    groups = plan_groups(segs, merge_sec=20, min_dur=0.0)
    assert sorted(i for g in groups for i in g) == list(range(10))
    assert all(len({segs[i]["spk"] for i in g}) == 1 for g in groups)
    assert all(len(g) <= 4 for g in groups)  # MERGE_CAP


def test_plan_groups_skips_short_and_empty_lines():
    segs = [{"start": 0, "end": 0.5, "tr": "Evet", "spk": 0},      # min_dur'dan kısa
            {"start": 1, "end": 3.0, "tr": "", "spk": 0},          # çevrilememiş
            {"start": 4, "end": 6.0, "tr": "Tamam o zaman", "spk": 0}]
    assert plan_groups(segs, merge_sec=0, min_dur=1.0) == [[2]]
