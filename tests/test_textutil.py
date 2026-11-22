from tonguedub.textutil import REFUSAL, clean_desc, drop_descriptions, syllables


def test_clean_desc_removes_sound_event_descriptions():
    assert clean_desc("(laughs) Hello there.") == "Hello there."
    assert clean_desc("*cough*") == ""
    assert clean_desc("- Merhaba [müzik] dünya ♪ la la ♪") == "Merhaba dünya"


def test_drop_descriptions_keeps_timing_of_spoken_words():
    words = [{"word": " (laughs"}, {"word": ")"}, {"word": " ok"}]
    assert [w["word"] for w in drop_descriptions(words)] == [" ok"]


def test_refusal_does_not_match_normal_translations():
    assert not REFUSAL.search("- Üzgünüm.")
    assert not REFUSAL.search("Bunu sana yardımcı olmak için söylüyorum")  # 'yardımcı olamam' değil
    assert REFUSAL.search("Bunu çeviremem.")
    assert REFUSAL.search("I can't help with that")


def test_syllable_count_is_vowel_count():
    assert syllables("Hadi ama bana yardım et") == 9
    assert syllables("") == 0
