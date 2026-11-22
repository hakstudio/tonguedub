"""Metin yardımcıları (ses/ağ gerektirmez; birim testleri burada)."""
from __future__ import annotations

import re

DESC_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|♪[^♪]*♪?")
VALID = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]{2}")
REFUSAL = re.compile(r"(çeviremem|yardımcı olamam|I can't (help|assist|translate)|I cannot (help|assist|translate))", re.I)
TURK_V = set("aeıioöuüAEIİOÖUÜ")


def clean_desc(text: str) -> str:
    """[...], (...), *...*, ♪ biçimli ses olayı betimlemelerini ve baştaki '-' işaretini temizler."""
    t = DESC_RE.sub(" ", text)
    return re.sub(r"\s+", " ", t).strip(" -–")


def drop_descriptions(words: list[dict]) -> list[dict]:
    """Kelime düzeyinde: açılış ([ ( * ♪) ile kapanış arasındaki kelimeleri at (zamanlama korunur)."""
    out, inside = [], False
    op, cl = re.compile(r"[\(\[\*♪]"), re.compile(r"[\)\]\*♪]")
    for w in words:
        t = w["word"]
        if inside:
            if cl.search(t):
                inside = False
            continue
        m = op.search(t)
        if m:
            if not cl.search(t[m.end():]):
                inside = True
            continue
        out.append(w)
    return out


def syllables(t: str) -> int:
    """Türkçe'de hece sayısı = ünlü sayısı (yazım ses uyumlu)."""
    return sum(c in TURK_V for c in t)
