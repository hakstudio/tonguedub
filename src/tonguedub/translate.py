"""Aşama 3: DeepSeek ile bağlamlı, süreye duyarlı çeviri + hece tabanlı kısaltma."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from .config import Settings, resolve_api_key
from .constants import SOFT_RATE, TR_CHARS_PER_SEC, TTS_SYL_RATE
from .errors import TonguedubError
from .textutil import REFUSAL, clean_desc, syllables

SYS = ("Sen profesyonel bir EN->TR dublaj çevirmenisin. "
       "Diyaloğu olduğu gibi, sadık ve doğal, günlük konuşma Türkçesiyle çevir; yumuşatma, ekleme ya da çıkarma yapma. "
       "Her satırın Türkçesi konuşulduğunda orijinal süreye sığmalı: verilen max_karakter sınırını aşma, "
       "gerekirse kısalt. Çeviri yapılmış bir senaryo gibi değil, ana dili Türkçe olan biri gibi konuş. "
       "Cevap SADECE JSON dizisi olsun: [{\"i\":<id>,\"tr\":\"...\"}, ...]. Açıklama yazma.")
SYS_SHORT = ("Sen dublaj için Türkçe cümleleri kısaltan bir editörsün. Anlamı, tonu ve söyleyiş düzeyini "
             "olduğu gibi koru, yumuşatma. Cümleleri konuşma diliyle, verilen max_hece sınırına sığacak "
             "şekilde kısalt. Zaten sığıyorsa aynen bırak. Cevap SADECE JSON dizisi: [{\"i\":<id>,\"tr\":\"...\"}]")

MAX_BAD_RATIO = 0.5  # satırların bundan fazlası çevrilemediyse çıktı kullanılamaz: dur


def ask(cfg: Settings, msgs: list[dict], model: str | None = None) -> str:
    t = cfg.translate
    r = requests.post(
        f"{t.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {resolve_api_key(cfg)}"},
        json={"model": model or t.model, "messages": msgs, "temperature": t.temperature,
              "thinking": {"type": "disabled"}},
        timeout=t.timeout)
    if r.status_code in (401, 402, 403):
        raise TonguedubError(f"Çeviri API'si isteği reddetti (HTTP {r.status_code}): API anahtarı, bakiye veya yetki sorunu. "
                          "config.toml [translate] ayarlarını kontrol et.")
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def translate_chunk(cfg: Settings, segs: list[dict], k: int, B: int) -> list[str]:
    chunk = segs[k:k + B]
    prev = "\n".join(s["en"] for s in segs[max(0, k - 4):k])
    nxt = "\n".join(s["en"] for s in segs[k + B:k + B + 2])
    items = [{"i": k + j, "en": s["en"],
              "max_karakter": max(12, int((s["end"] - s["start"]) * TR_CHARS_PER_SEC))}
             for j, s in enumerate(chunk)]
    user = (f"Önceki cümleler (bağlam, çevirme):\n{prev}\n\n" if prev else "") + \
           (f"Sonraki cümleler (bağlam, çevirme):\n{nxt}\n\n" if nxt else "") + \
           "Çevrilecek:\n" + json.dumps(items, ensure_ascii=False)
    fb = cfg.translate.fallback_model
    for attempt in range(5):
        try:
            m = fb if (attempt >= 3 and fb) else None  # son 2 denemede reddedilirse hızlı modele düş
            txt = ask(cfg, [{"role": "system", "content": SYS}, {"role": "user", "content": user}], m)
            arr = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
            if len(arr) == len(chunk):
                out = [a["tr"].strip() for a in arr]
                if not any(REFUSAL.search(x) for x in out):
                    return out
        except TonguedubError:
            raise  # kimlik/bakiye hatası: tekrar denemenin anlamı yok
        except Exception:
            pass
    return [""] * len(chunk)


def shorten_pass(cfg: Settings, done: list[dict]) -> list[dict]:
    """Zamana sığmayacak satırları (hece tahminine göre) LLM'e kısalttır. Türkçe İngilizceden uzun olduğu için
    taşma/hızlandırma/kesme sorununun kaynağında çözer."""
    min_dur = cfg.pipeline.min_dur
    flagged = []
    for i, s in enumerate(done):
        d = s["end"] - s["start"]
        if d < min_dur or not s["tr"]:
            continue
        nxt = done[i + 1]["start"] if i + 1 < len(done) else s["end"] + 2
        slot = max(d, min(nxt - s["start"] - 0.05, d + 0.6))
        n = syllables(s["tr"])
        if n >= 4 and n / TTS_SYL_RATE > slot * SOFT_RATE * 1.1:
            flagged.append((i, max(3, int(slot * TTS_SYL_RATE * SOFT_RATE))))
    if not flagged:
        print("  süre kontrolü: kısaltılacak satır yok", flush=True)
        return done

    def one(batch):
        items = [{"i": i, "tr": done[i]["tr"], "en": done[i]["en"], "max_hece": mh} for i, mh in batch]
        user = "Kısalt:\n" + json.dumps(items, ensure_ascii=False)
        for _ in range(3):
            try:
                txt = ask(cfg, [{"role": "system", "content": SYS_SHORT}, {"role": "user", "content": user}])
                arr = json.loads(re.search(r"\[.*\]", txt, re.S).group(0))
                return {a["i"]: clean_desc(a["tr"]) for a in arr}
            except TonguedubError:
                raise
            except Exception:
                pass
        return {}

    batches = [flagged[k:k + 12] for k in range(0, len(flagged), 12)]
    with ThreadPoolExecutor(min(8, len(batches))) as ex:
        res = list(ex.map(one, batches))
    n = 0
    for r in res:
        for i, t in r.items():
            if isinstance(i, int) and 0 <= i < len(done) and t and 0 < syllables(t) < syllables(done[i]["tr"]):
                done[i]["tr_uzun"] = done[i]["tr"]
                done[i]["tr"] = t
                n += 1
    print(f"  süre kontrolü: {len(flagged)} satır sığmıyordu, {n} tanesi kısaltıldı", flush=True)
    return done


def stage_translate(cfg: Settings, segs: list[dict], W: Path) -> list[dict]:
    f = W / "translated.json"
    if f.exists():
        return json.loads(f.read_text())
    print("[3/5] Çeviri (DeepSeek API)...")
    resolve_api_key(cfg)  # anahtar yoksa ağ işlemlerinden önce, açık mesajla dur
    B = cfg.translate.batch_size
    starts = list(range(0, len(segs), B))
    with ThreadPoolExecutor(min(cfg.translate.max_threads, max(1, len(starts)))) as ex:
        res = list(ex.map(lambda k: translate_chunk(cfg, segs, k, B), starts))
    done, bad = [], 0
    for k, tr in zip(starts, res):
        for j, t in enumerate(tr):
            s = dict(segs[k + j])
            s["tr"] = clean_desc(t)
            bad += (s["tr"] == "")
            done.append(s)
    print(f"  {len(done)} satır çevrildi, {bad} satır çevrilemedi (orijinal ses kalır)")
    if done and bad / len(done) > MAX_BAD_RATIO:
        raise TonguedubError(f"Satırların %{bad / len(done) * 100:.0f}'i çevrilemedi; ağ, API anahtarı veya model ayarını kontrol et.")
    done = shorten_pass(cfg, done)
    f.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    return done
