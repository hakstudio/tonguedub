"""Aşama 2: WhisperKit ile kelime zamanlı transkripsiyon ve cümlelere bölme."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .audio import tail
from .config import Settings, expand
from .errors import TonguedubError
from .textutil import drop_descriptions


def whisper_command(cfg: Settings, audio: Path, report_dir: Path) -> list[str]:
    t = cfg.transcribe
    cmd = [t.whisperkit_cli, "transcribe", "--audio-path", str(audio), "--language", t.language,
           "--word-timestamps", "--report", "--report-path", str(report_dir)]
    if t.model_path:
        mp = expand(t.model_path)
        if not mp.is_dir():
            raise TonguedubError(f"Whisper model klasörü yok: {mp}\nconfig.toml [transcribe] model_path ayarını düzelt "
                              "ya da boş bırak (model ilk çalıştırmada indirilir).")
        cmd += ["--model-path", str(mp)]
    else:
        cmd += ["--model", t.model, "--download-model-path", str(expand(t.model_download_dir))]
    return cmd


def group_sentences(words: list[dict]) -> list[dict]:
    """Kelimeleri duraklama (0,8 sn), uzunluk (12 sn) ve noktalamaya göre cümlelere böler."""
    segs, cur = [], []

    def flush():
        if not cur:
            return
        text = "".join(w["word"] for w in cur).strip()
        if re.fullmatch(r"[\-\s]*[\*\[\(].*[\*\]\)][\s\.]*", text):
            pass  # *...*, [...], (...): konuşma değil, ses olayı betimlemesi; dublajlama
        elif len(re.sub(r"[^A-Za-z]", "", text)) >= 2:
            segs.append({"start": round(cur[0]["start"], 2), "end": round(cur[-1]["end"], 2), "en": text})
        cur.clear()

    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > 0.8 or w["end"] - cur[0]["start"] > 12):
            flush()
        cur.append(w)
        if re.search(r"[.?!]$", w["word"].strip()):
            flush()
    flush()
    return segs


def stage_transcribe(cfg: Settings, audio: Path, W: Path) -> list[dict]:
    f = W / "segments.json"
    if f.exists():
        return json.loads(f.read_text())
    print("[2/5] Transkripsiyon (whisperkit)...")
    rep = W / "wk"
    rep.mkdir(exist_ok=True)
    cmd = whisper_command(cfg, audio, rep)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)  # stdout transkript içerebilir: gösterilmez
    except FileNotFoundError as e:
        raise TonguedubError(f"'{cfg.transcribe.whisperkit_cli}' bulunamadı. Kur: brew install whisperkit-cli") from e
    if p.returncode != 0:
        raise TonguedubError(f"whisperkit-cli başarısız oldu (kod {p.returncode}):\n{tail(p.stderr)}")
    reports = list(rep.glob("*.json"))
    if not reports:
        raise TonguedubError("whisperkit-cli rapor üretmedi (ses boş ya da model yüklenemedi olabilir).")
    d = json.loads(reports[0].read_text())
    words = []
    for sg in d["segments"]:
        if sg.get("noSpeechProb", 0) > 0.6 or sg.get("avgLogprob", 0) < -1.3:
            continue  # konuşma dışı seslerde Whisper'ın uydurma ürettiği yerler
        words.extend(sg.get("words", []))
    segs = group_sentences(drop_descriptions(words))
    f.write_text(json.dumps(segs, ensure_ascii=False, indent=1))
    return segs
