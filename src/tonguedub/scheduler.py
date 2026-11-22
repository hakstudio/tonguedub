"""Kuyruk: videoları sıraya alır, kaynakları (bellek/GPU/CPU) boş bırakmadan boru hattı gibi çalıştırır.

  tonguedub add video.mp4 [seçenekler]   # sıraya ekle (çalışan kuyruk yoksa başlatır)
  tonguedub status                       # durum tablosu
  tonguedub clear                        # biten/başarısız kuyruk kayıtlarını temizle
  tonguedub clean                        # sahipsiz çalışma klasörlerini ve eski logları sil (aktif işlere dokunmaz)

Aşamalar (her video için): hazırlık (demucs+Whisper+çeviri: CPU/nöral motor/ağ) -> TTS+mix (GPU).
Kurallar:
  * aynı anda en fazla 1 hazırlık ve 1 TTS işi
  * TTS süreç sayısı: sırada başka video yoksa tam kapasite (RAM'e göre auto: 24 GB -> 3), varsa bir eksik
  * hazırlık, TTS sürerken yalnızca bellek yeterince boşsa başlar (boş RAM >= %30)
  * kuyruk dışında çalışan başka bir tonguedub işi varsa TTS başlatılmaz (belleği zorlamamak için)
Ortam: TONGUEDUB_FAKE=1 -> gerçek iş yerine kısa 'sleep' çalıştırır (mantık testi için).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .config import Settings, expand, max_workers, private_dir, work_path
from .errors import TonguedubError

FAKE = bool(os.environ.get("TONGUEDUB_FAKE"))
MEM_FREE_FOR_PREP = 30  # TTS sürerken hazırlık için gereken boş RAM yüzdesi
ACTIVE_STATES = ("queued", "prep", "prepped", "tts")
# Çalışan bir tonguedub iş sürecini (run / _tts-worker) yönetim komutlarından ayırır.
JOB_PROC = re.compile(r"-m tonguedub(?!\s+(queue|status|add|clear|clean|doctor)\b)")


class Queue:
    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.root = expand(cfg.paths.queue_dir)
        self.jobs = self.root / "jobs"
        self.logs = self.root / "logs"
        self.pidf = self.root / "scheduler.pid"
        self.events = self.root / "events.log"

    # ---------- durum dosyaları ----------
    def ensure_dirs(self) -> None:
        private_dir(self.root)
        for d in (self.jobs, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, job: dict) -> None:
        p = self.jobs / f"{job['id']}.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1))
        os.replace(tmp, p)

    def load_jobs(self) -> list[dict]:
        out = []
        for p in sorted(self.jobs.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                pass
        return out

    def event(self, msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self.events, "a") as f:
            f.write(line + "\n")

    def scheduler_running(self) -> bool:
        try:
            return alive(int(self.pidf.read_text()))
        except Exception:
            return False

    # ---------- süreç başlatma ----------
    def build_cmd(self, job: dict, stage: str, workers: int | None = None) -> list[str]:
        opts = list(job["opts"])
        if FAKE:
            return ["sleep", "4" if stage == "prep" else "8"]
        base = [sys.executable, "-m", "tonguedub", "run", job["video"]]
        if stage == "prep":
            return base + ["--until", "translate"] + opts
        if workers and "--workers" not in opts:
            opts += ["--workers", str(workers)]
        return ["caffeinate", "-i"] + base + opts

    def start(self, job: dict, stage: str, workers: int | None = None) -> subprocess.Popen:
        log = open(self.logs / f"{job['id']}.{stage}.log", "w")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        p = subprocess.Popen(self.build_cmd(job, stage, workers), stdout=log, stderr=subprocess.STDOUT, env=env)
        job.update(state=stage, pid=p.pid, **{f"{stage}_t0": time.time()}, workers=workers)
        self.save(job)
        self.event(f"{Path(job['video']).name}: {'hazırlık' if stage == 'prep' else f'TTS+mix ({workers} süreç)'} başladı")
        return p

    # ---------- zamanlayıcı ----------
    def run_scheduler(self) -> None:
        self.ensure_dirs()
        self.pidf.write_text(str(os.getpid()))
        self.prune_logs()
        procs: dict[str, subprocess.Popen] = {}
        for j in self.load_jobs():  # yeniden başlatma: sahipsiz kalan durumları geri al
            if j["state"] in ("prep", "tts") and not alive(j.get("pid", -1)):
                j["state"] = "queued" if j["state"] == "prep" else "prepped"
                self.save(j)
        idle_since = None
        self.event("kuyruk başladı")
        try:
            while True:
                for j in self.load_jobs():  # biten süreçleri işle
                    p = procs.get(j["id"])
                    if p is not None and p.poll() is not None:
                        rc = p.returncode
                        procs.pop(j["id"])
                        dur = (time.time() - j.get(f"{j['state']}_t0", time.time())) / 60
                        name = Path(j["video"]).name
                        if rc != 0:
                            j["state"] = "failed"
                            self.event(f"{name}: HATA (kod {rc}) -> {self.logs}/{j['id']}.*.log")
                        elif j["state"] == "prep":
                            j["state"] = "prepped"
                            j["prep_min"] = round(dur, 1)
                            self.event(f"{name}: hazırlık bitti ({dur:.1f} dk)")
                        else:
                            j["state"] = "done"
                            j["tts_min"] = round(dur, 1)
                            self.event(f"{name}: TAMAMLANDI (TTS+mix {dur:.1f} dk)")
                        self.save(j)
                jobs = self.load_jobs()

                def by(s, jobs=jobs):
                    return [j for j in jobs if j["state"] == s]

                my_pids = [p.pid for p in procs.values()]
                if not by("tts") and by("prepped") and not external_jobs(my_pids):  # GPU boşsa TTS
                    j = by("prepped")[0]
                    waiting = len(by("queued")) + len(by("prep"))
                    w = max_workers() if waiting == 0 else max(1, max_workers() - 1)
                    procs[j["id"]] = self.start(j, "tts", w)
                jobs = self.load_jobs()  # hazırlık: sırada video varsa, hazırlık boşsa ve bellek yetiyorsa
                if not by("prep") and by("queued"):
                    if not by("tts") or free_pct() >= MEM_FREE_FOR_PREP:
                        j = by("queued")[0]
                        procs[j["id"]] = self.start(j, "prep")
                if any(j["state"] in ACTIVE_STATES for j in jobs):
                    idle_since = None
                else:
                    idle_since = idle_since or time.time()
                    if time.time() - idle_since > 30:
                        self.event("iş kalmadı, kuyruk kapandı")
                        break
                time.sleep(1 if FAKE else 3)
        finally:
            self.pidf.unlink(missing_ok=True)

    # ---------- komutlar ----------
    def add(self, videos: list[str], opts: list[str]) -> None:
        self.ensure_dirs()
        for v in videos:
            vp = Path(v).expanduser().resolve()
            if not vp.is_file():
                raise TonguedubError(f"dosya yok: {vp}")
            jid = time.strftime("%Y%m%d-%H%M%S") + f"-{len(list(self.jobs.glob('*.json'))):03d}"
            self.save({"id": jid, "video": str(vp), "opts": list(opts), "state": "queued", "added": time.time()})
            print(f"sıraya eklendi: {vp.name}")
        if not self.scheduler_running():
            log = open(self.root / "scheduler.log", "a")
            subprocess.Popen([sys.executable, "-m", "tonguedub", "queue"], stdout=log, stderr=log, start_new_session=True)
            print("kuyruk arka planda başlatıldı")

    def status(self) -> None:
        self.ensure_dirs()
        print(f"kuyruk: {'ÇALIŞIYOR' if self.scheduler_running() else 'kapalı'} | boş RAM %{free_pct()}")
        names = {"queued": "sırada", "prep": "hazırlık", "prepped": "TTS bekliyor", "tts": "TTS+mix", "done": "bitti", "failed": "HATA"}
        for j in self.load_jobs():
            extra = ""
            if j["state"] == "tts":
                extra = f" ({j.get('workers')} süreç)"
            if j["state"] == "done":
                extra = f" (hazırlık {j.get('prep_min', '?')} dk, TTS+mix {j.get('tts_min', '?')} dk)"
            print(f"  {Path(j['video']).name:28} {names[j['state']]}{extra}")

    def prune_logs(self, keep_days: int = 7) -> None:
        """Eski kuyruk günlüklerini (ilerleme çıktıları) siler; yeni olanlar kalır."""
        cut = time.time() - keep_days * 86400
        for f in self.logs.glob("*.log"):
            if f.stat().st_mtime < cut:
                f.unlink(missing_ok=True)

    def clean(self) -> None:
        """Sahipsiz çalışma klasörlerini (cache/ara dosyalar) ve eski logları temizler; aktif işlere dokunmaz."""
        self.ensure_dirs()
        sched_pids = []
        try:
            sched_pids = [int(self.pidf.read_text())] if self.scheduler_running() else []
        except Exception:
            pass
        if external_jobs(sched_pids):
            print("  çalışan bir tonguedub işi var; çalışma klasörlerine dokunulmadı (iş bitince tekrar dene).")
            self.prune_logs()
            return
        active = {work_path(self.cfg, Path(j["video"])).name for j in self.load_jobs() if j["state"] in ACTIVE_STATES}
        wd = expand(self.cfg.paths.work_dir)
        freed = 0
        for d in sorted(wd.iterdir()) if wd.is_dir() else []:
            if not d.is_dir():
                continue
            if d.name in active:
                print(f"  atlandı (aktif): {d.name}")
                continue
            sz = dir_size(d)
            shutil.rmtree(d, ignore_errors=True)
            freed += sz
            print(f"  silindi: {d.name} ({sz / 2**20:.0f} MB)")
        self.prune_logs()
        print(f"toplam {freed / 2**30:.2f} GB temizlendi")

    def clear(self) -> None:
        for j in self.load_jobs():
            if j["state"] in ("done", "failed"):
                (self.jobs / f"{j['id']}.json").unlink(missing_ok=True)
        print("temizlendi")


# ---------- yardımcılar ----------
def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def free_pct() -> int:
    if FAKE:
        return 90
    try:
        out = subprocess.check_output(["memory_pressure"], text=True)
        return int(re.search(r"free percentage: (\d+)%", out).group(1))
    except Exception:
        return 50


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def process_table() -> list[tuple[int, int, str]]:
    rows = []
    for line in subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True).splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if m:
            rows.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return rows


def external_jobs(my_pids: list[int]) -> bool:
    """Kuyruğun başlatmadığı, çalışan bir tonguedub iş süreci (run/_tts-worker) var mı?"""
    if FAKE:
        return False
    rows = process_table()
    mine = set(my_pids) | {os.getpid()}
    grew = True
    while grew:
        grew = False
        for pid, ppid, _ in rows:
            if ppid in mine and pid not in mine:
                mine.add(pid)
                grew = True
    return any(JOB_PROC.search(c) and pid not in mine for pid, _, c in rows)
