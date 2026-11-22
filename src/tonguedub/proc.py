"""Alt süreç yönetimi: hata/Ctrl+C durumunda arkada süreç ve ara dosya bırakmamak için."""
from __future__ import annotations

import subprocess
import time


class ChildProcs:
    def __init__(self) -> None:
        self._procs: list[subprocess.Popen] = []

    def add(self, p: subprocess.Popen) -> subprocess.Popen:
        self._procs.append(p)
        return p

    def terminate_all(self, grace: float = 5.0) -> None:
        live = [p for p in self._procs if p.poll() is None]
        for p in live:
            p.terminate()
        t0 = time.time()
        while live and time.time() - t0 < grace:
            live = [p for p in live if p.poll() is None]
            time.sleep(0.1)
        for p in live:
            p.kill()
        self._procs.clear()


CHILDREN = ChildProcs()
