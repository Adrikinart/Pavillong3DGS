"""Background GPU/system sampler -> JSONL (nvidia-smi + psutil-free stdlib)."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

from ..core.atomicio import append_jsonl


class SystemMonitor:
    """Samples GPU utilization/memory/temp/power + host load into a JSONL file."""

    def __init__(self, out_path: Path, interval_s: float = 15.0) -> None:
        self.out_path = out_path
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_gpu(self) -> list[dict]:
        try:
            r = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if r.returncode != 0:
                return []
            out = []
            for line in r.stdout.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 6:
                    out.append({"gpu": int(p[0]), "util": _f(p[1]), "mem_used_mib": _f(p[2]),
                                "mem_total_mib": _f(p[3]), "temp_c": _f(p[4]), "power_w": _f(p[5])})
            return out
        except Exception:
            return []

    def _loop(self) -> None:
        while not self._stop.is_set():
            rec = {"ts": time.time(), "gpus": self._sample_gpu(),
                   "loadavg": os.getloadavg()[0] if hasattr(os, "getloadavg") else None}
            append_jsonl(self.out_path, rec)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def _f(x: str):
    try:
        return float(x)
    except ValueError:
        return None
