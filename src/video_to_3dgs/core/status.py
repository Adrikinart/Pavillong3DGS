"""Per-stage status files. The JSON status file IS the completion marker.

Invariant enforced by the runner (see runner.py): RUNNING is written *before* a
stage does work; COMPLETED is written *only after* output validation passes. A
crash therefore leaves a stale RUNNING or a FAILED — never a COMPLETED.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .atomicio import atomic_write_json, read_json

SCHEMA_VERSION = 1


class StageStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class StatusRecord:
    stage: str
    state: str = StageStatus.PENDING.value
    fingerprint: str | None = None
    attempt: int = 0
    host: str | None = None
    pid: int | None = None
    slurm_job_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    note: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_status(path: str | Path, stage: str) -> StatusRecord:
    """Read a status file, or return a fresh PENDING record if absent/corrupt."""
    p = Path(path)
    if not p.exists():
        return StatusRecord(stage=stage)
    try:
        data = read_json(p)
        # tolerate unknown future fields
        known = {f: data.get(f) for f in StatusRecord.__dataclass_fields__ if f in data}
        rec = StatusRecord(stage=data.get("stage", stage))
        for k, v in known.items():
            setattr(rec, k, v)
        return rec
    except Exception:
        return StatusRecord(stage=stage)


def write_status(path: str | Path, rec: StatusRecord) -> None:
    atomic_write_json(path, rec.to_dict())


def process_alive(pid: int | None, slurm_job_id: str | None) -> bool:
    """Best-effort liveness check for stale-RUNNING recovery.

    A RUNNING status whose owning process/job is dead is treated as FAILED so a
    killed job never blocks re-runs forever.
    """
    if slurm_job_id:
        try:
            out = subprocess.run(
                ["squeue", "--noheader", "--job", str(slurm_job_id).split("_")[0]],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0:
                return bool(out.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # no slurm here (login/local) — fall through to pid check
    if pid:
        try:
            os.kill(int(pid), 0)
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            return False
        except OSError:
            return False
    return False
