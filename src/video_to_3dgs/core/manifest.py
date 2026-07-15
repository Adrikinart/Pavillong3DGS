"""Append-only dataset manifest — the provenance record for a run.

Header (dataset id, videos, software block) is written once at run init; one
record per stage execution is appended over time. Writes are serialized with a
simple lockfile and made atomic.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .atomicio import atomic_write_json, read_json
from .status import utcnow

SCHEMA_VERSION = 1


class _FileLock:
    """Minimal cross-process lock via O_CREAT|O_EXCL (works on NFS well enough
    for a single-user tool). Times out rather than deadlocking."""

    def __init__(self, target: Path, timeout: float = 30.0) -> None:
        self.lock_path = Path(str(target) + ".lock")
        self.timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> "_FileLock":
        deadline = time.time() + self.timeout
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() > deadline:
                    # stale lock: steal it (single-user tool, best effort)
                    try:
                        self.lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                time.sleep(0.1)

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None:
            os.close(self._fd)
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


class Manifest:
    """Read/modify/atomic-write wrapper over ``manifest.json``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return read_json(self.path)

    def init_header(self, *, dataset_id: str, videos: list[dict[str, Any]],
                    software: dict[str, Any]) -> None:
        """Create the manifest with its static header if it does not yet exist."""
        with _FileLock(self.path):
            if self.path.exists():
                data = read_json(self.path)
            else:
                data = {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_id": dataset_id,
                    "created_at": utcnow(),
                    "videos": videos,
                    "software": software,
                    "stages": [],
                }
            # refresh mutable header fields on re-init (e.g. new software probe)
            data.setdefault("stages", [])
            data["dataset_id"] = dataset_id
            if videos:
                data["videos"] = videos
            if software:
                data["software"] = software
            atomic_write_json(self.path, data)

    def append_stage(self, record: dict[str, Any]) -> None:
        with _FileLock(self.path):
            data = self.load() or {"schema_version": SCHEMA_VERSION, "stages": []}
            data.setdefault("stages", []).append({"recorded_at": utcnow(), **record})
            atomic_write_json(self.path, data)

    def set_field(self, key: str, value: Any) -> None:
        with _FileLock(self.path):
            data = self.load() or {"schema_version": SCHEMA_VERSION, "stages": []}
            data[key] = value
            atomic_write_json(self.path, data)
