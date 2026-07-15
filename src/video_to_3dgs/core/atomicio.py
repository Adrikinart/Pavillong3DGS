"""Atomic file IO and checksum helpers.

On the NFS mount used by this project, half-written files are a real hazard
(node preemption, NFS stalls). Every durable write goes through a ``.tmp`` file
that is fsync'd and then ``os.replace``d into place — an atomic operation on a
single filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    """Streamed sha256 of a file (never loads the whole file into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_str(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    """Atomically write ``data`` to ``path`` (write tmp -> fsync -> replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | os.PathLike, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | os.PathLike, obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, sort_keys=False, default=str))


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON string for fingerprinting (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def append_jsonl(path: str | os.PathLike, obj: Any) -> None:
    """Append one JSON record + newline. Not atomic, but append-only is crash-safe
    enough for metrics/event streams (a torn tail line is ignored on read)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def iter_jsonl(path: str | os.PathLike) -> Iterable[Any]:
    """Yield records from a JSONL file, skipping a possibly-torn final line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # torn tail (crash mid-write) — stop; earlier records are intact
                break
