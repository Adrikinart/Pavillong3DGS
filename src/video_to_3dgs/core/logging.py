"""Structured logging: human-readable console + machine-readable JSONL sink."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from .atomicio import append_jsonl

_CONFIGURED = False


class _JsonlHandler(logging.Handler):
    """Mirror log records into a JSONL file for machine consumption."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rec: dict[str, Any] = {
                "ts": self.formatTime(record) if hasattr(self, "formatTime") else record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            if record.exc_info:
                rec["exc"] = self.format(record)
            for k, v in getattr(record, "extra_fields", {}).items():
                rec[k] = v
            append_jsonl(self.path, rec)
        except Exception:  # never let logging crash the pipeline
            pass


def configure_logging(verbose: bool = False, jsonl_path: str | Path | None = None) -> None:
    """Configure the root ``video_to_3dgs`` logger. Idempotent."""
    global _CONFIGURED
    root = logging.getLogger("video_to_3dgs")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    # Console handler (always refreshed so --verbose takes effect on re-call)
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, _JsonlHandler):
            root.removeHandler(h)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                                      datefmt="%H:%M:%S"))
    root.addHandler(ch)
    if jsonl_path is not None:
        # avoid duplicate JSONL handlers for the same path
        if not any(isinstance(h, _JsonlHandler) and h.path == Path(jsonl_path)
                   for h in root.handlers):
            root.addHandler(_JsonlHandler(jsonl_path))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str = "video_to_3dgs") -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    if name == "video_to_3dgs":
        return logging.getLogger(name)
    return logging.getLogger(f"video_to_3dgs.{name}")
