"""Training metrics: append-only JSONL (durable) + optional TensorBoard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.atomicio import append_jsonl


class MetricsLogger:
    def __init__(self, jsonl_path: Path, tb_dir: Path | None = None,
                 enable_tb: bool = True):
        self.jsonl_path = jsonl_path
        self._tb = None
        if enable_tb and tb_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb = SummaryWriter(log_dir=str(tb_dir))
            except Exception:
                self._tb = None

    def log(self, step: int, scalars: dict[str, Any], *, kind: str = "train") -> None:
        rec = {"step": step, "kind": kind, **scalars}
        append_jsonl(self.jsonl_path, rec)
        if self._tb is not None:
            for k, v in scalars.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(f"{kind}/{k}", v, step)

    def log_histogram(self, step: int, name: str, values) -> None:
        if self._tb is not None:
            try:
                self._tb.add_histogram(name, values, step)
            except Exception:
                pass

    def log_image(self, step: int, name: str, chw) -> None:
        if self._tb is not None:
            try:
                self._tb.add_image(name, chw, step)
            except Exception:
                pass

    def close(self) -> None:
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
