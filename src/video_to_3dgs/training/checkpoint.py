"""Atomic checkpointing with integrity verification and latest-valid discovery."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from ..core.atomicio import sha256_file

_CKPT_RE = re.compile(r"ckpt_(\d+)\.pt$")


def save_checkpoint(ckpt_dir: Path, params, optimizers, step: int,
                    extra: dict[str, Any] | None = None) -> Path:
    """Atomically save a checkpoint + sha256 sidecar, update ckpt_latest symlink."""
    import torch
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "params": {k: v.detach().cpu() for k, v in params.items()},
        "optimizers": {k: opt.state_dict() for k, opt in optimizers.items()},
        "extra": extra or {},
    }
    target = ckpt_dir / f"ckpt_{step:07d}.pt"
    tmp = target.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, target)
    # sha256 sidecar (guards against SIGKILL mid-write on resume)
    (ckpt_dir / f"ckpt_{step:07d}.sha256").write_text(sha256_file(target))
    # update latest symlink atomically
    latest = ckpt_dir / "ckpt_latest.pt"
    tmp_link = ckpt_dir / "ckpt_latest.pt.tmplink"
    if tmp_link.exists() or tmp_link.is_symlink():
        tmp_link.unlink()
    try:
        tmp_link.symlink_to(target.name)
        os.replace(tmp_link, latest)
    except OSError:
        # filesystem without symlinks: write a pointer file instead
        (ckpt_dir / "ckpt_latest.txt").write_text(target.name)
    return target


def _verify(path: Path) -> bool:
    side = path.with_suffix(".sha256")
    if side.exists():
        try:
            if sha256_file(path) != side.read_text().strip():
                return False
        except OSError:
            return False
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
        return "params" in state and "step" in state
    except Exception:
        return False


def find_latest_valid(ckpt_dir: Path) -> Path | None:
    """Return the newest checkpoint that passes integrity + load checks."""
    if not ckpt_dir.exists():
        return None
    ckpts = []
    for p in ckpt_dir.iterdir():
        m = _CKPT_RE.search(p.name)
        if m:
            ckpts.append((int(m.group(1)), p))
    for _, p in sorted(ckpts, reverse=True):
        if _verify(p):
            return p
    return None


def load_checkpoint(path: Path, params, optimizers) -> int:
    """Load state into params/optimizers in place; return the step."""
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for k, v in state["params"].items():
            if k in params:
                params[k].data = v.to(params[k].device)
    for k, opt in optimizers.items():
        if k in state.get("optimizers", {}):
            try:
                opt.load_state_dict(state["optimizers"][k])
            except Exception:
                pass
    return int(state["step"])
