"""Provenance capture: git, python/torch/cuda/driver versions, package freeze."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .atomicio import sha256_str


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def git_info(repo_root: str | Path) -> dict[str, Any]:
    repo_root = str(repo_root)
    sha = _run(["git", "-C", repo_root, "rev-parse", "HEAD"])
    status = _run(["git", "-C", repo_root, "status", "--porcelain"])
    branch = _run(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "git_sha": sha,
        "git_branch": branch,
        "git_dirty": bool(status) if status is not None else None,
    }


def nvidia_driver_info() -> dict[str, Any]:
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader",
    ])
    gpus: list[dict[str, Any]] = []
    if out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0],
                    "driver_version": parts[1],
                    "memory_total": parts[2],
                    "compute_cap": parts[3],
                })
    return {"gpus": gpus, "n_gpus": len(gpus)}


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch  # noqa: F401

        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        try:
            info["arch_list"] = torch.cuda.get_arch_list()
            info["sm_120_supported"] = any("120" in a for a in info["arch_list"])
        except Exception:
            info["arch_list"] = None
        if torch.cuda.is_available():
            devs = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devs.append({
                    "index": i, "name": p.name, "major": p.major, "minor": p.minor,
                    "total_memory": p.total_memory,
                })
            info["devices"] = devs
    except Exception as e:  # torch not installed (e.g. CPU login env without GPU wheels)
        info["torch"] = None
        info["error"] = str(e)
    return info


def package_freeze() -> tuple[str, str]:
    """Return (freeze_text, sha256)."""
    txt = _run([sys.executable, "-m", "pip", "freeze"], timeout=60) or ""
    return txt, sha256_str(txt)


def software_block(repo_root: str | Path) -> dict[str, Any]:
    from .. import __version__

    freeze_txt, freeze_sha = package_freeze()
    block: dict[str, Any] = {
        "framework_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pip_freeze_sha256": freeze_sha,
    }
    block.update(git_info(repo_root))
    block.update(torch_info())
    block["nvidia"] = nvidia_driver_info()
    return block, freeze_txt
