"""Environment & CUDA/GPU inspection and smoke tests.

`inspect-env` writes a JSON report and, with --gpu-check, runs:
  * a CUDA tensor forward smoke test,
  * a CUDA backward-pass smoke test,
  * a gsplat import + tiny rasterization test (compiles the sm_120 extension),
  * arch-compatibility check (PyTorch arch_list contains the device's sm).
It never hardcodes a compute capability — it detects the device and configures
`TORCH_CUDA_ARCH_LIST` accordingly for any JIT compilation.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from .core.atomicio import atomic_write_json
from .core.logging import get_logger

log = get_logger("environment")


def _cmd(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "rc": r.returncode,
                "stdout": r.stdout.strip(), "stderr": r.stderr.strip()[:2000]}
    except FileNotFoundError:
        return {"cmd": " ".join(cmd), "rc": 127, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": " ".join(cmd), "rc": 124, "stdout": "", "stderr": "timeout"}


def collect_shell_env() -> dict[str, Any]:
    """The shell-level inspection required by the task spec."""
    out: dict[str, Any] = {
        "hostname": platform.node(),
        "uname": platform.platform(),
        "python": platform.python_version(),
        "which_python": _cmd(["which", "python"]).get("stdout")
        or _cmd(["which", "python3"]).get("stdout"),
        "cwd": os.getcwd(),
    }
    for key, cmd in {
        "nvidia_smi": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                       "--format=csv,noheader"],
        "nvidia_smi_L": ["nvidia-smi", "-L"],
        "nvcc": ["nvcc", "--version"],
        "conda": ["conda", "--version"],
        "ffmpeg": ["ffmpeg", "-version"],
        "colmap": ["colmap", "-h"],
        "sinfo": ["sinfo", "-s"],
    }.items():
        out[key] = _cmd(cmd)
    # scratch locations
    user = os.environ.get("USER", "user")
    scratch: dict[str, bool] = {}
    for c in [os.environ.get("SLURM_TMPDIR"), os.environ.get("TMPDIR"),
              f"/scratch/{user}", f"/local_scratch/{user}", f"/var/tmp/{user}"]:
        if c:
            scratch[c] = Path(c).exists() and os.access(c, os.W_OK)
    out["scratch_candidates"] = scratch
    out["relevant_env"] = {k: v for k, v in os.environ.items()
                           if k.startswith(("SLURM", "CUDA", "TORCH", "V2GS")) or k in ("TMPDIR",)}
    return out


def torch_probe() -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        try:
            info["arch_list"] = torch.cuda.get_arch_list()
        except Exception:
            info["arch_list"] = None
        if torch.cuda.is_available():
            devs = []
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                devs.append({"index": i, "name": p.name, "cc": f"{p.major}.{p.minor}",
                             "sm": f"sm_{p.major}{p.minor}", "total_memory": p.total_memory})
            info["devices"] = devs
    except Exception as e:
        info["torch"] = None
        info["import_error"] = str(e)
    return info


def set_arch_env_for_devices(info: dict[str, Any]) -> str | None:
    """Set TORCH_CUDA_ARCH_LIST from detected devices (for gsplat JIT). Returns it."""
    devs = info.get("devices") or []
    if not devs:
        return None
    archs = sorted({f"{d['cc']}" for d in devs})
    val = ";".join(archs)  # e.g. "12.0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = val
    return val


def cuda_smoke_tests() -> dict[str, Any]:
    """Forward + backward CUDA smoke tests. Returns pass/fail per test."""
    res: dict[str, Any] = {"forward": None, "backward": None, "error": None}
    try:
        import torch
        if not torch.cuda.is_available():
            res["error"] = "cuda not available"
            return res
        dev = torch.device("cuda:0")
        a = torch.randn(256, 256, device=dev)
        b = torch.randn(256, 256, device=dev)
        c = (a @ b).sum()
        torch.cuda.synchronize()
        res["forward"] = bool(torch.isfinite(c).item())
        # backward
        x = torch.randn(128, 128, device=dev, requires_grad=True)
        y = (x * x).sum()
        y.backward()
        torch.cuda.synchronize()
        res["backward"] = bool(x.grad is not None and torch.isfinite(x.grad).all().item())
    except Exception as e:  # e.g. "no kernel image available" on arch mismatch
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def gsplat_smoke_test() -> dict[str, Any]:
    """Import gsplat and run a tiny rasterization (compiles the CUDA ext on sm_120)."""
    res: dict[str, Any] = {"import": None, "rasterize": None, "version": None, "error": None}
    try:
        import torch
        import gsplat
        res["import"] = True
        res["version"] = getattr(gsplat, "__version__", "?")
        if not torch.cuda.is_available():
            res["error"] = "cuda not available; skipped rasterize"
            return res
        dev = torch.device("cuda:0")
        N = 100
        means = torch.randn(N, 3, device=dev)
        quats = torch.randn(N, 4, device=dev)
        scales = torch.rand(N, 3, device=dev) * 0.1
        opacities = torch.rand(N, device=dev)
        colors = torch.rand(N, 3, device=dev)
        K = torch.tensor([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]], device=dev)[None]
        viewmat = torch.eye(4, device=dev)[None]
        out = gsplat.rasterization(means, quats, scales, opacities, colors,
                                   viewmat, K, width=256, height=256)
        img = out[0]
        torch.cuda.synchronize()
        res["rasterize"] = bool(torch.isfinite(img).all().item())
    except Exception as e:
        if res["import"] is None:
            res["import"] = False
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def inspect_environment(gpu_check: bool = False, repo_root: str | Path | None = None) -> dict[str, Any]:
    from .core.provenance import git_info

    report: dict[str, Any] = {
        "shell": collect_shell_env(),
        "torch": torch_probe(),
    }
    if repo_root:
        report["git"] = git_info(repo_root)
    arch = set_arch_env_for_devices(report["torch"])
    report["torch_cuda_arch_list"] = arch
    # arch compatibility: does the wheel contain kernels for the device's sm?
    devs = report["torch"].get("devices") or []
    arch_list = report["torch"].get("arch_list") or []
    compat = []
    for d in devs:
        want = d["sm"].replace("sm_", "")  # "120"
        ok = any(want in a for a in arch_list)
        compat.append({"device": d["name"], "sm": d["sm"], "supported_by_wheel": ok})
    report["arch_compatibility"] = compat
    if gpu_check:
        report["cuda_smoke"] = cuda_smoke_tests()
        report["gsplat_smoke"] = gsplat_smoke_test()
    return report


def write_report(report: dict[str, Any], out_path: str | Path) -> None:
    atomic_write_json(out_path, report)
    log.info("environment report written to %s", out_path)


def summarize(report: dict[str, Any]) -> str:
    t = report.get("torch", {})
    lines = [
        f"host      : {report['shell'].get('hostname')}",
        f"python    : {report['shell'].get('python')}",
        f"torch     : {t.get('torch')} (cuda build {t.get('torch_cuda_build')})",
        f"cuda avail: {t.get('cuda_available')}",
    ]
    for d in t.get("devices") or []:
        lines.append(f"  gpu     : {d['name']} {d['sm']} {d['total_memory'] // (1024**2)} MiB")
    for c in report.get("arch_compatibility", []):
        flag = "OK" if c["supported_by_wheel"] else "MISSING KERNELS"
        lines.append(f"  arch    : {c['sm']} supported_by_wheel={c['supported_by_wheel']} [{flag}]")
    if "cuda_smoke" in report:
        s = report["cuda_smoke"]
        lines.append(f"cuda smoke: fwd={s.get('forward')} bwd={s.get('backward')} err={s.get('error')}")
    if "gsplat_smoke" in report:
        g = report["gsplat_smoke"]
        lines.append(f"gsplat    : import={g.get('import')} rasterize={g.get('rasterize')} "
                     f"v={g.get('version')} err={g.get('error')}")
    return "\n".join(lines)
