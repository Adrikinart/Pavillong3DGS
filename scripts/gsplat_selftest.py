"""Verify gsplat works on the current GPU node, in isolation.

Run this as a SHORT job before submitting a long training run on a node that has
not built the gsplat CUDA extension yet. It (a) triggers the JIT build — the only
unbounded-RAM step in our pipeline — in a throwaway job rather than inside a long
training run, and (b) proves the kernels actually execute on this GPU's arch.

Slurm on this cluster reports RealMemory=1 and gathers no memory accounting, so a
runaway build can exhaust node RAM and take the node NOT_RESPONDING. Always pair
this with a low MAX_JOBS (e.g. 4).

Usage (on a GPU node):
    MAX_JOBS=4 python scripts/gsplat_selftest.py
"""

import time

import torch
import torch.nn.functional as F


def main() -> int:
    dev = "cuda"
    if not torch.cuda.is_available():
        print("FAIL cuda_not_available")
        return 1
    p = torch.cuda.get_device_properties(0)
    print(f"GPU {p.name} | VRAM_GB {p.total_memory / 1e9:.0f} | SM {p.major}.{p.minor} "
          f"| torch {torch.__version__}")

    t0 = time.time()
    import gsplat                     # triggers the JIT build if not cached
    build_s = time.time() - t0

    n = 2000
    means = torch.randn(n, 3, device=dev)
    means[:, 2] += 4.0                # put them in front of the camera
    quats = F.normalize(torch.randn(n, 4, device=dev), dim=-1)
    scales = torch.rand(n, 3, device=dev) * 0.05
    opacities = torch.rand(n, device=dev)
    colors = torch.rand(n, 3, device=dev)
    viewmats = torch.eye(4, device=dev)[None]
    Ks = torch.tensor([[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]]], device=dev)

    out, alpha, info = gsplat.rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmats, Ks=Ks, width=320, height=240)
    torch.cuda.synchronize()

    ok = torch.isfinite(out).all().item()
    print(f"GSPLAT_OK={ok} import_build_s={build_s:.0f} render={tuple(out.shape)} "
          f"peak_vram_MB={torch.cuda.max_memory_allocated() / 1e6:.0f} "
          f"total_s={time.time() - t0:.0f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
