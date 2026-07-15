"""Validation/evaluation: render held-out cameras and compute image metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .losses import psnr, ssim


class _LPIPS:
    _net = None

    @classmethod
    def get(cls, device: str):
        if cls._net is None:
            try:
                import lpips
                cls._net = lpips.LPIPS(net="alex").to(device).eval()
            except Exception:
                cls._net = False
        return cls._net


def evaluate_split(render_fn: Callable, dataset, device: str, *,
                   out_dir: Path | None = None, compute_lpips: bool = True,
                   masked: bool = True, max_images: int | None = None,
                   save_renders: bool = True) -> dict[str, Any]:
    """Render each sample, compare to GT, return aggregate + per-view metrics."""
    import torch

    n = len(dataset)
    idxs = list(range(n))
    if max_images is not None and n > max_images:
        step = n / max_images
        idxs = sorted({min(n - 1, int(i * step)) for i in range(max_images)})

    lpips_net = _LPIPS.get(device) if compute_lpips else False
    per_view: list[dict[str, Any]] = []
    if out_dir is not None and save_renders:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in idxs:
        gt, mask = dataset.load_image(i)
        gt = gt.to(device)
        mask_t = mask.to(device) if (masked and mask is not None) else None
        with torch.no_grad():
            render = render_fn(i).clamp(0, 1)
        p = psnr(render, gt, mask_t)
        s = float(ssim(render.permute(2, 0, 1), gt.permute(2, 0, 1)))
        lp = None
        if lpips_net:
            with torch.no_grad():
                a = render.permute(2, 0, 1)[None] * 2 - 1
                b = gt.permute(2, 0, 1)[None] * 2 - 1
                lp = float(lpips_net(a, b).item())
        rec = {"name": dataset.samples[i].name, "psnr": round(p, 3),
               "ssim": round(s, 4), "lpips": round(lp, 4) if lp is not None else None}
        per_view.append(rec)
        if out_dir is not None and save_renders:
            _save_png(render, out_dir / f"{Path(dataset.samples[i].name).stem}.png")

    def _avg(key):
        vals = [r[key] for r in per_view if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "n_views": len(per_view),
        "psnr": _avg("psnr"), "ssim": _avg("ssim"), "lpips": _avg("lpips"),
        "per_view": per_view,
    }


def _save_png(hwc, path: Path) -> None:
    import numpy as np
    from PIL import Image
    arr = (hwc.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
