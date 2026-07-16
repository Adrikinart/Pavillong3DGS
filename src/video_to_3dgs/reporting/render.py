"""Render a trained gsplat checkpoint from arbitrary cameras (novel views)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CheckpointRenderer:
    """Loads Gaussians from a checkpoint and renders any PathCamera to an image."""

    def __init__(self, layout, train_run_id: str, sh_degree: int, device: str = "cuda",
                 near: float = 0.01, far: float = 1e10, crop_box: dict | None = None,
                 crop_margin: float = 0.1):
        import torch  # noqa: F401

        from ..training.checkpoint import find_latest_valid, load_checkpoint
        from ..training.dataset import ColmapDataset
        from ..training.gaussians import create_splats
        from ..training.gsplat_backend import GsplatBackend

        self.device = device
        self.sh_degree = sh_degree
        self._backend = GsplatBackend()
        import gsplat
        self._gsplat = gsplat

        ckpt = find_latest_valid(layout.checkpoints_dir(train_run_id))
        if ckpt is None:
            raise FileNotFoundError(f"no valid checkpoint under {layout.checkpoints_dir(train_run_id)}")
        self.checkpoint = ckpt
        self._layout = layout
        self._train_run_id = train_run_id
        self._load_checkpoint = load_checkpoint
        ds = ColmapDataset(layout, "train", downscale=1)
        self.params, _ = create_splats(ds.points, ds.point_colors, sh_degree, device)
        load_checkpoint(ckpt, self.params, {})
        self.dataset = ds
        self.near = near
        self.far = far
        self._crop = None
        if crop_box:
            import numpy as _np
            self._crop = (_np.array(crop_box["min"]) - crop_margin,
                          _np.array(crop_box["max"]) + crop_margin)
        self._active = None
        self._apply_crop()

    def _apply_crop(self) -> None:
        """Keep only Gaussians inside the object crop box (removes floaters from
        beauty renders). Falls back to all Gaussians if the box is unset/empty."""
        if self._crop is None:
            self._active = self.params
            return
        import torch
        lo, hi = self._crop
        m = self.params["means"]
        lo_t = torch.tensor(lo, dtype=m.dtype, device=m.device)
        hi_t = torch.tensor(hi, dtype=m.dtype, device=m.device)
        inside = ((m >= lo_t) & (m <= hi_t)).all(dim=1)
        if int(inside.sum()) < 100:      # box too tight — keep everything
            self._active = self.params
            return
        idx = inside.nonzero(as_tuple=True)[0]
        self._active = {k: v[idx] for k, v in self.params.items()}

    def load(self, checkpoint) -> None:
        """Reload Gaussian params from a specific checkpoint (for progression)."""
        self._load_checkpoint(checkpoint, self.params, {})
        self.checkpoint = checkpoint
        self._apply_crop()

    def checkpoints(self) -> list:
        """All saved checkpoints for this run, ordered by iteration."""
        import re
        d = self._layout.checkpoints_dir(self._train_run_id)
        out = []
        for p in d.iterdir():
            m = re.search(r"ckpt_(\d+)\.pt$", p.name)
            if m:
                out.append((int(m.group(1)), p))
        return [p for _, p in sorted(out)]

    def render(self, cam) -> np.ndarray:
        """Render a PathCamera -> HxWx3 uint8 RGB image."""
        import torch

        vm = torch.from_numpy(cam.viewmat.astype(np.float32)).to(self.device)
        K = torch.from_numpy(cam.K.astype(np.float32)).to(self.device)
        with torch.no_grad():
            r, _, _ = self._backend._rasterize(self._gsplat, self._active, vm, K,
                                               cam.width, cam.height, self.sh_degree,
                                               self.near, self.far)
        img = r[0].clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8)

    def n_gaussians(self) -> int:
        return int(self.params["means"].shape[0])
