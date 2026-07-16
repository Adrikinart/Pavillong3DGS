"""Render a trained gsplat checkpoint from arbitrary cameras (novel views)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CheckpointRenderer:
    """Loads Gaussians from a checkpoint and renders any PathCamera to an image."""

    def __init__(self, layout, train_run_id: str, sh_degree: int, device: str = "cuda",
                 near: float = 0.01, far: float = 1e10):
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
        ds = ColmapDataset(layout, "train", downscale=1)
        self.params, _ = create_splats(ds.points, ds.point_colors, sh_degree, device)
        load_checkpoint(ckpt, self.params, {})
        self.dataset = ds
        self.near = near
        self.far = far

    def render(self, cam) -> np.ndarray:
        """Render a PathCamera -> HxWx3 uint8 RGB image."""
        import torch

        vm = torch.from_numpy(cam.viewmat.astype(np.float32)).to(self.device)
        K = torch.from_numpy(cam.K.astype(np.float32)).to(self.device)
        with torch.no_grad():
            r, _, _ = self._backend._rasterize(self._gsplat, self.params, vm, K,
                                               cam.width, cam.height, self.sh_degree,
                                               self.near, self.far)
        img = r[0].clamp(0, 1).cpu().numpy()
        return (img * 255).astype(np.uint8)

    def n_gaussians(self) -> int:
        return int(self.params["means"].shape[0])
