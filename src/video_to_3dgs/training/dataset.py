"""COLMAP-format dataset loader for gsplat training.

Reads the (undistorted) sparse model + images + optional masks, applies the
scene-normalization similarity transform, and exposes per-split camera samples
with world-to-camera viewmats (OpenCV convention) and intrinsics — exactly what
``gsplat.rasterization`` consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import colmap_io


def _center_from_viewmat(viewmat: "np.ndarray") -> "np.ndarray":
    """Camera center in world coords from a world->camera matrix."""
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return -R.T @ t


@dataclass
class CameraSample:
    name: str
    viewmat: "np.ndarray"     # (4,4) world->camera
    K: "np.ndarray"           # (3,3)
    width: int
    height: int
    image_path: Path
    mask_path: Path | None


class ColmapDataset:
    def __init__(self, layout, split: str, *, use_masks: bool = False,
                 downscale: int = 1, cache_images: bool = True):
        import torch  # local import so CPU CLI need not have torch

        self.layout = layout
        self.split = split
        self.use_masks = use_masks
        self.downscale = max(1, int(downscale))
        self._torch = torch

        cams, imgs, pts = colmap_io.read_model(layout.colmap_sparse0)
        self.transform, self.scale, self.R_up = self._load_transform(layout)

        # split filter
        names = self._read_split(layout, split)
        by_name = {im.name: im for im in imgs.values()}
        sel = [by_name[n] for n in names if n in by_name]
        if not sel:  # e.g. evaluate on 'train' when no split file: use all
            sel = list(imgs.values())

        self.samples: list[CameraSample] = []
        for im in sorted(sel, key=lambda i: i.name):
            cam = cams[im.camera_id]
            vm, K = self._transform_camera(im, cam)
            img_path = layout.colmap_images / im.name
            mp = None
            if use_masks and layout.masks_dir.exists():
                cand = layout.masks_dir / (Path(im.name).stem + ".png")
                mp = cand if cand.exists() else None
            self.samples.append(CameraSample(im.name, vm, K, cam.width, cam.height, img_path, mp))

        # initial point cloud (normalized frame)
        if pts:
            P = np.array([p.xyz for p in pts.values()])
            C = np.array([p.rgb for p in pts.values()], dtype=np.float32) / 255.0
            self.points = (self.transform[:3, :3] @ P.T).T + self.transform[:3, 3]
            self.point_colors = C
        else:
            self.points = np.zeros((0, 3))
            self.point_colors = np.zeros((0, 3), dtype=np.float32)

        self._cache: dict[int, Any] = {}
        self._cache_images = cache_images

    # -------------------------------------------------------------- #
    @staticmethod
    def _load_transform(layout):
        tf_path = layout.normalize_transform
        if tf_path.exists():
            d = json.loads(tf_path.read_text())
            T = np.array(d["transform"], dtype=np.float64)
            scale = float(d.get("scale", 1.0))
            R_up = T[:3, :3] / scale if scale else np.eye(3)
            return T, scale, R_up
        return np.eye(4), 1.0, np.eye(3)

    @staticmethod
    def _read_split(layout, split: str) -> list[str]:
        p = layout.split_file(split)
        if not p.exists():
            return []
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]

    def _transform_camera(self, im, cam):
        Rwc_old = im.rotmat()
        center_old = im.camera_center()
        Rcw_new = self.R_up @ Rwc_old.T
        center_new = self.transform[:3, :3] @ center_old + self.transform[:3, 3]
        Rwc_new = Rcw_new.T
        t_new = -Rwc_new @ center_new
        vm = np.eye(4)
        vm[:3, :3] = Rwc_new
        vm[:3, 3] = t_new
        K = cam.K().copy()
        if self.downscale > 1:
            K[:2, :] /= self.downscale
        return vm, K

    # -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.samples)

    def scene_extent(self) -> float:
        """Scene radius from CAMERA positions (3DGS `spatial_lr_scale` convention).

        Must NOT use points.max(): COLMAP leaves far outlier points that inflate it
        ~10x, which then (a) scales the means LR ~10x too high (positions jitter ->
        blurry renders) and (b) makes gsplat's grow_scale3d treat every Gaussian as
        'small' -> clone-only, never split -> Gaussians never shrink. Camera extent
        is robust and is what the densification threshold is calibrated against.
        """
        centers = np.array([_center_from_viewmat(s.viewmat) for s in self.samples])
        if len(centers):
            c = centers.mean(axis=0)
            return float(np.linalg.norm(centers - c, axis=1).max()) * 1.1
        return 1.0

    def load_image(self, idx: int):
        torch = self._torch
        if idx in self._cache:
            img, mask = self._cache[idx]
        else:
            from PIL import Image
            s = self.samples[idx]
            im = Image.open(s.image_path).convert("RGB")
            if self.downscale > 1:
                im = im.resize((s.width // self.downscale, s.height // self.downscale),
                               Image.BILINEAR)
            img = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)  # HWC
            mask = None
            if s.mask_path is not None:
                m = Image.open(s.mask_path).convert("L")
                if self.downscale > 1:
                    m = m.resize((s.width // self.downscale, s.height // self.downscale),
                                 Image.NEAREST)
                mask = torch.from_numpy((np.asarray(m, dtype=np.float32) / 255.0))[..., None]
            if self._cache_images:
                self._cache[idx] = (img, mask)
        return img, mask

    def camera_tensors(self, idx: int, device: str):
        torch = self._torch
        s = self.samples[idx]
        vm = torch.from_numpy(s.viewmat.astype(np.float32)).to(device)
        K = torch.from_numpy(s.K.astype(np.float32)).to(device)
        w = s.width // self.downscale
        h = s.height // self.downscale
        return vm, K, w, h
