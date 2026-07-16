"""Camera-path generators for novel-view rendering (orbit, interpolated).

All poses are produced in the *normalized* scene frame (the frame the trained
Gaussians live in: scene roughly centered at the origin, orbit radius ~1). Each
path element is a (viewmat world->camera, K, width, height) tuple, matching what
``gsplat.rasterization`` and the training dataset use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathCamera:
    viewmat: np.ndarray  # (4,4) world->camera (OpenCV convention)
    K: np.ndarray        # (3,3)
    width: int
    height: int


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """World->camera matrix (OpenCV: +x right, +y down, +z forward/into scene)."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)          # forward (camera +z)
    r = np.cross(f, up)
    r = r / (np.linalg.norm(r) + 1e-9)          # right (camera +x)
    d = np.cross(f, r)                          # down (camera +y)
    R = np.stack([r, d, f], axis=0)             # rows = camera axes in world
    t = -R @ eye
    vm = np.eye(4)
    vm[:3, :3] = R
    vm[:3, 3] = t
    return vm


def scene_frame_from_dataset(dataset) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Estimate (center, up, radius, mean_K) in the normalized frame from a dataset.

    Uses the transformed camera centers/points already exposed by ColmapDataset.
    """
    centers = np.array([_camera_center(s.viewmat) for s in dataset.samples])
    center = np.median(centers, axis=0) if len(centers) else np.zeros(3)
    # up = mean of camera "up" (negative image-plane y) directions
    ups = [-s.viewmat[1, :3] for s in dataset.samples]
    up = np.mean(ups, axis=0) if ups else np.array([0.0, 0.0, 1.0])
    up = up / (np.linalg.norm(up) + 1e-9)
    radius = float(np.median(np.linalg.norm(centers - center, axis=1))) if len(centers) else 1.0
    K = np.median(np.stack([s.K for s in dataset.samples]), axis=0) if dataset.samples else np.eye(3)
    return center, up, radius, K


def _camera_center(viewmat: np.ndarray) -> np.ndarray:
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    return -R.T @ t


def orbit_path(center: np.ndarray, up: np.ndarray, radius: float, K: np.ndarray,
               width: int, height: int, *, n_frames: int = 120,
               elevation_deg: float = 20.0, radius_scale: float = 1.2) -> list[PathCamera]:
    """A closed horizontal orbit around ``center`` at a fixed elevation."""
    up = up / (np.linalg.norm(up) + 1e-9)
    # build an orthonormal basis with `up` as the vertical axis
    a = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e0 = np.cross(up, a); e0 /= np.linalg.norm(e0) + 1e-9
    e1 = np.cross(up, e0)
    r = radius * radius_scale
    elev = np.deg2rad(elevation_deg)
    cams: list[PathCamera] = []
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        planar = np.cos(theta) * e0 + np.sin(theta) * e1
        eye = center + r * (np.cos(elev) * planar + np.sin(elev) * up)
        cams.append(PathCamera(_look_at(eye, center, up), K.copy(), width, height))
    return cams


def resize_intrinsics(K: np.ndarray, src_wh: tuple[int, int],
                      dst_wh: tuple[int, int]) -> np.ndarray:
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K = K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K
