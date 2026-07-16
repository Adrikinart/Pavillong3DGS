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


@dataclass
class ObjectFrame:
    center: np.ndarray   # object bbox center (normalized frame)
    size: np.ndarray     # object bbox size
    front: np.ndarray    # unit viewing dir from cameras toward the object
    up: np.ndarray
    fit_distance: float   # camera distance that frames the whole object


def object_frame_from_dataset(dataset, K: np.ndarray, width: int, height: int,
                              margin: float = 1.2) -> ObjectFrame:
    """Object bbox + the 'front' viewing side + a distance that frames the whole
    object in view (so orbit/overview cameras aren't stuck in close-up)."""
    centers = np.array([_camera_center(s.viewmat) for s in dataset.samples])
    pts = dataset.points if len(getattr(dataset, "points", [])) else centers
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    center = (lo + hi) / 2.0
    size = hi - lo
    cam_centroid = centers.mean(axis=0) if len(centers) else center - np.array([0, 0, 1.0])
    front = center - cam_centroid
    n = np.linalg.norm(front)
    front = front / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    ups = [-s.viewmat[1, :3] for s in dataset.samples]
    up = np.mean(ups, axis=0) if ups else np.array([0.0, 1.0, 0.0])
    up = up / (np.linalg.norm(up) + 1e-9)
    # distance so the object's largest face fits the (smaller) field of view
    fx, fy = K[0, 0], K[1, 1]
    fov_x = 2 * np.arctan(width / (2 * fx))
    fov_y = 2 * np.arctan(height / (2 * fy))
    radius = 0.5 * float(np.linalg.norm(np.sort(size)[-2:]))   # half-diagonal of 2 largest dims
    fit = margin * radius / max(np.tan(min(fov_x, fov_y) / 2), 1e-3)
    return ObjectFrame(center=center, size=size, front=front, up=up, fit_distance=fit)


def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = axis
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def front_arc_path(obj: ObjectFrame, K: np.ndarray, width: int, height: int, *,
                   n_frames: int = 120, arc_deg: float = 80.0,
                   elevation_deg: float = 12.0) -> list[PathCamera]:
    """A gentle azimuth arc across the captured FRONT of the object (not a full
    360 — the back was never seen), at the framing distance, looking at center.
    Reveals the relief via parallax while keeping the whole object in view."""
    base_eye = obj.center - obj.fit_distance * obj.front   # straight-on overview
    cams: list[PathCamera] = []
    half = np.deg2rad(arc_deg) / 2.0
    elev = np.deg2rad(elevation_deg)
    for i in range(n_frames):
        # smooth back-and-forth sweep so the clip loops
        t = np.sin(2 * np.pi * i / n_frames)          # -1..1
        az = t * half
        R_az = _rot_about_axis(obj.up, az)
        right = np.cross(obj.front, obj.up)
        R_el = _rot_about_axis(right, elev * np.cos(2 * np.pi * i / n_frames))
        eye = obj.center + R_el @ (R_az @ (base_eye - obj.center))
        cams.append(PathCamera(_look_at(eye, obj.center, obj.up), K.copy(), width, height))
    return cams


def overview_camera(obj: ObjectFrame, K: np.ndarray, width: int, height: int) -> PathCamera:
    """A single straight-on camera that frames the whole object."""
    eye = obj.center - obj.fit_distance * obj.front
    return PathCamera(_look_at(eye, obj.center, obj.up), K.copy(), width, height)


def resize_intrinsics(K: np.ndarray, src_wh: tuple[int, int],
                      dst_wh: tuple[int, int]) -> np.ndarray:
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    K = K.copy()
    K[0, :] *= sx
    K[1, :] *= sy
    return K
