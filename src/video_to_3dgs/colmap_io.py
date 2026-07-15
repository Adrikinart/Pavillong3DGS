"""Readers/writers for COLMAP's sparse model format (.bin and .txt).

Self-contained (no pycolmap dependency) — fewer wheels to build on Blackwell.
Format reference: COLMAP `src/colmap/scene/reconstruction.cc`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# COLMAP camera model id -> (name, num_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        """3x3 intrinsics (uses the pinhole part of the model)."""
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",
                          "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        else:  # PINHOLE / OPENCV / FULL_OPENCV / fisheye variants
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


@dataclass
class Image:
    id: int
    qvec: np.ndarray   # (4,) w,x,y,z
    tvec: np.ndarray   # (3,)
    camera_id: int
    name: str
    xys: np.ndarray = None          # (M,2)
    point3D_ids: np.ndarray = None  # (M,)

    def rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

    def world_to_camera(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotmat()
        T[:3, 3] = self.tvec
        return T

    def camera_center(self) -> np.ndarray:
        R = self.rotmat()
        return -R.T @ self.tvec


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track_length: int


def qvec2rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def _read(f, num: int, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, f.read(num))


def read_cameras_binary(path: str | Path) -> dict[int, Camera]:
    cams: dict[int, Camera] = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, 24, "iiQQ")
            name, nparams = CAMERA_MODELS[model_id]
            params = np.array(_read(f, 8 * nparams, "d" * nparams))
            cams[cam_id] = Camera(cam_id, name, w, h, params)
    return cams


def read_images_binary(path: str | Path) -> dict[int, Image]:
    images: dict[int, Image] = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            img_id = _read(f, 4, "i")[0]
            qvec = np.array(_read(f, 32, "dddd"))
            tvec = np.array(_read(f, 24, "ddd"))
            cam_id = _read(f, 4, "i")[0]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            npts = _read(f, 8, "Q")[0]
            data = _read(f, 24 * npts, "ddq" * npts)
            xys = np.array(data).reshape(npts, 3)[:, :2] if npts else np.zeros((0, 2))
            p3d = np.array(data).reshape(npts, 3)[:, 2].astype(np.int64) if npts else np.zeros((0,), np.int64)
            images[img_id] = Image(img_id, qvec, tvec, cam_id, name.decode("utf-8"), xys, p3d)
    return images


def read_points3D_binary(path: str | Path) -> dict[int, Point3D]:
    points: dict[int, Point3D] = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "Q")[0]
        for _ in range(n):
            pid = _read(f, 8, "Q")[0]
            xyz = np.array(_read(f, 24, "ddd"))
            rgb = np.array(_read(f, 3, "BBB"))
            err = _read(f, 8, "d")[0]
            track_len = _read(f, 8, "Q")[0]
            _ = _read(f, 8 * track_len, "ii" * track_len)  # track (image_id, point2D_idx)
            points[pid] = Point3D(pid, xyz, rgb, err, track_len)
    return points


def read_model(sparse_dir: str | Path) -> tuple[dict[int, Camera], dict[int, Image], dict[int, Point3D]]:
    """Read a sparse model dir, preferring .bin, falling back to .txt (not impl here)."""
    d = Path(sparse_dir)
    cams = read_cameras_binary(d / "cameras.bin")
    imgs = read_images_binary(d / "images.bin")
    pts = read_points3D_binary(d / "points3D.bin")
    return cams, imgs, pts


def model_stats(sparse_dir: str | Path, n_input_images: int | None = None) -> dict[str, Any]:
    cams, imgs, pts = read_model(sparse_dir)
    errors = [p.error for p in pts.values()]
    track_lengths = [p.track_length for p in pts.values()]
    obs_per_image = [int((im.point3D_ids >= 0).sum()) for im in imgs.values()] if imgs else [0]
    stats = {
        "n_registered_images": len(imgs),
        "n_points3D": len(pts),
        "mean_reprojection_error": float(np.mean(errors)) if errors else None,
        "mean_track_length": float(np.mean(track_lengths)) if track_lengths else None,
        "mean_observations_per_image": float(np.mean(obs_per_image)) if obs_per_image else None,
        "cameras": {cid: {"model": c.model, "width": c.width, "height": c.height,
                          "params": c.params.tolist()} for cid, c in cams.items()},
    }
    if n_input_images:
        stats["n_input_images"] = n_input_images
        stats["registration_ratio"] = len(imgs) / max(1, n_input_images)
    return stats
