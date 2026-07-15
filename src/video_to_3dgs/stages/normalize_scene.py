"""Stage: object-centric scene normalization.

Produces a similarity transform (4x4) mapping COLMAP world coordinates into a
normalized frame centered on the object with unit-ish scale and a chosen up
axis. The transform and its inverse are stored so results can be mapped back to
COLMAP / Blender / Unity coordinates and reused for downstream annotations.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext
from .. import colmap_io

_UP_VECTORS = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1],
               "-x": [-1, 0, 0], "-y": [0, -1, 0], "-z": [0, 0, -1]}


class NormalizeSceneStage(Stage):
    name = "normalize_scene"
    depends_on = ("validate_colmap",)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("colmap_sparse", ctx.layout.colmap_sparse0, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("normalize_transform", ctx.layout.normalize_transform, "file")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.normalize_scene.model_dump()

    def run(self, ctx: StageContext) -> dict[str, Any]:
        cfg = ctx.config.normalize_scene
        cams, imgs, pts = colmap_io.read_model(ctx.layout.colmap_sparse0)
        centers = np.array([im.camera_center() for im in imgs.values()])
        P = np.array([p.xyz for p in pts.values()]) if pts else centers.copy()

        # robust center
        if cfg.method == "cameras":
            center = np.median(centers, axis=0)
        elif cfg.method == "points":
            center = _robust_center(P, cfg.outlier_std)
        else:  # pca falls back to points center
            center = _robust_center(P, cfg.outlier_std)

        # scale so the chosen percentile radius of cameras -> 1.0
        cam_r = np.linalg.norm(centers - center, axis=1)
        radius = np.percentile(cam_r, cfg.scale_percentile) + 1e-9
        scale = 1.0 / radius

        # up axis
        if cfg.up_axis:
            up = np.array(_UP_VECTORS[cfg.up_axis], dtype=np.float64)
            R = _rotation_to_up(_estimate_up(imgs), up)
        else:
            up = _estimate_up(imgs)
            R = np.eye(3)  # keep COLMAP orientation, just record estimated up

        # similarity transform: x_norm = scale * R @ (x - center)
        T = np.eye(4)
        T[:3, :3] = scale * R
        T[:3, 3] = -scale * R @ center
        Tinv = np.linalg.inv(T)

        # near/far from normalized camera-to-scene distances
        norm_centers = (T[:3, :3] @ centers.T).T + T[:3, 3]
        norm_pts = (T[:3, :3] @ P.T).T + T[:3, 3]
        dists = np.linalg.norm(norm_centers[:, None, :] - norm_pts[None,
                               np.random.choice(len(norm_pts), min(2000, len(norm_pts)), replace=False), :],
                               axis=2) if len(norm_pts) else np.array([[1.0]])
        near = float(max(1e-3, np.percentile(dists, 1)))
        far = float(np.percentile(dists, 99))

        # crop box (normalized) from point cloud percentiles
        lo = np.percentile(norm_pts, 2, axis=0).tolist() if len(norm_pts) else [-1, -1, -1]
        hi = np.percentile(norm_pts, 98, axis=0).tolist() if len(norm_pts) else [1, 1, 1]

        out = {
            "method": cfg.method,
            "center_colmap": center.tolist(),
            "scale": float(scale),
            "radius_colmap": float(radius),
            "up_axis_estimated_colmap": _estimate_up(imgs).tolist(),
            "up_axis_target": up.tolist(),
            "transform": T.tolist(),        # colmap -> normalized
            "transform_inverse": Tinv.tolist(),
            "near": near,
            "far": far,
            "crop_box_normalized": {"min": lo, "max": hi},
            "n_cameras": len(imgs),
            "n_points": len(pts),
            "conventions": {
                "handedness": "right-handed (COLMAP)",
                "note": "x_norm = transform @ [x_colmap; 1]. See docs for Blender/Unity.",
            },
        }
        atomic_write_json(ctx.layout.normalize_transform, out)
        ctx.logger.info("normalized scene: scale=%.4g radius=%.4g near=%.3f far=%.3f",
                        scale, radius, near, far)
        return {"scale": round(float(scale), 5), "near": round(near, 3), "far": round(far, 3)}


def _robust_center(P: np.ndarray, std: float) -> np.ndarray:
    c = np.median(P, axis=0)
    d = np.linalg.norm(P - c, axis=1)
    keep = d < (d.mean() + std * d.std())
    return P[keep].mean(axis=0) if keep.any() else c


def _estimate_up(imgs) -> np.ndarray:
    """Estimate world up as the mean of camera -y axes (image-plane up)."""
    ups = []
    for im in imgs.values():
        R = im.rotmat()
        ups.append(-R[1, :])  # camera up in world
    if not ups:
        return np.array([0.0, 0.0, 1.0])
    u = np.mean(ups, axis=0)
    n = np.linalg.norm(u)
    return u / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def _rotation_to_up(src_up: np.ndarray, dst_up: np.ndarray) -> np.ndarray:
    a = src_up / (np.linalg.norm(src_up) + 1e-9)
    b = dst_up / (np.linalg.norm(dst_up) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
