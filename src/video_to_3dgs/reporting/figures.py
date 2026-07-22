"""Static figure generators (matplotlib). Each returns the output path or None.

Robust to missing inputs: a figure that can't be built logs and returns None
rather than failing the whole run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.atomicio import iter_jsonl


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# --------------------------------------------------------------------------- #
# F1 — training curves
# --------------------------------------------------------------------------- #
def training_curves(metrics_jsonl: Path, out: Path) -> Path | None:
    if not Path(metrics_jsonl).exists():
        return None
    train = [r for r in iter_jsonl(metrics_jsonl) if r.get("kind") == "train"]
    val = [r for r in iter_jsonl(metrics_jsonl) if r.get("kind") == "val"]
    if not train:
        return None
    plt = _mpl()
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    def series(rows, key):
        xs = [r["step"] for r in rows if r.get(key) is not None]
        ys = [r[key] for r in rows if r.get(key) is not None]
        return xs, ys

    ax[0, 0].plot(*series(train, "loss")); ax[0, 0].set_title("loss"); ax[0, 0].set_yscale("log")
    ax[0, 1].plot(*series(train, "l1"), label="L1")
    ax[0, 1].plot(*series(train, "ssim"), label="SSIM"); ax[0, 1].legend(); ax[0, 1].set_title("L1 / SSIM (train)")
    ax[0, 2].plot(*series(train, "psnr"), color="gray", alpha=0.6, label="train")
    if val:
        ax[0, 2].plot(*series(val, "psnr"), "o-", color="C1", label="val")
    ax[0, 2].legend(); ax[0, 2].set_title("PSNR")
    if val:
        ax[1, 0].plot(*series(val, "ssim"), "o-"); ax[1, 0].set_title("val SSIM")
    ax[1, 1].plot(*series(train, "n_gaussians"), color="C2")
    ax[1, 1].set_title("# Gaussians (densify/prune)")
    ax[1, 2].plot(*series(train, "iters_per_s"), color="C3")
    ax[1, 2].set_title("iters / sec")
    for a in ax.flat:
        a.set_xlabel("iteration"); a.grid(alpha=0.3)
    fig.suptitle("Training curves")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# F2 — qualitative comparisons (GT | render | error)
# --------------------------------------------------------------------------- #
def qualitative_comparison(eval_json: Path, renders_dir: Path, gt_dir: Path,
                           out: Path, split: str = "test") -> Path | None:
    if not Path(eval_json).exists():
        return None
    data = json.loads(Path(eval_json).read_text())
    pv = data.get("splits", {}).get(split, {}).get("per_view", [])
    pv = [p for p in pv if p.get("psnr") is not None]
    if not pv:
        return None
    pv_sorted = sorted(pv, key=lambda p: p["psnr"])
    picks = [("worst", pv_sorted[0]), ("median", pv_sorted[len(pv_sorted) // 2]),
             ("best", pv_sorted[-1])]
    from PIL import Image
    plt = _mpl()
    fig, ax = plt.subplots(3, 3, figsize=(12, 11))
    for row, (label, p) in enumerate(picks):
        stem = Path(p["name"]).stem
        rp = renders_dir / f"{stem}.png"
        gp = _find_gt(gt_dir, p["name"])
        if not rp.exists() or gp is None:
            continue
        render = np.asarray(Image.open(rp).convert("RGB"), dtype=np.float32) / 255
        gt = np.asarray(Image.open(gp).convert("RGB").resize(
            (render.shape[1], render.shape[0])), dtype=np.float32) / 255
        err = np.abs(render - gt).mean(axis=2)
        ax[row, 0].imshow(gt); ax[row, 0].set_ylabel(f"{label}\nPSNR {p['psnr']}")
        ax[row, 1].imshow(render)
        im = ax[row, 2].imshow(err, cmap="inferno", vmin=0, vmax=0.5)
        if row == 0:
            ax[0, 0].set_title("Ground truth"); ax[0, 1].set_title("Render")
            ax[0, 2].set_title("Error |render-GT|")
        for c in range(3):
            ax[row, c].set_xticks([]); ax[row, c].set_yticks([])
    fig.suptitle(f"Held-out ({split}) qualitative comparison")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def _find_gt(gt_dir: Path, name: str) -> Path | None:
    for cand in (gt_dir / name, gt_dir / Path(name).name):
        if cand.exists():
            return cand
    return None


# --------------------------------------------------------------------------- #
# F3 — model statistics + per-view metrics
# --------------------------------------------------------------------------- #
def gaussian_stats(checkpoint: Path, out: Path) -> Path | None:
    if not Path(checkpoint).exists():
        return None
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    p = state["params"]
    opacity = torch.sigmoid(p["opacities"]).numpy()
    scale = torch.exp(p["scales"]).numpy().reshape(-1)
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(opacity, bins=60, color="C0"); ax[0].set_title("opacity"); ax[0].set_xlabel("opacity")
    ax[1].hist(np.log10(np.clip(scale, 1e-6, None)), bins=60, color="C1")
    ax[1].set_title("scale (log10)"); ax[1].set_xlabel("log10 scale")
    fig.suptitle(f"Gaussian statistics ({len(opacity)} gaussians)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def per_view_metrics(eval_json: Path, out: Path, split: str = "test") -> Path | None:
    if not Path(eval_json).exists():
        return None
    data = json.loads(Path(eval_json).read_text())
    pv = data.get("splits", {}).get(split, {}).get("per_view", [])
    pv = [p for p in pv if p.get("psnr") is not None]
    if not pv:
        return None
    plt = _mpl()
    names = [Path(p["name"]).stem[-6:] for p in pv]
    psnr = [p["psnr"] for p in pv]
    ssim = [p.get("ssim") for p in pv]
    lpips = [p.get("lpips") for p in pv]
    fig, ax = plt.subplots(3, 1, figsize=(max(6, len(pv) * 0.35), 9), sharex=True)
    ax[0].bar(names, psnr, color="C0"); ax[0].axhline(np.mean(psnr), ls="--", color="k")
    ax[0].set_ylabel("PSNR")
    ax[1].bar(names, ssim, color="C1"); ax[1].set_ylabel("SSIM")
    if any(v is not None for v in lpips):
        ax[2].bar(names, [v or 0 for v in lpips], color="C2")
    ax[2].set_ylabel("LPIPS"); ax[2].set_xlabel("held-out view")
    for a in ax:
        a.tick_params(axis="x", rotation=90); a.grid(alpha=0.3)
    fig.suptitle(f"Per-view metrics ({split})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# F4 — Gaussian centers (top / side)
# --------------------------------------------------------------------------- #
def gaussian_centers(checkpoint: Path, out: Path, crop_box: dict | None = None) -> Path | None:
    if not Path(checkpoint).exists():
        return None
    import torch
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    xyz = state["params"]["means"].numpy()
    if len(xyz) > 50000:
        xyz = xyz[np.random.choice(len(xyz), 50000, replace=False)]
    plt = _mpl()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(xyz[:, 0], xyz[:, 1], s=0.3, alpha=0.3, c="C0"); ax[0].set_title("top (XY)")
    ax[1].scatter(xyz[:, 0], xyz[:, 2], s=0.3, alpha=0.3, c="C0"); ax[1].set_title("side (XZ)")
    if crop_box:
        _draw_box(ax[0], crop_box, (0, 1)); _draw_box(ax[1], crop_box, (0, 2))
    for a in ax:
        a.set_aspect("equal"); a.grid(alpha=0.3)
    fig.suptitle("Gaussian centers (normalized frame)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    return out


def _draw_box(ax, crop_box: dict, axes: tuple[int, int]) -> None:
    lo, hi = np.array(crop_box["min"]), np.array(crop_box["max"])
    i, j = axes
    xs = [lo[i], hi[i], hi[i], lo[i], lo[i]]
    ys = [lo[j], lo[j], hi[j], hi[j], lo[j]]
    ax.plot(xs, ys, "r-", lw=1)


# --------------------------------------------------------------------------- #
# F5 — SfM result: sparse point cloud + camera frusta
# --------------------------------------------------------------------------- #
def camera_poses(sparse_dir: Path, out: Path, *, max_points: int = 60000,
                 frustum_scale: float = 0.05, elev: float = 20.0,
                 azim: float = -60.0, title: str | None = None) -> Path | None:
    """3D view + top-down view of the COLMAP solution: RGB point cloud with a
    frustum per registered camera. CPU-only (colmap_io + matplotlib)."""
    sparse_dir = Path(sparse_dir)
    if not (sparse_dir / "points3D.bin").exists():
        return None
    from ..colmap_io import read_model
    cams, images, points = read_model(sparse_dir)

    xyz = np.array([p.xyz for p in points.values()])
    rgb = np.array([p.rgb for p in points.values()]) / 255.0
    # robust crop: sparse SfM clouds have far outliers that would flatten the plot
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    ims = sorted(images.values(), key=lambda im: im.name)
    centers = np.array([im.camera_center() for im in ims])
    extent = float(np.linalg.norm(np.ptp(np.concatenate([xyz, centers]), axis=0)))
    fr = frustum_scale * extent

    # canonical frame: z = trajectory-plane normal signed by the images' up
    # direction, x = dominant trajectory axis — so the plot is right side up
    # regardless of COLMAP's arbitrary world axes
    _, _, vt = np.linalg.svd(centers - centers.mean(0), full_matrices=False)
    up_mean = np.mean([im.rotmat().T @ np.array([0.0, -1.0, 0.0]) for im in ims],
                      axis=0)
    e3 = vt[2] if np.dot(vt[2], up_mean) >= 0 else -vt[2]
    e1 = vt[0]
    e2 = np.cross(e3, e1)
    Rw = np.stack([e1, e2, e3])

    # frustum corner rays in world space, one pyramid per registered image
    segs = []
    for im in ims:
        cam = cams[im.camera_id]
        K = cam.K()
        corners_px = np.array([[0, 0], [cam.width, 0], [cam.width, cam.height],
                               [0, cam.height]], dtype=np.float64)
        d = np.column_stack([(corners_px[:, 0] - K[0, 2]) / K[0, 0],
                             (corners_px[:, 1] - K[1, 2]) / K[1, 1],
                             np.ones(4)])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        c = Rw @ im.camera_center()
        world = c + (Rw @ im.rotmat().T @ (d.T * fr)).T
        segs += [[c, w] for w in world]
        segs += [[world[i], world[(i + 1) % 4]] for i in range(4)]
    xyz = xyz @ Rw.T
    centers = centers @ Rw.T

    # view from behind the cameras, looking toward the point cloud
    obj_dir = xyz.mean(0) - centers.mean(0)
    if np.linalg.norm(obj_dir[:2]) > 1e-3 * extent:
        azim = float(np.degrees(np.arctan2(obj_dir[1], obj_dir[0]))) + 210.0

    plt = _mpl()
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    fig = plt.figure(figsize=(14, 7))

    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=0.4, alpha=0.6,
                linewidths=0, rasterized=True)
    ax3.add_collection3d(Line3DCollection(segs, colors="crimson", lw=0.4, alpha=0.6))
    ax3.plot(centers[:, 0], centers[:, 1], centers[:, 2], "-", color="crimson",
             lw=0.8, alpha=0.8)
    allp = np.concatenate([xyz, centers])
    span = np.ptp(allp, axis=0)
    mid, half = allp.min(0) + span / 2, span.max() / 2
    ax3.set_xlim(mid[0] - half, mid[0] + half)
    ax3.set_ylim(mid[1] - half, mid[1] + half)
    ax3.set_zlim(mid[2] - half, mid[2] + half)
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=elev, azim=azim)
    ax3.set_axis_off()
    ax3.set_title(f"{len(ims)} cameras · {len(points):,} points")

    # top-down: the canonical frame's xy is the camera-trajectory plane
    ax2 = fig.add_subplot(1, 2, 2)
    p2, c2 = xyz[:, :2], centers[:, :2]
    ax2.scatter(p2[:, 0], p2[:, 1], c=rgb, s=0.4, alpha=0.6, linewidths=0,
                rasterized=True)
    ax2.plot(c2[:, 0], c2[:, 1], "-", color="crimson", lw=1.0, alpha=0.8)
    ax2.scatter(c2[:, 0], c2[:, 1], color="crimson", s=6, zorder=3)
    ax2.set_aspect("equal")
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title("top view (camera trajectory plane)")

    fig.suptitle(title or "SfM: camera poses + sparse point cloud")
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out
