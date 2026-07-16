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
