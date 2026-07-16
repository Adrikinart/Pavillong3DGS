"""Aggregate all metrics for a run into a summary JSON + a Markdown table.

Pulls from: eval.json (held-out image quality + efficiency), colmap
sfm_stats/validation_report (reconstruction), train_result.json + status/train.json
(training), and manifest.json (provenance).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ..core.atomicio import atomic_write_json, read_json


def _safe_load(p: Path) -> dict[str, Any]:
    try:
        return read_json(p) if p.exists() else {}
    except Exception:
        return {}


def aggregate(layout, train_run_id: str) -> dict[str, Any]:
    ev = _safe_load(layout.eval_json(train_run_id))
    colmap = _safe_load(layout.colmap_dir / "sfm_stats.json")
    colmap_val = _safe_load(layout.colmap_report)
    train_res = _safe_load(layout.training_dir(train_run_id) / "train_result.json")
    train_status = _safe_load(layout.status_file("train"))
    manifest = _safe_load(layout.manifest)

    summary: dict[str, Any] = {
        "dataset_id": layout.dataset_id,
        "train_run_id": train_run_id,
        "image_quality": {},
        "efficiency": {},
        "reconstruction": {},
        "training": {},
        "provenance": {},
    }

    # image quality (per split, aggregate + dispersion)
    for split, res in ev.get("splits", {}).items():
        pv = [p for p in res.get("per_view", []) if p.get("psnr") is not None]
        q = {"n_views": res.get("n_views"), "psnr": res.get("psnr"),
             "ssim": res.get("ssim"), "lpips": res.get("lpips"),
             "render_fps": res.get("render_fps")}
        for key in ("psnr", "ssim", "lpips"):
            vals = [p[key] for p in pv if p.get(key) is not None]
            if len(vals) > 1:
                q[f"{key}_std"] = round(statistics.pstdev(vals), 4)
                q[f"{key}_min"] = round(min(vals), 4)
                q[f"{key}_max"] = round(max(vals), 4)
        summary["image_quality"][split] = q

    model = ev.get("model", {})
    summary["efficiency"] = {
        "n_gaussians": model.get("n_gaussians"),
        "checkpoint_mb": round((model.get("checkpoint_bytes") or 0) / 1e6, 2),
        "peak_vram_gb": round((model.get("peak_vram_bytes") or 0) / 1e9, 3),
    }
    summary["reconstruction"] = {
        "n_input_images": colmap.get("n_input_images"),
        "n_registered_images": colmap.get("n_registered_images"),
        "registration_ratio": colmap.get("registration_ratio"),
        "mean_reprojection_error_px": colmap.get("mean_reprojection_error"),
        "n_points3D": colmap.get("n_points3D"),
        "mean_track_length": colmap.get("mean_track_length"),
        "validation_passed": colmap_val.get("passed"),
    }
    summary["training"] = {
        "status": train_res.get("status"),
        "final_step": (train_res.get("metrics") or {}).get("final_step"),
        "best_val_psnr": (train_res.get("metrics") or {}).get("best_val_psnr"),
        "n_gaussians": train_res.get("n_gaussians"),
        "duration_s": train_status.get("duration_s"),
    }
    sw = manifest.get("software", {})
    summary["provenance"] = {
        "git_sha": sw.get("git_sha"), "torch": sw.get("torch"),
        "cuda": sw.get("torch_cuda_build"), "gpu": (sw.get("nvidia", {}).get("gpus") or [{}])[0].get("name"),
        "framework_version": sw.get("framework_version"),
    }
    return summary


def write_summary_and_table(layout, train_run_id: str) -> tuple[Path, Path]:
    summary = aggregate(layout, train_run_id)
    mdir = layout.training_dir(train_run_id) / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    js = mdir / "metrics_summary.json"
    atomic_write_json(js, summary)
    md = mdir / "metrics_table.md"
    md.write_text(_render_table(summary), encoding="utf-8")
    return js, md


def _render_table(s: dict[str, Any]) -> str:
    lines = [f"# Metrics — {s['dataset_id']} / {s['train_run_id']}", ""]
    for split, q in s.get("image_quality", {}).items():
        lines += [f"## Held-out image quality ({split})", "",
                  "| metric | value | std | min | max |", "|---|---|---|---|---|"]
        for k in ("psnr", "ssim", "lpips"):
            lines.append(f"| {k.upper()} | {q.get(k)} | {q.get(k+'_std','')} | "
                         f"{q.get(k+'_min','')} | {q.get(k+'_max','')} |")
        lines += [f"| render FPS | {q.get('render_fps')} | | | |",
                  f"| # views | {q.get('n_views')} | | | |", ""]
    e = s["efficiency"]
    lines += ["## Efficiency", "",
              f"- # Gaussians: {e.get('n_gaussians')}",
              f"- checkpoint: {e.get('checkpoint_mb')} MB",
              f"- peak VRAM: {e.get('peak_vram_gb')} GB", ""]
    r = s["reconstruction"]
    lines += ["## Reconstruction (COLMAP)", "",
              f"- registered: {r.get('n_registered_images')}/{r.get('n_input_images')} "
              f"(ratio {r.get('registration_ratio')})",
              f"- mean reproj error: {r.get('mean_reprojection_error_px')} px",
              f"- points: {r.get('n_points3D')}, mean track length: {r.get('mean_track_length')}", ""]
    t = s["training"]
    lines += ["## Training", "",
              f"- status: {t.get('status')}, final step: {t.get('final_step')}",
              f"- best val PSNR: {t.get('best_val_psnr')}",
              f"- duration: {t.get('duration_s')} s", ""]
    p = s["provenance"]
    lines += ["## Provenance", "",
              f"- git: `{p.get('git_sha')}` | torch {p.get('torch')} (cuda {p.get('cuda')}) | "
              f"GPU {p.get('gpu')}", ""]
    return "\n".join(lines)
