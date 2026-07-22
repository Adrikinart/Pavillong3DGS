"""Stage: evaluate a trained model on held-out views (never on train metrics)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext
from .train import resolve_train_run_id


class EvaluateStage(Stage):
    name = "evaluate"
    depends_on = ("train",)
    needs_gpu = True

    def _tr(self, ctx: StageContext) -> str:
        return resolve_train_run_id(ctx)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("checkpoints", ctx.layout.checkpoints_dir(self._tr(ctx)), "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("eval_json", ctx.layout.eval_json(self._tr(ctx)), "file")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return {"train_run_id": self._tr(ctx), **ctx.config.evaluate.model_dump()}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        import time

        import torch

        from ..training.checkpoint import find_latest_valid, load_checkpoint
        from ..training.dataset import ColmapDataset
        from ..training.gaussians import create_splats
        from ..training.gsplat_backend import GsplatBackend
        from ..training.validation import evaluate_split

        import gsplat  # noqa: F401

        tr = self._tr(ctx)
        device = ctx.params.get("device", "cuda")
        ecfg = ctx.config.evaluate
        ckpt = find_latest_valid(ctx.layout.checkpoints_dir(tr))
        if ckpt is None:
            from ..core.errors import InputValidationError
            raise InputValidationError("no valid checkpoint to evaluate")

        # rebuild params from checkpoint
        train_ds = ColmapDataset(ctx.layout, "train", downscale=ctx.config.train.image_downscale)
        params, optimizers = create_splats(train_ds.points, train_ds.point_colors,
                                           ctx.config.train.sh_degree, device)
        step = load_checkpoint(ckpt, params, optimizers)
        backend = GsplatBackend()

        # Restore the per-image appearance model, if the run used one. Without
        # this the held-out views are scored against the UNCORRECTED render while
        # training optimised corrected ones — a systematic photometric penalty
        # that has nothing to do with reconstruction quality. Each view is scored
        # under the mean appearance of its OWN source clip (metadata + that clip's
        # training images only; the held-out pixels are never used).
        app_model, clip_to_train_idx = None, {}
        if ctx.config.train.appearance_embedding:
            from ..training.appearance import AppearanceModel, clip_key
            try:
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
                sd = (state.get("extra") or {}).get("appearance")
                if sd and "grids" in sd:                    # bilateral grid
                    from ..training.bilateral_grid import BilateralGrid
                    n, _, gl, gh, gw = sd["grids"].shape
                    app_model = BilateralGrid(n, grid_w=gw, grid_h=gh, grid_l=gl).to(device)
                elif sd:                                     # affine latents
                    app_model = AppearanceModel(sd["embed.weight"].shape[0],
                                                dim=sd["embed.weight"].shape[1]).to(device)
                if sd:
                    app_model.load_state_dict(sd)
                    app_model.eval()
                    for j, s in enumerate(train_ds.samples):
                        clip_to_train_idx.setdefault(clip_key(s.name), []).append(j)
                    ctx.logger.info("appearance model restored (%s, clips: %s)",
                                    type(app_model).__name__,
                                    ", ".join(sorted(clip_to_train_idx)))
            except Exception as e:  # noqa: BLE001
                ctx.logger.warning("could not restore appearance model: %s", e)

        near = 0.01
        if ctx.layout.normalize_transform.exists():
            d = json.loads(ctx.layout.normalize_transform.read_text())
            near = max(1e-3, float(d.get("near", 0.01)) * 0.5)

        results: dict[str, Any] = {"train_run_id": tr, "checkpoint": str(ckpt),
                                   "step": step, "splits": {}}
        peak_vram = 0
        for split in ecfg.splits:
            ds = ColmapDataset(ctx.layout, split, use_masks=ctx.config.train.use_masks,
                               downscale=ctx.config.train.image_downscale)
            if len(ds) == 0:
                ctx.logger.warning("split '%s' empty; skipping", split)
                continue
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.time()

            def render_fn(i, ds=ds):
                vm, K, w, h = ds.camera_tensors(i, device)
                r, _, _ = backend._rasterize(gsplat, params, vm, K, w, h,
                                             ctx.config.train.sh_degree, near, 1e10)
                r = r[0]
                if app_model is not None:
                    from ..training.appearance import clip_key
                    r = app_model.canonical_for(
                        r, clip_to_train_idx.get(clip_key(ds.samples[i].name)))
                return r

            res = evaluate_split(render_fn, ds, device,
                                 out_dir=ctx.layout.renders_dir(tr) / f"eval_{split}",
                                 compute_lpips=ecfg.compute_lpips,
                                 masked=ecfg.masked_metrics and ctx.config.train.use_masks)
            dt = time.time() - t0
            res["render_fps"] = round(len(ds) / dt, 2) if dt > 0 else None
            if torch.cuda.is_available():
                peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))
            results["splits"][split] = res
            ctx.logger.info("eval[%s]: psnr=%s ssim=%s lpips=%s fps=%s",
                            split, res["psnr"], res["ssim"], res["lpips"], res["render_fps"])

        # model stats
        n_gauss = int(params["means"].shape[0])
        ckpt_size = ckpt.stat().st_size
        results["model"] = {"n_gaussians": n_gauss, "checkpoint_bytes": ckpt_size,
                            "peak_vram_bytes": peak_vram}
        atomic_write_json(ctx.layout.eval_json(tr), results)

        self._append_registry(ctx, tr, results)
        try:
            from ..monitoring.report import write_report
            write_report(ctx, tr, results)
        except Exception as e:
            ctx.logger.warning("report generation failed: %s", e)

        primary = ecfg.splits[0] if ecfg.splits else None
        ps = results["splits"].get(primary, {}) if primary else {}
        return {"n_gaussians": n_gauss, "psnr": ps.get("psnr"), "ssim": ps.get("ssim"),
                "lpips": ps.get("lpips")}

    def _append_registry(self, ctx: StageContext, tr: str, results: dict[str, Any]) -> None:
        import csv

        from datetime import datetime, timezone

        reg = ctx.repo_root / "experiments" / "registry.csv"
        reg.parent.mkdir(parents=True, exist_ok=True)
        primary = ctx.config.evaluate.splits[0] if ctx.config.evaluate.splits else "test"
        ps = results["splits"].get(primary, {})
        colmap = {}
        try:
            from ..core.atomicio import read_json
            sfm = ctx.layout.colmap_dir / "sfm_stats.json"
            colmap = read_json(sfm) if sfm.exists() else {}
        except Exception:
            colmap = {}
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_id": tr, "dataset_id": ctx.layout.dataset_id,
            "backend": ctx.config.train.backend,
            "mapper_backend": ctx.config.run_colmap.mapper_backend,
            "max_iterations": ctx.config.train.max_iterations,
            "n_registered": colmap.get("n_registered_images"),
            "registration_ratio": colmap.get("registration_ratio"),
            "num_gaussians": results["model"]["n_gaussians"],
            "psnr": ps.get("psnr"), "ssim": ps.get("ssim"), "lpips": ps.get("lpips"),
            "render_fps": ps.get("render_fps"),
            "peak_vram_gb": round((results["model"]["peak_vram_bytes"] or 0) / 1e9, 2),
            "gpu_model": ctx.config.profile.gpu_type or "", "git_commit": _git(ctx),
            "config_resolved": str(ctx.layout.config_resolved),
            "checkpoint_path": results["checkpoint"],
            "report_path": str(ctx.layout.report_dir(tr) / "report.md"),
        }
        _append_row(reg, row)


def _append_row(reg, row: dict) -> None:
    """Append one row to the CSV registry; if the header changed, rotate the old
    file to .bak and start fresh (so the schema can evolve cleanly)."""
    import csv

    fieldnames = list(row.keys())
    header = ",".join(fieldnames)
    if reg.exists():
        first = reg.read_text(encoding="utf-8").splitlines()[:1]
        if first and first[0].strip() != header:
            reg.rename(reg.with_suffix(".csv.bak"))
    write_header = not reg.exists()
    with open(reg, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def _git(ctx: StageContext) -> str:
    from ..core.provenance import git_info
    return git_info(ctx.repo_root).get("git_sha") or ""
