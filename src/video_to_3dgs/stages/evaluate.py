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
                return r[0]

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

        reg = ctx.repo_root / "experiments" / "registry.csv"
        reg.parent.mkdir(parents=True, exist_ok=True)
        primary = ctx.config.evaluate.splits[0] if ctx.config.evaluate.splits else "test"
        ps = results["splits"].get(primary, {})
        row = {
            "run_id": tr, "dataset_id": ctx.layout.dataset_id,
            "backend": ctx.config.train.backend, "git_commit": _git(ctx),
            "gpu_model": ctx.config.profile.gpu_type or "", "num_gaussians": results["model"]["n_gaussians"],
            "psnr": ps.get("psnr"), "ssim": ps.get("ssim"), "lpips": ps.get("lpips"),
            "render_fps": ps.get("render_fps"),
            "peak_vram_bytes": results["model"]["peak_vram_bytes"],
            "checkpoint_path": results["checkpoint"],
            "report_path": str(ctx.layout.report_dir(tr) / "report.md"),
        }
        exists = reg.exists()
        with open(reg, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)


def _git(ctx: StageContext) -> str:
    from ..core.provenance import git_info
    return git_info(ctx.repo_root).get("git_sha") or ""
