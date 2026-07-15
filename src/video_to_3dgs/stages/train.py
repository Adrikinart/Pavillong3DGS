"""Stage: train a 3DGS model with the configured backend (default gsplat)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext


def resolve_train_run_id(ctx: StageContext) -> str:
    tr = ctx.params.get("train_run_id") or ctx.config.train.train_run_id
    if tr:
        return tr
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ctx.config.train.backend}_{stamp}"


class TrainStage(Stage):
    name = "train"
    depends_on = ("split_dataset",)
    needs_gpu = True

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [
            Artifact("colmap_sparse", ctx.layout.colmap_sparse0, "dir"),
            Artifact("split_train", ctx.layout.split_file("train"), "file"),
        ]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        tr = resolve_train_run_id(ctx)
        return [Artifact("checkpoints", ctx.layout.checkpoints_dir(tr), "dir")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return {"train_run_id": resolve_train_run_id(ctx), **ctx.config.train.model_dump()}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        from ..training.backend import TrainContext, get_backend

        tr = resolve_train_run_id(ctx)
        ctx.logger.info("training backend=%s run_id=%s", ctx.config.train.backend, tr)
        backend = get_backend(ctx.config.train.backend)
        backend.validate_env()

        device = ctx.params.get("device", "cuda")
        tctx = TrainContext(
            layout=ctx.layout, config=ctx.config, train_cfg=ctx.config.train,
            train_run_id=tr, device=device, logger=ctx.logger,
            resume=not ctx.force,
        )
        # write resolved training config for provenance
        atomic_write_json(ctx.layout.training_dir(tr) / "config_train.json",
                          {"train_run_id": tr, **ctx.config.train.model_dump()})

        result = backend.train(tctx)
        atomic_write_json(ctx.layout.training_dir(tr) / "train_result.json", {
            "status": result.status, "n_gaussians": result.n_gaussians,
            "final_checkpoint": str(result.final_checkpoint), "metrics": result.metrics,
        })
        if result.status == "PREEMPTED":
            # do not mark COMPLETED: raise so the runner records a re-runnable state
            from ..core.errors import StageExecutionError
            raise StageExecutionError("training preempted; checkpoint saved, resume to continue")
        return {"train_run_id": tr, "status": result.status,
                "n_gaussians": result.n_gaussians, **(result.metrics or {})}
