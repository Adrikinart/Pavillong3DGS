"""Stage: train a 3DGS model with the configured backend (default gsplat)."""

from __future__ import annotations

from typing import Any

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext


def resolve_train_run_id(ctx: StageContext) -> str:
    """Deterministic training-run id, STABLE across calls and processes.

    Must not depend on wall-clock time: the runner calls declared_outputs() both
    before and after run(), and evaluate/export run in separate processes — a
    timestamped id would point them at different directories. Default to one run
    per backend; pass --train-run-id (or train.train_run_id) for named/parallel
    trainings (e.g. sweeps)."""
    tr = ctx.params.get("train_run_id") or ctx.config.train.train_run_id
    return tr or f"{ctx.config.train.backend}_run"


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
        # If upstream data/config changed since the last training (e.g. a different
        # SfM model), existing checkpoints are stale -> train fresh instead of
        # resuming poses from a different reconstruction. Detect via a fingerprint
        # file stored next to the checkpoints.
        import shutil
        ckdir = ctx.layout.checkpoints_dir(tr)
        fp_file = ckdir / ".train_fingerprint"
        cur_fp = self.fingerprint(ctx)
        has_ckpts = ckdir.exists() and any(ckdir.glob("ckpt_*.pt"))
        # Resume ONLY when the checkpoints provably belong to this exact config+data
        # (fingerprint file present and matching). Missing file (legacy checkpoints)
        # or a mismatch => unknown/stale provenance => clear and train fresh, so we
        # never resume poses from a different reconstruction.
        fp_ok = fp_file.exists() and fp_file.read_text().strip() == cur_fp
        if has_ckpts and not fp_ok:
            ctx.logger.warning("existing checkpoints are stale/unverified (fingerprint "
                               "missing or changed) -> clearing for fresh training")
            shutil.rmtree(ctx.layout.training_dir(tr), ignore_errors=True)
        ckdir.mkdir(parents=True, exist_ok=True)
        fp_file.write_text(cur_fp)

        # convenience pointer to the most recent training (easy to find/monitor)
        latest = ctx.layout.trainings_dir / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(tr)
        except OSError:
            (ctx.layout.trainings_dir / "latest.txt").write_text(tr)

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
