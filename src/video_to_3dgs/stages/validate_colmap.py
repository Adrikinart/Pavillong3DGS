"""Stage: validate a COLMAP reconstruction against configurable quality gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.atomicio import atomic_write_json
from ..core.errors import OutputValidationError
from ..core.stage import Artifact, Stage, StageContext
from .. import colmap_io


class ValidateColmapStage(Stage):
    name = "validate_colmap"
    depends_on = ("run_colmap",)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("colmap_sparse", ctx.layout.colmap_sparse0, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("colmap_report", ctx.layout.colmap_report, "file")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.validate_colmap.model_dump()

    def run(self, ctx: StageContext) -> dict[str, Any]:
        v = ctx.config.validate_colmap
        log = ctx.logger
        cams, imgs, pts = colmap_io.read_model(ctx.layout.colmap_sparse0)

        # count sibling models (fragmentation)
        n_models = sum(1 for p in ctx.layout.colmap_sparse.iterdir()
                       if p.is_dir() and (p / "images.bin").exists())
        n_input = self._n_input(ctx)
        stats = colmap_io.model_stats(ctx.layout.colmap_sparse0, n_input_images=n_input)

        warnings: list[str] = []
        checks: dict[str, bool] = {}

        ratio = stats.get("registration_ratio", 0.0)
        checks["registration_ratio"] = ratio >= v.minimum_registration_ratio
        if not checks["registration_ratio"]:
            warnings.append(f"low registration ratio {ratio:.2f} < {v.minimum_registration_ratio}")

        rep = stats.get("mean_reprojection_error") or 999
        checks["reprojection_error"] = rep <= v.maximum_mean_reprojection_error_px
        if not checks["reprojection_error"]:
            warnings.append(f"high reproj error {rep:.2f}px > {v.maximum_mean_reprojection_error_px}")

        checks["sparse_points"] = stats["n_points3D"] >= v.minimum_sparse_points
        if not checks["sparse_points"]:
            warnings.append(f"few sparse points {stats['n_points3D']} < {v.minimum_sparse_points}")

        checks["single_model"] = n_models <= 1 or v.allow_multiple_models
        if n_models > 1:
            warnings.append(f"{n_models} disconnected models found")

        # camera-jump detection: large gaps between consecutive camera centers
        centers = np.array([im.camera_center() for im in
                            sorted(imgs.values(), key=lambda i: i.name)])
        jumps = 0
        if len(centers) > 2:
            d = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            med = np.median(d) + 1e-9
            jumps = int((d > 8 * med).sum())
            if jumps:
                warnings.append(f"{jumps} large camera jumps (>8x median step)")

        # implausible intrinsics: focal wildly off image size
        for cid, c in cams.items():
            K = c.K()
            f = K[0, 0]
            if f < 0.3 * c.width or f > 5.0 * c.width:
                warnings.append(f"camera {cid} implausible focal {f:.0f} for width {c.width}")

        passed = all(checks.values())
        report = {
            "passed": passed,
            "checks": checks,
            "warnings": warnings,
            "n_models": n_models,
            "camera_jumps": jumps,
            "stats": stats,
        }
        atomic_write_json(ctx.layout.colmap_report, report)

        try:
            self._plot_trajectory(ctx, centers, pts)
        except Exception as e:  # non-fatal
            log.warning("trajectory plot failed: %s", e)

        for w in warnings:
            log.warning("colmap validation: %s", w)
        if not passed:
            msg = f"COLMAP validation gates failed: {warnings}"
            if v.hard_fail:
                raise OutputValidationError(msg)
            log.warning("%s (hard_fail=false -> continuing)", msg)
        else:
            log.info("colmap validation PASSED (%d imgs, %.2fpx reproj, %d pts)",
                     stats["n_registered_images"], rep, stats["n_points3D"])
        return {"passed": passed, "n_warnings": len(warnings),
                "registration_ratio": round(ratio, 3)}

    def _n_input(self, ctx: StageContext) -> int:
        d = ctx.layout.frames_filtered_dir
        if not d.exists():
            return 0
        return sum(1 for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    @staticmethod
    def _plot_trajectory(ctx: StageContext, centers: np.ndarray, pts) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        if len(pts):
            P = np.array([p.xyz for p in pts.values()])
            idx = np.random.choice(len(P), min(5000, len(P)), replace=False)
            ax1.scatter(P[idx, 0], P[idx, 1], s=0.5, c="gray", alpha=0.4)
        if len(centers):
            ax1.plot(centers[:, 0], centers[:, 1], "-r", lw=1)
            ax1.scatter(centers[:, 0], centers[:, 1], s=8, c="blue")
        ax1.set_title("top view (XY)")
        ax1.set_aspect("equal")
        ax2 = fig.add_subplot(122)
        if len(centers):
            ax2.plot(centers[:, 0], centers[:, 2], "-r", lw=1)
            ax2.scatter(centers[:, 0], centers[:, 2], s=8, c="blue")
        ax2.set_title("side view (XZ)")
        ax2.set_aspect("equal")
        fig.tight_layout()
        out = ctx.layout.colmap_dir / "trajectory.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
