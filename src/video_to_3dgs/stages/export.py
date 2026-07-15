"""Stage: export a trained model (.ply, cameras, transforms, conventions doc)."""

from __future__ import annotations

import json
import shutil
from typing import Any

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext
from .train import resolve_train_run_id


class ExportStage(Stage):
    name = "export"
    depends_on = ("train",)
    needs_gpu = False  # ply export is CPU-only (loads checkpoint on CPU)

    def _tr(self, ctx: StageContext) -> str:
        return resolve_train_run_id(ctx)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("checkpoints", ctx.layout.checkpoints_dir(self._tr(ctx)), "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("export_dir", ctx.layout.exports_dir(self._tr(ctx)), "dir")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return {"train_run_id": self._tr(ctx), **ctx.config.export.model_dump()}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        from ..training.backend import TrainContext, get_backend
        from ..training.checkpoint import find_latest_valid

        tr = self._tr(ctx)
        out = ctx.layout.exports_dir(tr)
        out.mkdir(parents=True, exist_ok=True)
        ckpt = find_latest_valid(ctx.layout.checkpoints_dir(tr))
        if ckpt is None:
            from ..core.errors import InputValidationError
            raise InputValidationError("no valid checkpoint to export")

        produced: list[str] = []
        fmts = ctx.config.export.formats

        if "ply" in fmts:
            backend = get_backend(ctx.config.train.backend)
            tctx = TrainContext(layout=ctx.layout, config=ctx.config,
                                train_cfg=ctx.config.train, train_run_id=tr,
                                device="cpu", logger=ctx.logger)
            ply = out / "point_cloud.ply"
            backend.export_ply(tctx, ckpt, ply)
            produced.append("point_cloud.ply")
            ctx.logger.info("exported %s (%.1f MB)", ply.name, ply.stat().st_size / 1e6)

        if "transforms" in fmts and ctx.layout.normalize_transform.exists():
            shutil.copy2(ctx.layout.normalize_transform, out / "normalize_transform.json")
            produced.append("normalize_transform.json")

        if "cameras" in fmts:
            self._export_cameras(ctx, out)
            produced.append("cameras.json")

        self._write_conventions(out)
        produced.append("COORDINATES.md")
        atomic_write_json(out / "export_manifest.json",
                          {"train_run_id": tr, "checkpoint": str(ckpt), "produced": produced})
        return {"n_exports": len(produced), "formats": produced}

    def _export_cameras(self, ctx: StageContext, out) -> None:
        from .. import colmap_io
        cams, imgs, _ = colmap_io.read_model(ctx.layout.colmap_sparse0)
        recs = []
        for im in sorted(imgs.values(), key=lambda i: i.name):
            c = cams[im.camera_id]
            recs.append({
                "name": im.name, "camera_id": im.camera_id,
                "qvec_wxyz": im.qvec.tolist(), "tvec": im.tvec.tolist(),
                "camera_center": im.camera_center().tolist(),
                "model": c.model, "width": c.width, "height": c.height,
                "K": c.K().tolist(),
            })
        atomic_write_json(out / "cameras.json", {"cameras": recs})

    @staticmethod
    def _write_conventions(out) -> None:
        text = """# Coordinate conventions for this export

## Source (COLMAP / gsplat)
- Right-handed world coordinates.
- Camera extrinsics stored as world-to-camera (COLMAP `images.bin`): `x_cam = R * x_world + t`.
- `qvec` is `(w, x, y, z)`. `camera_center = -R^T t`.
- The `.ply` Gaussian positions are in the **normalized** frame
  (`x_norm = transform @ x_colmap`); see `normalize_transform.json` for the 4x4
  `transform` and its inverse to map back to COLMAP coordinates.

## Units & scale
- The normalization scales cameras so the 95th-percentile orbit radius ~= 1.0.
  Multiply by `radius_colmap` (in `normalize_transform.json`) to recover COLMAP units.
  COLMAP units are an arbitrary SfM scale (not metric) unless you added a scale bar.

## Blender import
- Blender is right-handed, Z-up. COLMAP/gsplat here is right-handed with the
  estimated up axis recorded in `normalize_transform.json` (`up_axis_estimated_colmap`).
  Apply a rotation aligning that up vector to +Z, then import the `.ply`.

## Unity import
- Unity is left-handed, Y-up. Convert by negating one axis (e.g. Z) to flip
  handedness, then map the recorded up axis to +Y. Camera forward in COLMAP is +Z
  (into the scene); Unity camera forward is +Z as well but with flipped handedness,
  so invert X after the handedness flip.

## Limitations
- Gaussian splats are a view-dependent radiance representation, not a watertight mesh.
- For mesh/geometry workflows use a geometry-oriented backend (2DGS/SuGaR) — not
  wired in this iteration.
"""
        (out / "COORDINATES.md").write_text(text, encoding="utf-8")
