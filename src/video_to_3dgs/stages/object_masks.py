"""Stage: per-view subject masks by projecting a 3D volume (mesh or box).

Runs after ``normalize_scene`` because it needs camera poses. Writes PNG masks into the
same ``masks/`` directory the training dataset reads, so ``train.use_masks: true`` picks
them up with no further wiring.

Kept separate from ``generate_masks`` on purpose: that stage feeds SfM, and on a capture
whose subject is featureless (chrome) while the scene beside it is not (a checkerboard),
masking before SfM discards the features that give good poses. Here the masks affect only
the loss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.atomicio import atomic_write_json
from ..core.errors import InputValidationError
from ..core.stage import Artifact, Stage, StageContext


class ObjectMasksStage(Stage):
    name = "object_masks"
    depends_on = ("normalize_scene",)
    needs_gpu = False

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("colmap_sparse", ctx.layout.colmap_sparse0, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        if not ctx.config.object_masks.enabled:
            return []
        return [Artifact("object_masks", ctx.layout.masks_dir, "dir")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.object_masks.model_dump()

    def run(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.object_masks
        if not c.enabled:
            ctx.logger.info("object masks disabled; skipping")
            return {"enabled": False}

        import cv2

        from ..training.dataset import ColmapDataset

        pts = self._source_points(ctx, c)
        out = ctx.layout.masks_dir
        out.mkdir(parents=True, exist_ok=True)

        # masks_dir is per-DATASET, not per-config, so two configs that share object_name
        # but define different subject volumes overwrite each other's masks. That is easy to
        # do by accident (a "helmet" config and a "helmet + base" config on one capture) and
        # silent: training would simply use whichever volume was written last. Record the
        # volume that produced these masks and say so when it changes.
        prev = out / "object_mask_diagnostics.json"
        if prev.exists():
            try:
                import json
                old = json.loads(prev.read_text()).get("volume")
                if old and old != self._volume_key(c):
                    ctx.logger.warning(
                        "object masks: overwriting masks built for a DIFFERENT subject "
                        "volume (%s -> %s). Any training run that reused this dataset dir "
                        "with the old volume must be re-run to stay consistent; give the "
                        "configs different object_name values to keep both.",
                        old, self._volume_key(c))
            except Exception:                      # diagnostics are advisory, never fatal
                pass

        seen: set[str] = set()
        fracs: list[float] = []
        empty: list[str] = []
        for split in ("train", "val", "test"):
            try:
                ds = ColmapDataset(ctx.layout, split, cache_images=False)
            except Exception:                      # a split may not exist yet
                continue
            for s in ds.samples:
                if s.name in seen:
                    continue
                seen.add(s.name)
                mask = self._project(cv2, pts, s, c)
                frac = float((mask > 0).mean())
                fracs.append(frac)
                if frac == 0.0:
                    empty.append(s.name)
                cv2.imwrite(str(out / (Path(s.name).stem + ".png")), mask)

        if not fracs:
            raise InputValidationError("object_masks: no camera samples found")

        f = np.asarray(fracs)
        diag = {
            "n_masks": len(fracs), "n_empty": len(empty),
            "coverage_min": float(f.min()), "coverage_p50": float(np.percentile(f, 50)),
            "coverage_max": float(f.max()),
            "empty_views": empty[:20],
            "source": c.source,
            "volume": self._volume_key(c),
        }
        atomic_write_json(out / "object_mask_diagnostics.json", diag)
        ctx.logger.info("object masks: %d written, coverage min/p50/max %.1f%%/%.1f%%/%.1f%%, "
                        "%d empty", len(fracs), 100 * f.min(),
                        100 * np.percentile(f, 50), 100 * f.max(), len(empty))
        # Empty masks are not fatal -- on an orbit a few frames genuinely aim past the
        # subject -- but they contribute no loss, so say so rather than let them vanish.
        if empty:
            ctx.logger.warning("object masks: %d view(s) project empty (subject out of "
                               "frame); they contribute nothing to the loss: %s",
                               len(empty), ", ".join(empty[:5]))
        if c.min_area_fraction and float(np.percentile(f, 50)) < c.min_area_fraction:
            ctx.logger.warning("object masks: median coverage %.1f%% is below "
                               "min_area_fraction %.1f%% -- is the volume correct?",
                               100 * np.percentile(f, 50), 100 * c.min_area_fraction)
        return diag

    # ------------------------------------------------------------------ #
    @staticmethod
    def _volume_key(c) -> str:
        """Short identity of the subject volume, for detecting a silent redefinition."""
        if c.source == "box":
            return f"box:{c.box_center}:{c.box_half_extent}:d{c.dilate_px}"
        return f"mesh:{c.mesh_path}:s{c.splat_px}:d{c.dilate_px}"

    def _source_points(self, ctx: StageContext, c) -> np.ndarray:
        if c.source == "box":
            if c.box_center is None or c.box_half_extent is None:
                raise InputValidationError(
                    "object_masks: source='box' needs box_center and box_half_extent")
            centre = np.asarray(c.box_center, dtype=np.float64)
            h = float(c.box_half_extent)
            offs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                            dtype=np.float64)
            return centre[None, :] + offs * h

        if not c.mesh_path:
            raise InputValidationError("object_masks: source='mesh' needs mesh_path")
        mesh_path = Path(c.mesh_path)
        if not mesh_path.is_absolute():
            mesh_path = ctx.layout.run_dir / mesh_path
        if not mesh_path.exists():
            raise InputValidationError(f"object_masks: mesh not found at {mesh_path}")
        try:
            import open3d as o3d
        except ImportError as e:                   # optional dep: only the mesh path needs it
            raise InputValidationError(
                "object_masks: source='mesh' needs open3d (pip install open3d). "
                "For a self-contained alternative that needs no extra dependency and no "
                "prior reconstruction, use source: box with box_center/box_half_extent -- "
                "looser (a cube projects to a hexagon larger than the object) but adequate."
            ) from e
        m = o3d.io.read_triangle_mesh(str(mesh_path))
        pts = np.asarray(m.vertices)
        if len(pts) == 0:
            raise InputValidationError(f"object_masks: {mesh_path} has no vertices")
        # Strip isolated TSDF speckles: projected, each paints a stray blob of "subject"
        # onto background pixels.
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        clean, _ = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
        pts = np.asarray(clean.points)
        if len(pts) > 40000:                       # the outline is what matters
            pts = pts[:: len(pts) // 40000 + 1]
        ctx.logger.info("object masks: %d vertices from %s", len(pts), mesh_path.name)
        return pts

    def _project(self, cv2, pts: np.ndarray, s, c) -> np.ndarray:
        cam = (s.viewmat[:3, :3] @ pts.T).T + s.viewmat[:3, 3]
        front = cam[:, 2] > 1e-6
        mask = np.zeros((s.height, s.width), dtype=np.uint8)
        if not front.any():
            return mask
        cam = cam[front]
        uv = (s.K @ cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]

        if c.source == "box":
            hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
        else:
            ij = np.round(uv).astype(np.int64)
            ok = ((ij[:, 0] >= 0) & (ij[:, 0] < s.width)
                  & (ij[:, 1] >= 0) & (ij[:, 1] < s.height))
            if not ok.any():
                return mask
            mask[ij[ok, 1], ij[ok, 0]] = 255
            r = max(c.splat_px, 1)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.dilate(mask, k)
            ff = mask.copy()
            cv2.floodFill(ff, np.zeros((s.height + 2, s.width + 2), np.uint8), (0, 0), 255)
            mask = mask | cv2.bitwise_not(ff)      # fill interior holes between samples

        if c.dilate_px > 0:
            k = np.ones((c.dilate_px * 2 + 1,) * 2, np.uint8)
            mask = cv2.dilate(mask, k)
        return mask
