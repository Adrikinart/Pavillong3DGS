"""Stage: object masking (rembg / imported / none) with diagnostics."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..core.atomicio import atomic_write_json
from ..core.errors import InputValidationError
from ..core.stage import Artifact, Stage, StageContext


class GenerateMasksStage(Stage):
    name = "generate_masks"
    depends_on = ("filter_frames",)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("frames_filtered", ctx.layout.frames_filtered_dir, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        # masks optional overall: the artifact is optional when masking disabled
        optional = not ctx.config.generate_masks.enabled
        return [Artifact("masks", ctx.layout.masks_dir, "dir", optional=optional)]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.generate_masks.model_dump()

    def run(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.generate_masks
        if not c.enabled or c.backend == "none":
            ctx.logger.info("masking disabled; skipping")
            return {"enabled": False}

        frames = sorted([p for p in ctx.layout.frames_filtered_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        out = ctx.layout.masks_dir
        out.mkdir(parents=True, exist_ok=True)

        if c.backend == "imported":
            n = self._import_masks(ctx, frames, out)
        elif c.backend == "rembg":
            n = self._rembg(ctx, frames, out)
        elif c.backend == "sam2":
            raise NotImplementedError("sam2 backend not yet implemented; use rembg or imported")
        else:
            raise InputValidationError(f"unknown mask backend {c.backend}")

        diag = self._diagnostics(ctx, frames, out)
        atomic_write_json(out / "mask_diagnostics.json", diag)
        ctx.logger.info("masks: %d generated (%d empty, %d full, %d border-touch)",
                        n, diag["n_empty"], diag["n_full"], diag["n_border"])
        return {"enabled": True, "backend": c.backend, "n_masks": n,
                "n_empty": diag["n_empty"], "n_full": diag["n_full"]}

    def _import_masks(self, ctx, frames, out) -> int:
        c = ctx.config.generate_masks
        if not c.imported_dir:
            raise InputValidationError("masking backend=imported requires imported_dir")
        src = Path(c.imported_dir)
        n = 0
        for f in frames:
            for cand in (src / (f.stem + ".png"), src / (f.name + ".png"), src / f.name):
                if cand.exists():
                    shutil.copy2(cand, out / (f.stem + ".png"))
                    n += 1
                    break
        return n

    def _rembg(self, ctx, frames, out) -> int:
        from rembg import new_session, remove
        from PIL import Image
        c = ctx.config.generate_masks
        session = new_session(c.model)
        n = 0
        for f in frames:
            img = Image.open(f).convert("RGB")
            res = remove(img, session=session, only_mask=True)  # single-channel mask
            m = np.array(res)
            if c.dilate_px > 0:
                m = _dilate(m, c.dilate_px)
            Image.fromarray(m).save(out / (f.stem + ".png"))
            n += 1
        return n

    @staticmethod
    def _diagnostics(ctx, frames, out) -> dict[str, Any]:
        from PIL import Image
        c = ctx.config.generate_masks
        n_empty = n_full = n_border = n_small = 0
        areas = []
        for f in frames:
            mp = out / (f.stem + ".png")
            if not mp.exists():
                continue
            m = np.array(Image.open(mp).convert("L")) > 127
            frac = float(m.mean())
            areas.append(frac)
            if frac < c.minimum_area_fraction:
                n_empty += 1
            if frac > c.maximum_area_fraction:
                n_full += 1
            if m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any():
                n_border += 1
        return {"n_masks": len(areas), "n_empty": n_empty, "n_full": n_full,
                "n_border": n_border, "n_small": n_small,
                "mean_area_fraction": float(np.mean(areas)) if areas else 0.0}


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    try:
        import cv2
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        return cv2.dilate(mask, k)
    except Exception:
        return mask
