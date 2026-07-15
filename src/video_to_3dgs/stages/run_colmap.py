"""Stage: COLMAP structure-from-motion.

Runs entirely on node-local scratch (the SQLite database is hostile to NFS) and
syncs the verified sparse model + undistorted images back to the run dir. Uses
the ``colmap`` CLI so it works with any install (conda, module, system). Tries a
bounded set of fallback configs before giving up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..core.errors import InputValidationError, StageExecutionError
from ..core.scratch import ScratchContext
from ..core.stage import Artifact, Stage, StageContext
from .. import colmap_io


def resolve_colmap_bin(configured: str = "colmap") -> str:
    """Locate the colmap binary robustly.

    Honors an explicit path/name if it resolves; otherwise falls back to the
    running interpreter's env prefix (``<sys.prefix>/bin/colmap``) so the stage
    works even when PATH does not include the env bin (a common Slurm pitfall).
    """
    if os.path.sep in configured or configured != "colmap":
        if Path(configured).exists() or shutil.which(configured):
            return configured
    found = shutil.which(configured)
    if found:
        return found
    cand = Path(sys.prefix) / "bin" / "colmap"
    if cand.exists():
        return str(cand)
    raise InputValidationError(
        "colmap binary not found on PATH nor in the env prefix "
        f"({cand}). Install colmap into the env (conda-forge) or set "
        "run_colmap.colmap_bin to its absolute path.")


def _run_colmap(args: list[str], log, colmap_bin: str = "colmap") -> None:
    cmd = [colmap_bin] + args
    log.info("colmap %s", args[0])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
    if r.returncode != 0:
        raise StageExecutionError(
            f"colmap {args[0]} failed (rc={r.returncode}): {r.stderr[-800:]}")


def _prepare_masks(frames_dir: Path, masks_src: Path, mask_dst: Path, log) -> bool:
    """COLMAP expects a mask named '<image>.png' per image (0 = ignore).

    Our masks are alpha PNGs keyed by frame stem. Build the expected names.
    """
    if not masks_src.exists():
        return False
    mask_dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in frames_dir.iterdir():
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        cand = masks_src / (img.stem + ".png")
        if cand.exists():
            shutil.copy2(cand, mask_dst / (img.name + ".png"))
            n += 1
    log.info("prepared %d colmap masks", n)
    return n > 0


class RunColmapStage(Stage):
    name = "run_colmap"
    depends_on = ("filter_frames",)
    needs_gpu = False  # CPU SIFT by default; GPU optional

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("frames_filtered", ctx.layout.frames_filtered_dir, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [
            Artifact("colmap_sparse", ctx.layout.colmap_sparse0, "dir"),
            Artifact("colmap_images", ctx.layout.colmap_images, "dir"),
        ]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.run_colmap.model_dump()

    def _n_input_images(self, ctx: StageContext) -> int:
        return sum(1 for p in ctx.layout.frames_filtered_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    def run(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.run_colmap
        log = ctx.logger
        colmap_bin = resolve_colmap_bin(c.colmap_bin)
        n_input = self._n_input_images(ctx)
        log.info("colmap=%s on %d images (matcher=%s gpu=%s)", colmap_bin, n_input,
                 c.matcher, c.use_gpu)

        attempts = [{"matcher": c.matcher}]
        if c.fallback_to_exhaustive and c.matcher != "exhaustive":
            attempts.append({"matcher": "exhaustive"})

        with ScratchContext(ctx.layout.dataset_id,
                            scratch_root=ctx.config.storage.scratch_root) as scr:
            # stage inputs onto scratch (resolves symlinks -> real images)
            images = scr.path("images")
            images.mkdir(parents=True, exist_ok=True)
            for p in ctx.layout.frames_filtered_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    shutil.copy2(p.resolve(), images / p.name)

            mask_dir = None
            if c.use_masks and ctx.layout.masks_dir.exists():
                md = scr.path("colmap_masks")
                if _prepare_masks(images, ctx.layout.masks_dir, md, log):
                    mask_dir = md

            last_err: Exception | None = None
            chosen: dict[str, Any] | None = None
            for i, att in enumerate(attempts):
                db = scr.path(f"database_{i}.db")
                sparse = scr.path(f"sparse_{i}")
                sparse.mkdir(parents=True, exist_ok=True)
                try:
                    self._reconstruct(scr, db, images, sparse, mask_dir, c, att["matcher"],
                                      log, colmap_bin)
                    model0 = sparse / "0"
                    if not (model0 / "images.bin").exists():
                        raise StageExecutionError("mapper produced no model")
                    stats = colmap_io.model_stats(model0, n_input_images=n_input)
                    log.info("attempt %d (%s): registered %d/%d, reproj=%.3f",
                             i, att["matcher"], stats["n_registered_images"], n_input,
                             stats["mean_reprojection_error"] or -1)
                    chosen = {"attempt": i, "matcher": att["matcher"], "stats": stats,
                              "db": db, "sparse0": model0}
                    # good enough? accept first attempt reaching the ratio gate
                    if stats.get("registration_ratio", 0) >= \
                            ctx.config.validate_colmap.minimum_registration_ratio:
                        break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    log.warning("colmap attempt %d (%s) failed: %s", i, att["matcher"], e)

            if chosen is None:
                scr.retain()
                raise StageExecutionError(f"all colmap attempts failed: {last_err}")

            # undistortion -> training-ready images + sparse
            final_images = scr.path("undist/images")
            final_sparse = scr.path("undist/sparse")
            if c.undistort:
                undist = scr.path("undist")
                _run_colmap(["image_undistorter", "--image_path", str(images),
                             "--input_path", str(chosen["sparse0"]),
                             "--output_path", str(undist), "--output_type", "COLMAP"],
                            log, colmap_bin)
                # undistorter writes undist/images and undist/sparse
            else:
                final_images = images
                final_sparse = chosen["sparse0"]

            # sync back: database, sparse/0, images
            ctx.layout.colmap_dir.mkdir(parents=True, exist_ok=True)
            pairs = [
                (chosen["db"], ctx.layout.colmap_db),
                (final_sparse, ctx.layout.colmap_sparse0),
                (final_images, ctx.layout.colmap_images),
            ]
            scr.sync_back(pairs)

            metrics = {**chosen["stats"], "chosen_matcher": chosen["matcher"],
                       "attempt": chosen["attempt"], "undistorted": c.undistort}
            # write the raw stats alongside for the validator
            from ..core.atomicio import atomic_write_json
            atomic_write_json(ctx.layout.colmap_dir / "sfm_stats.json", metrics)
            return {"n_registered_images": metrics["n_registered_images"],
                    "registration_ratio": round(metrics.get("registration_ratio", 0), 3),
                    "n_points3D": metrics["n_points3D"],
                    "mean_reproj_error": round(metrics.get("mean_reprojection_error") or -1, 3)}

    def _reconstruct(self, scr, db: Path, images: Path, sparse: Path, mask_dir,
                     c, matcher: str, log, colmap_bin: str) -> None:
        cb = colmap_bin
        gpu = "1" if c.use_gpu else "0"
        feat = ["feature_extractor", "--database_path", str(db), "--image_path", str(images),
                "--ImageReader.camera_model", c.camera_model,
                "--ImageReader.single_camera", "1" if c.single_camera else "0",
                "--SiftExtraction.use_gpu", gpu,
                "--SiftExtraction.max_num_features", str(c.sift_max_features)]
        if mask_dir is not None:
            feat += ["--ImageReader.mask_path", str(mask_dir)]
        _run_colmap(feat, log, cb)

        if matcher == "sequential":
            m = ["sequential_matcher", "--database_path", str(db),
                 "--SiftMatching.use_gpu", gpu,
                 "--SequentialMatching.overlap", str(c.sequential_overlap)]
            if c.loop_detection and c.vocab_tree_path:
                m += ["--SequentialMatching.loop_detection", "1",
                      "--SequentialMatching.vocab_tree_path", c.vocab_tree_path]
        elif matcher == "vocab_tree":
            m = ["vocab_tree_matcher", "--database_path", str(db),
                 "--SiftMatching.use_gpu", gpu]
            if c.vocab_tree_path:
                m += ["--VocabTreeMatching.vocab_tree_path", c.vocab_tree_path]
        else:  # exhaustive
            m = ["exhaustive_matcher", "--database_path", str(db),
                 "--SiftMatching.use_gpu", gpu]
        _run_colmap(m, log, cb)

        _run_colmap(["mapper", "--database_path", str(db), "--image_path", str(images),
                     "--output_path", str(sparse)], log, cb)
