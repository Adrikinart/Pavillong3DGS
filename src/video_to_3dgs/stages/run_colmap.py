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


_IMG_EXT = (".jpg", ".jpeg", ".png")


def frame_images(d: Path) -> list[Path]:
    """Real frame images in a dir: excludes contact sheets and sub-directories."""
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXT
                  and not p.name.startswith("contact_sheet"))


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
    # surface why the mapper registered nothing (e.g. "No good initial image pair")
    if args[0] == "mapper":
        out = (r.stdout or "") + (r.stderr or "")
        tail = "\n".join(out.splitlines()[-15:])
        log.info("mapper output tail:\n%s", tail)


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
        return len(frame_images(ctx.layout.frames_filtered_dir))

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
            for p in frame_images(ctx.layout.frames_filtered_dir):
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
                    cand = {"attempt": i, "matcher": att["matcher"], "stats": stats,
                            "db": db, "sparse0": model0}
                    # keep the BEST attempt (most registered images) — COLMAP's
                    # incremental mapper is nondeterministic, so a later attempt can
                    # be worse; never let it overwrite a better earlier result.
                    if chosen is None or (stats["n_registered_images"]
                                          > chosen["stats"]["n_registered_images"]):
                        chosen = cand
                    # good enough? stop once an attempt clears the acceptance gate
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
        # COLMAP >= 3.13 renamed the GPU flags: SiftExtraction/SiftMatching.use_gpu
        # -> FeatureExtraction/FeatureMatching.use_gpu (max_num_features stayed on
        # SiftExtraction). CPU-only builds omit these flags entirely, so on a build
        # without them we simply skip the flag (default is CPU).
        ext_gpu = self._gpu_flag(cb, "feature_extractor", "FeatureExtraction.use_gpu",
                                 "SiftExtraction.use_gpu")
        match_gpu = self._gpu_flag(cb, "exhaustive_matcher", "FeatureMatching.use_gpu",
                                   "SiftMatching.use_gpu")
        feat = ["feature_extractor", "--database_path", str(db), "--image_path", str(images),
                "--ImageReader.camera_model", c.camera_model,
                "--ImageReader.single_camera", "1" if c.single_camera else "0",
                "--SiftExtraction.max_num_features", str(c.sift_max_features)]
        if ext_gpu:
            feat += [f"--{ext_gpu}", gpu]
        if mask_dir is not None:
            feat += ["--ImageReader.mask_path", str(mask_dir)]
        _run_colmap(feat, log, cb)

        if matcher == "sequential":
            m = ["sequential_matcher", "--database_path", str(db),
                 "--SequentialMatching.overlap", str(c.sequential_overlap)]
            if c.loop_detection and c.vocab_tree_path:
                m += ["--SequentialMatching.loop_detection", "1",
                      "--SequentialMatching.vocab_tree_path", c.vocab_tree_path]
        elif matcher == "vocab_tree":
            m = ["vocab_tree_matcher", "--database_path", str(db)]
            if c.vocab_tree_path:
                m += ["--VocabTreeMatching.vocab_tree_path", c.vocab_tree_path]
        else:  # exhaustive
            m = ["exhaustive_matcher", "--database_path", str(db)]
        if match_gpu:
            m += [f"--{match_gpu}", gpu]
        _run_colmap(m, log, cb)

        # incremental mapping -> sparse/0
        _run_colmap(["mapper", "--database_path", str(db), "--image_path", str(images),
                     "--output_path", str(sparse)], log, cb)

    @staticmethod
    def _gpu_flag(colmap_bin: str, subcmd: str, new_name: str, old_name: str) -> str | None:
        """Return the use_gpu option name this colmap build accepts, or None (CPU build)."""
        try:
            h = subprocess.run([colmap_bin, subcmd, "--help"], capture_output=True,
                               text=True, timeout=30).stdout
        except Exception:
            return new_name
        if new_name.split(".")[0] in h and "use_gpu" in h:
            return new_name
        if old_name in h:
            return old_name
        return None

        _run_colmap(["mapper", "--database_path", str(db), "--image_path", str(images),
                     "--output_path", str(sparse)], log, cb)
