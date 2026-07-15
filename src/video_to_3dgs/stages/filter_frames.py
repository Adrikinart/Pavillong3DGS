"""Stage: score frames and filter out blurry / bad / near-duplicate ones.

Nothing is deleted: rejected frames are moved to ``rejected/`` with a reason,
and accepted frames are symlinked into ``frames_filtered/`` (the image source
for COLMAP). A coverage safeguard keeps at least ``min_kept`` frames.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext


def _load_gray(path: Path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def _blur_var(gray) -> float:
    import cv2
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ahash(gray, size: int = 8) -> int:
    import cv2
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class FilterFramesStage(Stage):
    name = "filter_frames"
    depends_on = ("extract_frames",)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("frames_dir", ctx.layout.frames_dir, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [
            Artifact("frames_filtered", ctx.layout.frames_filtered_dir, "dir"),
            Artifact("frame_scores", ctx.layout.frame_scores_csv, "file"),
        ]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.filter_frames.model_dump()

    def run(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.filter_frames
        frames = sorted([p for p in ctx.layout.frames_dir.iterdir()
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        if not frames:
            from ..core.errors import OutputValidationError
            raise OutputValidationError("no frames to filter")

        rows: list[dict[str, Any]] = []
        prev_hash: int | None = None
        for p in frames:
            gray = _load_gray(p)
            if gray is None:
                rows.append({"image": p.name, "reason": "unreadable", "accepted": False})
                continue
            blur = _blur_var(gray)
            lum = float(gray.mean())
            sat = float((gray >= 250).mean())
            under = float((gray <= 12).mean())
            ah = _ahash(gray)
            dup_dist = _hamming(ah, prev_hash) if prev_hash is not None else 64

            reasons: list[str] = []
            if c.enabled:
                if blur < c.blur_var_min:
                    reasons.append(f"blurry(var={blur:.0f}<{c.blur_var_min})")
                if lum < c.luminance_min:
                    reasons.append("too_dark")
                if lum > c.luminance_max:
                    reasons.append("too_bright")
                if sat > c.saturated_frac_max:
                    reasons.append("saturated")
                if under > c.underexposed_frac_max:
                    reasons.append("underexposed")
                if dup_dist < c.duplicate_phash_dist_min:
                    reasons.append(f"near_duplicate(d={dup_dist})")
            accepted = len(reasons) == 0
            if accepted:
                prev_hash = ah
            rows.append({"image": p.name, "blur_var": round(blur, 1), "luminance": round(lum, 1),
                         "saturated_frac": round(sat, 4), "underexposed_frac": round(under, 4),
                         "ahash": ah, "dup_dist": dup_dist,
                         "reason": ";".join(reasons), "accepted": accepted})

        # coverage safeguard: if too few accepted, relax by keeping highest-blur rejects
        accepted_rows = [r for r in rows if r.get("accepted")]
        if len(accepted_rows) < c.min_kept:
            deficit = c.min_kept - len(accepted_rows)
            rescuable = sorted([r for r in rows if not r.get("accepted") and "unreadable" not in r.get("reason", "")],
                               key=lambda r: r.get("blur_var", 0), reverse=True)[:deficit]
            for r in rescuable:
                r["accepted"] = True
                r["reason"] = (r.get("reason", "") + ";rescued_for_coverage").strip(";")
            ctx.logger.warning("coverage safeguard rescued %d frames to reach min_kept=%d",
                               len(rescuable), c.min_kept)
            accepted_rows = [r for r in rows if r.get("accepted")]

        # optional keep_fraction cap (drop worst-blur accepted)
        if c.keep_fraction is not None:
            cap = max(c.min_kept, int(len(rows) * c.keep_fraction))
            if len(accepted_rows) > cap:
                worst = sorted(accepted_rows, key=lambda r: r.get("blur_var", 0))[: len(accepted_rows) - cap]
                for r in worst:
                    r["accepted"] = False
                    r["reason"] = (r.get("reason", "") + ";keep_fraction_cap").strip(";")
                accepted_rows = [r for r in rows if r.get("accepted")]

        # build frames_filtered (symlinks) and move rejected — clean dir FIRST so the
        # CSV we write into it below is not removed by the rmtree.
        import shutil
        fdir = ctx.layout.frames_filtered_dir
        if fdir.exists():
            shutil.rmtree(fdir)
        fdir.mkdir(parents=True, exist_ok=True)

        # write CSV (into the freshly-created frames_filtered dir)
        self._write_csv(ctx.layout.frame_scores_csv, rows)
        rej = ctx.layout.rejected_dir
        rej.mkdir(parents=True, exist_ok=True)
        n_rejected = 0
        for r in rows:
            src = ctx.layout.frames_dir / r["image"]
            if r.get("accepted"):
                link = fdir / r["image"]
                try:
                    link.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, link)
            else:
                if src.exists():
                    shutil.copy2(src, rej / r["image"])
                n_rejected += 1

        if c.make_contact_sheets:
            try:
                self._contact_sheet(ctx, [r for r in rows if r.get("accepted")], "accepted")
                self._contact_sheet(ctx, [r for r in rows if not r.get("accepted")], "rejected")
            except Exception as e:  # non-fatal
                ctx.logger.warning("contact sheet generation failed: %s", e)

        atomic_write_json(fdir / "filter_summary.json",
                          {"n_total": len(rows), "n_accepted": len(accepted_rows),
                           "n_rejected": n_rejected})
        ctx.logger.info("filtered: %d/%d accepted, %d rejected",
                        len(accepted_rows), len(rows), n_rejected)
        return {"n_total": len(rows), "n_accepted": len(accepted_rows), "n_rejected": n_rejected}

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["image", "blur_var", "luminance", "saturated_frac", "underexposed_frac",
                "ahash", "dup_dist", "reason", "accepted"]
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        tmp.replace(path)

    @staticmethod
    def _contact_sheet(ctx: StageContext, rows: list[dict[str, Any]], name: str,
                       cols: int = 6, thumb: int = 200) -> None:
        from PIL import Image
        rows = rows[:60]
        if not rows:
            return
        n = len(rows)
        rws = (n + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rws * thumb), (30, 30, 30))
        for i, r in enumerate(rows):
            src = ctx.layout.frames_dir / r["image"]
            if not src.exists():
                continue
            im = Image.open(src).convert("RGB")
            im.thumbnail((thumb, thumb))
            sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
        out = ctx.layout.frames_filtered_dir / f"contact_sheet_{name}.jpg"
        sheet.save(out, quality=85)
