"""Stage: extract frames from source videos (rotation-aware, multi-video)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..core.atomicio import atomic_write_json
from ..core.paths import slugify
from ..core.stage import Artifact, Stage, StageContext
from ..media import extract_frames_ffmpeg, video_metadata


class ExtractFramesStage(Stage):
    name = "extract_frames"
    depends_on = ("inspect_video",)

    def _videos(self, ctx: StageContext) -> list[str]:
        return ctx.params.get("videos") or ctx.config.videos

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("video_metadata", ctx.layout.video_dir / "metadata.json", "file")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [
            Artifact("frames_dir", ctx.layout.frames_dir, "dir"),
            Artifact("frames_index", ctx.layout.frames_index, "file"),
        ]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.extract_frames
        return {"videos": self._videos(ctx), **c.model_dump()}

    def run(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.extract_frames
        frames_dir = ctx.layout.frames_dir
        # clean previous extraction for idempotency
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        vids = self._videos(ctx)
        all_records: list[dict[str, Any]] = []
        start = 0
        for v in vids:
            prefix = slugify(Path(v).stem)
            fps = c.target_fps if c.strategy == "fixed_fps" else None
            interval = c.frame_interval if c.strategy == "fixed_interval" else None
            if c.strategy == "max_frames":
                # derive an fps that yields ~max_frames over the clip duration
                dur = video_metadata(v)["duration_s"] or 1.0
                fps = max(0.1, c.max_frames / max(dur, 1e-3))
            resize = None if c.preserve_original_resolution else c.resize_long_edge
            recs = extract_frames_ffmpeg(
                v, frames_dir, prefix=prefix, fps=fps, frame_interval=interval,
                resize_long_edge=resize, honor_rotation=c.honor_rotation,
                jpeg_quality=c.jpeg_quality, output_format=c.output_format, start_index=start,
            )
            start += len(recs)
            all_records.extend(recs)
            ctx.logger.info("extracted %d frames from %s", len(recs), Path(v).name)

        # enforce max_frames cap by uniform subsampling across the full set
        if len(all_records) > c.max_frames:
            keep_idx = _uniform_indices(len(all_records), c.max_frames)
            keep = set(keep_idx)
            removed = 0
            kept_records = []
            for i, rec in enumerate(all_records):
                if i in keep:
                    kept_records.append(rec)
                else:
                    (frames_dir / rec["image"]).unlink(missing_ok=True)
                    removed += 1
            all_records = kept_records
            ctx.logger.info("capped to max_frames=%d (removed %d)", c.max_frames, removed)

        if len(all_records) < c.min_frames:
            ctx.logger.warning("only %d frames extracted (< min_frames=%d); consider a higher fps",
                               len(all_records), c.min_frames)

        atomic_write_json(ctx.layout.frames_index,
                          {"n_frames": len(all_records), "strategy": c.strategy,
                           "frames": all_records})
        return {"n_frames": len(all_records), "n_videos": len(vids)}


def _uniform_indices(n: int, k: int) -> list[int]:
    if k >= n:
        return list(range(n))
    step = n / k
    return sorted({min(n - 1, int(i * step)) for i in range(k)})
