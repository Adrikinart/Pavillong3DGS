"""Stage: inspect source videos with ffprobe and record provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.atomicio import atomic_write_json, sha256_file
from ..core.stage import Artifact, Stage, StageContext
from ..media import video_metadata


class InspectVideoStage(Stage):
    name = "inspect_video"
    depends_on = ()

    def _videos(self, ctx: StageContext) -> list[str]:
        return ctx.params.get("videos") or ctx.config.videos

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("video_metadata", ctx.layout.video_dir / "metadata.json", "file")]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return {"videos": self._videos(ctx),
                "min_duration_s": ctx.config.inspect_video.min_duration_s}

    def validate_inputs(self, ctx: StageContext) -> None:
        vids = self._videos(ctx)
        if not vids:
            from ..core.errors import InputValidationError
            raise InputValidationError("no source videos configured (set `videos:`)")
        for v in vids:
            if not Path(v).exists():
                from ..core.errors import InputValidationError
                raise InputValidationError(f"video not found: {v}")

    def run(self, ctx: StageContext) -> dict[str, Any]:
        vids = self._videos(ctx)
        records: list[dict[str, Any]] = []
        min_dur = ctx.config.inspect_video.min_duration_s
        for v in vids:
            meta = video_metadata(v)
            ctx.logger.info("video %s: %dx%d %s %.1fs fps=%.2f rot=%d",
                            Path(v).name, meta["width"], meta["height"], meta["codec"],
                            meta["duration_s"], meta["avg_fps"] or 0.0, meta["rotation"])
            if meta["duration_s"] < min_dur:
                ctx.logger.warning("video %s shorter than min_duration_s=%.2f", v, min_dur)
            if meta["variable_fps"]:
                ctx.logger.warning("video %s appears to be variable-frame-rate", v)
            meta["sha256"] = sha256_file(v)
            meta["bytes"] = Path(v).stat().st_size
            records.append(meta)

        out = ctx.layout.video_dir / "metadata.json"
        atomic_write_json(out, {"videos": records})
        # record in manifest header
        ctx.manifest.set_field("videos", records)
        return {"n_videos": len(records),
                "total_duration_s": round(sum(r["duration_s"] for r in records), 1)}
