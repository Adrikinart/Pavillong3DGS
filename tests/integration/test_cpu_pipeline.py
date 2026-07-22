"""Integration: synthesize a tiny video with ffmpeg, run CPU stages via the runner."""

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from video_to_3dgs.config.loader import load_config, make_layout
from video_to_3dgs.core.manifest import Manifest
from video_to_3dgs.core.runner import StageRunner
from video_to_3dgs.core.stage import StageContext
from video_to_3dgs.stages.extract_frames import ExtractFramesStage
from video_to_3dgs.stages.filter_frames import FilterFramesStage
from video_to_3dgs.stages.inspect_video import InspectVideoStage

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")
cv2 = pytest.importorskip("cv2")


def _make_video(path: Path) -> None:
    # 2s of moving test pattern, 10 fps, small
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=size=320x240:rate=10:duration=2", "-pix_fmt", "yuv420p", str(path)],
        check=True, timeout=60)


def _ctx(tmp_path, videos):
    cfg = load_config("configs/pipeline/templates/smoke_test.yaml", overrides=[
        "extract_frames.target_fps=5", "extract_frames.max_frames=8",
        "extract_frames.min_frames=3", "filter_frames.min_kept=3",
        "filter_frames.blur_var_min=1.0",
    ])
    lay = make_layout(cfg, tmp_path, "testvid")
    lay = lay.__class__(runs_root=tmp_path / "runs", dataset_id="testvid")
    lay.ensure_base_dirs()
    return StageContext(layout=lay, config=cfg, manifest=Manifest(lay.manifest),
                        logger=logging.getLogger("test"), repo_root=tmp_path,
                        params={"videos": videos})


def test_inspect_extract_filter(tmp_path):
    vid = tmp_path / "clip.mp4"
    _make_video(vid)
    ctx = _ctx(tmp_path, [str(vid)])
    runner = StageRunner(ctx)

    runner.execute(InspectVideoStage())
    assert (ctx.layout.video_dir / "metadata.json").exists()

    runner.execute(ExtractFramesStage())
    frames = list(ctx.layout.frames_dir.glob("*.jpg"))
    assert len(frames) >= 3

    runner.execute(FilterFramesStage())
    assert ctx.layout.frame_scores_csv.exists()
    kept = [p for p in ctx.layout.frames_filtered_dir.iterdir()
            if p.suffix.lower() == ".jpg"]
    assert len(kept) >= 3

    # idempotent rerun
    from video_to_3dgs.core.status import StageStatus
    assert runner.execute(InspectVideoStage()) == StageStatus.SKIPPED
