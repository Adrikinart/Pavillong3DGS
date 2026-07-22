"""Collect README/demo assets for one object into docs/assets/<object>/.

CPU-only (login-node safe). From an existing run directory it produces:
  poses.png     — SfM figure: sparse point cloud + camera frusta (always,
                  only needs colmap/sparse/0)
  orbit.gif     — GitHub-friendly GIF from the training's videos/orbit.mp4
                  (ffmpeg two-pass palette; skipped if the mp4 is missing)
  training_curves.png / qualitative.png / per_view_metrics.png /
  gaussian_stats.png / metrics_table.md — copied from the training's
                  figures/ and metrics/ dirs when present

Every asset is best-effort: missing inputs are reported and skipped, so the
script already works for an object whose training hasn't finished (it will
just emit poses.png).

Usage:
  python scripts/make_demo_assets.py <dataset_id> [--train-run latest]
         [--object NAME] [--out DIR] [--gif-width 640] [--gif-fps 12]

Examples:
  python scripts/make_demo_assets.py pavillon_hidetail_d252a0e5 \
         --train-run gsplat_hidetail_cap375k --object pavillon
  python scripts/make_demo_assets.py casque_orbit_07ccd886
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from video_to_3dgs.core.paths import RunLayout           # noqa: E402
from video_to_3dgs.reporting import figures              # noqa: E402

# training figure -> asset name (qualitative_test loses its split suffix)
FIGURE_COPIES = {
    "training_curves.png": "training_curves.png",
    "qualitative_test.png": "qualitative.png",
    "per_view_metrics.png": "per_view_metrics.png",
    "gaussian_stats.png": "gaussian_stats.png",
}


def make_gif(mp4: Path, out: Path, width: int, fps: int) -> Path | None:
    if not mp4.exists():
        return None
    filters = f"fps={fps},scale={width}:-1:flags=lanczos"
    with tempfile.TemporaryDirectory() as td:
        palette = Path(td) / "palette.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(mp4),
             "-vf", f"{filters},palettegen=stats_mode=diff", str(palette)],
            check=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-i", str(palette),
             "-lavfi", f"{filters}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
             "-loop", "0", str(out)],
            check=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dataset_id", help="run dir name under --runs-root")
    ap.add_argument("--runs-root", type=Path, default=REPO / "experiments" / "runs")
    ap.add_argument("--train-run", default="latest")
    ap.add_argument("--object", default=None,
                    help="asset folder name (default: dataset_id up to first '_')")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: docs/assets/<object>)")
    ap.add_argument("--gif-width", type=int, default=640)
    ap.add_argument("--gif-fps", type=int, default=12)
    args = ap.parse_args()

    layout = RunLayout(runs_root=args.runs_root, dataset_id=args.dataset_id)
    if not layout.run_dir.exists():
        ap.error(f"run dir not found: {layout.run_dir}")
    obj = args.object or args.dataset_id.split("_")[0]
    out_dir = args.out or REPO / "docs" / "assets" / obj
    out_dir.mkdir(parents=True, exist_ok=True)

    tdir = layout.training_dir(args.train_run)
    tr_name = tdir.resolve().name if tdir.exists() else args.train_run
    produced, skipped = [], []

    # --- poses.png (needs only the COLMAP sparse model) ---
    p = figures.camera_poses(layout.colmap_sparse0, out_dir / "poses.png",
                             title=f"{obj}: camera poses + sparse point cloud "
                                   f"({args.dataset_id})")
    (produced if p else skipped).append("poses.png")

    # --- orbit.gif from the training's orbit.mp4 ---
    g = make_gif(layout.videos_dir(args.train_run) / "orbit.mp4",
                 out_dir / "orbit.gif", args.gif_width, args.gif_fps)
    (produced if g else skipped).append("orbit.gif")

    # --- copies of the per-training report figures/metrics ---
    fdir = layout.figures_dir(args.train_run)
    for src_name, dst_name in FIGURE_COPIES.items():
        src = fdir / src_name
        if src.exists():
            shutil.copyfile(src, out_dir / dst_name)
            produced.append(dst_name)
        else:
            skipped.append(dst_name)
    table = layout.metrics_dir(args.train_run) / "metrics_table.md"
    if table.exists():
        shutil.copyfile(table, out_dir / "metrics_table.md")
        produced.append("metrics_table.md")
    else:
        skipped.append("metrics_table.md")

    print(f"object={obj}  train_run={tr_name}  ->  {out_dir}")
    for name in produced:
        size = (out_dir / name).stat().st_size
        print(f"  wrote  {name:22s} {size / 1e6:6.2f} MB")
    for name in skipped:
        print(f"  skip   {name} (input missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
