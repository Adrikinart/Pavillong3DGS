"""Validation measured on the whole frame versus on the subject alone.

The Casque training curves look alarming: validation PSNR peaks around iteration 7000,
declines for the next 20 000 steps, then jumps ~4.4 dB at the final iteration. Nothing is
wrong with the training. Scoring the same checkpoints on the helmet region instead shows a
monotone rise throughout -- the decline belongs entirely to the room behind the subject,
which stays unsettled until the noise scale decays at the end of the schedule.

That matters beyond one confusing plot:

* ``best_val`` and early stopping (``early_stop_patience``) both read the whole-frame
  number. It is disabled by default, but any patience of 3 or more would have terminated
  this healthy run somewhere around iteration 10 000.
* For an object capture the whole-frame metric is mostly a statement about the background,
  which is exactly the part nobody is going to look at.
* The subject reaches 22.8 dB by iteration 10 000 and gains only ~1.9 dB over the remaining
  20 000, so the useful stopping point depends entirely on which curve you read.

Requires masks on disk (scripts/make_box_masks.py) -- they are used only for measurement
here, not for training.

Usage:
  python scripts/object_vs_scene_val.py casque_orbit_07ccd886 casque_gsplat \
      --out docs/assets/casque/object_vs_scene_val.png
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("train_run")
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="checkpoint steps to score (default: every checkpoint found)")
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "docs" / "assets" / "casque" / "object_vs_scene_val.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gsplat

    from video_to_3dgs.core.paths import RunLayout
    from video_to_3dgs.training.checkpoint import load_checkpoint
    from video_to_3dgs.training.dataset import ColmapDataset
    from video_to_3dgs.training.gaussians import create_splats
    from video_to_3dgs.training.gsplat_backend import GsplatBackend
    from video_to_3dgs.training.validation import evaluate_split

    layout = RunLayout(REPO / "experiments" / "runs", args.dataset_id)
    train_ds = ColmapDataset(layout, "train", cache_images=False)
    full = ColmapDataset(layout, "val", downscale=1)
    masked = ColmapDataset(layout, "val", downscale=1, use_masks=True)
    if not any(s.mask_path for s in masked.samples):
        print("no masks on disk; run scripts/make_box_masks.py first")
        return 1

    backend = GsplatBackend()
    ckpt_dir = layout.checkpoints_dir(args.train_run)
    steps = args.steps
    if not steps:
        steps = sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("ckpt_0*.pt"))
    if not steps:
        print(f"no checkpoints under {ckpt_dir}")
        return 1

    xs, scene, obj = [], [], []
    for step in steps:
        p = ckpt_dir / f"ckpt_{step:07d}.pt"
        if not p.exists():
            continue
        params, _ = create_splats(train_ds.points, train_ds.point_colors, 3, "cuda")
        load_checkpoint(p, params, {})
        row = []
        for ds, use_mask in ((full, False), (masked, True)):
            def render(i, _ds=ds):
                return backend._rasterize(gsplat, params, *_ds.camera_tensors(i, "cuda"),
                                          3, 0.01, 1e10)[0][0]
            r = evaluate_split(render, ds, "cuda", out_dir=None, compute_lpips=False,
                               masked=use_mask, max_images=args.max_images)
            row.append(r["psnr"])
        xs.append(step); scene.append(row[0]); obj.append(row[1])
        print(f"step {step:>6}: whole frame {row[0]:6.2f} dB | subject only {row[1]:6.2f} dB")

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(xs, scene, marker="o", color="#7f7f7f", lw=2, label="whole frame (what we plot)")
    ax.plot(xs, obj, marker="o", color="#d62728", lw=2, label="subject only (the deliverable)")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("validation PSNR (dB)")
    ax.set_title("The declining validation curve is the room, not the subject", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="center left")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    if len(obj) > 1:
        print(f"subject: {obj[0]:.2f} -> {obj[-1]:.2f} dB (monotone: "
              f"{all(b >= a - 1e-6 for a, b in zip(obj, obj[1:]))})")
        print(f"scene:   {scene[0]:.2f} -> {scene[-1]:.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
