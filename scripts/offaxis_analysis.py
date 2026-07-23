"""Per-view quality against how far off-axis the camera was aimed from the subject.

Aggregate PSNR hides which views fail, and on both of our captures the failures are the
same kind of view: a close-up whose optical axis points well away from the subject's
centre, so the object is seen tangentially and fills the frame at a grazing angle.

Finding that variable took several wrong guesses, and the wrong ones are worth recording
because each sounded reasonable. On the Casque test split, correlation with per-view PSNR:

    camera distance to subject           r = +0.05
    angle to nearest training view       r = +0.12
    training-view density within 15 deg  r = +0.18
    **off-axis aim angle**               r = -0.74   (Spearman -0.87)

Distance fails because a close-up is not intrinsically hard; nearest-neighbour angle and
local density fail because coverage is only binding below a threshold. What actually hurts
is aiming *past* the subject: the surface is then seen at a grazing angle, where a Gaussian
splat's footprint is worst conditioned and where few training views land.

Two robustness checks, because a single strong r on 13 views is not much on its own, and
the script prints both so they cannot drift from this text:

* dropping the one catastrophic view leaves r = -0.78;
* the angles are bimodal (ten views at 3.7-11.2 deg, three at 23-29 deg), so a linear fit
  over the whole range risks encoding nothing but "near vs far" -- within the on-axis
  cluster alone it is still r = -0.55 (Spearman -0.77).

This is an observed association on one split of one capture, not a controlled result: the
off-axis views are also the closest ones, and nothing here separates the two.

Usage:
  python scripts/offaxis_analysis.py casque_orbit_07ccd886 casque_nodepth_cap6m \
      --out docs/assets/casque/offaxis.png
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("train_run")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "docs" / "assets" / "casque" / "offaxis.png")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from video_to_3dgs.core.paths import RunLayout
    from video_to_3dgs.reporting import cameras as cam_mod
    from video_to_3dgs.training.dataset import ColmapDataset

    layout = RunLayout(REPO / "experiments" / "runs", args.dataset_id)
    # Subject centre from the rig, not the point cloud -- see cameras.capture_geometry.
    centre = cam_mod.capture_geometry(
        ColmapDataset(layout, "train", cache_images=False)).center

    ev_path = layout.training_dir(args.train_run) / "eval.json"
    ev = {v["name"]: v for v in
          json.load(open(ev_path))["splits"][args.split]["per_view"]}

    angles, psnrs, names = [], [], []
    for s in ColmapDataset(layout, args.split, cache_images=False).samples:
        if s.name not in ev:
            continue
        cam = s.viewmat[:3, :3] @ centre + s.viewmat[:3, 3]
        if cam[2] <= 1e-6:                       # subject behind the camera
            continue
        u = s.K[0, 0] * cam[0] / cam[2] + s.K[0, 2]
        v = s.K[1, 1] * cam[1] / cam[2] + s.K[1, 2]
        off = np.hypot((u - s.K[0, 2]) / s.K[0, 0], (v - s.K[1, 2]) / s.K[1, 1])
        angles.append(float(np.degrees(np.arctan(off))))
        psnrs.append(ev[s.name]["psnr"])
        names.append(s.name)

    A, P = np.array(angles), np.array(psnrs)
    if len(A) < 3:
        print("not enough views")
        return 1
    r = float(np.corrcoef(A, P)[0, 1])
    rs = spearman(A, P)

    # Robustness. Two distinct worries, checked separately.
    # (1) the headline r resting on the single worst view;
    # (2) the x-values being bimodal -- a tight cluster plus a few far outliers -- in which
    #     case a linear fit encodes only "near vs far" and says nothing within the cluster.
    keep = A < A.max()
    r_drop = float(np.corrcoef(A[keep], P[keep])[0, 1]) if keep.sum() > 2 else float("nan")
    gap = A < 0.5 * (A.min() + A.max())
    r_cluster = (float(np.corrcoef(A[gap], P[gap])[0, 1]) if gap.sum() > 2
                 else float("nan"))
    rs_cluster = spearman(A[gap], P[gap]) if gap.sum() > 2 else float("nan")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.scatter(A, P, s=46, color="#d62728", zorder=3)
    if len(A) > 2:
        m, b = np.polyfit(A, P, 1)
        xs = np.linspace(A.min(), A.max(), 50)
        ax.plot(xs, m * xs + b, color="0.45", ls="--", lw=1.4, zorder=2,
                label=f"fit: {m:+.2f} dB/deg")
    worst = int(np.argmin(P))
    ax.annotate(names[worst].replace(".jpg", ""), (A[worst], P[worst]),
                textcoords="offset points", xytext=(-12, 12), fontsize=8, color="0.3",
                ha="right")
    ax.set_xlabel("off-axis aim angle: subject centre to optical axis (degrees)")
    ax.set_ylabel(f"{args.split} PSNR (dB)")
    ax.set_title("Views aimed past the subject are the ones that fail\n"
                 f"r = {r:+.2f} (Spearman {rs:+.2f}) · without worst view {r_drop:+.2f} · "
                 f"within on-axis cluster {r_cluster:+.2f}", fontsize=9.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    print(f"n={len(A)}  pearson {r:+.3f}  spearman {rs:+.3f}  "
          f"without-worst {r_drop:+.3f}")
    print(f"within the on-axis cluster only (n={int(gap.sum())}): "
          f"pearson {r_cluster:+.3f}  spearman {rs_cluster:+.3f}")
    lo, hi = P[A < 10], P[A >= 20]
    if len(lo) and len(hi):
        print(f"mean PSNR: on-axis <10deg {lo.mean():.2f} (n={len(lo)}) | "
              f">=20deg {hi.mean():.2f} (n={len(hi)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
