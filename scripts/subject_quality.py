"""Rank trained models by quality ON THE SUBJECT, whatever they were trained on.

Most comparisons in this project asked "which model is cheaper" or "which scores better on
whatever pixels it was trained on". Neither answers the question that matters when the
object is the deliverable: *which model renders the subject best?*

A masked model is scored on the subject by construction; a full-scene model is scored on the
whole frame, and its subject quality is never reported. Those two numbers are not comparable,
so the ranking everyone reaches for is the wrong one. This applies one mask to all of them
and computes the metric over the same pixels, from the renders already on disk -- no GPU and
no retraining.

PSNR and SSIM only: LPIPS needs a network and a GPU, and the point here is a cheap
apples-to-apples ranking rather than a perceptual study.

Usage:
  python scripts/subject_quality.py casque_orbit_07ccd886 \
      casque_helmet_masked casque_nodepth casque_nodepth_cap6m
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def masked_ssim(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    """Global SSIM restricted to the masked pixels (luma, no local windows).

    A windowed SSIM would straddle the mask boundary and mix in background, which is exactly
    what we are trying to exclude; the global form keeps every contributing pixel inside the
    subject at the cost of ignoring spatial structure.
    """
    ga = a.mean(axis=2)[m]
    gb = b.mean(axis=2)[m]
    if ga.size < 2:
        return float("nan")
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ma, mb = ga.mean(), gb.mean()
    va, vb = ga.var(), gb.var()
    cov = ((ga - ma) * (gb - mb)).mean()
    return float(((2 * ma * mb + c1) * (2 * cov + c2))
                 / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test")
    ap.add_argument("--erode", type=int, default=0,
                    help="erode the mask before scoring; the masks are dilated by design, "
                         "so a rim of background is otherwise credited to the subject")
    args = ap.parse_args()

    from PIL import Image

    root = REPO / "experiments" / "runs" / args.dataset_id
    masks_dir = root / "masks"
    rows = []
    for run in args.runs:
        rend = root / "trainings" / run / "renders" / f"eval_{args.split}"
        if not rend.exists():
            print(f"  {run}: no renders at {rend}")
            continue
        psnrs, ssims, names = [], [], []
        for f in sorted(rend.glob("*.png")):
            gt_p = root / "frames_filtered" / (f.stem + ".jpg")
            m_p = masks_dir / (f.stem + ".png")
            if not (gt_p.exists() and m_p.exists()):
                continue
            r = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
            gt = np.asarray(Image.open(gt_p).convert("RGB").resize((r.shape[1], r.shape[0])),
                            dtype=np.float64) / 255.0
            m = np.asarray(Image.open(m_p).convert("L").resize((r.shape[1], r.shape[0])))
            if args.erode > 0:
                import cv2
                m = cv2.erode(m, np.ones((args.erode * 2 + 1,) * 2, np.uint8))
            sel = m > 127
            if sel.sum() < 100:
                continue
            mse = ((r - gt) ** 2).mean(axis=2)[sel].mean()
            psnrs.append(10 * np.log10(1.0 / max(mse, 1e-12)))
            ssims.append(masked_ssim(r, gt, sel))
            names.append(f.stem)
        if psnrs:
            rows.append((run, float(np.mean(psnrs)), float(np.mean(ssims)),
                         len(psnrs), dict(zip(names, psnrs))))

    if not rows:
        print("nothing to compare")
        return 1

    rows.sort(key=lambda r: -r[1])
    print(f"\nSubject-region quality on the {args.split} split"
          f"{f' (mask eroded {args.erode}px)' if args.erode else ''}:\n")
    print(f"  {'run':32s} {'PSNR':>7} {'SSIM':>8} {'views':>6}")
    for run, p, s, n, _ in rows:
        print(f"  {run:32s} {p:7.2f} {s:8.4f} {n:>6}")

    # Paired against the best, because a 13-view mean hides whether a gap is consistent.
    import math
    import statistics as st
    best = rows[0]
    print(f"\n  paired against {best[0]}:")
    for run, p, s, n, per in rows[1:]:
        keys = sorted(set(best[4]) & set(per))
        d = [best[4][k] - per[k] for k in keys]
        if len(d) < 2:
            continue
        mm = st.mean(d)
        h = 2.18 * st.stdev(d) / math.sqrt(len(d))
        verdict = "TIE" if abs(mm) < h else "SIGNIFICANT"
        print(f"    vs {run:30s} {mm:+.2f} dB CI [{mm-h:+.2f}, {mm+h:+.2f}] "
              f"{sum(1 for v in d if v > 0)}/{len(d)}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
