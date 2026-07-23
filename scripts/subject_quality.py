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
    ap.add_argument("dataset_id",
                    help="default dataset; a run may override it as DATASET:RUN")
    ap.add_argument("runs", nargs="+",
                    help="train-run ids, or DATASET:RUN to compare ACROSS datasets "
                         "(e.g. a 2560 and a 4K reconstruction of the same subject)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--width", type=int, default=None,
                    help="resample every render and its ground truth to this width before "
                         "scoring. REQUIRED for a cross-resolution comparison: PSNR is not "
                         "resolution-invariant, since a larger image carries more "
                         "high-frequency content to get wrong, so a 4K model can look worse "
                         "than a 2560 one at identical visual quality.")
    ap.add_argument("--masks-from", default=None,
                    help="score EVERY run through this dataset's masks. Required for a "
                         "cross-dataset comparison: each dataset generates its own masks "
                         "(possibly from a different volume or source), so without this you "
                         "compare two models through two different apertures and the "
                         "difference is partly the masks. Masks are image-space and the "
                         "frame names match across resolutions, so one set serves all.")
    ap.add_argument("--erode", type=int, default=0,
                    help="erode the mask before scoring; the masks are dilated by design, "
                         "so a rim of background is otherwise credited to the subject")
    args = ap.parse_args()

    from PIL import Image

    rows = []
    for spec in args.runs:
        ds_id, _, run = spec.rpartition(":")
        ds_id = ds_id or args.dataset_id
        root = REPO / "experiments" / "runs" / ds_id
        masks_dir = (REPO / "experiments" / "runs" / args.masks_from / "masks"
                     if args.masks_from else root / "masks")
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
            rim = Image.open(f).convert("RGB")
            if args.width:
                rim = rim.resize((args.width, int(rim.height * args.width / rim.width)),
                                 Image.LANCZOS)
            r = np.asarray(rim, dtype=np.float64) / 255.0
            gt = np.asarray(Image.open(gt_p).convert("RGB").resize(
                (r.shape[1], r.shape[0]), Image.LANCZOS), dtype=np.float64) / 255.0
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
            label = run if ds_id == args.dataset_id else f"{ds_id.split('_')[-2]}:{run}"
            rows.append((label, float(np.mean(psnrs)), float(np.mean(ssims)),
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
            print(f"    vs {run:30s} only {len(d)} shared view(s) -- not comparable. "
                  f"Two datasets split independently share few held-out views; retrain both "
                  f"under split_dataset.strategy=periodic, which holds out by frame index.")
            continue
        mm = st.mean(d)
        # t(0.975) by sample size: a 3-view comparison needs 4.30, not 2.18, and using the
        # large-sample value there would understate the interval by nearly 2x.
        tcrit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36,
                 9: 2.31, 10: 2.26, 11: 2.23, 12: 2.20, 13: 2.18}.get(len(d), 2.05)
        h = tcrit * st.stdev(d) / math.sqrt(len(d))
        verdict = "TIE" if abs(mm) < h else "SIGNIFICANT"
        print(f"    vs {run:30s} {mm:+.2f} dB CI [{mm-h:+.2f}, {mm+h:+.2f}] "
              f"{sum(1 for v in d if v > 0)}/{len(d)}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
