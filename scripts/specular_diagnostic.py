"""Is the reconstruction error on this subject actually concentrated on its specular parts?

Before implementing a specular-aware method (GaussianShader, 3DGS-DR, Spec-Gaussian --- all
substantial changes to the rasteriser or the appearance model) it is worth checking that
specularity is what is costing us. "The chrome looks wrong" is an impression; this measures
it.

Method, and its limits. Within the subject mask, pixels are split by the *ground truth*
colour into three crude material classes:

  chrome  desaturated and bright        -- the mirror-like dome
  gold    saturated, red/green dominant -- the matte-ish crest and fittings
  dark    low luminance                 -- horsehair, shadow, the plume's base

That is a proxy, not a segmentation: a dark reflection on chrome lands in "dark", and gold
is itself somewhat specular. It is nevertheless enough to answer the decision at hand --- if
error per pixel is roughly uniform across classes, specular modelling is not the bottleneck
and the engineering effort belongs elsewhere.

Classifying on ground truth rather than on the render matters: classifying on the render
would let a bad reconstruction move pixels into whichever class explains its own error.

Usage:
  python scripts/specular_diagnostic.py casque_orbit_07ccd886 casque_helmet_masked
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def classify(gt: np.ndarray) -> dict[str, np.ndarray]:
    """Crude material classes from ground-truth colour. gt is HxWx3 in [0,1]."""
    lum = gt.mean(axis=2)
    mx, mn = gt.max(axis=2), gt.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    warm = (gt[..., 0] + gt[..., 1]) / 2 - gt[..., 2]      # gold is red+green over blue

    dark = lum < 0.25
    gold = (~dark) & (sat > 0.25) & (warm > 0.08)
    chrome = (~dark) & (~gold) & (sat < 0.25)
    other = ~(dark | gold | chrome)
    return {"chrome": chrome, "gold": gold, "dark": dark, "other": other}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("train_run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--erode", type=int, default=0,
                    help="erode the subject mask by N px before classifying. The masks are "
                         "DILATED by design (a tight silhouette clips the plume), so they "
                         "carry a rim of background; desaturated background would be "
                         "counted as 'chrome' and inflate its error. Erode to check.")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "docs" / "assets" / "casque" / "specular_error.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    runs = REPO / "experiments" / "runs" / args.dataset_id
    rend = runs / "trainings" / args.train_run / "renders" / f"eval_{args.split}"
    if not rend.exists():
        print(f"no renders at {rend}")
        return 1
    masks_dir = runs / "masks"

    totals = {k: [0.0, 0] for k in ("chrome", "gold", "dark", "other")}
    per_view = []
    for f in sorted(rend.glob("*.png")):
        r = np.asarray(Image.open(f).convert("RGB"), dtype=np.float64) / 255.0
        gt_p = runs / "frames_filtered" / (f.stem + ".jpg")
        if not gt_p.exists():
            continue
        gt = np.asarray(Image.open(gt_p).convert("RGB").resize((r.shape[1], r.shape[0])),
                        dtype=np.float64) / 255.0
        m_p = masks_dir / (f.stem + ".png")
        if m_p.exists():
            m = np.asarray(Image.open(m_p).convert("L").resize((r.shape[1], r.shape[0])))
            if args.erode > 0:
                import cv2
                k = np.ones((args.erode * 2 + 1,) * 2, np.uint8)
                m = cv2.erode(m, k)
            subject = m > 127
        else:
            subject = np.ones(r.shape[:2], dtype=bool)
        if subject.sum() < 100:
            continue

        se = ((r - gt) ** 2).mean(axis=2)
        cls = classify(gt)
        row = {"name": f.stem}
        for k, sel in cls.items():
            sel = sel & subject
            if sel.sum() == 0:
                continue
            totals[k][0] += float(se[sel].sum())
            totals[k][1] += int(sel.sum())
            row[k] = float(se[sel].mean())
        per_view.append(row)

    if not per_view:
        print("no comparable views found")
        return 1

    print(f"{'class':>8} {'pixels':>12} {'share':>7} {'MSE':>10} {'PSNR-equiv':>11}")
    labels, psnrs = [], []
    for k, (s, n) in totals.items():
        if n == 0:
            continue
        mse = s / n
        share = n / sum(v[1] for v in totals.values())
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        print(f"{k:>8} {n:>12,} {share:>6.1%} {mse:>10.5f} {psnr:>10.2f}")
        labels.append(k); psnrs.append(psnr)

    order = np.argsort(psnrs)
    labels = [labels[i] for i in order]; psnrs = [psnrs[i] for i in order]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colours = {"chrome": "#9aa5b1", "gold": "#d4a017", "dark": "#4a4a4a", "other": "#7fb3d5"}
    ax.bar(labels, psnrs, color=[colours.get(k, "#888") for k in labels], width=0.6)
    for i, p in enumerate(psnrs):
        ax.text(i, p + 0.15, f"{p:.1f}", ha="center", fontsize=9, weight="bold")
    ax.set_ylabel("PSNR-equivalent within the subject mask (dB)")
    ax.set_title(f"Where the error sits on the subject"
                 f"{f' (mask eroded {args.erode}px: interior only)' if args.erode else ''}\n"
                 "material classes from ground-truth colour — a proxy, not a segmentation",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    spread = max(psnrs) - min(psnrs)
    print(f"\nspread across classes: {spread:.2f} dB")
    # The decision needs the achievable gain, not the gap: a class that is 2 dB worse but
    # covers a third of the pixels is worth less than the gap suggests.
    tot_n = sum(v[1] for v in totals.values())
    cur = sum(v[0] for v in totals.values()) / max(tot_n, 1)
    best_mse = min(v[0] / v[1] for v in totals.values() if v[1])
    ideal = sum(min(v[0] / v[1], best_mse) * v[1] for v in totals.values() if v[1]) / max(tot_n, 1)
    print(f"subject PSNR now {10 * np.log10(1 / max(cur, 1e-12)):.2f} dB; if EVERY class "
          f"reached the best class it would be {10 * np.log10(1 / max(ideal, 1e-12)):.2f} dB "
          f"(+{10 * np.log10(cur / max(ideal, 1e-12)):.2f} dB).")
    print("That headroom is the ceiling for material-specific modelling here -- compare it "
          "against the cost before changing the rasteriser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
