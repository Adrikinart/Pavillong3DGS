"""The 2DGS distortion-loss collapse, and what the aggregate metric hides.

Read only the final test PSNR, and 2DGS on the Casque looks like a flat failure
(15.79 dB vs 20.35 for 3DGS) -- the same verdict 2DGS earned on the Pavillon. The
training trajectory says something completely different: the run was at 24.6 dB and
still climbing until the distortion and normal regularizers switched on at iteration
7000, after which it decayed monotonically for the remaining 23 k iterations. Turning
the distortion term off recovers +4.40 dB and brings 2DGS to parity with 3DGS.

This figure exists because the two readings support opposite engineering decisions:
"2DGS does not work on this capture" versus "2DGS works, and one loss weight was
wrong". Only the trajectory distinguishes them.

A note on which numbers are compared where. Validation renders a *deterministic*
evenly-spaced subset (``validation.py``), so val curves are comparable across runs --
but only across runs using the same densification strategy. The 3DGS runs use MCMC,
whose per-step noise injection makes a single val reading swing by several dB (one run
reads 16.02 at step 29 000 and 20.42 at 29 999). So the left panel compares the two
2DGS runs to each other (both ``default`` strategy, stable), and every cross-backend
claim in the right panel uses the held-out *test* split evaluated after training.

Usage:
  python scripts/dist_collapse_figure.py --out docs/assets/casque/2dgs_distortion.png
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[1]
TRAIN = REPO / "experiments" / "runs" / "casque_orbit_07ccd886" / "trainings"

REG_START = 7000          # dist_start_iter == normal_start_iter in casque_2dgs.yaml


def val_curve(run: str) -> tuple[list[int], list[float]]:
    steps, psnr = [], []
    p = TRAIN / run / "metrics.jsonl"
    if not p.exists():
        return steps, psnr
    for line in open(p):
        d = json.loads(line)
        if d.get("kind") == "val" and d.get("psnr") is not None:
            steps.append(d["step"])
            psnr.append(d["psnr"])
    return steps, psnr


def test_stat(run: str) -> tuple[float, float] | None:
    """Test PSNR and the 95 % CI half-width of its mean over held-out views."""
    p = TRAIN / run / "eval.json"
    if not p.exists():
        return None
    t = json.load(open(p))["splits"]["test"]
    pv = [v["psnr"] for v in t["per_view"]]
    n = len(pv)
    half = 2.18 * st.stdev(pv) / math.sqrt(n) if n > 1 else 0.0
    return t["psnr"], half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "docs" / "assets" / "casque" / "2dgs_distortion.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                 gridspec_kw={"width_ratios": [1.55, 1]})

    curves = [
        ("casque_2dgs", "2DGS, dist_lambda=1.0 (default)", "#d62728"),
        ("casque_2dgs_nodist", "2DGS, dist_lambda=0.0", "#2ca02c"),
    ]
    for run, label, colour in curves:
        s, y = val_curve(run)
        if s:
            ax.plot(s, y, label=label, color=colour, lw=2)

    ax.axvline(REG_START, ls="--", color="0.35", lw=1.4)
    # Place the callout at the top of the axis: the lower-left corner is where the
    # collapsed curve and the legend both live.
    ax.annotate("distortion + normal\nlosses switch on",
                xy=(REG_START, 1.0), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(8, -30),
                fontsize=9, color="0.25")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("validation PSNR (dB)")
    ax.set_title("2DGS collapses the moment the distortion loss engages")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    bars = [
        ("3DGS\n1.5M", "casque_gsplat", "#1f77b4"),
        ("2DGS\ndist=1.0", "casque_2dgs", "#d62728"),
        ("2DGS\ndist=0", "casque_2dgs_nodist", "#2ca02c"),
    ]
    xs, ys, es, cs, names = [], [], [], [], []
    for i, (name, run, colour) in enumerate(bars):
        stat = test_stat(run)
        if stat is None:
            continue
        xs.append(i); ys.append(stat[0]); es.append(stat[1])
        cs.append(colour); names.append(name)
    bx.bar(xs, ys, yerr=es, capsize=5, color=cs, width=0.62)
    for x, y, e in zip(xs, ys, es):
        bx.text(x, y + e + 0.45, f"{y:.2f}", ha="center", fontsize=9, weight="bold")
    bx.set_xticks(xs)
    bx.set_xticklabels(names, fontsize=9)
    bx.set_ylabel("test PSNR (dB)")
    bx.set_ylim(0, max(y + e for y, e in zip(ys, es)) * 1.16 if ys else 1)
    bx.set_title("Held-out test: 2DGS reaches parity\n(+4.40 dB from one loss weight)")
    bx.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    for name, run, _ in bars:
        s = test_stat(run)
        if s:
            print(f"  {run:22s} test PSNR {s[0]:.2f} +/- {s[1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
