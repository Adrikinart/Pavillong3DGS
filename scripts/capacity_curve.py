"""Capacity-vs-quality curves, with the paired statistics that decide them.

The single most transferable finding in this project is that the optimal Gaussian
budget is a property of the *capture*, not of the framework: the Pavillon (single-sided,
low parallax) peaks at **375 k** and degrades above it, while the Casque (full orbit)
climbs all the way to **6 M** before turning -- a 16x difference between two captures run
through the same pipeline. This script draws that comparison from the run directories so
the figure can never drift from the numbers.

It also plots the Casque sweep twice, with and without a monocular depth prior that turned
out to be harmful. That is deliberate: the prior's damage grows with capacity, so it did
not merely lower the curve, it flattened its top and made a still-climbing curve look like
it plateaued at 1.5 M. Keeping both series visible is the honest record of a confound that
changed a conclusion.

Two things it deliberately does that an eyeballed curve does not:

* **Error bars are the t-based 95 % CI of the mean over the held-out views.** With 13-28
  test views, differences under a few tenths of a dB are not resolvable, and a bare line
  plot invites reading them as real.
* **Adjacent points are compared *paired*** (same view, both models), which is far more
  sensitive than comparing two means with independent error bars, because it cancels the
  large view-to-view difficulty variation. The printed table reports those paired deltas;
  a delta whose CI spans zero is annotated as a tie on the plot.

Usage:
  python scripts/capacity_curve.py --out docs/assets/capacity_curve.png
  python scripts/capacity_curve.py --object casque --out docs/assets/casque/capacity_curve.png
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNS = REPO / "experiments" / "runs"

# Which training runs form each object's capacity sweep. Every run in a series must share
# the same dataset (same SfM solution, same held-out split) or the comparison is invalid --
# that is why the dataset dir is named once per series rather than per point.
SERIES = {
    "pavillon": {
        "dataset": "pavillon_hidetail_d252a0e5",
        "runs": ["gsplat_hidetail_cap190k", "gsplat_hidetail_cap375k",
                 "gsplat_hidetail_cap750k", "gsplat_hidetail_30k"],
        "label": "Pavillon (single-sided, low parallax)",
    },
    # Two Casque series, because the depth prior turned out to change the curve's SHAPE and
    # not merely its level. With the prior on, 1.5M -> 3M reads as a tie and the curve looks
    # like it plateaus; with it off, both steps are significant and it is still climbing.
    # Keeping both plotted is the point: the confound is visible rather than quietly fixed.
    "casque": {
        "dataset": "casque_orbit_07ccd886",
        "runs": ["casque_cap750k", "casque_gsplat", "casque_cap3m"],
        "label": "Casque orbit — with depth prior (confounded)",
    },
    "casque_nodepth": {
        "dataset": "casque_orbit_07ccd886",
        "runs": ["casque_nodepth_cap750k", "casque_nodepth",
                 "casque_nodepth_cap3m", "casque_nodepth_cap6m",
                 "casque_nodepth_cap12m"],
        "label": "Casque orbit — no depth prior (recommended)",
    },
    # The control that explains the other two curves. Identical pipeline, but the loss is
    # restricted to the helmet, so the model is no longer asked to reproduce the room.
    # Absolute PSNR here is NOT comparable to the other series (different pixel population);
    # only the SHAPE is, and the shape is flat across a 32x range.
    "casque_masked": {
        "dataset": "casque_orbit_07ccd886",
        "runs": ["casque_masked_190k", "casque_masked_375k", "casque_masked_750k",
                 "casque_masked_1500k", "casque_masked_6m"],
        "label": "Casque — loss masked to the helmet (level not comparable)",
    },
}

T95 = {  # two-sided t critical values, small-n honest rather than assuming z=1.96
    5: 2.78, 10: 2.26, 12: 2.20, 13: 2.18, 17: 2.11, 18: 2.11,
    27: 2.05, 28: 2.05, 32: 2.04, 33: 2.04, 52: 2.01, 53: 2.01,
}


def t_crit(n: int) -> float:
    """t for n-1 dof; fall back to the nearest tabulated n, then to 1.96 for large n."""
    if n in T95:
        return T95[n]
    if n > 60:
        return 1.96
    return T95[min(T95, key=lambda k: abs(k - n))]


def load_point(dataset: str, run: str) -> dict | None:
    d = RUNS / dataset / "trainings" / run
    ev, cf = d / "eval.json", d / "config_train.json"
    if not ev.exists() or not cf.exists():
        return None
    e = json.load(open(ev))
    test = e.get("splits", {}).get("test")
    if not test:
        return None
    cap = json.load(open(cf)).get("densification", {}).get("cap_max")
    per_view = {v["name"]: v["psnr"] for v in test.get("per_view", [])}
    psnrs = list(per_view.values())
    n = len(psnrs)
    sem = st.stdev(psnrs) / math.sqrt(n) if n > 1 else 0.0
    return {
        "run": run, "cap": cap, "n": n,
        "psnr": test["psnr"], "ssim": test["ssim"], "lpips": test["lpips"],
        "ci": t_crit(n) * sem, "per_view": per_view,
    }


def paired_delta(a: dict, b: dict) -> tuple[float, float, int, int]:
    """Paired mean delta b-a over shared views, its 95 % half-width, wins and n."""
    keys = sorted(set(a["per_view"]) & set(b["per_view"]))
    d = [b["per_view"][k] - a["per_view"][k] for k in keys]
    n = len(d)
    if n < 2:
        return 0.0, float("inf"), 0, n
    m = st.mean(d)
    half = t_crit(n) * st.stdev(d) / math.sqrt(n)
    return m, half, sum(1 for v in d if v > 0), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", choices=sorted(SERIES) + ["both"], default="both")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "docs" / "assets" / "capacity_curve.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(SERIES) if args.object == "both" else [args.object]
    series = {}
    for name in names:
        spec = SERIES[name]
        pts = [p for p in (load_point(spec["dataset"], r) for r in spec["runs"]) if p]
        pts = [p for p in pts if p["cap"]]
        if pts:
            series[name] = (spec["label"], sorted(pts, key=lambda p: p["cap"]))

    if not series:
        print("no capacity runs found")
        return 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"pavillon": "#1f77b4", "casque": "#f0a3a3",
              "casque_nodepth": "#d62728", "casque_masked": "#2ca02c"}

    for name, (label, pts) in series.items():
        caps = [p["cap"] / 1e6 for p in pts]
        c = colors.get(name, None)
        for ax, key, err in ((axes[0], "psnr", True), (axes[1], "ssim", False),
                             (axes[2], "lpips", False)):
            ys = [p[key] for p in pts]
            if err:
                ax.errorbar(caps, ys, yerr=[p["ci"] for p in pts], marker="o",
                            capsize=4, color=c, label=label, lw=2)
            else:
                ax.plot(caps, ys, marker="o", color=c, label=label, lw=2)
        # Mark the *recommended* operating point, not the raw argmax. The smallest budget
        # that is statistically tied with the best one is the right recommendation: a
        # capacity whose advantage sits inside the error bar is not an advantage, and the
        # smaller model is cheaper to store and render.
        best = max(pts, key=lambda p: p["psnr"])
        rec = best
        for p in pts:
            if p["cap"] >= best["cap"]:
                break
            m, half, _, _ = paired_delta(p, best)
            if abs(m) < half:          # tied with the best -> prefer the smaller model
                rec = p
                break
        # Wording matters: this marks the smallest budget tied *on PSNR*, which is not
        # automatically the recommendation -- SSIM/LPIPS can still separate the tied
        # points (they do for the Pavillon, where 375k is preferred over a PSNR-tied 190k).
        note = f"{rec['cap']/1e6:g}M" + ("" if rec is best else " (smallest PSNR-tied)")
        axes[0].annotate(note, (rec["cap"] / 1e6, rec["psnr"]),
                         textcoords="offset points", xytext=(6, -14),
                         fontsize=9, color=c, weight="bold")

    for ax, t, yl in ((axes[0], "PSNR (higher better)", "PSNR (dB)"),
                      (axes[1], "SSIM (higher better)", "SSIM"),
                      (axes[2], "LPIPS (lower better)", "LPIPS")):
        ax.set_xscale("log")
        # Label the budgets actually trained; log minor ticks are unreadable noise here.
        caps_all = sorted({p["cap"] / 1e6 for _, pts in series.values() for p in pts})
        ax.set_xticks(caps_all)
        ax.set_xticklabels([f"{c:g}M" for c in caps_all], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("Gaussian budget cap_max (log scale)")
        ax.set_ylabel(yl)
        ax.set_title(t)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("The optimal Gaussian budget is a property of the capture, not the framework\n"
                 "error bars: 95% CI of the mean over held-out views",
                 fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    # The statistics that justify reading (or not reading) each step of the curve.
    for name, (label, pts) in series.items():
        print(f"\n{label}  [{pts[0]['n']} test views]")
        print(f"  {'cap_max':>9} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
        for p in pts:
            print(f"  {p['cap']:>9,} {p['psnr']:>7.2f} {p['ssim']:>7.4f} {p['lpips']:>7.4f}")
        print("  paired step comparisons (same views, both models):")
        for a, b in zip(pts, pts[1:]):
            m, half, wins, n = paired_delta(a, b)
            tie = "TIE (CI spans 0)" if abs(m) < half else "significant"
            print(f"    {a['cap']:,} -> {b['cap']:,}: {m:+.2f} dB "
                  f"CI [{m-half:+.2f}, {m+half:+.2f}] wins {wins}/{n}  {tie}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
