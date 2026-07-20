"""Floater / room-bound analysis WITHOUT a GPU: parse both .ply models' Gaussian
centers + opacities and compare their spatial distribution against the room box
(the AABB the trainer constrains to). Baseline scatters Gaussians outside the
room (floaters); the regularized model is confined + floater-free.

Usage:  python scripts/floater_spatial.py <dataset_id> <baseline.ply> <reg.ply> <out.png>
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from video_to_3dgs.core.paths import RunLayout           # noqa: E402
from video_to_3dgs.training.dataset import ColmapDataset, _center_from_viewmat  # noqa: E402


def load_ply_centers(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        props, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property float"):
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(n * len(props) * 4), dtype="<f4").reshape(n, len(props))
    col = {p: i for i, p in enumerate(props)}
    means = data[:, [col["x"], col["y"], col["z"]]]
    opac = 1.0 / (1.0 + np.exp(-data[:, col["opacity"]]))          # sigmoid
    return means, opac


def main():
    dsid, base_ply, reg_ply, out_png = sys.argv[1:5]
    layout = RunLayout(runs_root=REPO / "experiments" / "runs", dataset_id=dsid)
    ds = ColmapDataset(layout, "train", use_masks=False)
    cams = np.array([_center_from_viewmat(s.viewmat) for s in ds.samples])
    pts = ds.points if len(ds.points) else cams
    allp = np.concatenate([pts, cams], axis=0)
    lo, hi = allp.min(0), allp.max(0)
    m = 0.5 * (hi - lo)
    box_lo, box_hi = lo - m, hi + m                                 # the trainer's room box

    def outside_frac(means):
        out = ((means < box_lo) | (means > box_hi)).any(axis=1)
        return out, 100.0 * out.mean()

    bm, bo = load_ply_centers(base_ply)
    rm, ro = load_ply_centers(reg_ply)
    b_out, b_pct = outside_frac(bm)
    r_out, r_pct = outside_frac(rm)

    # robust plot extent = 3x the room box (so far floaters are visible but not off-scale)
    c = (box_lo + box_hi) / 2
    span = (box_hi - box_lo)
    plo, phi = c - 1.5 * span, c + 1.5 * span

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    planes = [(0, 2, "top-down (x-z)"), (0, 1, "side (x-y)")]
    for row, (name, means, out, pct) in enumerate(
            [("baseline (no reg)", bm, b_out, b_pct), ("regularized (A3)", rm, r_out, r_pct)]):
        for coliax, (ax_, (a, bax, title)) in enumerate(zip(axes[row], planes)):
            # subsample for plotting speed
            idx = np.random.default_rng(0).choice(len(means), size=min(120000, len(means)), replace=False)
            mm = means[idx]; oo = out[idx]
            ax_.scatter(mm[~oo, a], mm[~oo, bax], s=0.4, c="#2c7fb8", alpha=0.25, label="in room")
            ax_.scatter(mm[oo, a], mm[oo, bax], s=0.8, c="#d7301f", alpha=0.5, label="OUT of room (floater)")
            # draw the room box
            bl, bh = box_lo, box_hi
            ax_.plot([bl[a], bh[a], bh[a], bl[a], bl[a]], [bl[bax], bl[bax], bh[bax], bh[bax], bl[bax]],
                     "k--", lw=1.2, label="room box")
            ax_.set_xlim(plo[a], phi[a]); ax_.set_ylim(plo[bax], phi[bax])
            ax_.set_title(f"{name} — {title}", fontsize=10)
            ax_.set_aspect("equal", "box")
            if row == 0 and coliax == 0:
                ax_.legend(loc="upper right", fontsize=7, markerscale=6)
    fig.suptitle(
        f"Gaussian centers vs room box   |   baseline: {len(bm):,} gaussians, {b_pct:.1f}% outside room"
        f"   →   reg: {len(rm):,} gaussians, {r_pct:.1f}% outside room",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_png, dpi=100)
    print(f"baseline: {len(bm):,} gaussians, {b_pct:.2f}% outside room box")
    print(f"reg     : {len(rm):,} gaussians, {r_pct:.2f}% outside room box")
    print("wrote", out_png)


if __name__ == "__main__":
    main()
