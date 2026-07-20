"""Floater A/B: render two 3DGS .ply models from the SAME oblique, pulled-back
camera (outside the capture cone) where 'flying' Gaussians are most visible, and
write a side-by-side PNG. Used to show the anti-floater/room-bound regularizers
(reg model) removing the background haze the baseline model carries.

Usage (on a GPU node):
  python scripts/floater_ab.py <dataset_id> <baseline.ply> <reg.ply> <out.png>
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from video_to_3dgs.core.paths import RunLayout  # noqa: E402
from video_to_3dgs.training.dataset import ColmapDataset  # noqa: E402
from video_to_3dgs.reporting.cameras import object_frame_from_dataset, _look_at, _rot_about_axis  # noqa: E402


def load_ply(path: Path):
    """Parse the standard 3DGS binary .ply -> render-ready gsplat tensors."""
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
    quats = data[:, [col[f"rot_{i}"] for i in range(4)]]
    scales = data[:, [col[f"scale_{i}"] for i in range(3)]]
    opac = data[:, col["opacity"]]
    fdc = data[:, [col[f"f_dc_{i}"] for i in range(3)]]                 # (N,3)
    frest_keys = sorted([k for k in props if k.startswith("f_rest_")],
                        key=lambda s: int(s.split("_")[-1]))
    frest = data[:, [col[k] for k in frest_keys]]                       # (N, 3*(K-1))
    kminus1 = frest.shape[1] // 3
    sh0 = fdc.reshape(n, 1, 3)
    shN = frest.reshape(n, 3, kminus1).transpose(0, 2, 1)               # (N,K-1,3)
    colors = np.concatenate([sh0, shN], axis=1)                          # (N,K,3)
    dev = "cuda"
    return {
        "means": torch.tensor(means, device=dev),
        "quats": torch.tensor(quats, device=dev),
        "scales": torch.exp(torch.tensor(scales, device=dev)),
        "opacities": torch.sigmoid(torch.tensor(opac, device=dev)),
        "colors": torch.tensor(colors, device=dev),
        "sh_degree": int(round(np.sqrt(sh0.shape[1] + kminus1))) - 1,
        "n": n,
    }


def render(g, viewmat, K, w, h):
    import gsplat
    out, _, _ = gsplat.rasterization(
        means=g["means"], quats=g["quats"], scales=g["scales"], opacities=g["opacities"],
        colors=g["colors"], viewmats=torch.tensor(viewmat, device="cuda")[None],
        Ks=torch.tensor(K, device="cuda")[None], width=w, height=h,
        sh_degree=g["sh_degree"], near_plane=0.01, far_plane=1e10,
        packed=False, rasterize_mode="antialiased")
    return out[0].clamp(0, 1).cpu().numpy()


def main():
    dsid, base_ply, reg_ply, out_png = sys.argv[1:5]
    layout = RunLayout(runs_root=REPO / "experiments" / "runs", dataset_id=dsid)
    ds = ColmapDataset(layout, "train", use_masks=False)
    W, H = 1280, 720
    K = ds.samples[0].K.copy().astype(np.float64)
    # scale intrinsics to the render size
    sx, sy = W / ds.samples[0].width, H / ds.samples[0].height
    K[0, :] *= sx; K[1, :] *= sy
    obj = object_frame_from_dataset(ds, K, W, H, margin=1.2)
    # oblique, pulled back 2.2x so the empty volume around the object is in frame
    eye = obj.center - 2.2 * obj.fit_distance * obj.front
    right = np.cross(obj.front, obj.up)
    R = _rot_about_axis(obj.up, np.deg2rad(32)) @ _rot_about_axis(right, np.deg2rad(18))
    eye = obj.center + R @ (eye - obj.center)
    vm = _look_at(eye, obj.center, obj.up)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.3))
    for a, (name, ply) in zip(ax, [("baseline (no reg)", base_ply), ("regularized (A3)", reg_ply)]):
        g = load_ply(Path(ply))
        img = render(g, vm, K, W, H)
        a.imshow(img); a.axis("off")
        a.set_title(f"{name}\n{g['n']:,} gaussians", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_png, dpi=100)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
