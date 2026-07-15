"""Initialize Gaussian parameters from a sparse point cloud (gsplat conventions)."""

from __future__ import annotations

import numpy as np

C0 = 0.28209479177387814  # SH DC constant


def rgb_to_sh(rgb):
    return (rgb - 0.5) / C0


def _knn_mean_dist(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to k nearest neighbors, for per-point scale init."""
    n = len(points)
    if n <= 1:
        return np.full((n,), 0.01, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        d, _ = tree.query(points, k=min(k + 1, n))
        d = d[:, 1:]  # drop self
        return np.clip(d.mean(axis=1), 1e-6, None).astype(np.float32)
    except Exception:
        c = points.mean(axis=0)
        extent = np.linalg.norm(points - c, axis=1).mean() + 1e-6
        return np.full((n,), extent / (n ** (1 / 3) + 1), dtype=np.float32)


def create_splats(points: np.ndarray, colors: np.ndarray, sh_degree: int,
                  device: str, lr_means: float = 1.6e-4, scene_scale: float = 1.0):
    """Build a ParameterDict of Gaussians + a per-group Adam optimizer dict."""
    import torch
    from torch import nn

    if len(points) == 0:  # degenerate: seed a small random cloud
        points = np.random.randn(1000, 3).astype(np.float32) * 0.1
        colors = np.full((1000, 3), 0.5, dtype=np.float32)

    N = len(points)
    means = torch.tensor(points, dtype=torch.float32)
    dist = torch.tensor(_knn_mean_dist(points), dtype=torch.float32)
    scales = torch.log(dist.clamp_min(1e-6))[:, None].repeat(1, 3)
    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((N,), 0.1))
    sh_dim = (sh_degree + 1) ** 2
    sh0 = torch.tensor(rgb_to_sh(colors), dtype=torch.float32)[:, None, :]  # (N,1,3)
    shN = torch.zeros(N, sh_dim - 1, 3)

    params = nn.ParameterDict({
        "means": nn.Parameter(means),
        "scales": nn.Parameter(scales),
        "quats": nn.Parameter(quats),
        "opacities": nn.Parameter(opacities),
        "sh0": nn.Parameter(sh0),
        "shN": nn.Parameter(shN),
    }).to(device)

    # per-parameter learning rates (3DGS conventions; means LR scaled by extent)
    lrs = {
        "means": lr_means * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20.0,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": params[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }
    return params, optimizers
