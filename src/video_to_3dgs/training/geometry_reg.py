"""Depth--normal consistency: a geometric signal for 3D Gaussian Splatting.

The photometric loss constrains only what the cameras see, and on a single-sided
capture that leaves the *shape* of the surface badly determined — primitives can sit
at plausible depths in wildly implausible orientations and still reproduce every
training image. Surface-aligned methods (2DGS, GOF, PGSR) fix this structurally by
changing the primitive or the rasteriser. Both are large builds and 2DGS
specifically fails on this capture, because a disk must *recover* an orientation
that a single-sided capture never observes.

This module takes the regulariser those methods use internally and applies it to
ordinary 3D Gaussians, where it is a soft self-consistency term rather than a hard
constraint:

1.  Each Gaussian has an implied normal — the axis of least variance, i.e. the
    column of its rotation matrix belonging to the smallest scale. Compositing
    those with the ordinary rasteriser (passing them as post-activation "colours"
    with ``sh_degree=None``) yields a rendered normal map.
2.  The rendered *depth* independently implies a normal at every pixel, via the
    cross product of the unprojected surface gradients.
3.  Disagreement between the two is penalised.

The two quantities come from the same model but through different routes — one from
primitive orientation, one from composited depth — so agreeing forces the primitives
to lie *along* the surface they collectively render. Nothing external is required,
which is why it works where multi-view normal evidence is unavailable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_normals(quats: torch.Tensor, scales_log: torch.Tensor,
                     viewmat: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    """Per-Gaussian normal in CAMERA space, oriented towards the camera.

    The normal is the axis of least variance. Ambiguity in sign is resolved by
    flipping normals that face away from the camera — an unoriented axis would make
    the consistency term penalise correct geometry half the time.
    """
    from gsplat.utils import normalized_quat_to_rotmat

    R = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))     # (N,3,3), wxyz
    smallest = scales_log.argmin(dim=-1)                          # (N,)
    n_world = torch.gather(
        R, 2, smallest[:, None, None].expand(-1, 3, 1)).squeeze(-1)  # (N,3)

    R_view = viewmat[:3, :3]
    n_cam = n_world @ R_view.T
    p_cam = means @ R_view.T + viewmat[:3, 3]                     # camera-space centres
    # flip so each normal points back towards the camera origin
    flip = (n_cam * p_cam).sum(-1, keepdim=True) > 0
    return torch.where(flip, -n_cam, n_cam)


def depth_to_normals(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Normals implied by a depth map, in camera space. ``depth`` is (H,W)."""
    h, w = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype), indexing="ij")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (xs - cx) / fx * depth
    y = (ys - cy) / fy * depth
    p = torch.stack([x, y, depth], dim=-1)                        # (H,W,3)

    # central differences; the border is dropped by the caller's valid mask
    dx = torch.zeros_like(p)
    dy = torch.zeros_like(p)
    dx[:, 1:-1] = p[:, 2:] - p[:, :-2]
    dy[1:-1, :] = p[2:, :] - p[:-2, :]
    n = torch.cross(dx, dy, dim=-1)
    n = F.normalize(n, dim=-1, eps=1e-8)
    flip = (n * p).sum(-1, keepdim=True) > 0
    return torch.where(flip, -n, n)


def normal_consistency_loss(rendered_normals: torch.Tensor, depth: torch.Tensor,
                            alphas: torch.Tensor, K: torch.Tensor,
                            alpha_min: float = 0.5) -> torch.Tensor:
    """Alpha-weighted ``1 - cos`` between primitive-implied and depth-implied normals.

    Pixels where little opacity accumulated are excluded: there the composited depth
    is not a noisy surface estimate but no estimate at all, and its finite-difference
    normal is meaningless.
    """
    n_depth = depth_to_normals(depth, K)
    n_rend = F.normalize(rendered_normals, dim=-1, eps=1e-8)

    valid = torch.zeros_like(depth, dtype=torch.bool)
    valid[1:-1, 1:-1] = True
    w = (alphas >= alpha_min) & valid & (depth > 0)
    if int(w.sum()) < 64:
        return depth.new_zeros(())

    cos = (n_rend * n_depth).sum(-1)
    return (1.0 - cos)[w].mean()
