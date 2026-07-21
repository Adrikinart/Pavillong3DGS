"""Per-camera pose refinement (SE(3) deltas learned jointly with the scene).

3DGS treats SfM poses as exact, but on a low-overlap capture they are not. A pose
error and a geometry error are indistinguishable to a photometric loss, so the
Gaussians end up absorbing calibration error as distorted geometry — smeared
surfaces, or detail that never resolves because consecutive views disagree about
where it is.

The remedy is to let each training camera learn a small rigid correction alongside
the scene, as in BARF :cite:`lin2021barf`. Each camera owns a 6-vector
:math:`(\\omega, \\tau)` mapped to SE(3) by the exponential map and left-multiplied
onto its world-to-camera matrix::

    T_refined = exp([omega, tau]) @ T_colmap

Initialised at zero, so training starts from exactly the SfM solution and departs
only if that reduces the loss.

**Held-out views are never refined.** Only training cameras have deltas; validation
and test poses stay as SfM produced them. Optimising a held-out camera's pose would
fit the evaluation image and inflate the metric — the pose is part of what a
reconstruction must predict, not an input to be tuned per test view. The corollary
is that pose refinement only helps held-out metrics if it improves the *scene*, not
if it merely re-aligns training cameras onto their own images.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def se3_exp(delta: torch.Tensor) -> torch.Tensor:
    """Exponential map from a 6-vector ``(omega, tau)`` to a 4x4 SE(3) matrix.

    Rotation uses Rodrigues' formula. The small-angle branch is handled with a
    Taylor expansion because ``sin(t)/t`` is numerically unstable as ``t -> 0`` —
    and ``t -> 0`` is exactly where this operates, since the deltas start at zero
    and stay small.
    """
    omega, tau = delta[:3], delta[3:]
    theta = torch.linalg.norm(omega)
    eye = torch.eye(3, device=delta.device, dtype=delta.dtype)

    if float(theta.detach()) < 1e-8:   # branch choice is not differentiable
        # exp(w^) ~ I + w^  to first order
        wx = _skew(omega)
        R = eye + wx
    else:
        k = omega / theta
        kx = _skew(k)
        R = eye + torch.sin(theta) * kx + (1 - torch.cos(theta)) * (kx @ kx)

    T = torch.zeros(4, 4, device=delta.device, dtype=delta.dtype)
    T[:3, :3] = R
    T[:3, 3] = tau
    T[3, 3] = 1.0
    return T


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


class PoseOptimizer(nn.Module):
    """One learnable SE(3) delta per training camera, initialised to identity."""

    def __init__(self, n_cameras: int):
        super().__init__()
        self.n_cameras = int(n_cameras)
        self.deltas = nn.Parameter(torch.zeros(self.n_cameras, 6))

    def forward(self, index: int, viewmat: torch.Tensor) -> torch.Tensor:
        """Apply camera ``index``'s learned correction to its world-to-camera matrix."""
        T = se3_exp(self.deltas[index].to(viewmat.dtype))
        return T @ viewmat

    @torch.no_grad()
    def magnitude(self) -> tuple[float, float]:
        """(mean rotation in degrees, mean translation in scene units).

        Diagnostic: near-zero means SfM was already consistent and the refinement is
        doing nothing; a large translation relative to the scene extent means it is
        absorbing something it probably should not.
        """
        rot = torch.linalg.norm(self.deltas[:, :3], dim=1)
        trans = torch.linalg.norm(self.deltas[:, 3:], dim=1)
        return float(torch.rad2deg(rot).mean()), float(trans.mean())
