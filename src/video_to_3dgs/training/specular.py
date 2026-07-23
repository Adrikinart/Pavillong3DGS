"""Reflection-direction specular head (Ref-NeRF style) for Gaussian splatting.

Spherical harmonics evaluated at the *viewing* direction are a poor basis for a mirror-like
surface: as the camera moves, a reflection sweeps across the surface far faster than a
low-order SH in view direction can follow, so the optimiser settles for a blurred average.
Ref-NeRF's observation (Verbin et al., CVPR 2022) is that the reflected radiance is much
better behaved as a function of the *reflection* direction

    r = v - 2 (v . n) n

which for a mirror points at whatever is actually being reflected. GaussianShader and
Ref-Gaussian carry the same idea into splatting.

This is a deliberately small version of that idea, chosen so it needs no change to the CUDA
rasteriser: each Gaussian carries a second, low-order SH bank evaluated at ``r`` and added
to the ordinary view-direction colour. We evaluate both banks in PyTorch and hand the
rasteriser precomputed RGB.

What it does NOT do, so nobody reads more into it than is there: no explicit BRDF, no
Fresnel term, no roughness-dependent lobe width, no environment map shared across the
scene. A per-Gaussian directional residual is strictly less expressive than any of those,
and on a surface whose reflection is dominated by a *distant* environment a shared env-map
would be the better model.

Measured motivation, on the Casque helmet: the chrome dome scores 22.3 dB against the gold
crest's 24.2, and if every material class reached the best one the subject would gain about
2.2 dB. That is the ceiling this is aiming at a part of -- see docs/reproduce_casque.md.

Cost: one extra SH evaluation per Gaussian per rendered view, plus ``(deg+1)^2 * 3``
parameters per Gaussian. At degree 2 that is 27 floats, about a 25 % parameter increase over
a degree-3 diffuse model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reflect_directions(quats: torch.Tensor, scales_log: torch.Tensor,
                       means: torch.Tensor, viewmat: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """World-space (view_dir, reflect_dir) per Gaussian.

    The normal is the Gaussian's axis of least variance, sign-resolved to face the camera.
    An unoriented axis would reflect half the primitives into the wrong hemisphere, which
    is worse than having no specular term at all.
    """
    from gsplat.utils import normalized_quat_to_rotmat

    R = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))          # (N,3,3)
    smallest = scales_log.argmin(dim=-1)
    n = torch.gather(R, 2, smallest[:, None, None].expand(-1, 3, 1)).squeeze(-1)
    n = F.normalize(n, dim=-1)

    cam_centre = -viewmat[:3, :3].T @ viewmat[:3, 3]                   # world camera origin
    v = F.normalize(means - cam_centre[None, :], dim=-1)               # camera -> Gaussian

    # Face the camera: v points away from the camera, so a camera-facing normal has n.v < 0.
    n = torch.where((n * v).sum(-1, keepdim=True) > 0, -n, n)
    r = v - 2.0 * (v * n).sum(-1, keepdim=True) * n
    return v, F.normalize(r, dim=-1)


def specular_colors(gsplat, params: dict, viewmat: torch.Tensor,
                    sh_degree_now: int, spec_degree: int) -> torch.Tensor:
    """Per-Gaussian RGB: diffuse SH at the view direction + specular SH at the reflection.

    Returns post-activation colours in [0, 1], matching what ``gsplat.rasterization``
    produces internally for the ``sh_degree`` path, so the two are interchangeable.
    """
    from gsplat import spherical_harmonics

    diffuse_coeffs = torch.cat([params["sh0"], params["shN"]], dim=1)  # (N,K,3)
    v, r = reflect_directions(params["quats"], params["scales"], params["means"], viewmat)

    # gsplat's rasteriser evaluates SH at the direction from camera to Gaussian, then adds
    # the 0.5 DC offset and clamps at zero. Reproduce exactly, or enabling the head would
    # silently shift every colour.
    rgb = spherical_harmonics(sh_degree_now, v, diffuse_coeffs) + 0.5
    spec = spherical_harmonics(spec_degree, r, params["sh_spec"])
    return torch.clamp_min(rgb + spec, 0.0)


def init_specular_bank(n: int, degree: int, device) -> torch.Tensor:
    """Zero-initialised specular coefficients: ``(N, (degree+1)^2, 3)``.

    Zero matters. At initialisation the head must be an exact no-op so that turning it on
    changes nothing until it has learned something -- otherwise a run with the head enabled
    is not comparable to one without, and any measured difference is partly just a different
    starting point.
    """
    k = (degree + 1) ** 2
    return torch.zeros(n, k, 3, device=device)
