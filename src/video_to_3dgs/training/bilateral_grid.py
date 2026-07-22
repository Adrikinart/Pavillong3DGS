"""Bilateral-grid appearance correction (Wang et al., SIGGRAPH 2024).

Our affine appearance model (``appearance.py``) applies one 3x3 matrix and bias to
every pixel of an image. That is deliberately weak — it provably cannot encode
geometry — but it is also unable to represent anything *spatially varying*:
vignetting, a local auto-exposure response, or a white-balance shift that differs
between the sunlit and shadowed halves of a frame.

A bilateral grid is the standard middle ground, and is what Nerfstudio now uses for
Splatfacto. Each image owns a low-resolution 3D grid over
:math:`(x, y, \\text{luma})`; each cell holds an affine colour transform, and a pixel
is corrected by the transform obtained from trilinearly interpolating ("slicing")
that grid at its own position and intensity. Because the grid is coarse — typically
16x16x8 — the correction varies smoothly across the image and cannot represent
high-frequency, per-pixel edits.

That coarseness is the whole safety argument, and it is weaker than the affine
model's. An affine map carries exactly zero spatial information (it commutes with
any pixel permutation); a bilateral grid carries a little, bounded by the grid
resolution. A total-variation penalty on the grid keeps it smooth, and the grid
should stay coarse: with enough cells it could begin memorising each training view
and eroding the multi-view constraint that makes reconstruction possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# BT.601 luma, matching the usual bilateral-grid guide channel
_LUMA = (0.299, 0.587, 0.114)


class BilateralGrid(nn.Module):
    """Per-image low-resolution grid of affine colour transforms.

    Grid shape is ``(n_images, 12, L, H, W)``: twelve channels per cell hold a 3x4
    affine transform (3x3 matrix + bias), stored as a *residual* around the identity
    so an all-zero grid is exactly a no-op at initialisation.
    """

    # NOTE: do not name any helper ``_apply`` - nn.Module._apply is the internal
    # hook used by .to()/.cuda()/.float(), and shadowing it breaks device moves.
    def __init__(self, n_images: int, grid_w: int = 16, grid_h: int = 16,
                 grid_l: int = 8):
        super().__init__()
        self.n_images = int(n_images)
        self.shape = (grid_l, grid_h, grid_w)
        self.grids = nn.Parameter(torch.zeros(self.n_images, 12, grid_l, grid_h, grid_w))

    # ------------------------------------------------------------------ #
    def forward(self, image_index: int, rgb: torch.Tensor) -> torch.Tensor:
        """Apply image ``image_index``'s grid to an (H,W,3) render."""
        return self._apply_grid(self.grids[image_index], rgb)

    def canonical(self, rgb: torch.Tensor, indices=None) -> torch.Tensor:
        """Apply the mean grid of a subset (default: all images).

        As with the affine model, held-out views are scored under an appearance that
        does not depend on the held-out pixels — using the view's own grid would fit
        the evaluation image.
        """
        g = self.grids if indices is None or len(indices) == 0 else \
            self.grids[torch.as_tensor(list(indices), device=self.grids.device,
                                       dtype=torch.long)]
        return self._apply_grid(g.mean(dim=0), rgb)

    def canonical_for(self, rgb: torch.Tensor, indices) -> torch.Tensor:
        """API-compatible with :class:`AppearanceModel`: score a held-out view under
        the mean appearance of its own source clip."""
        return self.canonical(rgb, indices)

    # ------------------------------------------------------------------ #
    def _apply_grid(self, grid: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        h, w, _ = rgb.shape
        dev, dt = rgb.device, rgb.dtype

        luma = (rgb[..., 0] * _LUMA[0] + rgb[..., 1] * _LUMA[1]
                + rgb[..., 2] * _LUMA[2]).clamp(0, 1)
        ys = torch.linspace(-1, 1, h, device=dev, dtype=dt)[:, None].expand(h, w)
        xs = torch.linspace(-1, 1, w, device=dev, dtype=dt)[None, :].expand(h, w)
        zs = luma * 2 - 1                                  # guide axis -> [-1,1]

        # grid_sample expects (N,C,L,H,W) and coordinates ordered (x, y, z)
        coords = torch.stack([xs, ys, zs], dim=-1)[None, None]        # (1,1,H,W,3)
        sliced = F.grid_sample(grid[None], coords, mode="bilinear",
                               padding_mode="border", align_corners=True)
        sliced = sliced[0, :, 0].permute(1, 2, 0)                     # (H,W,12)

        matrix = sliced[..., :9].reshape(h, w, 3, 3)
        bias = sliced[..., 9:]
        eye = torch.eye(3, device=dev, dtype=dt).expand(h, w, 3, 3)
        matrix = matrix + eye                                          # residual
        return (matrix @ rgb[..., None]).squeeze(-1) + bias

    # ------------------------------------------------------------------ #
    def tv_loss(self) -> torch.Tensor:
        """Total variation over the grid, keeping the correction smooth."""
        g = self.grids
        return (g.diff(dim=2).abs().mean() + g.diff(dim=3).abs().mean()
                + g.diff(dim=4).abs().mean())

    @torch.no_grad()
    def drift(self) -> float:
        """Mean absolute departure from the identity transform."""
        return float(self.grids.abs().mean())
