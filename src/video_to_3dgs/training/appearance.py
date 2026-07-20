"""Per-image appearance modelling for multi-clip captures.

Merging several handheld clips of one object is the cheapest way to buy coverage,
but each clip carries its own auto-exposure / white-balance response. A 3DGS model
has no way to represent "the same surface, two different camera responses": the
photometric loss then fights itself, residual gradients stay high everywhere,
densification runs away and quality collapses (this is exactly what happened to the
3-clip Pavillon merge, which plateaued around PSNR 14 while a single clip reached
~24).

The standard remedy is a *generative latent optimisation* (GLO) style latent per
training image, decoded into a low-capacity photometric correction that is applied
to the render before the loss — as in NeRF-W's appearance embeddings. Keeping the
correction global and affine per image is the important part: a 3x3 colour matrix
plus bias can absorb exposure, gain and white-balance drift, but it has nowhere to
hide geometry, so it cannot explain away structure the Gaussians should be
learning.

Evaluation uses the *mean* training embedding, giving a single canonical appearance
rather than letting the model fit a held-out image's exposure (which would leak
test information).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def clip_key(name: str) -> str:
    """Source-clip identifier for a frame name.

    ``img_9647_000008.jpg`` -> ``img_9647``. Frames are named
    ``<clip>_<frameindex>``, so stripping the trailing index groups every frame
    that came from one video — and therefore one auto-exposure/white-balance
    regime.
    """
    stem = Path(name).stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


class AppearanceModel(nn.Module):
    """Per-image latent -> global affine colour transform (3x3 matrix + bias).

    Initialised to the identity transform so training starts unbiased and the
    correction only departs from identity if it actually reduces the loss.
    """

    def __init__(self, n_images: int, dim: int = 16, hidden: int = 64):
        super().__init__()
        self.n_images = int(n_images)
        self.embed = nn.Embedding(self.n_images, dim)
        nn.init.zeros_(self.embed.weight)          # all images start identical
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 12),
        )
        # zero last layer -> forward() returns exactly the identity transform
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    # ------------------------------------------------------------------ #
    def _transform(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.mlp(latent)
        eye = torch.eye(3, device=p.device, dtype=p.dtype)
        matrix = p[..., :9].reshape(3, 3) + eye     # residual around identity
        bias = p[..., 9:]
        return matrix, bias

    def forward(self, image_index: int, rgb: torch.Tensor) -> torch.Tensor:
        """Apply image ``image_index``'s correction to an (H,W,3) render."""
        idx = torch.as_tensor(int(image_index), device=rgb.device)
        matrix, bias = self._transform(self.embed(idx))
        return rgb @ matrix.T + bias

    def canonical(self, rgb: torch.Tensor) -> torch.Tensor:
        """Apply the mean training appearance — used for validation/eval so the
        model is scored on one canonical appearance instead of fitting the
        held-out image's own exposure."""
        matrix, bias = self._transform(self.embed.weight.mean(dim=0))
        return rgb @ matrix.T + bias

    def canonical_for(self, rgb: torch.Tensor, indices) -> torch.Tensor:
        """Apply the mean appearance of a SUBSET of training images.

        Used to score a held-out view under the appearance of *its own source
        clip* rather than a global average across clips. This leaks nothing: it
        uses the view's clip identity (metadata known a priori) and that clip's
        TRAINING images, never the held-out image's own pixels. Averaging across
        clips instead penalises every view when the clips genuinely differ, which
        is a measurement artefact rather than a property of the reconstruction.
        """
        if indices is None or len(indices) == 0:
            return self.canonical(rgb)
        idx = torch.as_tensor(list(indices), device=self.embed.weight.device,
                              dtype=torch.long)
        matrix, bias = self._transform(self.embed.weight[idx].mean(dim=0))
        return rgb @ matrix.T + bias

    @torch.no_grad()
    def drift(self) -> float:
        """Mean absolute departure from identity — a diagnostic. Values near 0 mean
        the clips were already photometrically consistent (the correction is doing
        nothing); large values mean it is absorbing real exposure differences."""
        matrix, bias = self._transform(self.embed.weight.mean(dim=0))
        eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
        return float((matrix - eye).abs().mean() + bias.abs().mean())
