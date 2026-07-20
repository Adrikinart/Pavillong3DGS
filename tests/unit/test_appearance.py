"""Per-image appearance embeddings: the invariants that keep them safe.

The correction must (a) start as the identity so training is unbiased, (b) be able
to absorb a global exposure/white-balance change, and (c) be incapable of encoding
geometry — otherwise it could explain away structure the Gaussians should learn.
"""

import pytest

torch = pytest.importorskip("torch")

from video_to_3dgs.training.appearance import AppearanceModel  # noqa: E402


def _render(h=8, w=8):
    g = torch.linspace(0, 1, h * w).reshape(h, w)
    return torch.stack([g, g.flip(0), g.flip(1)], dim=-1)  # (H,W,3)


def test_starts_as_identity():
    m = AppearanceModel(n_images=5, dim=8)
    rgb = _render()
    for i in range(5):
        assert torch.allclose(m(i, rgb), rgb, atol=1e-6)
    assert m.drift() == pytest.approx(0.0, abs=1e-6)


def test_canonical_matches_identity_at_init():
    m = AppearanceModel(n_images=3, dim=8)
    rgb = _render()
    assert torch.allclose(m.canonical(rgb), rgb, atol=1e-6)


def test_can_absorb_a_global_exposure_change():
    """A per-image latent should learn the gain+offset mapping a render to a
    differently-exposed version of itself.

    The target is deliberately an exact affine function of the render (no clamping
    — saturation is not affine, so no affine transform could reproduce it and the
    test would be asserting something impossible).
    """
    m = AppearanceModel(n_images=1, dim=8)
    rgb = _render()
    target = rgb * 0.6 + 0.05                       # exposure down + lift
    opt = torch.optim.Adam(m.parameters(), lr=5e-2)
    for _ in range(500):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(m(0, rgb), target)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-4, f"failed to fit an affine exposure change: mse={loss.item():.2e}"
    assert m.drift() > 0.01, "drift diagnostic should register a real correction"


def test_is_spatially_uniform_so_it_cannot_encode_geometry():
    """The transform is one affine map applied to every pixel: permuting pixels
    and transforming must equal transforming and permuting."""
    m = AppearanceModel(n_images=2, dim=8)
    with torch.no_grad():  # perturb away from identity
        for p in m.mlp[-1].parameters():
            p.add_(torch.randn_like(p) * 0.1)
    rgb = _render()
    perm = torch.randperm(rgb.shape[0])
    assert torch.allclose(m(1, rgb[perm]), m(1, rgb)[perm], atol=1e-6)


def test_state_round_trips_for_resume():
    a = AppearanceModel(n_images=4, dim=8)
    with torch.no_grad():
        a.embed.weight.add_(torch.randn_like(a.embed.weight))
        for p in a.mlp[-1].parameters():
            p.add_(torch.randn_like(p) * 0.1)
    b = AppearanceModel(n_images=4, dim=8)
    b.load_state_dict(a.state_dict())
    rgb = _render()
    assert torch.allclose(a(2, rgb), b(2, rgb), atol=1e-6)
