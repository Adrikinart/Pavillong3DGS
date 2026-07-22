"""Bilateral grid: identity init, spatial expressiveness, and its capacity bound.

The grid is deliberately more expressive than the affine model, so the safety
argument is different: it CAN vary spatially, but only smoothly, bounded by grid
resolution. These tests pin both halves of that.
"""
import pytest

torch = pytest.importorskip("torch")

from video_to_3dgs.training.bilateral_grid import BilateralGrid  # noqa: E402


def _img(h=24, w=32):
    g = torch.linspace(0.1, 0.9, h * w).reshape(h, w)
    return torch.stack([g, g.flip(0), g.flip(1)], dim=-1)


def test_starts_as_identity():
    bg = BilateralGrid(3)
    rgb = _img()
    for i in range(3):
        assert torch.allclose(bg(i, rgb), rgb, atol=1e-6)
    assert bg.drift() == pytest.approx(0.0)
    assert bg.tv_loss().item() == pytest.approx(0.0)


def test_canonical_matches_identity_at_init():
    bg = BilateralGrid(4)
    rgb = _img()
    assert torch.allclose(bg.canonical(rgb), rgb, atol=1e-6)


def test_can_represent_a_SPATIALLY_VARYING_correction():
    """This is the capability the affine model lacks: fit a left-right brightness
    ramp (a stand-in for vignetting), which no single global transform can match."""
    bg = BilateralGrid(1, grid_w=16, grid_h=16, grid_l=8)
    rgb = _img()
    h, w, _ = rgb.shape
    ramp = torch.linspace(0.6, 1.4, w)[None, :, None]
    target = (rgb * ramp).clamp(0, 1)
    opt = torch.optim.Adam(bg.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(bg(0, rgb), target)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3, f"failed to fit a spatial ramp: mse={loss.item():.2e}"
    assert bg.drift() > 1e-3


def test_correction_is_band_limited_compared_to_a_per_pixel_control():
    """Capacity bound, measured against a control.

    The property is that the correction cannot carry detail finer than the grid.
    Measuring that with a downsample/upsample round trip alone is confounded: the
    round trip is lossy even for a perfectly band-limited signal. So we run the
    SAME round trip on white noise of comparable energy and require the grid's
    correction to survive it far better. A per-pixel correction -- the thing that
    could memorise a training view -- would behave like the control.
    """
    import torch.nn.functional as F

    def roundtrip_ratio(field, gh, gw):
        f = field.permute(2, 0, 1)[None]
        rt = F.interpolate(F.interpolate(f, size=(gh, gw), mode="area"),
                           size=f.shape[-2:], mode="bilinear", align_corners=True)
        return ((f - rt).abs().mean() / f.abs().mean().clamp_min(1e-8)).item()

    gw = gh = 8
    bg = BilateralGrid(1, grid_w=gw, grid_h=gh, grid_l=4)
    with torch.no_grad():
        bg.grids.add_(torch.randn_like(bg.grids) * 0.2)

    flat = torch.full((24, 32, 3), 0.5)      # uniform guide isolates the SPATIAL axes
    field = bg(0, flat) - flat
    grid_ratio = roundtrip_ratio(field, gh, gw)

    torch.manual_seed(0)
    control = torch.randn_like(field) * field.abs().mean()
    control_ratio = roundtrip_ratio(control, gh, gw)

    assert field.abs().mean() > 1e-3, "perturbed grid should produce a real correction"
    assert grid_ratio < 0.5 * control_ratio, (
        f"correction is not meaningfully smoother than per-pixel noise: "
        f"grid {grid_ratio:.3f} vs control {control_ratio:.3f}")


def test_tv_loss_penalises_a_rough_grid():
    bg = BilateralGrid(1)
    with torch.no_grad():
        bg.grids.add_(torch.randn_like(bg.grids))
    assert bg.tv_loss().item() > 0.1


def test_module_survives_a_device_move():
    """Regression: a helper named ``_apply`` shadows ``nn.Module._apply``, the hook
    PyTorch uses for .to()/.cuda()/.float(). The collision is invisible on CPU-only
    tests and only explodes when the model is moved to the GPU, minutes into a run.
    """
    bg = BilateralGrid(2).to("cpu").float()
    rgb = _img()
    assert torch.allclose(bg(0, rgb), rgb, atol=1e-6)
    assert BilateralGrid._apply is torch.nn.Module._apply, \
        "do not shadow nn.Module._apply"


def test_api_matches_the_affine_appearance_model():
    """Both appearance models are used interchangeably by the trainer and the
    evaluate stage, so they must expose the same surface."""
    from video_to_3dgs.training.appearance import AppearanceModel
    bg, af = BilateralGrid(3), AppearanceModel(3)
    for name in ("forward", "canonical", "canonical_for", "drift"):
        assert hasattr(bg, name) and hasattr(af, name), f"missing {name}"
    rgb = _img()
    assert torch.allclose(bg.canonical_for(rgb, [0, 1]), rgb, atol=1e-6)


def test_state_round_trips_for_resume():
    a = BilateralGrid(2)
    with torch.no_grad():
        a.grids.add_(torch.randn_like(a.grids) * 0.1)
    b = BilateralGrid(2)
    b.load_state_dict(a.state_dict())
    rgb = _img()
    assert torch.allclose(a(1, rgb), b(1, rgb), atol=1e-6)
