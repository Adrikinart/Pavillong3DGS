"""Depth-normal consistency: geometry that must be right or the loss is a no-op.

Sign conventions are the trap here. An unoriented normal axis makes the loss punish
correct geometry half the time, and it fails silently -- training still runs.
"""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gsplat")

from video_to_3dgs.training.geometry_reg import (  # noqa: E402
    depth_to_normals, gaussian_normals, normal_consistency_loss)


def _identity_view():
    return torch.eye(4)


def _K(f=100.0, cx=16.0, cy=16.0):
    return torch.tensor([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def test_depth_normals_of_a_frontoparallel_plane_face_the_camera():
    """A plane at constant depth has normal (0,0,-1) in camera space."""
    depth = torch.full((32, 32), 2.0)
    n = depth_to_normals(depth, _K())
    inner = n[2:-2, 2:-2]
    assert torch.allclose(inner, torch.tensor([0.0, 0.0, -1.0]).expand_as(inner), atol=1e-4)


def test_depth_normals_tilt_with_a_slanted_plane():
    """Depth increasing along +x tilts the normal in x; it must stay unit and still
    face the camera."""
    xs = torch.arange(32, dtype=torch.float32)
    depth = 2.0 + 0.01 * xs[None, :].expand(32, 32)
    n = depth_to_normals(depth, _K())[4:-4, 4:-4]
    assert torch.allclose(n.norm(dim=-1), torch.ones(n.shape[:2]), atol=1e-4)
    assert (n[..., 2] < 0).all(), "normals must face the camera"
    assert n[..., 0].abs().mean() > 1e-3, "a slanted plane must tilt the normal"


def test_gaussian_normal_is_the_axis_of_least_variance():
    """A pancake flattened along world z has normal +/-z."""
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])          # identity, wxyz
    scales_log = torch.log(torch.tensor([[0.5, 0.5, 0.01]]))  # thin in z
    means = torch.tensor([[0.0, 0.0, 3.0]])               # in front of camera
    n = gaussian_normals(quats, scales_log, _identity_view(), means)
    assert torch.allclose(n.abs(), torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)


def test_gaussian_normals_are_oriented_towards_the_camera():
    """Sign must be resolved, otherwise the loss penalises correct geometry."""
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    scales_log = torch.log(torch.tensor([[0.5, 0.5, 0.01]] * 2))
    means = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 7.0]])
    n = gaussian_normals(quats, scales_log, _identity_view(), means)
    assert (n[:, 2] < 0).all(), "normals must point back towards the camera"


def test_loss_is_zero_when_normals_agree_and_positive_when_they_do_not():
    depth = torch.full((32, 32), 2.0)
    alphas = torch.ones(32, 32)
    agree = torch.tensor([0.0, 0.0, -1.0]).expand(32, 32, 3).contiguous()
    disagree = torch.tensor([0.0, 0.0, 1.0]).expand(32, 32, 3).contiguous()
    assert normal_consistency_loss(agree, depth, alphas, _K()).item() == pytest.approx(0.0, abs=1e-4)
    assert normal_consistency_loss(disagree, depth, alphas, _K()).item() == pytest.approx(2.0, abs=1e-3)


def test_low_alpha_pixels_are_excluded():
    """Where nothing accumulated, composited depth is not an estimate at all."""
    depth = torch.full((32, 32), 2.0)
    bad = torch.tensor([0.0, 0.0, 1.0]).expand(32, 32, 3).contiguous()
    assert normal_consistency_loss(bad, depth, torch.zeros(32, 32), _K()).item() == pytest.approx(0.0)
