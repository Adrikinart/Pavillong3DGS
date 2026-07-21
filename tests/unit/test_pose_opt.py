"""SE(3) pose refinement: the properties that make it safe to enable.

The failure modes here are silent — a wrong exponential map still trains, it just
distorts geometry — so the maths is pinned down directly.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from video_to_3dgs.training.pose_opt import PoseOptimizer, se3_exp  # noqa: E402


def test_zero_delta_is_identity():
    T = se3_exp(torch.zeros(6))
    assert torch.allclose(T, torch.eye(4), atol=1e-7)


def test_small_angle_branch_matches_the_general_one():
    """The Taylor branch exists because sin(t)/t is unstable near 0; it must agree
    with the general formula just above the switch-over."""
    for theta in (1e-7, 1e-6, 1e-5, 1e-4):
        d = torch.zeros(6)
        d[2] = theta
        T = se3_exp(d)
        # rotation about z by theta
        expect = torch.eye(4)
        expect[0, 0] = math.cos(theta); expect[0, 1] = -math.sin(theta)
        expect[1, 0] = math.sin(theta); expect[1, 1] = math.cos(theta)
        assert torch.allclose(T, expect, atol=1e-6), f"mismatch at theta={theta}"


def test_rotation_is_orthonormal_and_right_handed():
    torch.manual_seed(0)
    for _ in range(20):
        d = torch.randn(6) * 0.4
        R = se3_exp(d)[:3, :3]
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5), "R must be orthonormal"
        assert torch.det(R).item() == pytest.approx(1.0, abs=1e-5), "det(R) must be +1"


def test_known_rotation_90_degrees_about_z():
    d = torch.zeros(6); d[2] = math.pi / 2
    R = se3_exp(d)[:3, :3]
    got = R @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(got, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)


def test_translation_component_is_passed_through():
    d = torch.zeros(6); d[3:] = torch.tensor([0.1, -0.2, 0.3])
    assert torch.allclose(se3_exp(d)[:3, 3], torch.tensor([0.1, -0.2, 0.3]), atol=1e-7)


def test_optimizer_starts_as_a_no_op():
    """Training must begin from exactly the SfM solution."""
    po = PoseOptimizer(n_cameras=5)
    vm = torch.eye(4); vm[:3, 3] = torch.tensor([0.3, -0.4, 1.2])
    for i in range(5):
        assert torch.allclose(po(i, vm), vm, atol=1e-7)
    rot, trans = po.magnitude()
    assert rot == pytest.approx(0.0) and trans == pytest.approx(0.0)


def test_gradients_reach_the_deltas():
    """Gradient must flow to the used camera and to no other.

    The loss is deliberately ASYMMETRIC. An obvious choice like ((T - 2I)**2).sum()
    silently tests nothing at zero-init: there dL/dtau = 2*tau = 0 and the two skew
    terms cancel (dL/domega_3 = 4*omega_3 = 0), so the gradient vanishes because the
    loss is at a stationary point, not because anything is broken.
    """
    po = PoseOptimizer(n_cameras=3)
    vm = torch.eye(4)
    weights = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    loss = (po(1, vm) * weights).sum()
    loss.backward()
    g = po.deltas.grad
    assert g is not None and g[1].abs().sum() > 0, "camera 1 must receive gradient"
    assert g[0].abs().sum() == 0 and g[2].abs().sum() == 0, "other cameras must not"


def test_magnitude_reports_degrees_and_units():
    po = PoseOptimizer(n_cameras=2)
    with torch.no_grad():
        po.deltas[0, 2] = math.radians(3.0)      # 3 deg about z
        po.deltas[1, 2] = math.radians(1.0)
        po.deltas[:, 3] = 0.02                   # 0.02 translation on x
    rot, trans = po.magnitude()
    assert rot == pytest.approx(2.0, abs=1e-3)   # mean of 3 and 1 degrees
    assert trans == pytest.approx(0.02, abs=1e-6)
