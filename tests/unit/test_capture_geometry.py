"""Tests for the rig-geometry measurement that picks a novel-view camera path.

The Casque orbit video was rendered with the single-sided path and framed on the whole
room, which left the helmet a few pixels wide. The fix keys off measured geometry rather
than the declared ``capture_mode`` (which defaults to "orbit" and was left at that even
for the single-sided Pavillon, so it cannot be trusted). These tests build both rig types
synthetically and pin the discrimination and the recovered subject centre.
"""

from __future__ import annotations

import numpy as np
import pytest

from video_to_3dgs.reporting import cameras as cam


class _Sample:
    def __init__(self, viewmat):
        self.viewmat = viewmat
        self.K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
        self.width, self.height = 640, 480


class _DS:
    def __init__(self, samples):
        self.samples = samples
        self.points = np.zeros((0, 3))


def _look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=0)
    vm = np.eye(4)
    vm[:3, :3] = R
    vm[:3, 3] = -R @ eye
    return vm


def _orbit_rig(centre, radius=2.0, n=24):
    out = []
    for i in range(n):
        a = 2 * np.pi * i / n
        eye = centre + radius * np.array([np.cos(a), np.sin(a), 0.25])
        out.append(_Sample(_look_at(eye, centre)))
    return _DS(out)


def _single_sided_rig(target, distance=2.0, n=24):
    """Cameras sweeping across one face of a wall: all look roughly the same way."""
    out = []
    for i in range(n):
        lateral = (i / (n - 1) - 0.5) * 2.0
        eye = target + np.array([-distance, lateral, 0.2 * lateral])
        out.append(_Sample(_look_at(eye, target + np.array([0.0, lateral * 0.6, 0.0]))))
    return _DS(out)


def test_orbit_rig_is_detected_and_centre_recovered():
    centre = np.array([0.3, -0.2, 0.1])
    geom = cam.capture_geometry(_orbit_rig(centre))
    assert geom.is_orbit
    # Not ~1.0: the rig has elevation, so the shared downward tilt does not cancel.
    # The real Casque orbit measures 0.61 by the same statistic.
    assert geom.inwardness > 0.6, "inward-looking cameras should largely cancel"
    assert np.allclose(geom.center, centre, atol=1e-6)
    assert geom.cam_distance == pytest.approx(2.0 * np.sqrt(1 + 0.25 ** 2), rel=1e-3)


def test_single_sided_rig_is_not_treated_as_an_orbit():
    geom = cam.capture_geometry(_single_sided_rig(np.zeros(3)))
    assert not geom.is_orbit
    assert geom.inwardness < 0.4, "aligned view directions must not cancel"


def test_the_two_rig_types_are_separated_by_a_wide_margin():
    """Guards the threshold: it should sit in a gap, not between two near-ties."""
    orbit = cam.capture_geometry(_orbit_rig(np.zeros(3))).inwardness
    flat = cam.capture_geometry(_single_sided_rig(np.zeros(3))).inwardness
    assert orbit - flat > 0.4, f"margin too small to place a threshold ({flat}, {orbit})"


def test_centre_is_independent_of_where_the_points_are():
    """The subject centre must come from camera orientations, not the point cloud.

    On the real Casque the sparse points concentrate on a checkerboard target rather than
    the featureless chrome subject, so any point-derived centre lands on the wrong thing.
    """
    centre = np.array([0.1, 0.4, -0.2])
    ds = _orbit_rig(centre)
    ds.points = np.random.default_rng(0).normal(size=(500, 3)) * 5.0 + 50.0
    geom = cam.capture_geometry(ds)
    assert np.allclose(geom.center, centre, atol=1e-6)


def test_degenerate_rig_does_not_raise():
    """All-parallel axes make the least-squares system singular; fall back, don't crash."""
    # Cameras side by side all staring down +x: the axes never intersect, so the
    # least-squares system is singular. (Aiming them along the up axis instead would
    # only exercise a degeneracy in the test's own look-at helper.)
    samples = [_Sample(_look_at(np.array([-5.0, float(i), 0.0]),
                                np.array([0.0, float(i), 0.0])))
               for i in range(4)]
    geom = cam.capture_geometry(_DS(samples))
    assert np.all(np.isfinite(geom.center))
    assert not geom.is_orbit
    # The fallback must stay near the rig. An ill-conditioned solve returns a finite but
    # astronomically distant point, which would place the fly-around outside the scene.
    assert np.linalg.norm(geom.center) < 20.0


def test_empty_dataset_is_safe():
    geom = cam.capture_geometry(_DS([]))
    assert not geom.is_orbit and np.all(np.isfinite(geom.center))
