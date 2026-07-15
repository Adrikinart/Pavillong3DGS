"""Coordinate transforms, frame scores, and split logic (numpy-only)."""

import numpy as np
import pytest

from video_to_3dgs.colmap_io import qvec2rotmat


def test_qvec_identity():
    R = qvec2rotmat(np.array([1.0, 0, 0, 0]))
    assert np.allclose(R, np.eye(3))


def test_qvec_orthonormal():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    R = qvec2rotmat(q / np.linalg.norm(q))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_rotation_to_up_aligns():
    from video_to_3dgs.stages.normalize_scene import _rotation_to_up
    src = np.array([0.0, 0.0, 1.0])
    dst = np.array([0.0, 1.0, 0.0])
    R = _rotation_to_up(src, dst)
    aligned = R @ src
    assert np.allclose(aligned, dst, atol=1e-6)


def test_rotation_to_up_same_vector():
    from video_to_3dgs.stages.normalize_scene import _rotation_to_up
    v = np.array([0.0, 0.0, 1.0])
    assert np.allclose(_rotation_to_up(v, v), np.eye(3))


def test_uniform_indices_cap():
    from video_to_3dgs.stages.extract_frames import _uniform_indices
    idx = _uniform_indices(100, 10)
    assert len(idx) == 10
    assert idx[0] == 0 and idx[-1] <= 99
    assert _uniform_indices(5, 10) == [0, 1, 2, 3, 4]


def test_ahash_hamming():
    cv2 = pytest.importorskip("cv2")
    from video_to_3dgs.stages.filter_frames import _ahash, _hamming
    a = np.zeros((32, 32), np.uint8)
    b = a.copy()
    b[:16] = 255
    ha, hb = _ahash(a), _ahash(b)
    assert _hamming(ha, ha) == 0
    assert _hamming(ha, hb) > 0


def test_blur_var_discriminates():
    cv2 = pytest.importorskip("cv2")
    from video_to_3dgs.stages.filter_frames import _blur_var
    rng = np.random.default_rng(0)
    sharp = (rng.random((128, 128)) * 255).astype(np.uint8)
    blurry = cv2.GaussianBlur(sharp, (21, 21), 8)
    assert _blur_var(sharp) > _blur_var(blurry)
