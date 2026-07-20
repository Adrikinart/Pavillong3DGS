"""blur_var must not depend on extraction resolution (regression).

Extracting the Pavillon clip at 2560px instead of 1600px once dropped the median
blur score from ~92 to ~36, so the calibrated `blur_var_min: 50` rejected 308/400
frames as "blurry" and starved COLMAP of coverage. `_blur_var` therefore measures
at a fixed reference scale.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from video_to_3dgs.stages.filter_frames import _blur_var  # noqa: E402


def _textured(h: int, w: int) -> np.ndarray:
    """Deterministic high-frequency texture (stands in for carved relief)."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = (127 + 90 * np.sin(xx / 3.0) * np.cos(yy / 4.0)).astype(np.uint8)
    return img


def test_blur_var_is_scale_invariant():
    base = _textured(900, 1600)
    big = cv2.resize(base, (3200, 1800), interpolation=cv2.INTER_CUBIC)

    v_base, v_big = _blur_var(base), _blur_var(big)
    # Without normalization the upscaled variant scores several times lower.
    assert v_big > 0.4 * v_base, (
        f"blur score collapsed with resolution: {v_base:.1f} -> {v_big:.1f}; "
        "a threshold calibrated at one resize_long_edge would mis-filter at another")


def test_blur_var_still_ranks_blurry_below_sharp():
    sharp = _textured(900, 1600)
    blurry = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4)
    assert _blur_var(blurry) < _blur_var(sharp)
