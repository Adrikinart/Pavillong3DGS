"""Explicit object box in train.bounds (concentrate the budget on a small subject)."""
import pytest
from video_to_3dgs.config.schema import BoundsCfg, TrainCfg


def test_explicit_box_parses_and_is_optional():
    b = BoundsCfg()
    assert b.box_center is None and b.box_half_extent is None    # default: auto room box
    b2 = BoundsCfg(box_center=[-0.17, 0.17, 0.22], box_half_extent=0.45)
    assert b2.box_center == [-0.17, 0.17, 0.22] and b2.box_half_extent == 0.45


def test_train_cfg_accepts_bounds_box():
    c = TrainCfg(bounds={"box_center": [0.0, 0.1, 0.2], "box_half_extent": 0.4})
    assert c.bounds.box_half_extent == 0.4
