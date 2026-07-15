"""Checkpoint save/latest-valid/load (requires torch; skipped otherwise)."""

import pytest

torch = pytest.importorskip("torch")

from video_to_3dgs.training.checkpoint import find_latest_valid, load_checkpoint, save_checkpoint


def _make_params():
    from torch import nn
    return nn.ParameterDict({"means": nn.Parameter(torch.randn(10, 3))})


def _make_opt(params):
    return {"means": torch.optim.Adam([params["means"]], lr=1e-3)}


def test_save_and_find_latest(tmp_path):
    params = _make_params()
    opts = _make_opt(params)
    save_checkpoint(tmp_path, params, opts, 100)
    save_checkpoint(tmp_path, params, opts, 200)
    latest = find_latest_valid(tmp_path)
    assert latest is not None and "0000200" in latest.name


def test_corrupt_checkpoint_skipped(tmp_path):
    params = _make_params()
    opts = _make_opt(params)
    save_checkpoint(tmp_path, params, opts, 100)
    save_checkpoint(tmp_path, params, opts, 200)
    # corrupt the newest checkpoint's bytes -> sha mismatch -> skipped
    (tmp_path / "ckpt_0000200.pt").write_bytes(b"garbage")
    latest = find_latest_valid(tmp_path)
    assert latest is not None and "0000100" in latest.name


def test_load_restores_step(tmp_path):
    params = _make_params()
    opts = _make_opt(params)
    save_checkpoint(tmp_path, params, opts, 321)
    p2 = _make_params()
    o2 = _make_opt(p2)
    step = load_checkpoint(tmp_path / "ckpt_0000321.pt", p2, o2)
    assert step == 321
    assert torch.allclose(p2["means"], params["means"])
