"""Every inference path must reconstruct the specular bank before loading a checkpoint.

This pins a bug that cost a wrong conclusion. The specular head was wired into the training
loop but not into ``evaluate``, which rebuilds ``params`` with ``create_splats`` -- a
function that knows nothing about the head. Loading a checkpoint into a dict lacking the
``sh_spec`` key silently drops it, so evaluation rendered the diffuse bank alone. That is
not the trained model: with a head, part of the appearance lives in the specular bank.

The failure was quiet and looked exactly like the method being bad -- test PSNR fell
3.66 dB while validation, computed inside the trainer where the head *is* applied, showed a
tie. Two held-out splits disagreeing that violently is the signature of an evaluation bug,
not of a modelling result.

These are source-level checks because exercising the real path needs a GPU and a trained
checkpoint. They are cheap insurance against the same omission recurring in a third place.
"""

from __future__ import annotations

import inspect


def test_evaluate_attaches_the_bank_before_loading():
    from video_to_3dgs.stages import evaluate

    src = inspect.getsource(evaluate)
    assert "attach_specular_bank" in src, "evaluate must reconstruct the specular bank"
    attach = src.index("attach_specular_bank(params")
    load = src.index("load_checkpoint(ckpt, params")
    assert attach < load, "the bank must be attached BEFORE load_checkpoint, or it is dropped"


def test_evaluate_renders_with_the_head():
    from video_to_3dgs.stages import evaluate

    assert "spec_degree=spec_degree" in inspect.getsource(evaluate), \
        "evaluate must pass spec_degree to the rasteriser"


def test_report_renderer_attaches_and_uses_the_bank():
    from video_to_3dgs.reporting import render

    src = inspect.getsource(render)
    assert "attach_specular_bank" in src
    assert "spec_degree=self.spec_degree" in src, \
        "novel-view renders must apply the head too, or videos disagree with metrics"


def test_attach_returns_none_without_a_head(tmp_path):
    """A checkpoint with no head must leave params untouched and disable the branch."""
    import torch

    from video_to_3dgs.training.specular import attach_specular_bank

    ckpt = tmp_path / "c.pt"
    torch.save({"params": {"means": torch.zeros(4, 3), "sh0": torch.zeros(4, 1, 3)}}, ckpt)
    params = {"means": torch.zeros(4, 3)}
    assert attach_specular_bank(params, ckpt, "cpu") is None
    assert "sh_spec" not in params


def test_attach_infers_degree_from_the_checkpoint(tmp_path):
    """Degree comes from the stored shape, not from config: a checkpoint must render the
    way it was trained even if the config has since changed."""
    import torch

    from video_to_3dgs.training.specular import attach_specular_bank

    for degree in (1, 2, 3):
        k = (degree + 1) ** 2
        ckpt = tmp_path / f"c{degree}.pt"
        torch.save({"params": {"means": torch.zeros(7, 3),
                               "sh_spec": torch.zeros(7, k, 3)}}, ckpt)
        params = {"means": torch.zeros(7, 3)}
        got = attach_specular_bank(params, ckpt, "cpu")
        assert got == degree, f"expected degree {degree}, got {got}"
        assert params["sh_spec"].shape == (7, k, 3)
