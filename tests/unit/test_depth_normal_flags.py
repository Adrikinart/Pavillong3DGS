"""The depth-render / depth-loss flag split in the gsplat training loop.

A depth ablation (depth prior off, normal-consistency on) crashed at iteration 7000 with
``AttributeError: 'NoneType' object has no attribute 'get'``. One flag was answering two
questions: *do we need a depth render?* -- which normal-consistency also needs -- and *does
the depth loss apply?*. When normal-consistency switched on it turned the combined flag
true and sent a ``None`` depth bank into the loss block.

The combination had never been run: both captures shipped with the prior enabled, so the
bug sat latent behind the recommended configuration and only surfaced when the prior was
ablated. These tests pin the truth table directly, which is cheap; reproducing it through
the trainer would need a GPU and 7000 iterations.
"""

from __future__ import annotations

import pytest


def _flags(*, bank_present: bool, step: int, depth_start: int,
           normal_enabled: bool, normal_start: int) -> tuple[bool, bool]:
    """Mirror of the flag logic in ``GsplatBackend.train``.

    Kept as a small pure function so the truth table can be asserted without a GPU. If the
    trainer's logic changes, this must change with it -- the docstring above is the reason
    the two must stay in step.
    """
    want_normal = normal_enabled and step >= normal_start
    want_depth_loss = bank_present and step >= depth_start
    want_depth_render = want_depth_loss or want_normal
    return want_depth_render, want_depth_loss


def test_depth_off_with_normals_on_renders_depth_but_applies_no_depth_loss():
    """The exact configuration that crashed."""
    render, loss = _flags(bank_present=False, step=7000, depth_start=2000,
                          normal_enabled=True, normal_start=7000)
    assert render, "normal-consistency still needs the depth render"
    assert not loss, "no depth bank means the depth loss must not run"


def test_depth_off_and_normals_off_needs_no_depth_render():
    render, loss = _flags(bank_present=False, step=9000, depth_start=2000,
                          normal_enabled=False, normal_start=7000)
    assert not render and not loss


def test_depth_on_before_its_start_iteration_applies_no_loss():
    render, loss = _flags(bank_present=True, step=1000, depth_start=2000,
                          normal_enabled=False, normal_start=7000)
    assert not loss
    assert not render


def test_depth_on_after_start_applies_the_loss():
    render, loss = _flags(bank_present=True, step=2000, depth_start=2000,
                          normal_enabled=False, normal_start=7000)
    assert render and loss


@pytest.mark.parametrize("step", [0, 1999, 2000, 6999, 7000, 30000])
def test_loss_never_runs_without_a_bank(step):
    """The invariant that actually prevents the crash, across the whole schedule."""
    _, loss = _flags(bank_present=False, step=step, depth_start=2000,
                     normal_enabled=True, normal_start=7000)
    assert not loss


def test_trainer_still_separates_the_two_flags():
    """Guard against the flags being re-merged in the trainer source.

    A pure-function mirror can silently drift from the code it mirrors, so this checks the
    real module still distinguishes the two names.
    """
    import inspect

    from video_to_3dgs.training import gsplat_backend

    src = inspect.getsource(gsplat_backend.GsplatBackend.train)
    assert "want_depth_loss" in src, "trainer no longer separates loss from render"
    assert "if want_depth_loss:" in src, "depth loss block must be guarded by the loss flag"
