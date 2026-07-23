"""Model selection should track the subject, not the background, when both are measurable.

On an object filmed inside a room the whole-frame validation metric is dominated by the
background. Measured on the Casque: between iterations 10k and 25k the whole-frame number
falls 16.56 -> 15.90 dB while the helmet rises 22.83 -> 23.54. Reading the whole-frame
curve, a healthy run looks like it is degrading -- and both ``best_val`` and the
early-stopping check read it, so any ``early_stop_patience`` of 3 or more would have
terminated that run around iteration 10 000.

The trainer therefore also scores a subject-only pass whenever masks exist, and selects on
it. These tests pin the selection logic and the reporting contract, which are cheap to
check here and expensive to notice if they silently regress (the failure mode is a slightly
worse model, not a crash).
"""

from __future__ import annotations

import inspect

from video_to_3dgs.training import gsplat_backend


def _train_source() -> str:
    return inspect.getsource(gsplat_backend.GsplatBackend.train)


def test_subject_validation_dataset_is_built_when_masks_exist():
    src = _train_source()
    assert "val_obj_ds" in src, "subject-only validation dataset was removed"
    # Only when training itself is unmasked -- if the loss is already masked, the ordinary
    # val metric is the subject metric and a second pass would be pure waste.
    assert "if not cfg.use_masks and layout.masks_dir.exists():" in src


def test_both_metrics_are_reported():
    """Whole-frame must stay logged: the gap between the two is the diagnostic."""
    src = _train_source()
    assert '"psnr": vres["psnr"]' in src, "whole-frame val must still be logged"
    assert 'entry["psnr_object"]' in src, "subject val must be logged alongside"
    assert 'entry["ssim_object"]' in src


def test_selection_and_early_stopping_use_the_subject_metric():
    src = _train_source()
    assert "sel_psnr" in src, "selection metric variable was removed"
    assert "health.check_improvement(sel_psnr" in src, \
        "early stopping must read the selection metric, not the whole-frame one"
    assert "best_val = {\"psnr\": sel_psnr" in src, \
        "best_val must record the selection metric"


def test_selection_falls_back_to_whole_frame_without_masks():
    """No masks -> no subject metric -> selection must still work on the whole frame."""
    src = _train_source()
    assert 'sel_psnr, sel_what = vres["psnr"], "val"' in src, \
        "the no-mask fallback initialiser was removed"


def test_which_metric_selected_is_recorded():
    """A number is not interpretable unless the run says which metric produced it."""
    src = _train_source()
    assert '"metric": sel_what' in src


def test_val_render_binds_its_dataset_explicitly():
    """The renderer is reused across two datasets; an implicit capture would render the
    wrong camera if their orderings ever diverged."""
    src = _train_source()
    assert "def _make_val_render(ds" in src, "val renderer must take its dataset"
    assert "_make_val_render(val_obj_ds)" in src, \
        "the subject pass must bind the subject dataset, not inherit val_ds"
