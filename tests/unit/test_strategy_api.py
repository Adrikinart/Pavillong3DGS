"""Both densification strategies must be callable the way the trainer calls them.

DefaultStrategy and MCMCStrategy have genuinely different signatures - MCMC takes no
scene_scale at init and wants the current LR at step time instead of `packed`. A
mismatch only shows up on a GPU node minutes into a run, so pin it here.
"""
import inspect

import pytest

gsplat = pytest.importorskip("gsplat")


def test_default_strategy_signature():
    assert "scene_scale" in inspect.signature(gsplat.DefaultStrategy.initialize_state).parameters
    p = inspect.signature(gsplat.DefaultStrategy.step_post_backward).parameters
    assert "packed" in p and "lr" not in p


def test_mcmc_strategy_signature():
    # no scene_scale: the MCMC budget is absolute rather than scale-relative
    assert "scene_scale" not in inspect.signature(gsplat.MCMCStrategy.initialize_state).parameters
    p = inspect.signature(gsplat.MCMCStrategy.step_post_backward).parameters
    assert "lr" in p, "MCMC needs the current LR to scale its SGLD noise"


def test_mcmc_accepts_the_constructor_args_the_config_exposes():
    p = inspect.signature(gsplat.MCMCStrategy.__init__).parameters
    for k in ("cap_max", "noise_lr", "refine_start_iter", "refine_stop_iter",
              "refine_every", "min_opacity"):
        assert k in p, f"config exposes {k} but gsplat.MCMCStrategy does not accept it"
