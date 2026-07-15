from pathlib import Path

import pytest

from video_to_3dgs.config.loader import (
    _apply_overrides,
    _deep_merge,
    derive_dataset_id,
    load_config,
)
from video_to_3dgs.core.errors import ConfigError

CONFIGS = ["smoke_test", "scene_pavillon", "object_default",
           "object_high_quality", "object_turntable"]


@pytest.mark.parametrize("name", CONFIGS)
def test_all_pipeline_configs_validate(name):
    cfg = load_config(f"configs/pipeline/{name}.yaml")
    assert cfg.train.backend in ("gsplat", "splatfacto", "orig_3dgs")
    assert 0.0 < cfg.split_dataset.train_fraction < 1.0


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 20, "z": 30}}
    out = _deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    # base is not mutated
    assert base["a"]["y"] == 2


def test_override_scalar_coercion():
    cfg = {"train": {"seed": 0, "mixed_precision": False}}
    out = _apply_overrides(cfg, ["train.seed=7", "train.mixed_precision=true"])
    assert out["train"]["seed"] == 7
    assert out["train"]["mixed_precision"] is True


def test_override_via_load_config():
    cfg = load_config("configs/pipeline/smoke_test.yaml",
                      overrides=["train.max_iterations=123"])
    assert cfg.train.max_iterations == 123


def test_invalid_config_raises():
    with pytest.raises(ConfigError):
        load_config("configs/pipeline/smoke_test.yaml",
                    overrides=["train.backend=not_a_backend"])


def test_dataset_id_deterministic():
    cfg = load_config("configs/pipeline/smoke_test.yaml")
    a = derive_dataset_id(cfg, "deadbeef")
    b = derive_dataset_id(cfg, "deadbeef")
    assert a == b and a.endswith("deadbeef")


def test_frozen_config_immutable():
    cfg = load_config("configs/pipeline/smoke_test.yaml")
    with pytest.raises(Exception):
        cfg.train.max_iterations = 999  # frozen pydantic model
