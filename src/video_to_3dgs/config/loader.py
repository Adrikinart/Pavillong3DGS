"""Layered config loading: defaults -> profile -> user YAML -> --set overrides."""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from ..core.errors import ConfigError
from ..core.paths import RunLayout, slugify
from .schema import PipelineConfig


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file is not a mapping: {path}")
    return data


def _load_packaged_defaults() -> dict:
    try:
        text = (resources.files("video_to_3dgs.config.defaults")
                / "pipeline.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


def _load_profile(profile: str | None) -> dict:
    if not profile:
        return {}
    try:
        text = (resources.files("video_to_3dgs.config.defaults.profiles")
                / f"{profile}.yaml").read_text(encoding="utf-8")
        return yaml.safe_load(text) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


def _coerce_scalar(val: str) -> Any:
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` dotted overrides."""
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got: {item}")
        key, val = item.split("=", 1)
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ConfigError(f"--set path collides with scalar at {p}")
        node[parts[-1]] = _coerce_scalar(val)
    return out


def load_config(user_cfg: str | Path | None = None, *, profile: str | None = None,
                overrides: list[str] | None = None) -> PipelineConfig:
    """Merge all layers and validate into a frozen PipelineConfig."""
    merged = _load_packaged_defaults()

    # profile selection: explicit arg, else user cfg's `profile.name`, else SLURM partition
    user_data: dict = {}
    if user_cfg:
        user_data = _load_yaml(user_cfg)
    prof = profile or user_data.get("profile", {}).get("name")
    if not prof and os.environ.get("SLURM_JOB_PARTITION"):
        prof = os.environ["SLURM_JOB_PARTITION"]
    merged = _deep_merge(merged, _load_profile(prof))
    merged = _deep_merge(merged, user_data)
    if overrides:
        merged = _apply_overrides(merged, overrides)

    try:
        return PipelineConfig.model_validate(merged)
    except Exception as e:  # pydantic ValidationError -> ConfigError with context
        raise ConfigError(f"invalid configuration: {e}") from e


def derive_dataset_id(cfg: PipelineConfig, video_sha: str | None = None) -> str:
    """dataset_id = slug(object_name or first video stem)_<sha8>."""
    if cfg.dataset_id:
        return cfg.dataset_id
    if cfg.object_name:
        base = slugify(cfg.object_name)
    elif cfg.videos:
        base = slugify(Path(cfg.videos[0]).stem)
    else:
        base = "dataset"
    suffix = (video_sha or "")[-8:] if video_sha else "00000000"
    return f"{base}_{suffix}"


def make_layout(cfg: PipelineConfig, repo_root: Path, dataset_id: str) -> RunLayout:
    runs_root = Path(cfg.storage.runs_root)
    if not runs_root.is_absolute():
        runs_root = repo_root / runs_root
    return RunLayout(runs_root=runs_root, dataset_id=dataset_id)


def freeze_config(cfg: PipelineConfig, layout: RunLayout, force: bool = False) -> None:
    """Write config_resolved.yaml if absent. On re-run, keep the frozen file
    authoritative (a mid-run source edit must not silently change semantics) and
    warn on drift — unless ``force`` is set, which re-freezes from the current
    sources (``prepare --force`` / ``run-all --force``)."""
    from ..core.logging import get_logger

    layout.run_dir.mkdir(parents=True, exist_ok=True)
    target = layout.config_resolved
    payload = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)
    if target.exists() and not force:
        existing = target.read_text(encoding="utf-8")
        if existing.strip() != payload.strip():
            get_logger("config").warning(
                "resolved config differs from frozen %s; frozen file is authoritative. "
                "Re-run with --force to apply the edited config.", target)
        return
    if target.exists() and force:
        get_logger("config").info("re-freezing config (--force) at %s", target)
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, target)


def load_frozen_config(layout: RunLayout) -> PipelineConfig:
    if not layout.config_resolved.exists():
        raise ConfigError(f"no frozen config at {layout.config_resolved}; run `prepare` first")
    return PipelineConfig.model_validate(_load_yaml(layout.config_resolved))
