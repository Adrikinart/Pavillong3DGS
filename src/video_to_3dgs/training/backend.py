"""Training backend abstraction + registry.

Core code depends only on this interface. Concrete backends (gsplat now,
splatfacto/orig-3dgs later) are imported lazily so the CPU-only CLI can list
stages without importing torch/gsplat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import logging

    from ..config.schema import PipelineConfig, TrainCfg
    from ..core.paths import RunLayout


@dataclass
class TrainContext:
    layout: "RunLayout"
    config: "PipelineConfig"
    train_cfg: "TrainCfg"
    train_run_id: str
    device: str
    logger: "logging.Logger"
    resume: bool = True
    # optional callback set by the runner to check for preemption
    preempted: Callable[[], bool] = field(default=lambda: False)


@dataclass
class TrainResult:
    final_checkpoint: Path | None
    metrics: dict[str, Any]
    n_gaussians: int
    status: str = "COMPLETED"  # COMPLETED | PREEMPTED | FAILED


class TrainingBackend:
    name: str = "base"

    def validate_env(self) -> None:
        """Assert the required stack (torch cu128, gsplat, sm_120) is present."""
        raise NotImplementedError

    def train(self, ctx: TrainContext) -> TrainResult:
        raise NotImplementedError

    def export_ply(self, ctx: TrainContext, checkpoint: Path, out: Path) -> Path:
        raise NotImplementedError


_REGISTRY: dict[str, Callable[[], TrainingBackend]] = {}


def register(name: str, factory: Callable[[], TrainingBackend]) -> None:
    _REGISTRY[name] = factory


def get_backend(name: str) -> TrainingBackend:
    if name not in _REGISTRY:
        _register_builtins()
    if name not in _REGISTRY:
        raise ValueError(f"unknown training backend '{name}'. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def _register_builtins() -> None:
    def _gsplat() -> TrainingBackend:
        from .gsplat_backend import GsplatBackend
        return GsplatBackend()
    register("gsplat", _gsplat)

    def _splatfacto() -> TrainingBackend:
        from .adapters.nerfstudio_splatfacto import SplatfactoBackend
        return SplatfactoBackend()
    register("splatfacto", _splatfacto)
