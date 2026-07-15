"""Training health checks: convert pathological states into hard failures."""

from __future__ import annotations

import math

from ..core.errors import TrainingHealthError


class HealthMonitor:
    def __init__(self, *, gaussian_cap: int, gaussian_floor: int = 50,
                 no_improve_patience: int = 0, max_scale: float = 1e3):
        self.gaussian_cap = gaussian_cap
        self.gaussian_floor = gaussian_floor
        self.no_improve_patience = no_improve_patience
        self.max_scale = max_scale
        self._best_psnr = -math.inf
        self._stale_vals = 0
        self.nan_count = 0

    def check_loss(self, loss_value: float, step: int) -> None:
        if not math.isfinite(loss_value):
            self.nan_count += 1
            raise TrainingHealthError(f"non-finite loss at step {step}: {loss_value}")

    def check_gaussian_count(self, n: int, step: int) -> None:
        if n > self.gaussian_cap:
            raise TrainingHealthError(
                f"gaussian count exploded at step {step}: {n} > cap {self.gaussian_cap}")
        if n < self.gaussian_floor:
            raise TrainingHealthError(
                f"gaussian count collapsed at step {step}: {n} < floor {self.gaussian_floor}")

    def check_scales(self, max_scale_value: float, step: int) -> None:
        if max_scale_value > self.max_scale:
            # a warning-level condition; caller decides. We raise only if extreme.
            if max_scale_value > self.max_scale * 100:
                raise TrainingHealthError(
                    f"gaussian scale diverged at step {step}: {max_scale_value:.1f}")

    def check_improvement(self, val_psnr: float, step: int) -> bool:
        """Returns True if training should early-stop. Raises nothing by default."""
        if val_psnr > self._best_psnr + 1e-3:
            self._best_psnr = val_psnr
            self._stale_vals = 0
            return False
        self._stale_vals += 1
        if self.no_improve_patience and self._stale_vals >= self.no_improve_patience:
            return True
        return False
