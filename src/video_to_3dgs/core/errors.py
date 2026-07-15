"""Typed exceptions for the pipeline. Each maps to a clear CLI exit code."""

from __future__ import annotations


class V2GSError(Exception):
    """Base class for all framework errors."""

    exit_code: int = 1


class ConfigError(V2GSError):
    """Configuration is missing, malformed, or fails validation."""

    exit_code = 2


class InputValidationError(V2GSError):
    """A stage's declared inputs are missing or invalid (fail fast before work)."""

    exit_code = 3


class OutputValidationError(V2GSError):
    """A stage produced missing/corrupt outputs (gate before marking COMPLETED)."""

    exit_code = 4


class IntegrityError(V2GSError):
    """Checksum / file-count mismatch on transfer or checkpoint load."""

    exit_code = 5


class StageExecutionError(V2GSError):
    """A stage's ``run`` raised. Wraps the original exception."""

    exit_code = 6


class EnvironmentError_(V2GSError):
    """Environment/CUDA/dependency check failed."""

    exit_code = 7


class TrainingHealthError(V2GSError):
    """A training health check tripped a hard failure (NaN, exploding count, ...)."""

    exit_code = 8
