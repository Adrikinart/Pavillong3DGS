"""Slurm preemption/timeout handling.

Slurm sends SIGTERM (and optionally SIGUSR1 via ``--signal``) before killing a
job. We catch it, set a flag, and the training loop checkpoints + exits 0 at the
next safe point so ``--requeue`` can cleanly resubmit.
"""

from __future__ import annotations

import signal


class PreemptionHandler:
    def __init__(self) -> None:
        self.preempted = False
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            try:
                signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass  # not in main thread / unsupported
        self._installed = True

    def _handler(self, signum, frame) -> None:  # noqa: ANN001
        self.preempted = True

    def __call__(self) -> bool:
        return self.preempted
