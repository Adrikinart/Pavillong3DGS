"""StageRunner: executes stages while enforcing the status invariants.

Guarantees:
  * RUNNING is written before work; COMPLETED only after ``validate_outputs``.
  * A COMPLETED stage with a matching fingerprint is SKIPPED unless ``--force``.
  * A stale RUNNING (owning pid/job dead) is treated as re-runnable.
  * ``run_all`` topo-sorts by ``depends_on`` and slices with from/to/only.
"""

from __future__ import annotations

import os
import platform
import socket
import time
import traceback
from typing import Any, Iterable

from .errors import V2GSError
from .stage import Stage, StageContext
from .status import (
    StageStatus,
    StatusRecord,
    process_alive,
    read_status,
    utcnow,
    write_status,
)


class StageRunner:
    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx
        self.layout = ctx.layout
        self.log = ctx.logger

    # ------------------------------------------------------------------ #
    def _status_path(self, stage: Stage):
        return self.layout.status_file(stage.name)

    def _should_skip(self, stage: Stage, rec: StatusRecord, fp: str) -> bool:
        if self.ctx.force:
            return False
        if rec.state == StageStatus.COMPLETED.value and rec.fingerprint == fp:
            return True
        return False

    def _recover_stale(self, stage: Stage, rec: StatusRecord) -> None:
        """If a prior run left RUNNING but its owner is gone, demote to FAILED."""
        if rec.state == StageStatus.RUNNING.value:
            if not process_alive(rec.pid, rec.slurm_job_id):
                self.log.warning(
                    "stage %s was RUNNING but owner (pid=%s job=%s) is dead -> re-running",
                    stage.name, rec.pid, rec.slurm_job_id,
                )
                rec.state = StageStatus.FAILED.value
                rec.error = "stale RUNNING recovered"
                write_status(self._status_path(stage), rec)

    # ------------------------------------------------------------------ #
    def execute(self, stage: Stage) -> StageStatus:
        """Run a single stage, honoring resume/force/dry-run. Returns final state."""
        self.layout.ensure_base_dirs()
        sp = self._status_path(stage)
        rec = read_status(sp, stage.name)
        self._recover_stale(stage, rec)
        rec = read_status(sp, stage.name)

        fp = stage.fingerprint(self.ctx)

        if self._should_skip(stage, rec, fp):
            self.log.info("SKIP  %s (COMPLETED, fingerprint match)", stage.name)
            return StageStatus.SKIPPED

        # Fail-fast input validation (leaves prior state untouched)
        stage.validate_inputs(self.ctx)

        if self.ctx.dry_run:
            outs = ", ".join(a.key for a in stage.declared_outputs(self.ctx)) or "(none)"
            self.log.info("DRY   %s -> would produce: %s", stage.name, outs)
            return StageStatus.PENDING

        # ---- mark RUNNING before any work ----
        started = time.time()
        rec = StatusRecord(
            stage=stage.name,
            state=StageStatus.RUNNING.value,
            fingerprint=fp,
            attempt=(rec.attempt or 0) + 1,
            host=socket.gethostname() or platform.node(),
            pid=os.getpid(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            started_at=utcnow(),
            inputs=[{"key": a.key, "path": str(a.path), "checksum": a.checksum()}
                    for a in stage.declared_inputs(self.ctx)],
        )
        write_status(sp, rec)
        self.log.info("RUN   %s (attempt %d)", stage.name, rec.attempt)

        try:
            metrics = stage.run(self.ctx) or {}
            stage.validate_outputs(self.ctx)  # <-- gate before COMPLETED
        except V2GSError as e:
            self._mark_failed(stage, rec, started, e)
            raise
        except Exception as e:  # noqa: BLE001 - convert to failure status then re-raise
            self._mark_failed(stage, rec, started, e)
            raise

        rec.state = StageStatus.COMPLETED.value
        rec.finished_at = utcnow()
        rec.duration_s = round(time.time() - started, 2)
        rec.metrics = metrics
        rec.outputs = [{"key": a.key, "path": str(a.path), "checksum": a.checksum(),
                        "n_files": a.count_files()} for a in stage.declared_outputs(self.ctx)]
        rec.error = None
        rec.traceback = None
        write_status(sp, rec)
        self.ctx.manifest.append_stage({
            "stage": stage.name, "state": rec.state, "params": stage.stage_params(self.ctx),
            "fingerprint": fp, "metrics": metrics, "host": rec.host,
            "slurm_job_id": rec.slurm_job_id, "started_at": rec.started_at,
            "finished_at": rec.finished_at, "duration_s": rec.duration_s,
        })
        self.log.info("DONE  %s in %.1fs %s", stage.name, rec.duration_s or 0.0,
                      _fmt_metrics(metrics))
        return StageStatus.COMPLETED

    def _mark_failed(self, stage: Stage, rec: StatusRecord, started: float,
                     exc: Exception) -> None:
        rec.state = StageStatus.FAILED.value
        rec.finished_at = utcnow()
        rec.duration_s = round(time.time() - started, 2)
        rec.error = f"{type(exc).__name__}: {exc}"
        rec.traceback = traceback.format_exc()
        write_status(self._status_path(stage), rec)
        self.ctx.manifest.append_stage({
            "stage": stage.name, "state": rec.state, "params": stage.stage_params(self.ctx),
            "error": rec.error, "host": rec.host, "slurm_job_id": rec.slurm_job_id,
            "started_at": rec.started_at, "finished_at": rec.finished_at,
        })
        self.log.error("FAIL  %s: %s", stage.name, rec.error)

    # ------------------------------------------------------------------ #
    def run_all(self, stages: list[Stage], *, from_stage: str | None = None,
                to_stage: str | None = None, only: str | None = None) -> dict[str, str]:
        ordered = topo_sort(stages)
        selected = _slice_stages(ordered, from_stage, to_stage, only)
        results: dict[str, str] = {}
        for stage in selected:
            state = self.execute(stage)
            results[stage.name] = state.value
        return results


def _fmt_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    items = list(metrics.items())[:4]
    return "{" + ", ".join(f"{k}={v}" for k, v in items) + "}"


def topo_sort(stages: Iterable[Stage]) -> list[Stage]:
    """Deterministic topological sort by ``depends_on`` (small linear DAG)."""
    by_name = {s.name: s for s in stages}
    visited: dict[str, int] = {}  # 0=visiting, 1=done
    order: list[Stage] = []

    def visit(s: Stage) -> None:
        st = visited.get(s.name)
        if st == 1:
            return
        if st == 0:
            raise V2GSError(f"cycle in stage graph at {s.name}")
        visited[s.name] = 0
        for dep in s.depends_on:
            if dep in by_name:
                visit(by_name[dep])
        visited[s.name] = 1
        order.append(s)

    for s in stages:
        visit(s)
    return order


def _slice_stages(ordered: list[Stage], from_stage: str | None,
                  to_stage: str | None, only: str | None) -> list[Stage]:
    if only:
        return [s for s in ordered if s.name == only]
    names = [s.name for s in ordered]
    lo = names.index(from_stage) if from_stage in names else 0
    hi = names.index(to_stage) + 1 if to_stage in names else len(ordered)
    return ordered[lo:hi]
