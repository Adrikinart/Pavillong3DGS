"""Stage abstraction: the unit of pipeline work.

A stage declares its inputs/outputs, does work into temp locations, and is the
*only* thing that knows how to produce its outputs. It never writes its own
status file — the StageRunner does, which is what guarantees the "a failed stage
never marks itself COMPLETED" invariant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from .atomicio import canonical_json, sha256_file, sha256_str
from .errors import InputValidationError, OutputValidationError
from .paths import RunLayout

if TYPE_CHECKING:
    from ..config.schema import PipelineConfig
    from .manifest import Manifest


@dataclass(frozen=True)
class Artifact:
    """A named input or output of a stage."""

    key: str
    path: Path
    kind: Literal["file", "dir"] = "file"
    optional: bool = False

    def exists(self) -> bool:
        return self.path.exists()

    def checksum(self) -> str | None:
        """Content checksum: sha256 for files; a manifest-hash for directories."""
        if not self.path.exists():
            return None
        if self.kind == "file":
            return sha256_file(self.path)
        # directory: hash the sorted (relpath, size) listing — cheap and stable,
        # avoids reading gigabytes of frames/images just to fingerprint.
        entries: list[str] = []
        for p in sorted(self.path.rglob("*")):
            if p.is_file():
                try:
                    entries.append(f"{p.relative_to(self.path)}:{p.stat().st_size}")
                except OSError:
                    continue
        return sha256_str("\n".join(entries))

    def count_files(self) -> int:
        if self.kind == "file":
            return 1 if self.path.exists() else 0
        return sum(1 for p in self.path.rglob("*") if p.is_file())


@dataclass
class StageContext:
    """Everything a stage needs to run, resolved for one dataset run."""

    layout: RunLayout
    config: "PipelineConfig"
    manifest: "Manifest"
    logger: logging.Logger
    repo_root: Path
    dry_run: bool = False
    force: bool = False
    verbose: bool = False
    # Free-form runtime params passed from the CLI (e.g. train_run_id, source videos)
    params: dict[str, Any] = field(default_factory=dict)


class Stage:
    """Base class for pipeline stages. Subclasses implement ``run``."""

    name: ClassVar[str] = "stage"
    depends_on: ClassVar[tuple[str, ...]] = ()
    #: whether this stage requires a GPU (informs scheduling / where it runs)
    needs_gpu: ClassVar[bool] = False

    # ---- declared IO (override) ----
    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return []

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return []

    # ---- the actual work (override); returns metrics dict for the manifest ----
    def run(self, ctx: StageContext) -> dict[str, Any]:
        raise NotImplementedError

    # ---- params used for fingerprinting (override to expose the knobs) ----
    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return {}

    # ---- default validation hooks ----
    def validate_inputs(self, ctx: StageContext) -> None:
        for art in self.declared_inputs(ctx):
            if art.optional:
                continue
            if not art.exists():
                raise InputValidationError(
                    f"[{self.name}] required input '{art.key}' missing at {art.path}"
                )

    def validate_outputs(self, ctx: StageContext) -> None:
        for art in self.declared_outputs(ctx):
            if art.optional:
                continue
            if not art.exists():
                raise OutputValidationError(
                    f"[{self.name}] expected output '{art.key}' missing at {art.path}"
                )
            if art.kind == "dir" and art.count_files() == 0:
                raise OutputValidationError(
                    f"[{self.name}] output dir '{art.key}' is empty at {art.path}"
                )

    # ---- fingerprint: params + input checksums ----
    def fingerprint(self, ctx: StageContext) -> str:
        payload = {
            "stage": self.name,
            "params": self.stage_params(ctx),
            "inputs": {a.key: a.checksum() for a in self.declared_inputs(ctx)},
        }
        return sha256_str(canonical_json(payload))
