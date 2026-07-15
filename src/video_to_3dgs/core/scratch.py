"""ScratchContext: run high-I/O work on node-local disk, sync verified results back.

The COLMAP SQLite database and large temporary image sets are hostile to the NFS
mount. This context copies declared inputs to node-local scratch, lets the stage
work there, then verifies + atomically promotes outputs back to the run dir. It
deletes scratch only after every output is verified — on failure it retains the
scratch dir (logged) for debugging.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from .atomicio import sha256_file
from .errors import IntegrityError
from .logging import get_logger
from .paths import resolve_scratch_root

log = get_logger("scratch")


class ScratchContext:
    def __init__(self, dataset_id: str, scratch_root: str | None = None,
                 subdir: str = "v2g") -> None:
        root = resolve_scratch_root(scratch_root)
        self.work = root / subdir / dataset_id / uuid.uuid4().hex
        self._synced_ok = False
        self._retain = False

    def __enter__(self) -> "ScratchContext":
        self.work.mkdir(parents=True, exist_ok=True)
        log.info("scratch workspace: %s", self.work)
        return self

    # -- staging in --
    def stage_in(self, src: Path, name: str) -> Path:
        """Copy an input file/dir into scratch, return its scratch path."""
        dst = self.work / name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return dst

    def path(self, name: str) -> Path:
        p = self.work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # -- syncing back --
    def sync_back(self, pairs: list[tuple[Path, Path]]) -> None:
        """Promote (scratch_src, run_dst) outputs atomically with verification.

        Files: sha256-verified then os.replace. Directories: copied to a
        ``.partial`` sibling, then os.replace over the destination.
        """
        for src, dst in pairs:
            if not src.exists():
                raise IntegrityError(f"scratch output missing: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                partial = dst.with_name(dst.name + ".partial")
                if partial.exists():
                    shutil.rmtree(partial)
                shutil.copytree(src, partial)
                if dst.exists():
                    shutil.rmtree(dst)
                partial.replace(dst)
            else:
                partial = dst.with_name(dst.name + ".partial")
                shutil.copy2(src, partial)
                if sha256_file(src) != sha256_file(partial):
                    raise IntegrityError(f"checksum mismatch syncing {src} -> {dst}")
                partial.replace(dst)
            log.info("synced %s -> %s", src.name, dst)
        self._synced_ok = True

    def retain(self) -> None:
        self._retain = True

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None and self._synced_ok and not self._retain:
            shutil.rmtree(self.work, ignore_errors=True)
            log.info("scratch cleaned: %s", self.work)
        else:
            log.warning("scratch RETAINED for debugging: %s (exc=%s synced=%s)",
                        self.work, exc_type.__name__ if exc_type else None, self._synced_ok)
