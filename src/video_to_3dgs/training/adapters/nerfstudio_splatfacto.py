"""Optional Nerfstudio Splatfacto adapter (thin wrapper around `ns-train`).

Not implemented in this iteration. The primary backend is direct gsplat, which
avoids Nerfstudio's tiny-cuda-nn Blackwell friction. This stub documents the
intended shape: shell out to `ns-train splatfacto`, then parse its outputs into
a TrainResult without touching core code.
"""

from __future__ import annotations

from pathlib import Path

from ..backend import TrainContext, TrainingBackend, TrainResult


class SplatfactoBackend(TrainingBackend):
    name = "splatfacto"

    def validate_env(self) -> None:
        import shutil
        if shutil.which("ns-train") is None:
            raise RuntimeError(
                "nerfstudio ('ns-train') not installed. Install nerfstudio in the env "
                "or use backend=gsplat (recommended on Blackwell).")

    def train(self, ctx: TrainContext) -> TrainResult:
        raise NotImplementedError(
            "splatfacto adapter is a stub; use backend=gsplat. To implement: convert "
            "the COLMAP model to nerfstudio format, run `ns-train splatfacto`, then map "
            "its eval json into TrainResult.")

    def export_ply(self, ctx: TrainContext, checkpoint: Path, out: Path) -> Path:
        raise NotImplementedError("splatfacto export not implemented; use backend=gsplat")
