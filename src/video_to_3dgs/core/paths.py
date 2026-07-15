"""RunLayout: the single authority for on-disk paths within a dataset run.

No stage hardcodes paths; they all resolve through a RunLayout instance so the
directory structure is defined in exactly one place.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


def slugify(text: str) -> str:
    """Filesystem-safe slug: lowercase, alnum + underscores."""
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "dataset"


@dataclass(frozen=True)
class RunLayout:
    """Resolves every path under a single dataset run directory."""

    runs_root: Path
    dataset_id: str

    @property
    def run_dir(self) -> Path:
        return self.runs_root / self.dataset_id

    # --- top-level artifacts ---
    @property
    def config_resolved(self) -> Path:
        return self.run_dir / "config_resolved.yaml"

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def environment_json(self) -> Path:
        return self.run_dir / "environment.json"

    # --- per-stage bookkeeping ---
    @property
    def status_dir(self) -> Path:
        return self.run_dir / "status"

    def status_file(self, stage: str) -> Path:
        return self.status_dir / f"{stage}.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def stage_log(self, stage: str) -> Path:
        return self.logs_dir / f"{stage}.log"

    def stage_events(self, stage: str) -> Path:
        return self.logs_dir / f"{stage}.jsonl"

    # --- data artifacts ---
    @property
    def video_dir(self) -> Path:
        return self.run_dir / "video"

    @property
    def frames_dir(self) -> Path:
        return self.run_dir / "frames"

    @property
    def frames_index(self) -> Path:
        return self.run_dir / "frames" / "frames_index.json"

    @property
    def frames_filtered_dir(self) -> Path:
        return self.run_dir / "frames_filtered"

    @property
    def frame_scores_csv(self) -> Path:
        return self.run_dir / "frames_filtered" / "frame_scores.csv"

    @property
    def rejected_dir(self) -> Path:
        return self.run_dir / "rejected"

    @property
    def masks_dir(self) -> Path:
        return self.run_dir / "masks"

    @property
    def colmap_dir(self) -> Path:
        return self.run_dir / "colmap"

    @property
    def colmap_db(self) -> Path:
        return self.colmap_dir / "database.db"

    @property
    def colmap_sparse(self) -> Path:
        return self.colmap_dir / "sparse"

    @property
    def colmap_sparse0(self) -> Path:
        return self.colmap_dir / "sparse" / "0"

    @property
    def colmap_images(self) -> Path:
        # undistorted images used for training
        return self.colmap_dir / "images"

    @property
    def colmap_report(self) -> Path:
        return self.colmap_dir / "validation_report.json"

    @property
    def normalized_dir(self) -> Path:
        return self.run_dir / "normalized"

    @property
    def normalize_transform(self) -> Path:
        return self.normalized_dir / "transform.json"

    @property
    def splits_dir(self) -> Path:
        return self.run_dir / "splits"

    def split_file(self, split: str) -> Path:
        return self.splits_dir / f"{split}.txt"

    # --- training / eval / export (namespaced by train_run_id) ---
    @property
    def trainings_dir(self) -> Path:
        return self.run_dir / "trainings"

    def training_dir(self, train_run_id: str) -> Path:
        return self.trainings_dir / train_run_id

    def checkpoints_dir(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "checkpoints"

    def ckpt_latest(self, train_run_id: str) -> Path:
        return self.checkpoints_dir(train_run_id) / "ckpt_latest.pt"

    def metrics_jsonl(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "metrics.jsonl"

    def tensorboard_dir(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "tb"

    def renders_dir(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "renders"

    def eval_json(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "eval.json"

    def system_monitoring(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "system_monitoring.jsonl"

    def report_dir(self, train_run_id: str) -> Path:
        return self.training_dir(train_run_id) / "report"

    def exports_dir(self, train_run_id: str) -> Path:
        return self.run_dir / "exports" / train_run_id

    def ensure_base_dirs(self) -> None:
        for d in (self.run_dir, self.status_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def resolve_scratch_root(explicit: str | None = None) -> Path:
    """Pick a node-local scratch root, honoring the documented preference order."""
    if explicit and explicit != "auto":
        return Path(os.path.expandvars(explicit))
    user = os.environ.get("USER", "user")
    candidates = [
        os.environ.get("SLURM_TMPDIR"),
        os.environ.get("TMPDIR"),
        f"/scratch/{user}",
        f"/local_scratch/{user}",
        f"/var/tmp/{user}",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        try:
            if p.exists() and os.access(p, os.W_OK):
                return p
        except OSError:
            continue
    # last resort: a writable var/tmp we create
    fallback = Path(f"/var/tmp/{user}")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback
