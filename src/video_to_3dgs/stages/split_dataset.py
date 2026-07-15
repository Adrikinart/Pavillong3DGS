"""Stage: pose-aware train/val/test split (no adjacent-frame leakage)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.atomicio import atomic_write_json
from ..core.stage import Artifact, Stage, StageContext
from .. import colmap_io


class SplitDatasetStage(Stage):
    name = "split_dataset"
    depends_on = ("normalize_scene",)

    def declared_inputs(self, ctx: StageContext) -> list[Artifact]:
        return [Artifact("colmap_sparse", ctx.layout.colmap_sparse0, "dir")]

    def declared_outputs(self, ctx: StageContext) -> list[Artifact]:
        return [
            Artifact("split_train", ctx.layout.split_file("train"), "file"),
            Artifact("split_val", ctx.layout.split_file("val"), "file"),
            Artifact("split_test", ctx.layout.split_file("test"), "file"),
        ]

    def stage_params(self, ctx: StageContext) -> dict[str, Any]:
        return ctx.config.split_dataset.model_dump()

    def run(self, ctx: StageContext) -> dict[str, Any]:
        c = ctx.config.split_dataset
        cams, imgs, pts = colmap_io.read_model(ctx.layout.colmap_sparse0)
        images = sorted(imgs.values(), key=lambda i: i.name)
        names = [im.name for im in images]
        centers = np.array([im.camera_center() for im in images])
        n = len(names)
        if n < 6:
            raise ValueError(f"too few registered images to split: {n}")

        if c.strategy == "random":
            rng = np.random.default_rng(c.seed)
            perm = rng.permutation(n)
            n_test = max(1, int(round(n * c.test_fraction)))
            n_val = max(1, int(round(n * c.validation_fraction)))
            test_idx = set(perm[:n_test].tolist())
            val_idx = set(perm[n_test:n_test + n_val].tolist())
        elif c.strategy == "periodic":
            k = max(2, c.holdout_every)
            test_idx = set(range(0, n, k))
            val_idx = set(range(k // 2, n, k)) - test_idx
        else:  # pose_aware: spread holdouts evenly around the orbit (by azimuth)
            center = np.median(centers, axis=0)
            rel = centers - center
            azi = np.arctan2(rel[:, 1], rel[:, 0])
            order = np.argsort(azi)
            n_test = max(1, int(round(n * c.test_fraction)))
            n_val = max(1, int(round(n * c.validation_fraction)))
            test_pos = np.linspace(0, n - 1, n_test).round().astype(int)
            test_idx = set(order[test_pos].tolist())
            remaining = [int(i) for i in order if int(i) not in test_idx]
            val_pos = np.linspace(0, len(remaining) - 1, n_val).round().astype(int)
            val_idx = {remaining[p] for p in val_pos}

        train_idx = [i for i in range(n) if i not in test_idx and i not in val_idx]

        # coverage guard: every split must be non-empty
        if not train_idx or not val_idx or not test_idx:
            raise ValueError("a split ended up empty; adjust fractions/holdout_every")

        splits = {
            "train": [names[i] for i in train_idx],
            "val": [names[i] for i in sorted(val_idx)],
            "test": [names[i] for i in sorted(test_idx)],
        }
        for name, lst in splits.items():
            self._write_list(ctx.layout.split_file(name), lst)

        atomic_write_json(ctx.layout.splits_dir / "split_summary.json",
                          {"strategy": c.strategy, "counts": {k: len(v) for k, v in splits.items()}})
        try:
            self._plot(ctx, centers, train_idx, sorted(val_idx), sorted(test_idx))
        except Exception as e:
            ctx.logger.warning("split plot failed: %s", e)

        ctx.logger.info("split: train=%d val=%d test=%d (%s)",
                        len(splits["train"]), len(splits["val"]), len(splits["test"]), c.strategy)
        return {k: len(v) for k, v in splits.items()}

    @staticmethod
    def _write_list(path, items: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text("\n".join(items) + "\n", encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _plot(ctx, centers, train_idx, val_idx, test_idx) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(centers[train_idx, 0], centers[train_idx, 1], s=12, c="gray", label="train")
        ax.scatter(centers[val_idx, 0], centers[val_idx, 1], s=25, c="blue", label="val")
        ax.scatter(centers[test_idx, 0], centers[test_idx, 1], s=25, c="red", label="test")
        ax.set_aspect("equal")
        ax.legend()
        ax.set_title("camera split (top view)")
        fig.savefig(ctx.layout.splits_dir / "split_cameras.png", dpi=100)
        plt.close(fig)
