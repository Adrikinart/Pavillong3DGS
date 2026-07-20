"""Monocular-depth priors for sparse-view 3DGS (A3).

Low-overlap captures give the photometric loss too little multi-view signal, so
geometry drifts (floaters, wrong-depth blobs). A monocular depth network
(DepthAnything-v2) predicts a per-frame *relative* depth that we regularize the
rendered depth against, scale/shift-invariantly (Pearson correlation — the
SparseGS/FSGS idea). The prior carries no metric scale, only depth *ordering*,
which is exactly the geometry signal missing here.

Depth maps are computed once and cached to the run dir (``<training>/depth_prior/``)
so restarts and multiple training runs on the same frames reuse them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("video_to_3dgs.depth")


class DepthPriorBank:
    """Per-frame monocular-depth targets, keyed by image name.

    Targets are stored so that *larger == farther* (we negate DepthAnything's
    disparity), matching the rendered expected-depth ordering, so a positive
    Pearson correlation is the optimisation target.
    """

    def __init__(self, targets: dict[str, "np.ndarray"]):
        self._targets = targets  # name -> (H,W) float32, normalized to ~[0,1]

    def __len__(self) -> int:
        return len(self._targets)

    def get(self, name: str, device, shape_hw):
        """Return the target as a (H,W) tensor on ``device``, resized to shape_hw."""
        import torch
        import torch.nn.functional as F

        t = self._targets.get(name)
        if t is None:
            return None
        x = torch.from_numpy(t).to(device)[None, None]  # (1,1,h,w)
        if x.shape[-2:] != tuple(shape_hw):
            x = F.interpolate(x, size=tuple(shape_hw), mode="bilinear",
                              align_corners=False)
        return x[0, 0]

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, samples, cache_dir: Path, model: str, device: str,
              long_edge: int = 518) -> "DepthPriorBank | None":
        """Compute (or load cached) monocular depth for every training sample.

        Returns None if the depth model can't be loaded (the caller then trains
        without the prior rather than crashing).
        """
        cache_dir.mkdir(parents=True, exist_ok=True)
        targets: dict[str, np.ndarray] = {}
        todo = []
        for s in samples:
            cf = cache_dir / f"{Path(s.name).stem}.npy"
            if cf.exists():
                try:
                    targets[s.name] = np.load(cf)
                    continue
                except Exception:  # noqa: BLE001 - corrupt cache -> recompute
                    pass
            todo.append((s, cf))

        if todo:
            pipe = cls._load_pipeline(model, device)
            if pipe is None:
                return None
            from PIL import Image
            for i, (s, cf) in enumerate(todo):
                try:
                    img = Image.open(s.image_path).convert("RGB")
                except Exception as e:  # noqa: BLE001
                    log.warning("depth: cannot open %s: %s", s.name, e)
                    continue
                disp = cls._infer(pipe, img)                 # (H,W), larger=closer
                tgt = cls._to_target(disp)                   # larger=farther, ~[0,1]
                np.save(cf, tgt)
                targets[s.name] = tgt
                if (i + 1) % 20 == 0 or i + 1 == len(todo):
                    log.info("depth prior: %d/%d frames", i + 1, len(todo))

        if not targets:
            return None
        log.info("depth prior ready: %d frames (cache=%s)", len(targets), cache_dir)
        return cls(targets)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_pipeline(model: str, device: str):
        try:
            import torch
            from transformers import pipeline
            dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
            return pipeline("depth-estimation", model=model, device=dev)
        except Exception as e:  # noqa: BLE001
            log.warning("depth prior disabled (could not load %s: %s)", model, e)
            return None

    @staticmethod
    def _infer(pipe, img) -> np.ndarray:
        out = pipe(img)
        d = out["predicted_depth"] if "predicted_depth" in out else out["depth"]
        arr = np.asarray(d, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    @staticmethod
    def _to_target(disp: np.ndarray) -> np.ndarray:
        """DepthAnything returns disparity (larger=closer). Convert to a
        depth-ordered target (larger=farther), robustly normalized to ~[0,1]."""
        d = disp.astype(np.float32)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return 1.0 - d  # invert: far surfaces -> large target


def pearson_depth_loss(rendered_depth, target, mask=None):
    """1 - Pearson correlation between rendered depth and the mono-depth target.

    Scale/shift invariant, so the metric-less monocular prior only constrains
    relative depth ordering. ``rendered_depth``/``target`` are (H,W); ``mask``
    (H,W bool) restricts to valid/object pixels.
    """
    import torch

    x = rendered_depth.reshape(-1)
    y = target.reshape(-1)
    if mask is not None:
        m = mask.reshape(-1) > 0.5
        if int(m.sum()) < 32:
            return rendered_depth.new_zeros(())
        x, y = x[m], y[m]
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm() + 1e-6
    corr = (x * y).sum() / denom
    return 1.0 - corr
