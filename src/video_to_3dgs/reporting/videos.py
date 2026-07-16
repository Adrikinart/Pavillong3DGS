"""Video generators: V1 novel-view orbit, V3 training-progression."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_ITER_RE = re.compile(r"val_(\d+)")


def _writer(out: Path, fps: int):
    import imageio.v2 as imageio
    out.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(out), fps=fps, macro_block_size=None, codec="libx264",
                              quality=8)


def _even(img: np.ndarray) -> np.ndarray:
    """Crop to even width/height (libx264 requires it) and drop any alpha."""
    img = img[..., :3]
    h, w = img.shape[:2]
    return np.ascontiguousarray(img[: h - (h % 2), : w - (w % 2)])


def orbit_video(renderer, out: Path, *, n_frames: int = 120, elevation_deg: float = 20.0,
                radius_scale: float = 1.2, fps: int = 30, width: int = 960,
                height: int = 540) -> Path | None:
    """Render a closed orbit around the scene from a trained checkpoint."""
    from . import cameras as cam_mod

    center, up, radius, K = cam_mod.scene_frame_from_dataset(renderer.dataset)
    # scale intrinsics from a source view to the requested render size
    src = renderer.dataset.samples[0]
    K = cam_mod.resize_intrinsics(K, (src.width, src.height), (width, height))
    path = cam_mod.orbit_path(center, up, radius, K, width, height, n_frames=n_frames,
                              elevation_deg=elevation_deg, radius_scale=radius_scale)
    w = _writer(out, fps)
    try:
        for c in path:
            w.append_data(_even(renderer.render(c)))
    finally:
        w.close()
    return out


def progression_video(training_dir: Path, out: Path, *, fps: int = 10) -> Path | None:
    """Assemble the same held-out camera rendered across training (renders/val_*/).

    The validation step renders a deterministic set of cameras every
    validation_interval, so picking one camera name and ordering by iteration
    yields a 'reconstruction forming' clip — no extra checkpoints needed.
    """
    renders = Path(training_dir) / "renders"
    val_dirs = sorted([d for d in renders.glob("val_*") if d.is_dir()],
                      key=lambda d: int(_ITER_RE.search(d.name).group(1)))
    if len(val_dirs) < 2:
        return None
    # choose a camera present in the most val dirs (usually all)
    from collections import Counter
    counts: Counter = Counter()
    for d in val_dirs:
        for p in d.glob("*.png"):
            counts[p.name] += 1
    if not counts:
        return None
    cam_name = counts.most_common(1)[0][0]
    import imageio.v2 as imageio
    frames = [(int(_ITER_RE.search(d.name).group(1)), d / cam_name)
              for d in val_dirs if (d / cam_name).exists()]
    if len(frames) < 2:
        return None
    w = _writer(out, fps)
    try:
        for it, fp in frames:
            img = imageio.imread(fp)
            w.append_data(_even(_annotate(img, f"iter {it}")))
    finally:
        w.close()
    return out


def _annotate(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small iteration label into the top-left (best-effort)."""
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img[..., :3].astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 8 + 7 * len(text), 18], fill=(0, 0, 0))
        d.text((3, 3), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return img[..., :3].astype(np.uint8)
