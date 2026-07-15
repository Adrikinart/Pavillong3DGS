"""ffprobe / ffmpeg helpers for video inspection and frame extraction."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    pass


def ffprobe(path: str | Path) -> dict[str, Any]:
    """Return the full ffprobe JSON (format + streams) for a media file."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as e:
        raise MediaError("ffprobe not found on PATH") from e
    if r.returncode != 0:
        raise MediaError(f"ffprobe failed for {path}: {r.stderr[:400]}")
    return json.loads(r.stdout)


def _first_video_stream(probe: dict[str, Any]) -> dict[str, Any]:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    raise MediaError("no video stream found")


def _parse_fps(rate: str | None) -> float | None:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/")
        den_f = float(den)
        return float(num) / den_f if den_f else None
    try:
        return float(rate)
    except ValueError:
        return None


def video_metadata(path: str | Path) -> dict[str, Any]:
    """Condensed, pipeline-relevant metadata for one video."""
    probe = ffprobe(path)
    v = _first_video_stream(probe)
    fmt = probe.get("format", {})
    # rotation: check tags and side_data (iPhone displaymatrix)
    rotation = 0
    tags = v.get("tags", {})
    if "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except ValueError:
            pass
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            try:
                rotation = int(sd["rotation"])
            except (ValueError, TypeError):
                pass
    avg_fps = _parse_fps(v.get("avg_frame_rate"))
    r_fps = _parse_fps(v.get("r_frame_rate"))
    nb_frames = v.get("nb_frames")
    duration = float(fmt.get("duration") or v.get("duration") or 0.0)
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    return {
        "path": str(path),
        "codec": v.get("codec_name"),
        "width": int(v.get("width", 0)),
        "height": int(v.get("height", 0)),
        "avg_fps": avg_fps,
        "r_fps": r_fps,
        "variable_fps": (avg_fps is not None and r_fps is not None
                         and abs(avg_fps - r_fps) > 0.01),
        "duration_s": duration,
        "nb_frames": int(nb_frames) if nb_frames and str(nb_frames).isdigit() else None,
        "bit_rate": fmt.get("bit_rate"),
        "rotation": rotation,
        "pix_fmt": v.get("pix_fmt"),
        "color_space": v.get("color_space"),
        "has_audio": has_audio,
    }


def extract_frames_ffmpeg(video: str | Path, out_dir: Path, *, prefix: str,
                          fps: float | None = None, frame_interval: int | None = None,
                          resize_long_edge: int | None = None, honor_rotation: bool = True,
                          jpeg_quality: int = 95, output_format: str = "jpg",
                          start_index: int = 0) -> list[dict[str, Any]]:
    """Extract frames with ffmpeg. Returns per-frame index records.

    Rotation: ffmpeg auto-applies display-matrix rotation by default. We keep
    that behavior when ``honor_rotation`` is True (so iPhone portrait clips come
    out upright) and disable it with ``-noautorotate`` otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = video_metadata(video)
    src_fps = meta["avg_fps"] or meta["r_fps"] or 30.0

    vf_parts: list[str] = []
    if fps is not None:
        vf_parts.append(f"fps={fps}")
    elif frame_interval is not None:
        vf_parts.append(f"select=not(mod(n\\,{frame_interval}))")
    if resize_long_edge is not None:
        # scale so the long edge == resize_long_edge, preserve aspect, even dims
        vf_parts.append(
            f"scale='if(gt(iw,ih),{resize_long_edge},-2)':'if(gt(iw,ih),-2,{resize_long_edge})'")
    vf = ",".join(vf_parts) if vf_parts else "null"

    pattern = str(out_dir / f"{prefix}_%06d.{output_format}")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if not honor_rotation:
        cmd += ["-noautorotate"]
    cmd += ["-i", str(video), "-vf", vf, "-vsync", "vfr"]
    if output_format in ("jpg", "jpeg"):
        # ffmpeg qscale: 2 (best) .. 31 (worst); map 95->2, 50->16
        qscale = max(2, min(31, round((100 - jpeg_quality) / 3) + 2))
        cmd += ["-q:v", str(qscale)]
    cmd += ["-frame_pts", "0", pattern]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        raise MediaError(f"ffmpeg extraction failed for {video}: {r.stderr[:500]}")

    frames = sorted(out_dir.glob(f"{prefix}_*.{output_format}"))
    records: list[dict[str, Any]] = []
    eff_fps = fps if fps else (src_fps / frame_interval if frame_interval else src_fps)
    for i, fp in enumerate(frames):
        ts = i / eff_fps if eff_fps else None
        records.append({
            "image": fp.name,
            "source_video": str(video),
            "extraction_index": start_index + i,
            "approx_timestamp_s": round(ts, 3) if ts is not None else None,
            "method": f"fps={fps}" if fps else f"interval={frame_interval}",
            "rotation_applied": honor_rotation,
        })
    return records
