"""Crop an exported ``.ply`` (Gaussian cloud or TSDF mesh) to the object of interest.

The Casque capture is reconstructed *unmasked* on purpose: the checkerboard and the room
carry the SfM features that give good poses, while the chrome helmet has almost none
(see ``docs/reproduce_casque.md``). The cost of that decision is that the exported model
contains a whole auditorium, and the deliverable is the helmet.

So isolation happens here, at export, rather than during training. That ordering is a
measured result, not a preference: constraining the *optimiser* to a helmet-shaped box
(``train.bounds.box_center``) was tested and made things worse -- no sharper helmet, plus
boundary smearing, because with masks off the box fights the photometric loss over the
background the model still has to explain. Cropping afterwards costs nothing and throws
away only what we never wanted.

Two geometry types need different treatment:

* **Gaussian clouds** -- keep a splat if its centre is inside the box. Splats are small
  relative to the box, so centre-membership is a good proxy and avoids inventing
  half-clipped primitives.
* **Meshes** -- keep a triangle only if *all* of its vertices are inside, then drop the
  vertices no surviving face references. Keeping partially-inside triangles would leave
  spikes reaching out to vertices beyond the crop.

Usage:
  python scripts/crop_object.py IN.ply OUT.ply --center -0.17 0.17 0.22 --half-extent 0.5
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:  # pragma: no cover - dependency is only needed for this script
    PlyData = None


def crop_cloud(ply, lo: np.ndarray, hi: np.ndarray):
    """Keep vertices whose centre lies inside the box; preserve every property."""
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return PlyElement.describe(v[keep], "vertex"), int(keep.sum()), len(v)


def crop_mesh(ply, lo: np.ndarray, hi: np.ndarray):
    """Keep fully-inside triangles, then compact the vertex list and reindex faces."""
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)

    faces = ply["face"].data
    key = faces.dtype.names[0]                      # usually 'vertex_indices'
    idx = np.stack(faces[key])                      # (n_faces, 3)
    keep_face = inside[idx].all(axis=1)
    idx = idx[keep_face]

    used = np.unique(idx)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    idx = remap[idx]

    vert_el = PlyElement.describe(v[used], "vertex")
    face_arr = np.empty(len(idx), dtype=[(key, "i4", (3,))])
    face_arr[key] = idx
    face_el = PlyElement.describe(face_arr, "face")
    return [vert_el, face_el], len(used), len(v), int(keep_face.sum()), len(faces)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--center", type=float, nargs=3, required=True,
                    metavar=("X", "Y", "Z"),
                    help="box centre in the NORMALIZED frame (see exports/COORDINATES.md)")
    ap.add_argument("--half-extent", type=float, required=True,
                    help="half side length of the cube, same units as --center")
    args = ap.parse_args()

    if PlyData is None:
        print("this script needs plyfile:  pip install plyfile")
        return 2

    c = np.asarray(args.center, dtype=np.float64)
    lo, hi = c - args.half_extent, c + args.half_extent

    ply = PlyData.read(str(args.src))
    names = {e.name for e in ply.elements}
    src_mb = args.src.stat().st_size / 1e6

    if "face" in names and len(ply["face"].data):
        els, nv, nv0, nf, nf0 = crop_mesh(ply, lo, hi)
        PlyData(els, text=False).write(str(args.dst))
        print(f"mesh:  {nf0:,} -> {nf:,} faces   {nv0:,} -> {nv:,} vertices")
    else:
        el, nv, nv0 = crop_cloud(ply, lo, hi)
        PlyData([el], text=False).write(str(args.dst))
        print(f"cloud: {nv0:,} -> {nv:,} primitives")

    dst_mb = args.dst.stat().st_size / 1e6
    print(f"{args.src.name} ({src_mb:.1f} MB) -> {args.dst.name} ({dst_mb:.1f} MB)")
    if nv == 0:
        print("WARNING: nothing survived the crop -- check that --center is in the "
              "normalized frame, not COLMAP units (see exports/COORDINATES.md)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
