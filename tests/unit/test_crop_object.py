"""Regression tests for scripts/crop_object.py.

Cropping a mesh means dropping triangles *and* compacting the vertex array, which
requires reindexing every surviving face. That reindexing is the kind of code that fails
silently: a wrong remap still produces a well-formed .ply with a plausible triangle
count, but the faces point at the wrong vertices, and the damage only shows up as a
scrambled mesh in a viewer. These tests pin the invariant that matters -- surviving
triangles must still connect the same points in space they did before the crop.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

plyfile = pytest.importorskip("plyfile")

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "crop_object.py"


def _load():
    spec = importlib.util.spec_from_file_location("crop_object", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mesh_ply(vertices: np.ndarray, faces: np.ndarray):
    v = np.array([tuple(p) for p in vertices],
                 dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    f = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = faces
    return plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex"),
                            plyfile.PlyElement.describe(f, "face")])


def test_mesh_crop_keeps_only_fully_inside_faces_and_reindexes():
    mod = _load()
    # Two triangles: one entirely inside the unit box, one reaching far outside.
    verts = np.array([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0],    # inside
        [9.0, 9.0, 9.0], [9.1, 9.0, 9.0], [9.0, 9.1, 9.0],    # outside
        [0.05, 0.05, 0.0],                                     # inside, but used only
    ], dtype=np.float64)                                       # by a straddling face
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 0, 3]], dtype=np.int32)
    ply = _mesh_ply(verts, faces)

    lo = np.array([-1.0, -1.0, -1.0])
    hi = np.array([1.0, 1.0, 1.0])
    els, nv, nv0, nf, nf0 = mod.crop_mesh(ply, lo, hi)

    assert (nf0, nv0) == (3, 7)
    assert nf == 1, "only the fully-inside triangle should survive"
    # The straddling face is dropped, so vertex 6 is unreferenced and must be compacted
    # away along with the outside ones.
    assert nv == 3

    out_v = els[0].data
    out_f = np.stack(els[1].data["vertex_indices"])
    assert out_f.shape == (1, 3)
    assert out_f.min() >= 0 and out_f.max() < nv, "faces must index the compacted array"

    # The decisive check: the surviving triangle still spans the same three points.
    got = np.sort(np.stack([out_v["x"], out_v["y"], out_v["z"]], axis=1)[out_f[0]], axis=0)
    want = np.sort(verts[[0, 1, 2]], axis=0)
    assert np.allclose(got, want, atol=1e-6)


def test_cloud_crop_keeps_centres_inside_and_preserves_properties():
    mod = _load()
    # A cloud carrying an extra per-point property, as a Gaussian .ply does.
    v = np.array([(0.0, 0.0, 0.0, 0.5), (5.0, 0.0, 0.0, 0.25), (0.2, -0.2, 0.1, 0.75)],
                 dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    ply = plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")])

    el, nv, nv0 = mod.crop_cloud(ply, np.array([-1.0] * 3), np.array([1.0] * 3))

    assert (nv0, nv) == (3, 2)
    assert "opacity" in el.data.dtype.names, "non-positional properties must survive"
    assert np.allclose(sorted(el.data["opacity"]), [0.5, 0.75])


def test_empty_crop_is_reported_not_silently_written(tmp_path):
    """A box that matches nothing must not look like a successful crop."""
    mod = _load()
    v = np.array([(9.0, 9.0, 9.0, 1.0)],
                 dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")])
    ply = plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")])
    _, nv, nv0 = mod.crop_cloud(ply, np.array([-1.0] * 3), np.array([1.0] * 3))
    assert (nv0, nv) == (1, 0)   # main() turns this into a warning + non-zero exit
