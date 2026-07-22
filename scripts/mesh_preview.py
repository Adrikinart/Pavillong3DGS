"""Headless shaded previews of an exported mesh, for README/report figures.

Open3D's normal visualiser wants a GL context, which the login node does not have. This
uses ``RaycastingScene`` instead: it is pure CPU, needs no display, and gives a
deterministic image -- so the same command produces the same figure on any machine,
which matters for a documentation asset that must not silently drift from the mesh.

Shading is Lambertian on the geometric normal, with the light attached to the camera.
That deliberately shows *shape*, not colour: the question these previews answer is
whether the surface is coherent (does the mesh have the object's form, or is it a blob
of TSDF noise?), and vertex colours would hide exactly the defects worth seeing.

Usage:
  python scripts/mesh_preview.py mesh_helmet.ply --out docs/assets/casque/mesh_preview.png
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def look_at(eye, target, up):
    """Camera-to-world rotation whose -Z axis points from eye to target."""
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    n = np.linalg.norm(r)
    if n < 1e-8:                      # degenerate: view direction parallel to up
        up = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        n = np.linalg.norm(r)
    r /= n
    u = np.cross(r, f)
    return np.stack([r, u, -f], axis=1)


def render(scene, eye, target, up, width, height, fov_deg):
    import open3d as o3d

    R = look_at(eye, target, up)
    aspect = width / height
    t = np.tan(np.radians(fov_deg) / 2.0)
    ys, xs = np.mgrid[0:height, 0:width]
    x = (2 * (xs + 0.5) / width - 1) * t * aspect
    y = (1 - 2 * (ys + 0.5) / height) * t
    dirs = np.stack([x, y, -np.ones_like(x)], axis=-1)
    dirs = dirs @ R.T
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    rays = np.concatenate(
        [np.broadcast_to(eye.astype(np.float32), dirs.shape),
         dirs.astype(np.float32)], axis=-1).reshape(-1, 6)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

    hit = np.isfinite(ans["t_hit"].numpy()).reshape(height, width)
    nrm = ans["primitive_normals"].numpy().reshape(height, width, 3)

    # Head-light Lambertian; normals from raycasting are unoriented, so use |n.v|.
    shade = np.abs((nrm * -dirs).sum(-1)).clip(0, 1)
    img = np.full((height, width, 3), 1.0, dtype=np.float32)
    grey = 0.15 + 0.85 * shade
    img[hit] = grey[hit, None]
    return (img * 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=pathlib.Path, nargs="+",
                    help="one or more meshes; several are drawn as labelled rows sharing "
                         "identical viewpoints, so surfaces can be compared fairly")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="row labels, one per mesh (default: file stem)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--views", type=int, default=3, help="viewpoints around the object")
    ap.add_argument("--elevation", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=520, help="pixels per view (square)")
    ap.add_argument("--fov", type=float, default=45.0)
    ap.add_argument("--zoom", type=float, default=1.5,
                    help="camera distance in units of the object's bounding radius")
    ap.add_argument("--up", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="scene up axis; default reads up_axis_target from a sibling "
                         "normalize_transform.json, falling back to +Z")
    args = ap.parse_args()

    import open3d as o3d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meshes, labels, scenes = [], [], []
    for i, path in enumerate(args.mesh):
        m = o3d.io.read_triangle_mesh(str(path))
        if len(m.triangles) == 0:
            print(f"no triangles in {path}")
            return 1
        m.remove_duplicated_vertices()
        m.remove_degenerate_triangles()
        meshes.append(m)
        lab = args.labels[i] if args.labels and i < len(args.labels) else path.stem
        labels.append(f"{lab}\n{len(m.triangles):,} tri")
        s = o3d.t.geometry.RaycastingScene()
        s.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(m))
        scenes.append(s)
        print(f"{path.name}: {len(m.triangles):,} triangles")

    # Frame every row on the FIRST mesh's box so the rows share one camera path -- two
    # meshes of the same object otherwise get silently different scales, which would make
    # a surface-quality comparison meaningless.
    v = np.asarray(meshes[0].vertices)
    centre = 0.5 * (v.min(0) + v.max(0))
    radius = float(np.linalg.norm(v.max(0) - v.min(0)) / 2.0)

    # The normalized frame is NOT necessarily Z-up: normalize_scene only rotates when the
    # requested target up differs from the estimated one, so a capture whose gravity axis
    # was already consistent keeps an arbitrary up vector. Reading it from the transform
    # is what makes the orbit ring level instead of tumbling.
    if args.up is not None:
        up = np.asarray(args.up, dtype=float)
    else:
        up = np.array([0.0, 0.0, 1.0])
        tf = args.mesh[0].parent / "normalize_transform.json"
        if tf.exists():
            import json
            axis = json.load(open(tf)).get("up_axis_target")
            if axis:
                up = np.asarray(axis, dtype=float)
                print(f"up axis from {tf.name}: {np.round(up, 3)}")
    up = up / np.linalg.norm(up)
    # Build an orbit basis from the up axis rather than assuming the world axes are the
    # object's: azimuth must sweep the plane perpendicular to up.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, up)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = ref - np.dot(ref, up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)

    el = np.radians(args.elevation)
    dist = radius * args.zoom
    eyes = []
    for i in range(args.views):
        az = 2 * np.pi * i / args.views
        eyes.append(centre + dist * (np.cos(az) * np.cos(el) * e1
                                     + np.sin(az) * np.cos(el) * e2
                                     + np.sin(el) * up))

    rows = [[render(s, eye, centre, up, args.size, args.size, args.fov) for eye in eyes]
            for s in scenes]
    print(f"rendered {len(rows)} x {len(eyes)} views")

    nrow, ncol = len(rows), len(eyes)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.4 * nrow),
                             squeeze=False)
    for r, (row, label) in enumerate(zip(rows, labels)):
        for c, im in enumerate(row):
            ax = axes[r][c]
            ax.imshow(im)
            if r == 0:
                ax.set_title(f"{c * 360 // ncol}°", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
