"""Per-view masks by projecting a 3D object box, for isolating a subject in training.

This exists to test a specific hypothesis. The Casque orbit wants a Gaussian budget 16x
larger than the Pavillon (6 M vs 375 k), and the proposed explanation is that what is being
sized is not *the object* but *the amount of observed scene content*: the Casque model must
explain a whole auditorium seen across 310 degrees of azimuth, of which the helmet is a
small part. That predicts something falsifiable -- restrict the photometric loss to the
helmet and the optimum should collapse.

Why a projected box rather than a segmentation model. ``rembg`` was already evaluated on
this capture and rejected: its salient-object mask swings between 0.1 % and 45 % of the
frame across the orbit, defeated by chrome reflections, a wispy plume and a competing
stand. A mask that unreliable would confound the very measurement we want. Projecting a
box we have already validated in 3D is deterministic, has no failure mode that varies with
viewpoint, and its errors are geometric and inspectable.

Note this masks only the *training loss*. SfM keeps using the full frames, which is
essential here: the checkerboard carries nearly all the reliable features, while the chrome
helmet has almost none, so masked SfM would wreck the poses (see docs/reproduce_casque.md).

Usage:
  python scripts/make_box_masks.py casque_orbit_07ccd886 \
      --center -0.183 0.166 0.220 --half-extent 0.24 --dilate 24
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]


def box_corners(centre: np.ndarray, half: float) -> np.ndarray:
    o = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                 dtype=np.float64)
    return centre[None, :] + o * half


def project(corners: np.ndarray, viewmat: np.ndarray, K: np.ndarray):
    """Project the 8 corners; returns pixel coords of those in front of the camera."""
    cam = (viewmat[:3, :3] @ corners.T).T + viewmat[:3, 3]
    front = cam[:, 2] > 1e-6
    if not front.any():
        return None
    cam = cam[front]
    uv = (K @ cam.T).T
    return uv[:, :2] / uv[:, 2:3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_id")
    ap.add_argument("--center", type=float, nargs=3)
    ap.add_argument("--half-extent", type=float)
    ap.add_argument("--mesh", type=pathlib.Path,
                    help="cropped object mesh; its projected vertices give a far tighter "
                         "silhouette than a box hull (a cube in perspective projects to a "
                         "hexagon covering most of the frame). Strongly preferred.")
    ap.add_argument("--splat", type=int, default=7,
                    help="mesh mode: radius in px drawn per projected vertex, then closed; "
                         "large enough to bridge the gaps between samples")
    ap.add_argument("--dilate", type=int, default=16,
                    help="pixels of slack around the projected hull; the box is a coarse "
                         "bound and a tight mask would clip the plume")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--preview", type=pathlib.Path, default=None)
    args = ap.parse_args()

    import cv2

    from video_to_3dgs.core.paths import RunLayout
    from video_to_3dgs.training.dataset import ColmapDataset

    layout = RunLayout(REPO / "experiments" / "runs", args.dataset_id)
    out_dir = args.out_dir or layout.masks_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mesh is not None:
        import open3d as o3d
        m = o3d.io.read_triangle_mesh(str(args.mesh))
        pts = np.asarray(m.vertices)
        if len(pts) == 0:
            print(f"no vertices in {args.mesh}")
            return 1
        # A TSDF mesh carries isolated speckles of floating geometry. Projected, each one
        # paints a stray blob of "object" onto background pixels, so strip them first --
        # a mask with scattered false positives would supervise background as subject.
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        clean, keep = pcd.remove_statistical_outlier(nb_neighbors=24, std_ratio=2.0)
        removed = len(pts) - len(keep)
        pts = np.asarray(clean.points)
        print(f"outlier removal dropped {removed:,} of {removed + len(pts):,} vertices")
        # Subsample: the silhouette is set by the outline, and 100k+ projections per view
        # across 134 views is needlessly slow for a mask that gets closed anyway.
        if len(pts) > 40000:
            step = len(pts) // 40000 + 1
            pts = pts[::step]
        print(f"mesh silhouette from {len(pts):,} vertices")
        corners = pts
    else:
        if args.center is None or args.half_extent is None:
            print("need either --mesh or (--center and --half-extent)")
            return 2
        corners = box_corners(np.asarray(args.center, dtype=np.float64), args.half_extent)

    seen: set[str] = set()
    fracs: list[tuple[float, str]] = []
    previews: list[np.ndarray] = []
    for split in ("train", "val", "test"):
        ds = ColmapDataset(layout, split, cache_images=False)
        for s in ds.samples:
            if s.name in seen:
                continue
            seen.add(s.name)
            uv = project(corners, s.viewmat, s.K)
            mask = np.zeros((s.height, s.width), dtype=np.uint8)
            if uv is not None and len(uv) >= 3:
                if args.mesh is not None:
                    # Splat each projected vertex, then close: this follows the object's
                    # actual outline, including concavities a convex hull would fill in.
                    ij = np.round(uv).astype(np.int64)
                    ok = ((ij[:, 0] >= 0) & (ij[:, 0] < s.width)
                          & (ij[:, 1] >= 0) & (ij[:, 1] < s.height))
                    mask[ij[ok, 1], ij[ok, 0]] = 255
                    r = max(args.splat, 1)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
                    mask = cv2.dilate(mask, k)
                    # Fill interior holes left between sparse samples.
                    ff = mask.copy()
                    cv2.floodFill(ff, np.zeros((s.height + 2, s.width + 2), np.uint8),
                                  (0, 0), 255)
                    mask = mask | cv2.bitwise_not(ff)
                else:
                    hull = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
                    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
                if args.dilate > 0:
                    k = np.ones((args.dilate * 2 + 1,) * 2, np.uint8)
                    mask = cv2.dilate(mask, k)
            frac = float((mask > 0).mean())
            fracs.append((frac, s.name))
            cv2.imwrite(str(out_dir / (pathlib.Path(s.name).stem + ".png")), mask)
            if args.preview is not None and len(previews) < 4 and frac > 0:
                img = cv2.imread(str(s.image_path))
                if img is not None:
                    img = cv2.resize(img, (s.width, s.height))
                    tint = img.copy()
                    tint[mask > 0] = (0.55 * tint[mask > 0]
                                      + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
                    previews.append(cv2.resize(tint, (s.width // 4, s.height // 4)))

    f = np.array([x[0] for x in fracs])
    print(f"wrote {len(fracs)} masks -> {out_dir}")
    print(f"coverage: min {f.min():.1%}  p50 {np.percentile(f, 50):.1%}  "
          f"max {f.max():.1%}  empty {(f == 0).sum()}")
    # A mask that varies wildly across the orbit is the failure mode that disqualified
    # rembg here, so report the spread rather than only the mean.
    if f.max() > 0:
        print(f"spread (max/p50): {f.max() / max(np.percentile(f, 50), 1e-9):.2f}x")
    worst = sorted(fracs)[:3]
    print("smallest coverage:", ", ".join(f"{n} {v:.1%}" for v, n in worst))

    if previews:
        h = min(p.shape[0] for p in previews)
        strip = np.hstack([p[:h] for p in previews])
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), strip)
        print(f"preview -> {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
