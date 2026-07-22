"""Object-region metrics: score the panel, not the room.

Our held-out metrics are computed over the whole frame, but the frame is mostly
background --- brick wall, floor, a doorway --- which dilutes the signal from the
carved relief we actually care about. Flat brick scores high and inflates PSNR; the
far doorway scores low and deflates it; neither has anything to do with
reconstruction of the panel. With only ~28--33 test views this dilution is why so
many of our interventions land inside the +/-0.45 dB paired confidence interval and
read as ties.

This recomputes PSNR/SSIM/LPIPS over an *object region* defined purely from
geometry --- no segmentation model, fully reproducible: the axis-aligned bounding
box of the robust (2--98 percentile) SfM point cloud is projected into each view and
its 2D extent is the mask. That excludes the far background that causes the dilution
while including the panel and its immediate surround. It reuses the renders the
evaluate stage already saved, so no GPU and no retraining are needed.

Usage:
    python scripts/object_metrics.py <dataset_id> <train_run_id> [<train_run_id> ...]
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from video_to_3dgs.core.paths import RunLayout            # noqa: E402
from video_to_3dgs.training.dataset import ColmapDataset  # noqa: E402


def object_box(points: np.ndarray, margin: float = 0.05):
    lo, hi = np.percentile(points, 2, axis=0), np.percentile(points, 98, axis=0)
    pad = margin * (hi - lo)
    lo, hi = lo - pad, hi + pad
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return corners


def project_mask(corners, viewmat, K, w, h):
    """2D bounding rectangle of the object box in this view (bool HxW)."""
    cam = corners @ viewmat[:3, :3].T + viewmat[:3, 3]
    cam = cam[cam[:, 2] > 1e-3]
    if len(cam) < 2:
        return np.ones((h, w), bool)
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0 = np.clip(uv.min(0), [0, 0], [w, h])
    x1, y1 = np.clip(uv.max(0), [0, 0], [w, h])
    m = np.zeros((h, w), bool)
    m[int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))] = True
    return m


def _psnr(a, b, m):
    d = ((a - b) ** 2)[m]
    return 10 * np.log10(1.0 / max(d.mean(), 1e-12))


def _ssim_masked(a, b, m):
    """SSIM over the masked bounding rectangle (crop to the mask's extent)."""
    from video_to_3dgs.training.losses import ssim
    import torch
    ys, xs = np.where(m)
    if len(ys) < 64:
        return float("nan")
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    ta = torch.tensor(a[sl]).permute(2, 0, 1)
    tb = torch.tensor(b[sl]).permute(2, 0, 1)
    return float(ssim(ta, tb))


def main():
    dsid = sys.argv[1]
    run_ids = sys.argv[2:]
    layout = RunLayout(runs_root=REPO / "experiments" / "runs", dataset_id=dsid)
    ds = ColmapDataset(layout, "test", use_masks=False)
    corners = object_box(ds.points)
    by_name = {s.name: i for i, s in enumerate(ds.samples)}

    try:
        import lpips as _lp
        import torch
        lpips_net = _lp.LPIPS(net="alex").eval()
    except Exception:
        lpips_net = None

    for rid in run_ids:
        rdir = layout.renders_dir(rid) / "eval_test"
        if not rdir.exists():
            print(f"{rid}: no eval renders at {rdir}"); continue
        full, obj, sob, lps = [], [], [], []
        for f in sorted(rdir.glob("*.png")):
            name = next((n for n in by_name if Path(n).stem == f.stem), None)
            if name is None:
                continue
            i = by_name[name]
            gt, _ = ds.load_image(i)
            gt = gt.cpu().numpy()
            r = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(f)
                           .convert("RGB"), np.float32) / 255.0
            h, w = gt.shape[:2]
            if r.shape[:2] != (h, w):
                r = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(f)
                               .convert("RGB").resize((w, h)), np.float32) / 255.0
            vm, K, _, _ = ds.samples[i].viewmat, ds.samples[i].K, w, h
            m = project_mask(corners, vm, K, w, h)
            full.append(_psnr(gt, r, np.ones((h, w), bool)))
            obj.append(_psnr(gt, r, m))
            sob.append(_ssim_masked(gt, r, m))
            if lpips_net is not None:
                import torch
                ys, xs = np.where(m)
                sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                ta = torch.tensor(gt[sl] * 2 - 1).permute(2, 0, 1)[None].float()
                tb = torch.tensor(r[sl] * 2 - 1).permute(2, 0, 1)[None].float()
                with torch.no_grad():
                    lps.append(float(lpips_net(ta, tb).item()))
        full, obj, sob = np.array(full), np.array(obj), np.array(sob)
        frac = float(np.mean([project_mask(corners, ds.samples[by_name[n]].viewmat,
                                           ds.samples[by_name[n]].K, 100, 100).mean()
                              for n in list(by_name)[:1]]))
        print(f"{rid:28s}  full PSNR {full.mean():5.2f}  |  OBJECT PSNR {obj.mean():5.2f} "
              f"SSIM {np.nanmean(sob):.4f}"
              + (f" LPIPS {np.mean(lps):.4f}" if lps else "")
              + f"  (mask~{100*frac:.0f}% of frame)")


if __name__ == "__main__":
    main()
