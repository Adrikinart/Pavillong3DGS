"""Reporting: camera paths, even-crop, metrics aggregation (numpy/stdlib only)."""

import numpy as np

from video_to_3dgs.core.atomicio import atomic_write_json
from video_to_3dgs.core.paths import RunLayout


def test_orbit_path_geometry():
    from video_to_3dgs.reporting.cameras import orbit_path
    center = np.zeros(3)
    up = np.array([0.0, 0.0, 1.0])
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    cams = orbit_path(center, up, 1.0, K, 640, 480, n_frames=24, elevation_deg=20)
    assert len(cams) == 24
    for c in cams:
        R = c.viewmat[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)   # orthonormal
        # camera looks toward the center: forward (row 2) dot (center - eye) > 0
        eye = -R.T @ c.viewmat[:3, 3]
        fwd = R[2, :]
        assert float(fwd @ (center - eye)) > 0


def test_even_crop():
    from video_to_3dgs.reporting.videos import _even
    img = np.zeros((771, 1373, 3), dtype=np.uint8)
    out = _even(img)
    assert out.shape[0] % 2 == 0 and out.shape[1] % 2 == 0
    assert out.shape[:2] == (770, 1372)


def test_metrics_aggregate(tmp_path):
    from video_to_3dgs.reporting import metrics
    lay = RunLayout(runs_root=tmp_path, dataset_id="d1")
    tr = "gsplat_run"
    lay.ensure_base_dirs()
    atomic_write_json(lay.eval_json(tr), {
        "splits": {"test": {"n_views": 3, "psnr": 20.0, "ssim": 0.7, "lpips": 0.3,
                            "render_fps": 5.0,
                            "per_view": [{"name": "a", "psnr": 18, "ssim": 0.6, "lpips": 0.4},
                                         {"name": "b", "psnr": 20, "ssim": 0.7, "lpips": 0.3},
                                         {"name": "c", "psnr": 22, "ssim": 0.8, "lpips": 0.2}]}},
        "model": {"n_gaussians": 1000, "checkpoint_bytes": 2_000_000, "peak_vram_bytes": 3_000_000_000},
    })
    atomic_write_json(lay.colmap_dir / "sfm_stats.json",
                      {"n_input_images": 100, "n_registered_images": 90,
                       "registration_ratio": 0.9, "mean_reprojection_error": 0.5,
                       "n_points3D": 5000, "mean_track_length": 4.0})
    atomic_write_json(lay.training_dir(tr) / "train_result.json",
                      {"status": "COMPLETED", "n_gaussians": 1000,
                       "metrics": {"final_step": 30000, "best_val_psnr": 21.0}})
    s = metrics.aggregate(lay, tr)
    q = s["image_quality"]["test"]
    assert q["psnr"] == 20.0
    assert q["psnr_std"] is not None and q["psnr_min"] == 18 and q["psnr_max"] == 22
    assert s["reconstruction"]["n_registered_images"] == 90
    assert s["efficiency"]["n_gaussians"] == 1000
    assert s["training"]["best_val_psnr"] == 21.0
    # table + summary write cleanly
    js, md = metrics.write_summary_and_table(lay, tr)
    assert js.exists() and md.exists() and "Held-out image quality" in md.read_text()
