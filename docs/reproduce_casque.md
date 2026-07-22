# Reconstructing the Casque Saint-Georges

An object-centric orbit capture of a helmet (Saint-Georges), shot with a
**professional camera *and* an iPhone**. It is deliberately contrasted with the
Pavillon here, because the two captures sit at opposite ends of almost every axis and
therefore call for opposite settings.

<p align="center">
  <img src="assets/casque/poses.png" width="900" alt="Casque camera poses and sparse point cloud"><br>
  <em>The SfM solution: 134 registered cameras orbiting the helmet at two
  elevations, and the 36 k-point sparse cloud. Regenerate with
  <code>python scripts/make_demo_assets.py casque_orbit_07ccd886</code>.</em>
</p>

## Why this capture differs from the Pavillon

| Axis | Pavillon (scene) | Casque (object) | Consequence |
|---|---|---|---|
| Coverage | single-sided, low parallax | **full orbit**, high parallax | more capacity: **1.5 M**, not the Pavillon's 375k |
| Subject vs scene | relief carved *into* a wall | **free-standing** helmet | isolate the helmet **in 3D** (crop), not by mask — see below |
| Cameras | one iPhone clip | **pro camera + iPhone** | **appearance embeddings** reconcile the two responses |
| Surface method | 2DGS failed (no multi-view normals) | multi-view surface | **2DGS works** (with `dist_lambda: 0`) — use it for the mesh |
| Interest | the whole panel | the **helmet only** | spatial crop to the helmet; 360° orbit video |

The two things carried over from the Pavillon: **normal-consistency** (a cleaner mesh)
and the **anti-floater** prune. Everything else is inverted.

## Configs

- `configs/pipeline/casque/casque.yaml` — recommended 3DGS reconstruction.
- `configs/pipeline/casque/casque_2dgs.yaml` — surface mesh via 2DGS (the capture 2DGS
  was built for).

- `configs/pipeline/casque/casque_multiclip.yaml` — all clips merged, with per-image
  appearance embeddings and `single_camera: false` (the pro camera and iPhone have
  different intrinsics).

Both single-clip configs use GLOMAP, **masks off** (see step 2) and start from the
single 4K clip; the multi-clip config is the one that needs appearance embeddings.

## Steps

1. **Verify the footage.** The folder holds pro clips (`102A25xx.MOV`), iPhone clips
   (`IMG_8xxx.MOV`) and stills. Curate `videos:` down to the clean full orbits; short
   establishing clips can hurt. Check each with:
   ```bash
   python -m video_to_3dgs.cli inspect-video --config configs/pipeline/casque/casque.yaml
   ```

2. **Do NOT mask this capture** (a gate test settled it). rembg's salient-object model
   swings from 0.1% to 45% of the frame across the orbit — chrome reflections, a wispy
   horsehair plume and a competing stand defeat it. More importantly, the scene has a
   **checkerboard calibration target** (the best SfM features present) while the chrome
   helmet has almost none, so mask-only COLMAP would throw away the features that give
   good poses. Reconstruct the full scene and isolate the *free-standing* helmet
   spatially afterwards. (`rembg` needs `onnxruntime`; masking is the right tool only
   when the object is genuinely salient — a matte object on a plain background.)

3. **Run it** (env + node pre-flight identical to the Pavillon —
   see [reproduce_pavillon.md](reproduce_pavillon.md) §0–1):
   ```bash
   sbatch --partition=rtxpro --nodelist=GPURACK2 --gres=gpu:1 --cpus-per-task=8 \
          --export=ALL,MAX_JOBS=4,OMP_NUM_THREADS=4 \
          scripts/slurm/train.sbatch configs/pipeline/casque/casque.yaml
   ```

4. **Tune capacity for THIS object.** 375k was the Pavillon optimum *because* it was
   sparse-view; an orbit has real parallax, so the optimum is higher. Run the short
   sweep (reuses the SfM, so only training re-runs):
   ```bash
   for cap in 750000 1500000 3000000; do
     sbatch ... scripts/slurm/train.sbatch configs/pipeline/casque/casque.yaml \
            --force --from-stage train \
            --set train.densification.cap_max=$cap \
            --set train.train_run_id=casque_cap${cap}
   done
   ```
   Measured on the held-out split (13 test views):

   | `cap_max` | 750 k | **1.5 M** | 3 M |
   |---|---|---|---|
   | PSNR | 19.49 | **20.35** | 20.76 |
   | SSIM | 0.8585 | 0.8636 | 0.8653 |
   | LPIPS | 0.2892 | 0.2710 | 0.2612 |

   Read the *paired* per-view comparison, not the means: 750 k → 1.5 M is
   **+0.86 dB, CI [+0.34, +1.39]** (real), while 1.5 M → 3 M is **+0.41 dB,
   CI [−0.45, +1.26]** — a tie. So **1.5 M is the operating point**; 3 M doubles the
   model for a gain inside the error bar. Regenerate the curve and these statistics with:
   ```bash
   python scripts/capacity_curve.py --out docs/assets/capacity_curve.png
   ```
   The Pavillon's 375 k optimum does **not** transfer: copying it here costs over a
   decibel. The transferable rule is the method, not the number.

5. **Mesh.** Run `casque_2dgs.yaml` for a surface-aligned mesh; also export the 3DGS
   TSDF mesh (`export` emits `mesh.ply` automatically) and compare. Unlike the
   Pavillon, 2DGS has the multi-view coverage it needs here — it reaches **20.19 dB, a
   statistical tie with 3DGS** (−0.16 dB, CI [−1.26, +0.94]), which is the success
   condition for a surface method: same photometric quality, plus a real surface.

   **One setting decides this**, and it is already fixed in the config: `dist_lambda: 0`.
   With the 2DGS default of 1.0 the run peaks at 24.6 dB val and then collapses the
   instant the loss engages at iteration 7000, ending 4.40 dB lower.

   <p align="center">
     <img src="assets/casque/2dgs_distortion.png" width="900"
          alt="2DGS distortion-loss collapse and the recovered parity"><br>
     <em>Left: the collapse is caused, not gradual — it starts exactly at the
     regularizer's start iteration. Right: held-out test PSNR. Regenerate with
     <code>python scripts/dist_collapse_figure.py</code>.</em>
   </p>

## What to watch for

- **Appearance drift** (logged): if the pro/iPhone response differs a lot, the affine
  latents will show non-zero drift and are doing real work. If a *spatially varying*
  difference appears (vignetting), switch `appearance_model` to `bilateral`.
- **The object box is a tested negative here.** An explicit tight box
  (`train.bounds.box_center` / `box_half_extent`) around the helmet is implemented, and
  it *sounds* right for an object capture. It did not help: no sharper helmet, plus
  boundary smearing. The helmet is **data-limited, not budget-limited**, and with masks
  off the box fights the photometric loss over the background it still has to explain.
  Isolate the helmet by **cropping at export** instead. The box remains useful for
  genuinely mask-separable objects.
- **Capacity is the setting to tune per object**, and the Pavillon's answer does not
  carry over — see the sweep in step 4 and
  [technique_transfer.md](technique_transfer.md).
