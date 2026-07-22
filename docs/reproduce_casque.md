# Reconstructing the Casque Saint-Georges

An object-centric orbit capture of a helmet (Saint-Georges), shot with a
**professional camera *and* an iPhone**. It is deliberately contrasted with the
Pavillon here, because the two captures sit at opposite ends of almost every axis and
therefore call for opposite settings.

## Why this capture differs from the Pavillon

| Axis | Pavillon (scene) | Casque (object) | Consequence |
|---|---|---|---|
| Coverage | single-sided, low parallax | **full orbit**, high parallax | more Gaussian capacity is fine — do **not** copy 375k |
| Subject vs scene | relief carved *into* a wall | **free-standing** helmet | use **masks** to isolate the object |
| Cameras | one iPhone clip | **pro camera + iPhone** | **appearance embeddings** reconcile the two responses |
| Surface method | 2DGS failed (no multi-view normals) | multi-view surface | **2DGS should work** — try it for the mesh |
| Interest | the whole panel | the **helmet only** | mask-supervised training, 360° orbit video |

The two things carried over from the Pavillon: **normal-consistency** (a cleaner mesh)
and the **anti-floater** prune. Everything else is inverted.

## Configs

- `configs/pipeline/casque/casque.yaml` — recommended 3DGS reconstruction.
- `configs/pipeline/casque/casque_2dgs.yaml` — surface mesh via 2DGS (the capture 2DGS
  was built for).

Both enable masks, per-image appearance embeddings, GLOMAP, and `single_camera: false`
(the pro camera and iPhone have different intrinsics).

## Steps

1. **Verify the footage.** The folder holds pro clips (`102A25xx.MOV`), iPhone clips
   (`IMG_8xxx.MOV`) and stills. Curate `videos:` down to the clean full orbits; short
   establishing clips can hurt. Check each with:
   ```bash
   python -m video_to_3dgs.cli inspect-video --config configs/pipeline/casque/casque.yaml
   ```

2. **Check the masks before a long run.** The helmet must be the salient object for
   `rembg`; if the background competes, switch `generate_masks.backend` to `sam2`. The
   config sets `manual_review: true` so masks are written for inspection first.

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
   Pick the best on the held-out split. As a starting point, `cap_max ≈ 2× the sparse
   point count` has held on our data — verify, do not assume.

5. **Mesh.** Run `casque_2dgs.yaml` for a surface-aligned mesh; also export the 3DGS
   TSDF mesh (`export` emits `mesh.ply` automatically) and compare. Unlike the
   Pavillon, 2DGS has the multi-view coverage it needs here.

## What to watch for

- **Appearance drift** (logged): if the pro/iPhone response differs a lot, the affine
  latents will show non-zero drift and are doing real work. If a *spatially varying*
  difference appears (vignetting), switch `appearance_model` to `bilateral`.
- **Masks are the main risk.** A bad mask removes part of the helmet or leaks
  background into training. Review them (step 2) before committing GPU time.
- **An attractive object box** (pulling stray Gaussians toward the masked helmet) is a
  sensible addition *here* — the helmet is cleanly separable, unlike the Pavillon
  relief embedded in its wall. Not yet implemented; a natural next feature for masked
  object captures.
