# Technical decisions

## Backend: direct gsplat (not Nerfstudio splatfacto)
Blackwell `sm_120` + `tiny-cuda-nn` is a known pain point; Nerfstudio drags it in.
Direct gsplat (which splatfacto uses under the hood anyway) gives clean Blackwell
support and full control over checkpointing/health/preemption/metrics. Built on
the *packaged* `gsplat.rasterization` + `DefaultStrategy` — NOT `examples/
simple_trainer.py` (not installed, no API stability). Splatfacto/orig-3dgs remain
future adapters behind the `TrainingBackend` ABC.

## Environment: fresh `v2gs` conda env on /home, built by local-disk mamba
Isolated from the user's existing gs_jepa env. Built with the fast local-disk
mamba (login-node NFS conda is a storm) but installed into an NFS prefix so the
healthy-NFS GPU nodes can import it. torch cu128 for sm_120; in-env `cuda-nvcc
12.8` so gsplat JIT-builds for Blackwell.

## SfM: COLMAP via CLI on node-local scratch; self-contained model readers
COLMAP's SQLite DB is hostile to the NFS mount → all COLMAP work runs on
`/var/tmp/$USER` and verified outputs are synced back. Shelling to the `colmap`
CLI works with any install. `.bin` model readers are self-contained (`colmap_io.py`)
to avoid a pycolmap wheel dependency. **CPU SIFT by default** (GPU SIFT on sm_120
is unverified); GPU behind a flag with an exhaustive-matcher fallback.

## Orchestration: filesystem as source of truth (no daemon/DB)
Per-stage status JSON is the completion marker; the runner writes RUNNING before
work and COMPLETED only after output validation, so a crash never yields a false
COMPLETED. Fingerprint = sha256(params + input checksums) drives skip/rerun and
downstream invalidation. Matches a single-user cluster tool and survives reboots.

## Primary target: Pavillon scene (per user)
Masks default OFF for the scene config; point-based normalization. The object and
turntable configs keep masking fully enabled/mandatory.

## Config: pydantic v2, frozen, layered YAML
defaults → cluster profile → user YAML → `--set` overrides → frozen
`config_resolved.yaml`. All models frozen for immutability (stable fingerprints).

## Scratch strategy: sync-back only after verification
`ScratchContext` deletes node-local scratch only after every output is sha256-
verified and atomically promoted; on failure it retains scratch (logged) for
debugging.

## Multi-GPU: parallel experiments, not distributed single-run
Object-scale scenes train on one GPU; sweeps fan out via Slurm job arrays sharing
one COLMAP build.

## Two trainer bugs that caused blurry reconstructions (fixed)
Early full-scale runs produced soft/foggy renders (test PSNR ~14, LPIPS ~0.82)
even though the pipeline ran end-to-end. Root causes, both in the gsplat trainer:
1. **scene_scale from points.max()** — COLMAP outlier points inflated it ~10x
   (5-7 vs a true camera radius ~0.5). That scaled the means LR ~10x too high
   (positions jitter -> blur) AND inflated gsplat's grow_scale3d normalization so
   Gaussians were clone-only, never split (never shrink -> blur + runaway growth,
   PSNR fell as count grew). Fix: `ColmapDataset.scene_extent()` now uses the max
   **camera-center** distance * 1.1 (the 3DGS spatial_lr_scale convention).
2. **Missing position-LR decay** — added the standard ExponentialLR ~100x decay on
   the means LR so centers settle to sharp detail.
After both fixes, the single-orbit Pavillon reconstruction reaches test **PSNR
23.9 / SSIM 0.82 / LPIPS 0.245** with sharp renders (verified on the qualitative
figure). Diagnosed *using* the reporting subsystem's qualitative/stats figures.

## Data characterization + SOTA next steps (Pavillon carved panel)
The capture is a single-side, near-planar, low-overlap close-up of a bas-relief
carved wooden panel (ceiling stood vertical). Vanilla 3DGS now works (~24 PSNR
in-cone) but is fundamentally limited by the narrow cone / sparse overlap.
Adapted techniques to consider (framework has backend/SfM extension points):
robust SfM for low overlap (GLOMAP, MASt3R/DUSt3R/VGGT); surface-first
reconstruction for the relief (2DGS, SuGaR -> textured mesh); sparse-view
depth/normal priors (DepthAnything-v2/Marigold; FSGS/SparseGS/DNGaussian).

## SOTA A2 — 2DGS surface backend (implemented; underperforms on THIS capture)
The `2dgs` backend (gsplat.rasterization_2dgs + normal/distortion regularizers +
gradient_2dgs densification + Open3D TSDF mesh export) is implemented, API-validated
on GPU, and wired into the pipeline/export. BUT on the single-side, near-planar,
low-overlap carved-panel capture it produces flat, featureless renders (val render
inspected directly: uniform beige blocks, no relief) at PSNR ~13. Causes: 2DGS
needs good multi-view surface coverage this capture lacks; gradient_2dgs densifies
far too slowly here (~0.28M vs ~1.5M for 3DGS); and it likely needs 2DGS-specific
disk initialization (oriented normals) rather than the shared 3DGS init. Conclusion:
for this object, vanilla 3DGS (test PSNR ~24, sharp) is the deliverable; 2DGS is
kept as an experimental backend for better-covered captures and would need A3's
depth/normal priors to supply the missing surface signal.

## Monitoring lesson (from the user)
Judge a run early by LOOKING at the first/last val renders (saved under
trainings/<id>/renders/val_*/), not only the PSNR number — a glance at the image
reveals a dead-end (flat/blurry/floaters) in seconds. Applied going forward.

## SOTA A3 — geometry regularizers for the in-room, floater-prone capture
Three regularizers were added to the gsplat 3DGS trainer (config: `train.bounds`,
`train.floater`, `train.depth_prior`; ships in `pavillon_orbit_reg.yaml`), each
aimed at a failure mode of this single-side, low-overlap, in-a-room capture:
1. **Room bounds** — an AABB of (camera centers ∪ COLMAP points) expanded by
   `margin` (0.5). Gaussians drifting outside are opacity-suppressed each 500
   steps → pruned by the DefaultStrategy's opacity prune. (SOTA analog:
   scene-bound + visibility pruning.)
2. **Anti-floater** — Gaussians with scale > `max_scale_frac`·scene_extent (0.15)
   are opacity-suppressed during training; on clean completion a final HARD prune
   removes everything below `min_opacity` (0.02) or outside the room box, for a
   floater-free deliverable `.ply`. (SOTA analog: DNGaussian/FSGS floater removal.)
3. **Depth prior** — DepthAnything-v2 monocular depth (cached at `run_dir/
   depth_prior/`, HF model cached on NFS → GPU node runs offline) + a
   scale/shift-invariant **Pearson** depth loss on the rendered expected-depth
   (`RGB+ED`), ramped in at iter 2000, weight 0.1. Injects the depth *ordering*
   the sparse views lack. (SOTA analog: SparseGS/FSGS depth regularization.)

**Result (reg run vs the same-COLMAP baseline — a controlled comparison):**
final hard prune removed **20.2% of Gaussians (1.52M → 1.22M)**, `.ply` 374→**301
MB**, renders stay sharp (relief preserved). Test PSNR **22.3 / SSIM 0.80 / LPIPS
0.29** vs baseline 23.9 / 0.82 / 0.245; best **val** PSNR 24.5 (on par). The
~1.5 dB test drop is the **depth prior** trading photometric fit for geometric
consistency — the bounds+floater removal itself is nearly free. Net: the user's
two asks (keep Gaussians in the room; kill flying floaters) are delivered with a
cleaner, smaller model at a modest photometric cost. A **bounds+floater-only
(depth off)** variant is the lever to recover PSNR while keeping the floater
cleanup, if raw PSNR is preferred over geometric regularization.

**Where the 20% actually came from (CPU analysis of both `.ply`s):** the room box
barely triggered — only ~0.01% of *baseline* Gaussians sit outside the (generous)
AABB, so the box is a safety net, not the active lever here. The floaters on this
capture are *in-volume haze*, and the **anti-floater opacity/scale prune** removed
them: baseline had **18.2% of Gaussians below 0.02 opacity** (near-transparent) and
scales up to **1.0** (scene-spanning); the reg model has **0%** sub-0.02 haze, max
scale capped at **0.16**, and median opacity **0.51 → 0.88** — solid Gaussians, not
haze. Visually (scripts/floater_spatial.py) the baseline's diffuse streak toward the
cameras collapses to a tight, clean cluster. Takeaway: for room captures like this,
the scale+opacity floater prune does the work; the room AABB matters more for
captures where Gaussians actually escape to infinity.

Env note: `transformers` (depth pipeline) was pip-installed into `v2gs` on top of
the lockfile; add it to the env provisioning if the depth prior becomes standard.
GPU nodes: only the Blackwell nodes (GPURACK4/5) run this cu128 env — the 3090/4090
nodes have too-old a driver (see progress.md).
