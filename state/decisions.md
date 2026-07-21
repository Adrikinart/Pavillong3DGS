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
0.29** vs baseline 23.9 / 0.82 / 0.245; best **val** PSNR 24.5 (on par).

**Ablation settles what costs the PSNR (this corrects an earlier wrong guess).**
A depth-OFF run (`gsplat_reg_nodepth`, bounds+floater only, identical otherwise)
gives test **22.55 / 0.798 / 0.294**, prunes 18.0%, best val 24.45. So turning the
depth prior off recovers only **0.26 dB** — it is *not* the source of the ~1.4 dB
gap to baseline. The cost is the **final hard prune** itself: `min_opacity: 0.02`
removes faint Gaussians that still contribute to held-out views. Both regularized
runs land at ~22.3–22.6 regardless of the depth prior.

Practical consequence: the PSNR/cleanliness trade-off is tuned via
`floater.min_opacity`, not by disabling the depth prior. Lower it (0.02 → 0.005,
the densifier's own prune threshold) to keep most of the floater cleanup at a
smaller photometric cost; raise it for a leaner, cleaner `.ply`. The depth prior
is nearly free in PSNR terms and slightly *increases* floater removal (20.2% vs
18.0%), so it earns its place.

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

## Appearance embeddings work — but merging IMG_9649 does not help (negative result)
`train.appearance_embedding` is implemented (per-image GLO latent -> global affine
colour transform, identity-initialised, provably unable to encode geometry since it
commutes with any pixel permutation). The mechanism validates: SfM merged the two
clips at **332/332 registered, 0.921px reproj, 170 232 points, 266 train views**,
and `appearance_drift` settled at **0.05-0.066** (non-zero => the clips really do
differ photometrically and the latents are absorbing it).

**A measurement bug made this look far worse than it was (corrected).** The
`evaluate` stage rebuilt the Gaussians from the checkpoint and rendered *raw* — it
never restored the appearance model. So held-out views were scored on the
UNCORRECTED render while training had optimised corrected ones: a systematic
photometric penalty unrelated to reconstruction quality. Fixed by restoring the
latents in `evaluate.py` and scoring each view under the mean appearance of **its
own source clip** (`AppearanceModel.canonical_for`), using clip identity + that
clip's TRAINING images only — the held-out pixels are never used, unlike NeRF-W
which fits a latent on half the test image.

| multi-clip test | PSNR | SSIM | LPIPS |
|---|---|---|---|
| before (uncorrected — the bug) | 19.02 | 0.824 | 0.362 |
| **after (per-clip appearance)** | **22.03** | **0.833** | **0.342** |

That recovered **3.02 dB**, shrinking the gap to `gsplat_hidetail_30k` (23.08) from
4.06 dB to **1.05 dB**. An earlier entry here claimed "~3.7 dB is real geometric
degradation", derived from fitting a per-channel gain+bias to four validation
renders. **That was an overestimate and is retracted**: the real model applies a
full 3x3 cross-channel matrix over 33 test views, so it corrects more than that
cruder proxy could. The lesson is that a hand-rolled diagnostic is itself a model
and can be wrong — the direct measurement settled it.

**A real but modest regression remains**: median PSNR 24.48 -> 22.82, catastrophic
views (<18 dB) 11% -> **24%**, SSIM 0.846 -> 0.833. The per-clip breakdown localises
it: the ADDED clip's own test views score fine (img_9649: mean 23.68) while the
ORIGINAL clip's views degrade (img_9647: 21.81). Merging hurt the region that
previously had the entire Gaussian budget to itself. The clips are co-located
(camera centroids 0.39 apart vs a 0.46 cloud radius) but IMG_9649 adds only 48 train
views while enlarging the reconstructed volume (point extent [2.53, 9.61, 8.7]), so
1.5M Gaussians now cover more scene.

Conclusion: keep `gsplat_hidetail_30k` as the deliverable.

### 3DGS-MCMC densification: implemented, and at parity with the heuristic
`densification.strategy: mcmc` was a declared-but-unread flag; gsplat 1.5.3 ships
`MCMCStrategy`, so this was implementation rather than research. MCMC treats
primitives as samples, injects SGLD noise, and replaces clone/split heuristics with
relocation of dead primitives onto live ones; its `cap_max` is a target the sampler
fills rather than a ceiling growth happens to hit.

Run at our measured-optimal budget (375k), changing only the strategy:

| | Gaussians | PSNR | median | SSIM | LPIPS | <18 dB |
|---|---:|---:|---:|---:|---:|---:|
| ADC (default) | 358,878 | **24.92** | 25.13 | 0.8619 | 0.3132 | 0% |
| MCMC | 365,497 | 24.83 | **25.32** | **0.8623** | **0.3094** | 0% |

Paired per-view: MCMC − ADC = **−0.09 dB ± 0.45 (95% CI)** — *indistinguishable*.
MCMC is better on **17/28** views and on median, SSIM and LPIPS; ADC wins the mean
only because of a single −5.32 dB outlier view. With 28 held-out views we cannot
resolve differences below ~0.45 dB, so no quality claim is warranted either way.

**This is informative rather than disappointing.** It says the large gain we
measured came from the *budget itself*, not from how the budget is allocated —
consistent with the sparse-view overfitting explanation. MCMC is nonetheless the
better default going forward on operational grounds: it removes `grad_threshold`
(the main heuristic knob) and makes the budget explicit, which matters precisely
because the budget turned out to be the critical hyper-parameter. API note: MCMC
takes no `scene_scale` at init and needs the current LR at step time instead of
`packed`; `tests/unit/test_strategy_api.py` pins that on CPU since the mismatch
otherwise only appears minutes into a GPU run.

### Pose refinement does not help here (negative result)
`train.pose_optimization` is implemented: a learnable SE(3) delta per TRAINING
camera (BARF-style), identity-initialised, with held-out poses deliberately NOT
refined — optimising a test camera's pose fits the evaluation image and inflates the
metric. Measured on the best model (cap 375k), it is harmful:

| | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| cap375k baseline | **24.92** | **0.8619** | **0.3132** |
| + pose refinement | 23.94 | 0.8290 | 0.3301 |

The deltas barely moved — mean **0.036 deg** rotation and **0.0007** translation,
roughly one pixel at this resolution — yet cost ~1 dB. That is the failure mode
predicted when it was implemented: the refinement re-aligned training cameras onto
their own images, while held-out cameras keep their original poses, so the scene
drifts slightly relative to them. GLOMAP's poses were already **0.92 px** mean
reprojection error, i.e. there was no calibration error to recover. Pose refinement
is a remedy for bad poses; ours are not bad. Keep `pose_optimization: false` unless
SfM reprojection error is poor.

### Gaussian capacity is a REGULARIZER — cutting it 4x is the largest late-stage win
Having falsified dilution (below), we tested the reverse on the deliverable dataset.
Same data, same regularizers, same 30k iterations; only `cap_max` changes:

| cap_max | Gaussians | PSNR | median | SSIM | LPIPS | <18 dB |
|---|---:|---:|---:|---:|---:|---:|
| 1.5M | 1,244,835 | 23.08 | 24.48 | 0.8461 | 0.3096 | 11% |
| 750k |   653,831 | 24.32 | 25.10 | 0.8596 | **0.3025** | 11% |
| **375k** | **358,878** | **24.92** | 25.13 | **0.8619** | 0.3132 | **0%** |
| 190k |   192,475 | 24.86 | **25.18** | 0.8559 | 0.3382 | **0%** |

Cutting the budget 4x gained **1.8 dB PSNR**, improved SSIM, **eliminated the
catastrophic-view tail (11% -> 0%)** and shrank the model 3.5x (295 MB -> ~90 MB).
The same direction holds on multi-clip: 750k scores 22.37 (its best) vs 22.03 at
1.5M and 21.36 at 3.0M.

This is sparse-view overfitting. Depth is constrained only by disagreement between
views; a single-sided capture supplies little, so every surplus Gaussian is a
parameter free to sit at a wrong depth while still reproducing the training images.
Capacity buys solutions that fit training views and fail on held-out ones.

The curve turns rather than running away: at 190k PSNR and SSIM both fall back, so
375k is a genuine optimum and not just the smallest point tested. **LPIPS bottoms at
750k** and worsens monotonically as capacity shrinks — perceptual detail wants
capacity while accuracy/generalisation want less. For perceptual fidelity
of the carving, 750k is the better operating point; for accuracy and robustness,
375k. Recommended deliverable is now `gsplat_hidetail_cap375k`.

Method note: this was found by being wrong first. The dilution hypothesis predicted
*more* capacity would help, the experiment refuted it, and testing the opposite
direction produced the biggest gain since the resolution fix. The falsification was
worth more than the original hypothesis.

### Relief mesh via TSDF of 3DGS depth (works)
`GsplatBackend.export_mesh` fuses the model's composited depth (`RGB+ED`) over the
training cameras into a TSDF volume: 1.39M verts / 2.62M tris from 226/226 cameras,
with the panel's planks and carved cross-members clearly resolved. Alpha-masking is
essential (where little opacity accumulates the depth is not noisy but meaningless)
and the voxel size is derived from the robust point extent since the scene is in
normalized units. Caveats: volumetric primitives give a biased expected depth, so the
mesh is softer than a surface-aligned method would give, and the fusion includes
surrounding wall/floor. Also fixed: `export.py` selected CUDA only for the 2DGS
backend, so the gsplat mesh path would have been handed a CPU device.

### The budget-dilution hypothesis is FALSIFIED (cap_max experiment)
The reasoning above suggested the residual ~1 dB was Gaussian-budget dilution over
the 2.04x larger merged volume, and predicted that scaling `cap_max` 1.5M -> 3.0M
would let the original clip's views recover. `gsplat_multiclip_cap3m` tested exactly
that, with every other setting identical. **The prediction was wrong, and in the
opposite direction:**

| multi-clip | Gaussians | test PSNR | SSIM | median | <18 dB | img_9647 | img_9649 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cap 1.5M | 1.23 M | **22.03** | **0.833** | 22.82 | 24 % | 21.81 | 23.68 |
| cap 3.0M | 2.50 M | 21.36 | 0.824 | 21.23 | **33 %** | **21.11** | **23.15** |

Doubling the budget made **both** clips worse and raised the catastrophic-view rate
from 24% to 33%. So the multi-clip deficit is not dilution — capacity was never the
binding constraint.

**Interpretation: this is sparse-view overfitting, and it is theoretically expected.**
The compositing model is underdetermined when views share little parallax — depth is
constrained only by *disagreement* between views, and a single-sided capture supplies
little. Every extra Gaussian is another parameter that can sit at a wrong depth while
still reproducing the training images, so added capacity enlarges the space of
solutions that fit training views without generalising. Held-out views punish that,
which is precisely what the numbers show.

**Guidance reversed:** for low-parallax captures, capacity should be *constrained*,
not expanded. `cap_max` is a regularizer here, not a quality dial. The untested
follow-up is the opposite experiment — `cap_max` *below* 1.5M — and the same caution
applies to the single-clip model. Do not raise `cap_max` when merging clips.

## Resolution/coverage beats regularizer tuning (high-detail run)
Per-view analysis of the A3 model showed every catastrophic held-out view
(12.7-18.8 dB) was an EXTREME CLOSE-UP at a grazing angle, while medium-distance
views scored 25.6-26.5 — a spatial-frequency deficit. The pipeline was also
downscaling a 3840x2160 source to 1600px and using ~181 of ~5900 frames.
`pavillon_orbit_hidetail.yaml` raises this to 2560px / 400 sampled frames.

Result: SfM improves outright — **282/282 registered (100%) and 152 792 sparse
points**, vs 181/193 (94%) and 80 913. On its (larger, higher-res, harder) test
split the model reaches median PSNR **24.48** and SSIM **0.846** vs the A3 model's
23.43 / 0.796, best view 31.9 vs 26.5, and catastrophic views drop 17% -> 11%.
Cross-resolution PSNR/LPIPS are NOT comparable, so judge this on distribution.

**Remaining limitation is coverage, not resolution.** The views that still fail are
near-featureless wood plank at grazing angles filling the frame: low texture AND
minimal parallax with any other view. More pixels cannot fix a patch only one
camera saw well. The next real lever is more viewpoints — i.e. appearance
embeddings so all three Pavillon clips can be merged (~3x coverage), and/or pose
optimization. Both are currently **dead schema flags** (`train.appearance_embedding`,
`train.pose_optimization`) that are declared but never implemented.

Env note: `transformers` (depth pipeline) was pip-installed into `v2gs` on top of
the lockfile; add it to the env provisioning if the depth prior becomes standard.
GPU nodes: only the Blackwell nodes (GPURACK4/5) run this cu128 env — the 3090/4090
nodes have too-old a driver (see progress.md).
