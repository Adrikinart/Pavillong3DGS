# Progress

Last updated: 2026-07-16.

## Status: end-to-end pipeline WORKING on Blackwell (smoke passes)

The full pipeline has run green on GPURACK5 (RTX PRO 6000, sm_120):
COLMAP → validate → normalize → pose-aware split → gsplat train → evaluate →
export → report, with checkpoint/resume verified.

## Completed
- [x] Repo + cluster audit (docs/cluster_environment.md).
- [x] Core framework: RunLayout, atomic IO, status invariants, StageRunner
      (skip/force/resume, stale-RUNNING recovery, fingerprints), manifest,
      provenance, node-local ScratchContext.
- [x] Config: pydantic v2 frozen schema, layered YAML, cluster profiles, 5 configs.
      `--force` re-freezes config so edits apply.
- [x] `inspect-env` with CUDA fwd/bwd + gsplat rasterize + sm_120 wheel check.
- [x] Stages: inspect_video, extract_frames (rotation-aware), filter_frames
      (scores + coverage safeguard + contact sheets under _diagnostics/),
      generate_masks (rembg), run_colmap (scratch + fallbacks + best-attempt
      selection + COLMAP 3.13 option auto-detect), validate_colmap (gates + viz),
      normalize_scene (similarity transform + inverse), split_dataset (pose-aware).
- [x] Training: gsplat backend (masked L1+SSIM, DefaultStrategy densify/prune,
      atomic checkpoints + integrity + resume, JSONL+TB metrics, health hard-fails,
      SIGTERM->requeue), deterministic train_run_id.
- [x] evaluate (held-out PSNR/SSIM/LPIPS + masked, FPS/VRAM, registry.csv),
      export (.ply + cameras + transform + COORDINATES.md), monitoring, HTML/MD report.
- [x] CLI, Slurm sbatch (preprocess/train/evaluate/sweep), shell scripts,
      scripts/_activate_env.sh (CUDA_HOME/CPATH so gsplat JIT-builds on sm_120).
- [x] Tests: 39 unit + integration PASS in the CPU env.
- [x] Env `v2gs` on /home: torch 2.13.0+cu130 (sm_120), gsplat 1.5.3 built on sm_120,
      colmap 3.13. Bootstrap installs the pip layer via srun on a GPU node.
- [x] Blackwell validation on GPURACK5: CUDA fwd/bwd pass, gsplat rasterize pass.
- [x] **End-to-end smoke PASSES** (configs/pipeline/smoke_test.yaml, IMG_9647):
      COLMAP 46/126 @ 0.48px -> train (val PSNR 19.6) -> eval (test PSNR 17.73,
      SSIM 0.61, LPIPS 0.76, 1.02 render FPS) -> ply export -> report.
      Marker: experiments/SMOKE_TEST_OK.
- [x] **Resume verified**: dropped the final checkpoint, re-ran train -> "resumed
      from ckpt_0001000.pt at step 1001" -> completed.
- [x] Docs (README + 7 docs incl. COLMAP guide), CLAUDE.md NFS notes retired
      (fixed by 2026-07-15 reboot).

## Notes / remaining
- The Pavillon scene reconstructs but is view-dependent-hard; IMG_9648 is a
  weakly-connected walking clip that does NOT reconstruct (do not use it).
- COLMAP's incremental mapper is nondeterministic; the stage keeps the best attempt.
  For the full scene_pavillon run, exhaustive matching over all 3 clips is expected
  to register more images than the smoke's single-clip sequential pass.
- Full-quality runs (object_default/high_quality, scene_pavillon) are configured
  and ready to submit via scripts/slurm/*.sbatch; not yet run at full scale.
- Optional backends (nerfstudio splatfacto, orig-3dgs, 2DGS/SuGaR geometry) are
  stubbed behind the TrainingBackend ABC, not implemented.

## SOTA A1 — GLOMAP robust SfM (done)
Integrated `run_colmap.mapper_backend: colmap|glomap`. On the single-side
carved-panel close-ups GLOMAP registers **181/193** images vs **82** for
incremental COLMAP (same features/matches) — 2.2x coverage. Final orbit
reconstruction on 145 train views: test PSNR 23.0 / SSIM 0.81 / LPIPS 0.27
(median 23.3, best 28.0; the one weak held-out view is a hard close-up only
GLOMAP could register). Also fixed 3 resume-safety bugs surfaced by the re-run
(stale-checkpoint provenance, empty-loop guard). Deliverables regenerated:
exports/gsplat_run/point_cloud.ply + framed orbit/progression videos + figures.

## SOTA A2 — 2DGS surface backend (done; dead-end on THIS capture)
Implemented + GPU-validated; produces flat renders (PSNR ~13) on this single-side
low-overlap panel. Kept as an experimental backend. See decisions.md.

## SOTA A3 — geometry regularizers (done) — room bounds + anti-floater + depth prior
Config `pavillon_orbit_reg.yaml`, trainer knobs `train.{bounds,floater,depth_prior}`.
Reg run `gsplat_reg_30k` (same GLOMAP as baseline, controlled comparison):
- **20.2% floaters removed** by room-bounds + anti-floater hard prune (1.52M→1.22M
  Gaussians), `.ply` 374→**301 MB**, renders stay sharp.
- Test PSNR **22.3 / SSIM 0.80 / LPIPS 0.29** (best val 24.5) vs baseline 23.9 /
  0.82 / 0.245. Deliverables: exports/gsplat_reg_30k/point_cloud.ply + orbit/
  progression videos + 5 figures + report + eval.
- **Ablation `gsplat_reg_nodepth`** (bounds+floater, depth OFF): test **22.55 /
  0.798 / 0.294**, 18.0% pruned, best val 24.45. Depth-off recovers only 0.26 dB,
  so the ~1.4 dB gap to baseline is the **final hard prune** (`min_opacity 0.02`
  drops faint Gaussians that still serve held-out views) — NOT the depth prior.
  Tune `floater.min_opacity` to trade cleanliness vs PSNR. See decisions.md.
- Depth prior needs `transformers` (pip-installed into v2gs; model cached on NFS).

## Cluster/ops note (2026-07-16)
GPURACK5 (RTX PRO 6000) went **down** mid-visualize; the reg pipeline had already
completed (train/eval/export/visualize) so nothing was lost.
**Node driver constraint (important, verified by probing each node):** the `v2gs`
env (torch cu130) only runs where the driver is new enough. **GPURACK1 (3090),
GPURACK3 (4090) AND GPURACK4 (5080) all report driver 12080 "too old" →
`torch.cuda.is_available() == False`.** Being Blackwell is NOT sufficient —
GPURACK4 has a 5080 and still fails. **Only GPURACK2 (RTX PRO 4500 Blackwell,
verified SM 12.0 / torch 2.13.0+cu130) and GPURACK5 (RTX PRO 6000, currently
DOWN) can run GPU work.** Always probe a node before submitting.
CPU-only stages (extract/filter/COLMAP — SIFT is CPU by default) DO run on the
old-driver nodes, so reconstructing on one node and re-running `--from-stage
train` on a CUDA-capable node is a valid split.
Housekeeping: removed 4 superseded run dirs (~6 GB) + accumulated debug junk.
