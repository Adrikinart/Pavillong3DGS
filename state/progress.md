# Progress

Last updated: 2026-07-15.

## Completed
- [x] Repository + cluster audit (see docs/cluster_environment.md).
- [x] Architecture + plan (docs/, this repo).
- [x] Core framework: paths/RunLayout, atomicio, status, Stage ABC, StageRunner
      (skip/force/resume, stale-RUNNING recovery, fingerprints), manifest,
      provenance, structured logging, ScratchContext.
- [x] Config system: pydantic v2 frozen schema, layered YAML loader, cluster
      profiles (rtx30/40/50/rtxpro/login-cpu), 5 pipeline configs.
- [x] Environment inspection (`inspect-env`) with CUDA fwd/bwd + gsplat smoke
      tests and sm_120 wheel-compatibility check.
- [x] Stages: inspect_video, extract_frames (rotation-aware, multi-video),
      filter_frames (blur/exposure/duplicate + coverage safeguard + contact
      sheets), generate_masks (rembg/imported), run_colmap (scratch + fallbacks),
      validate_colmap (gates + trajectory viz), normalize_scene (similarity
      transform + inverse), split_dataset (pose-aware + viz).
- [x] Training: TrainingBackend ABC + registry, self-contained gsplat trainer
      (masked L1+SSIM, DefaultStrategy densify/prune, atomic checkpoints +
      integrity + resume, JSONL+TB metrics, health hard-fails, SIGTERM→requeue),
      dataset loader, gaussian init, validation/eval.
- [x] evaluate (held-out only, PSNR/SSIM/LPIPS + masked, FPS/VRAM, registry.csv),
      export (.ply + cameras + transforms + COORDINATES.md), monitoring
      (gpu_monitor, HTML/MD report).
- [x] Unified CLI (all stages + run-all/status/report).
- [x] Slurm sbatch (preprocess/train/evaluate/sweep) + shell scripts
      (bootstrap/inspect_cluster/run_smoke_test/run_pipeline/sync_scratch).
- [x] Tests: 35 unit + integration tests PASS in a CPU env (config, core, runner
      invariants, geometry, checkpoint, ffmpeg CPU pipeline).
- [x] CPU stages validated end-to-end on a real clip (inspect→extract→filter,
      idempotent SKIP verified).
- [x] Docs (README + 6 docs) + state files.

## In progress / on the cluster
- [ ] `v2gs` env build (torch cu128 + gsplat + colmap) — running via
      bootstrap_environment.sh (large NFS install).
- [ ] GPU validation on GPURACK5 (`inspect-env --gpu-check`) — pending env.
- [ ] End-to-end smoke test on GPURACK5 (COLMAP→train→eval→export→report) —
      pending env.
- [ ] Resume verification on GPU — pending env.

## Notes
- All GPU-dependent code is import-lazy so the CPU-only login node imports the
  package and runs preprocessing/tests without torch.
- Anything marked "pending env" is implemented and unit-tested but not yet run on
  a Blackwell GPU because the env build was still in progress.
