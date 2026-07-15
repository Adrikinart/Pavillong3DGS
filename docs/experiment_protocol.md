# Experiment protocol

## Reproducibility
Every run records (in `manifest.json` + `config_resolved.yaml` +
`logs/pip_freeze.txt`): resolved config, seeds, this repo's git SHA + dirty flag,
python/torch/CUDA-runtime versions, NVIDIA driver, GPU name + compute capability,
relevant env vars, package freeze hash, dataset manifest with per-video
checksums, command context, hostname, Slurm job id, and timestamps.

Reproduce a run: check out the recorded git SHA, rebuild the env from
`requirements-lock.txt`, and re-run with the same `config_resolved.yaml`
(`--dataset-id <id>` reuses the frozen config).

Determinism note: gsplat/CUDA rasterization is not bit-exact deterministic across
runs even with fixed seeds; metrics are stable to within noise. Seeds fix
data ordering and initialization.

## Evaluation policy
- Metrics computed **only on held-out** val/test views (pose-aware split).
- **Model selection uses validation metrics, never training loss.**
- Report PSNR/SSIM/LPIPS (+ masked variants when masks exist), render FPS, peak
  VRAM, model size, #gaussians. Per-view metrics + best/worst views are saved.
- One row per run is appended to `experiments/registry.csv`.

## Splitting
Pose-aware by default: held-out cameras are spread evenly around the orbit (by
azimuth) so val/test cover the full viewpoint range and no adjacent-frame leakage
occurs. `periodic` and `random` strategies are available.

## Sweeps (multi-GPU as parallel experiments)
For object-scale scenes, train on **one GPU** and parallelize *experiments*, not a
single training. Recommended flow:
1. `sbatch scripts/slurm/preprocess.sbatch <cfg>` — build COLMAP/masks once.
2. `sbatch scripts/slurm/sweep.sbatch <cfg>` — a job array trains variants
   (seeds/configs) against the shared dataset build; each writes its own
   `train_run_id` under `trainings/`.
Good sweep axes: frame fps, masking on/off, COLMAP matcher, densification
strategy, resolution, seed, backend.

## Configs
- `smoke_test.yaml` — smallest end-to-end (single clip, downscaled, ~1.5k iters).
- `scene_pavillon.yaml` — the Pavillon scene (masks off, point normalization).
- `object_default.yaml` — balanced object reconstruction (masks on).
- `object_high_quality.yaml` — higher res/iters, pose + appearance optimization.
- `object_turntable.yaml` — turntable capture (masks mandatory, exhaustive match).
Backend-independent knobs are typed; backend-specific knobs go in
`train.backend_opts`.
