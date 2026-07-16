# Pipeline architecture

## Principles
- **Filesystem is the source of truth** — no daemon, no database. State lives in
  per-stage status JSON + an append-only manifest under the run directory.
- **Every stage is idempotent and independently re-runnable.** A stage writes
  outputs to temp locations and atomically promotes them; re-running overwrites
  cleanly.
- **A stage reaches COMPLETED only after output validation passes.** The runner
  writes `RUNNING` before work and `COMPLETED` only after `validate_outputs`, so a
  crash leaves `RUNNING`/`FAILED`, never `COMPLETED`.
- **Node-local scratch for high-I/O**, durable `/home` for results.

## Modules (`src/video_to_3dgs/`)
- `core/` — `stage.py` (Stage ABC, Artifact, StageContext), `runner.py`
  (topo-order, skip/force/resume, stale-RUNNING recovery, fingerprints),
  `status.py`, `manifest.py`, `paths.py` (`RunLayout` — single path authority),
  `scratch.py` (`ScratchContext`), `atomicio.py`, `provenance.py`, `logging.py`.
- `config/` — pydantic v2 `schema.py` (frozen `PipelineConfig`), `loader.py`
  (layered YAML: defaults → profile → user YAML → `--set` → freeze to
  `config_resolved.yaml`).
- `stages/` — the 11 pipeline stages.
- `training/` — `backend.py` (`TrainingBackend` ABC + registry), `gsplat_backend.py`
  (self-contained trainer on packaged gsplat APIs), `dataset.py` (COLMAP reader +
  normalization), `gaussians.py`, `losses.py`, `checkpoint.py`, `metrics.py`,
  `health.py`, `signals.py`, `validation.py`, `adapters/`.
- `monitoring/` — `gpu_monitor.py`, `report.py`.
- `cli.py` — unified CLI.

## Run directory layout
```
experiments/runs/<dataset_id>/
  config_resolved.yaml         frozen merged config
  manifest.json                append-only provenance
  environment.json
  status/<stage>.json          per-stage status = completion marker
  logs/{<stage>.log,<stage>.jsonl,pip_freeze.txt}
  video/metadata.json
  frames/  frames_filtered/  rejected/  masks/
  colmap/{database.db, sparse/0/*.bin, images/, validation_report.json, trajectory.png}
  normalized/transform.json
  splits/{train,val,test}.txt  split_cameras.png
  trainings/<train_run_id>/{checkpoints/, metrics.jsonl, tb/, renders/, figures/, videos/, metrics/, eval.json, report/}
  trainings/latest -> <train_run_id>          # pointer to the most recent training
  exports/<train_run_id>/{point_cloud.ply, cameras.json, normalize_transform.json, COORDINATES.md}
```
`dataset_id = slug(object_name)_<8-char signature of video sizes/names>`.

**Run identity / monitoring.** Each training gets a descriptive, timestamped
`train_run_id`, e.g. `gsplat_glomap_30k_20260716-153045`
(`<backend>_<mapper>_<iters>k_<timestamp>`), generated once and **baked into the
frozen `config_resolved.yaml`** — so it is stable across stages and processes yet
unique per experiment (no overwrites). `trainings/latest` points at the newest.
Pin a name with `--train-run-id` (sweeps) or `train.train_run_id`. Cross-run
comparison lives in `experiments/registry.csv` (timestamp, dataset, backend,
mapper, registration ratio, #gaussians, PSNR/SSIM/LPIPS, VRAM, git, config path).
When a training's inputs change (e.g. a new SfM model), a `.train_fingerprint`
next to the checkpoints triggers a fresh train instead of resuming stale poses.

## Fingerprints & downstream invalidation
Each stage's fingerprint = `sha256(canonical(params) + input-artifact checksums)`.
`run-all` skips a `COMPLETED` stage whose fingerprint matches; changing a stage's
params (or an upstream output) changes the fingerprint and forces a re-run.

## Stage dependency graph
```
inspect_video → extract_frames → filter_frames → generate_masks
                                        ↓
                                    run_colmap → validate_colmap → normalize_scene → split_dataset
                                        ↓
                                     train → evaluate
                                          ↘ export
```
(`generate_masks` and `run_colmap` both consume `filter_frames`; masks feed COLMAP
and training when enabled.)

## Training backend
Direct gsplat trainer built on the **packaged** `gsplat.rasterization` +
`gsplat.DefaultStrategy` (not `examples/simple_trainer.py`). It owns the loop to
control atomic checkpointing, resume, health hard-fails (NaN / exploding or
collapsing gaussian count / diverging scale / OOM), structured JSONL + TensorBoard
metrics, fixed-camera validation, and SIGTERM→flush→exit-0 preemption handling.
New backends implement `TrainingBackend.{train,export_ply,validate_env}` with zero
core changes; backend-specific knobs ride in `TrainCfg.backend_opts`.
