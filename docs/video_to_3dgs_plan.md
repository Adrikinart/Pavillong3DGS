# video_to_3dgs — implementation plan

## Existing assets to preserve
Greenfield repo: only `CLAUDE.md` (machine/NFS rules) existed. Nothing else to
preserve. `CLAUDE.md` is kept and extended with a project section.

## Missing components (all built here)
Full framework: core orchestration, config system, 11 pipeline stages, gsplat
training backend, monitoring/reporting, CLI, Slurm scripts, tests, docs.

## Folder structure
See [pipeline_architecture.md](pipeline_architecture.md). `src/video_to_3dgs/`
with `core/ config/ stages/ training/ monitoring/ slurm/`; `configs/`, `scripts/`,
`tests/`, `docs/`, `state/`, `experiments/`.

## Baseline backend
**Direct gsplat** (self-contained trainer on packaged gsplat APIs). Chosen for
clean Blackwell/sm_120 support (no tiny-cuda-nn) and full control over
checkpointing/health/monitoring. Nerfstudio-splatfacto and original-3dgs are
future adapters behind the `TrainingBackend` ABC.

## Dependency strategy
- torch **cu128** + gsplat via `environment.yml` + `bootstrap_environment.sh`;
  pinned to `requirements-lock.txt`. Env built with local-disk mamba into an NFS
  prefix; gsplat CUDA ext JIT-compiled for `sm_120` with in-env `cuda-nvcc`.
- COLMAP from conda-forge, invoked via CLI (self-contained `.bin` readers in
  `colmap_io.py`, no pycolmap hard dep).

## Cluster integration
Slurm sbatch scripts (preprocess/train/evaluate/sweep) with `--requeue`,
`--signal=B:SIGTERM@120`, node-local scratch/caches, GRES per partition, env-var
driven. Auto-detects scheduler; local/interactive mode preserved.

## Dataset-processing stages
inspect_video → extract_frames (rotation-aware) → filter_frames (blur/exposure/
duplicate scoring, coverage safeguard) → generate_masks (rembg) → run_colmap
(scratch, fallbacks) → validate_colmap (gates + diagnostics) → normalize_scene
(similarity transform + inverse) → split_dataset (pose-aware).

## Monitoring
Structured JSONL (mandatory) + TensorBoard + GPU/system sampler + fixed-camera
validation renders; health checks with hard failures; final HTML/MD report;
`experiments/registry.csv`.

## Testing
Unit tests (config, paths, status, manifest, atomicio, runner invariants,
geometry, checkpoint) + an ffmpeg-driven CPU integration test + a
machine-verifiable end-to-end smoke test script.

## Milestones
1. Core + config ✅  2. CPU stages ✅  3. COLMAP + validation ✅
4. Normalize + split ✅  5. gsplat training + monitoring ✅
6. Evaluate + export + report ✅  7. CLI + Slurm ✅  8. Tests ✅
9. Env bootstrap + Blackwell validation → on cluster  10. End-to-end smoke on GPU → on cluster.

## Risks
Blackwell sm_120 toolchain (mitigated: detect arch, cu128, in-env nvcc, record
logs); COLMAP GPU SIFT on sm_120 (mitigated: CPU SIFT default); compute-node
internet (mitigated: fetch on login node only); NFS/SQLite (mitigated: node-local
scratch). See [troubleshooting.md](troubleshooting.md).

## Acceptance criteria
See README + task §29. Tracked in [../state/progress.md](../state/progress.md).
