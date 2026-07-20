# video_to_3dgs

Raw object/scene videos → trained, evaluated, exportable **3D Gaussian Splatting**
reconstructions, orchestrated across an SSH-accessible Slurm GPU cluster with
NVIDIA **Blackwell** GPUs.

The pipeline is modular, resumable, idempotent, scheduler-friendly, and
storage-efficient. Every stage is independently re-runnable and is the only
writer of its own status file, so a failed stage never marks itself complete.

```
raw videos → inspect → extract frames → filter → (mask) → COLMAP → validate
           → normalize → split → train (gsplat) → evaluate → export → report
```

> **Want to re-create the trained Pavillon model?** Follow
> **[docs/reproduce_pavillon.md](docs/reproduce_pavillon.md)** — a complete,
> config-driven recipe (env → data → GLOMAP → training with the A3 geometry
> regularizers → export → verification), including the hardware/driver
> constraints and expected metrics.

## 1. Supported capture assumptions
- One static physical object **or** a static scene; the camera moves around it.
- Multiple heights/elevations, background visible, pedestal allowed.
- Optional **turntable** mode (object rotates) — masks become mandatory.
- Objective: object-centric novel-view rendering; geometry usable downstream.

## 2. Recommended capture protocol
See [docs/capture_guide.md](docs/capture_guide.md). In short: keep the object
static, move the camera, multiple elevation orbits, maintain overlap, avoid pure
rotation and motion blur, lock focus/exposure/WB, no digital zoom.

## 3. Environment (Blackwell / sm_120)
Blackwell needs a **cu128** PyTorch build. `tiny-cuda-nn` is avoided; the default
backend is **direct gsplat**. Build the env once with the local-disk mamba
(fast solver, writes to an NFS prefix the GPU nodes can see):

```bash
scripts/bootstrap_environment.sh          # creates /home/$USER/envs/v2gs
```

Then validate on a Blackwell node:

```bash
srun --partition=rtxpro --nodelist=GPURACK5 --gres=gpu:1 --time=00:15:00 \
  /home/$USER/envs/v2gs/bin/python -m video_to_3dgs.cli inspect-env --gpu-check \
  --out experiments/env_gpurack5.json
```

`inspect-env` records driver, compute capability, torch/cuda, runs a CUDA
forward+backward smoke test and a gsplat rasterization test, and checks that the
installed PyTorch wheel contains kernels for the device's `sm_120`.

## 4. Storage
- High-I/O work (COLMAP SQLite, temp frames) runs on **node-local** `/var/tmp/$USER`.
- Durable outputs live under `experiments/runs/<dataset_id>/` on `/home`.
- The env build is one-time; caches (`TORCH_EXTENSIONS_DIR`) go on node-local scratch.

## 5. Quickstart — smoke test (end-to-end, machine-verifiable)
```bash
scripts/run_smoke_test.sh rtxpro GPURACK5
# on success writes experiments/SMOKE_TEST_OK and exits 0
```

## 6. Full reconstruction (Pavillon scene example)
```bash
# local (inside a GPU allocation):
scripts/run_pipeline.sh local configs/pipeline/scene_pavillon.yaml
# via Slurm (preprocess then train, dependency-chained):
scripts/run_pipeline.sh slurm configs/pipeline/scene_pavillon.yaml
```

## 7. CLI
```bash
python -m video_to_3dgs.cli inspect-env [--gpu-check --out env.json]
python -m video_to_3dgs.cli prepare              --config <cfg>
python -m video_to_3dgs.cli inspect-video        --config <cfg>
python -m video_to_3dgs.cli extract-frames       --config <cfg>
python -m video_to_3dgs.cli filter-frames        --config <cfg>
python -m video_to_3dgs.cli generate-masks       --config <cfg>
python -m video_to_3dgs.cli reconstruct          --config <cfg>   # COLMAP
python -m video_to_3dgs.cli validate-reconstruction --config <cfg>
python -m video_to_3dgs.cli normalize            --config <cfg>
python -m video_to_3dgs.cli split                --config <cfg>
python -m video_to_3dgs.cli train                --config <cfg>
python -m video_to_3dgs.cli evaluate             --config <cfg>
python -m video_to_3dgs.cli export               --config <cfg>
python -m video_to_3dgs.cli run-all              --config <cfg> [--from-stage S --to-stage S --only S]
python -m video_to_3dgs.cli status               --config <cfg>
python -m video_to_3dgs.cli report               --config <cfg> [--train-run-id ID]
```
Every stage supports `--dry-run`, `--force`, `--resume` (default), `--verbose`,
`--set key=value` overrides, and returns clear exit codes.

## 8. Monitoring
- Structured logs: `experiments/runs/<id>/logs/*.jsonl` (mandatory).
- TensorBoard: `tensorboard --logdir experiments/runs/<id>/trainings/<run>/tb`.
- GPU/system stats sampled to `system_monitoring.jsonl`.
- Fixed-camera validation renders under `trainings/<run>/renders/`.

## 9. Resuming
Training checkpoints atomically and resumes from the latest valid checkpoint
automatically (`--resume` is default). Slurm preemption (SIGTERM) flushes a
checkpoint and exits 0 for `--requeue`. Just resubmit the same job/command.

## 10. Evaluation & export
Evaluation is **only** on held-out views (PSNR/SSIM/LPIPS + masked variants,
render FPS, VRAM, model size). Model selection uses validation metrics, never
training loss. Export produces a native 3DGS `.ply`, camera json, the
normalization transform, and a coordinate-conventions doc (Blender/Unity).

## 11. Multiple experiments (sweeps)
Preprocess once, then fan out trainings as a Slurm job array
(`scripts/slurm/sweep.sbatch`) — different seeds/configs share the same COLMAP
build. One GPU per training (object-scale); multi-GPU is for parallel experiments.

## Further docs
- **[docs/reproduce_pavillon.md](docs/reproduce_pavillon.md) — reproduce the trained Pavillon model end-to-end (start here)**
- [docs/cluster_environment.md](docs/cluster_environment.md)
- [docs/pipeline_architecture.md](docs/pipeline_architecture.md)
- [docs/colmap_guide.md](docs/colmap_guide.md) — how to run COLMAP (it's env-provided, not a module)
- [docs/experiment_protocol.md](docs/experiment_protocol.md)
- [docs/troubleshooting.md](docs/troubleshooting.md)
- [docs/capture_guide.md](docs/capture_guide.md)
- [docs/video_to_3dgs_plan.md](docs/video_to_3dgs_plan.md)
