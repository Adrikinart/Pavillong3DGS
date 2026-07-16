# Troubleshooting

## Environment / Blackwell
**`CUDA error: no kernel image is available for execution`** or
`inspect-env` shows `supported_by_wheel=false` for `sm_120`.
→ The installed PyTorch wheel lacks Blackwell kernels. Install a **cu128** build:
`pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision`.
Re-run `inspect-env --gpu-check`.

**gsplat import/build fails on first GPU import.**
→ It JIT-compiles CUDA on first import and needs `nvcc`. The env ships
`cuda-nvcc=12.8`. Ensure `TORCH_CUDA_ARCH_LIST` is set (e.g. `12.0`) — the
sbatch scripts and `inspect-env` set it from the detected device. Check build
logs under `$TORCH_EXTENSIONS_DIR`.

**Conda is unbearably slow on the login node.**
→ Known NFS storm. Use the local-disk mamba (`scripts/bootstrap_environment.sh`
does this) as the solver; install the env into the NFS prefix so GPU nodes see it.

**Installing torch/gsplat.**
→ `bootstrap_environment.sh` installs the pip layer (torch/gsplat/…) via `srun`
on a GPU node (`V2GS_GPU_NODE`, default GPURACK5) — that node has `nvcc` and a GPU
so the gsplat CUDA extension builds and can be verified in place. (Historically
this also dodged a login-node NFS storm that throttled `/home` writes to ~200 KB/s;
that storm was fixed by the 2026-07-15 reboot, but building gsplat where the GPU is
remains the right call.) torch must be a cu128/cu130 build exposing `sm_120`.

## COLMAP
**All COLMAP attempts fail / few images registered.**
→ Check `colmap/validation_report.json` and `colmap/trajectory.png`. Common
causes: too few/blurry frames (raise `extract_frames.target_fps`, lower
`filter_frames.blur_var_min`), pure-rotation capture (need parallax), or wrong
matcher. The stage auto-falls-back to exhaustive matching once.

**GPU SIFT errors on sm_120.**
→ Default is CPU SIFT (`run_colmap.use_gpu: false`), which is robust. Only enable
GPU SIFT if your COLMAP build supports the device.

**COLMAP is slow / NFS stalls.**
→ It already runs on node-local scratch. Ensure `/var/tmp/$USER` is writable and
has space; check `scripts/sync_scratch.sh` for retained workspaces after a crash.

## Training
**Non-finite loss / training FAILED with a health error.**
→ Health checks hard-fail on NaN loss, exploding/collapsing gaussian count, or
diverging scale. Lower learning rates, reduce densification aggressiveness
(`densification.grad_threshold` up, `cap_max` down), or check the input data.

**Out-of-memory.**
→ Reduce `train.image_downscale` target resolution (raise `image_downscale`),
lower `densification.cap_max`, or use a bigger-VRAM node (GPURACK5, ~96 GB).

**Training didn't resume from where it stopped.**
→ Resume is automatic (`--resume` default). Confirm a valid checkpoint exists:
`ls trainings/<run>/checkpoints/`. A corrupt newest checkpoint is skipped (sha256
sidecar) and the previous valid one is used. `--force` restarts from scratch.

## Slurm
**Job preempted / requeued.**
→ Expected. SIGTERM triggers a checkpoint flush and exit-0; `--requeue` resubmits.
The stage resumes from the latest checkpoint. Status shows `RUNNING` with a
`preempted` note until it completes.

**A stage is stuck at RUNNING but nothing is running.**
→ Stale-RUNNING recovery: the runner checks the owning pid/`SLURM_JOB_ID`; if dead
it demotes to FAILED and re-runs. If it persists, delete the stale
`status/<stage>.json` and rerun.

## General
**Re-run one stage without redoing everything.**
→ `python -m video_to_3dgs.cli <stage> --config <cfg>` (upstream COMPLETED stages
are skipped). Use `--force` to redo a specific stage; downstream stages re-run
automatically when an upstream output changes.

**Where are the logs?**
→ `experiments/runs/<id>/logs/*.jsonl` (structured), Slurm logs under
`experiments/slurm_logs/`, training metrics in `trainings/<run>/metrics.jsonl`.
