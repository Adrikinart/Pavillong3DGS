# Working Instructions (repo root)

## Project: video_to_3dgs (video → 3D Gaussian Splatting framework)

This repo hosts `video_to_3dgs` (package under `src/video_to_3dgs/`): a modular,
resumable pipeline turning object/scene videos into trained 3DGS reconstructions
on the Slurm GPU cluster. Start at [README.md](README.md) and
[docs/pipeline_architecture.md](docs/pipeline_architecture.md).

Project-specific guidance for agents:
- **Two GPU envs, and which one you pick decides which nodes you can use** (verified
  2026-07-23). Build/refresh either with `scripts/bootstrap_environment.sh`; it installs
  the heavy pip layer via `srun` on a GPU node so the gsplat CUDA extension builds where
  `nvcc`+GPU are. Always `source scripts/_activate_env.sh` before GPU work (sets
  CUDA_HOME/CPATH so gsplat can JIT-build, and puts the env's `colmap` on PATH); it
  honours `V2GS_ENV_PREFIX`. Do not put torch/gsplat in `pyproject.toml`.

  | env | torch | runs on | VRAM |
  |---|---|---|---|
  | `~/envs/v2gs` (default) | 2.13.0+**cu130** | **GPURACK2** only (GPURACK5 is down) | 32 GB |
  | `~/envs/v2gs-cu128` | 2.11.0+**cu128** | **GPURACK4** (rtx50), and the rtx30/rtx40 nodes | 16.6 GB |

  ```bash
  sbatch --partition=rtx50 --export=ALL,V2GS_ENV_PREFIX=/home/$USER/envs/v2gs-cu128 ...
  ```
  Why two: cu130 needs a CUDA 13.0 driver, which only the rtxpro nodes have. Every other
  node reports driver 570.x / **CUDA 12.8** and fails with *"The NVIDIA driver on your
  system is too old (found version 12080)"* — `nvidia-smi` looks modern there, so probe
  with torch, don't infer. Both envs compile for `sm_120`; gsplat JIT-builds on the 5080
  in ~86 s. Keep a sweep **within one env** — torch 2.11 vs 2.13 is not a controlled
  comparison. Locks: `requirements-lock.txt` (cu130) and `requirements-lock-cu128.txt`.
  **Pin the torch build when you want cu128** (`torch==2.11.0+cu128`): `--index-url` plus
  `--extra-index-url` does not mean "prefer the first", pip takes the highest *version*
  across both, which is how the default env silently became cu130 and lost three nodes.
- **CPU-only work** (login node): the package imports without torch (GPU imports
  are lazy). A quick CPU test env can be made with pydantic/pyyaml/numpy/pillow/
  opencv/pytest; run `pytest tests/unit tests/integration`.
- **Never run COLMAP or SQLite on `/home`** — stages use node-local
  `/var/tmp/$USER` scratch via `core/scratch.py`. Keep it that way.
- **Default training backend is direct gsplat** (not Nerfstudio). See
  [state/decisions.md](state/decisions.md) before changing backends.
- **Never `git add -A`** (NFS rule below); `experiments/runs/`, checkpoints,
  frames, COLMAP DBs, and `*.MOV` are gitignored — stage explicit source paths.
- **GPURACK5 has been DOWN/NOT_RESPONDING since 2026-07-16** — do not target it. It was
  the default `--nodelist` in every sbatch script and in `bootstrap_environment.sh`, so
  jobs queued forever against a dead node while other GPUs sat free; Slurm reports that
  only as `ReqNodeNotAvail`, which reads like ordinary queueing. Those defaults are now
  removed: submit with a `--partition` and let the scheduler choose, pinning `--nodelist`
  only when you mean it. Reviving GPURACK5 needs an admin.

## Environment note (isiacluster)

The NFS `/home` mount was fixed by an admin reboot on 2026-07-15 — conda/pip on
`/home` work again (a directory `ls` is back to ~40 ms). The former NFSv4
`TEST_STATEID` storm and its local-disk-conda workaround are obsolete and were
removed. This box has **no GPU**; use it for the CPU test env, analysis, and
plotting, and run GPU work on the GPURACK nodes via Slurm.

Still relevant (an extension bug, not an NFS symptom, so kept post-reboot):
per-user `TMPDIR=/tmp/tmp-$(id -u)` is exported at the top of `~/.bashrc` so the
VS Code remote server / Codex extension don't collide with other users' IPC
sockets on this shared node.

## Git & filesystem hygiene on this NFS mount (learned 2026-07-13/14, keep even post-reboot)

The repo lives on the shared NFS volume, and `Data/` (~750 GB of splat corpora) plus
`gs_jepa/runs/` (~5 GB of checkpoints) sit inside the worktree. Rules that keep git and the
mount healthy:

- **Never `git add -A` / `git add .`** — it will try to hash gigabytes of `runs/`/`Data/`
  through NFS (one such attempt orphaned ~5k loose objects and stalled for minutes despite
  `.gitignore`, because ignore rules don't stop explicit adds of already-swept paths).
  Stage explicit paths; run artifacts are added per-directory as `*.json`/`*.jsonl` only.
- **Pushes can be dropped mid-upload** when NFS stalls pack streaming ("remote end hung up").
  Remedy: push in staged chunks — `git push origin <intermediate-sha>:refs/heads/<branch>`
  every ~15 commits, then the final `git push`.
- **`git checkout <branch>` may abort** on the other sub-project's dirty files; to merge a
  branch into `main` without touching the worktree, verify ancestry
  (`git merge-base --is-ancestor origin/main <branch>`) and push the ref directly:
  `git push origin <branch>:main`.
- Avoid repo-wide `du`/`find` that descend into `Data/` — they hang for minutes.
- Standing recommendation: `Data/` belongs on scratch/local storage, not NFS `/home`
  (the volume is at ~87% capacity); a symlink into the worktree keeps paths stable.
