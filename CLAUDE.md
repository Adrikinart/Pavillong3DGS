# Working Instructions (repo root)

## Project: video_to_3dgs (video → 3D Gaussian Splatting framework)

This repo hosts `video_to_3dgs` (package under `src/video_to_3dgs/`): a modular,
resumable pipeline turning object/scene videos into trained 3DGS reconstructions
on the Slurm GPU cluster. Start at [README.md](README.md) and
[docs/pipeline_architecture.md](docs/pipeline_architecture.md).

Project-specific guidance for agents:
- **GPU env is `v2gs`** at `/home/$USER/envs/v2gs` (torch **cu128/cu130** exposing
  Blackwell sm_120, gsplat, colmap 3.13). Build/refresh with
  `scripts/bootstrap_environment.sh`; it installs the heavy pip layer (torch/gsplat)
  via `srun` on a GPU node so the gsplat CUDA extension builds where `nvcc`+GPU are.
  Always `source scripts/_activate_env.sh` before GPU work (sets CUDA_HOME/CPATH so
  gsplat can JIT-build, and puts the env's `colmap` on PATH). Do not put torch/gsplat
  in `pyproject.toml`; they are provisioned by the env only.
- **CPU-only work** (login node): the package imports without torch (GPU imports
  are lazy). A quick CPU test env can be made with pydantic/pyyaml/numpy/pillow/
  opencv/pytest; run `pytest tests/unit tests/integration`.
- **Never run COLMAP or SQLite on `/home`** — stages use node-local
  `/var/tmp/$USER` scratch via `core/scratch.py`. Keep it that way.
- **Default training backend is direct gsplat** (not Nerfstudio). See
  [state/decisions.md](state/decisions.md) before changing backends.
- **Never `git add -A`** (NFS rule below); `experiments/runs/`, checkpoints,
  frames, COLMAP DBs, and `*.MOV` are gitignored — stage explicit source paths.
- Preferred smoke/heavy GPU node: **GPURACK5** (RTX PRO 6000, ~96 GB, idle).

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
