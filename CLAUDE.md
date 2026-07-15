# Working Instructions (repo root)

## Project: video_to_3dgs (video → 3D Gaussian Splatting framework)

This repo hosts `video_to_3dgs` (package under `src/video_to_3dgs/`): a modular,
resumable pipeline turning object/scene videos into trained 3DGS reconstructions
on the Slurm GPU cluster. Start at [README.md](README.md) and
[docs/pipeline_architecture.md](docs/pipeline_architecture.md).

Project-specific guidance for agents:
- **GPU env is `v2gs`** at `/home/$USER/envs/v2gs` (torch **cu128** for Blackwell
  sm_120, gsplat, colmap). Build/refresh with `scripts/bootstrap_environment.sh`
  (uses the local-disk mamba solver → NFS prefix). Do not put torch/gsplat in
  `pyproject.toml`; they are provisioned by the env only.
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

## Python environment on this machine (isiacluster) — READ FIRST

**Do not use `conda` from `~/miniconda3` on this box, and do not "fix" conda by reinstalling it.**

Since 2026-07-14 the NFS `/home` mount on isiacluster is pathologically slow (NFSv4
`TEST_STATEID` storm in the client kernel: every file open costs 40–400 ms, so the
NFS conda appears to hang forever). The fix requires an admin reboot of the node.
Until then, a working conda lives on the local disk:

```bash
source /var/tmp/AdrienK/conda-local.sh   # switches `conda` to /var/tmp/AdrienK/miniforge3
conda activate my3DGS                   # my3DGS may not exist yet so create it if not
```

Notes:

- The local `my3DGS` env mirrors `my3DGS/environment.yml` but with **CPU-only torch**
  (this box has no GPU). It is for tests, analysis, and plotting only.
- GPU training runs on the GPURACK nodes, whose NFS clients are healthy — they keep
  using the original `~/miniconda3/envs/my3DGS`. Do not modify that env from here.
- Anything touching many files under `/home` (pip installs into NFS envs, large
  `find`/`grep` sweeps, conda from `~/miniconda3`) will be extremely slow until the
  node is rebooted. Prefer short, targeted file access; the repo itself is small
  enough to be workable.
- The **Codex/ChatGPT VS Code extension** needed TWO fixes (2026-07-14):
  (1) its `~/.codex` state dir (four live SQLite DBs — SQLite over storming NFS hangs)
  now SYMLINKS to `/var/tmp/AdrienK/codex-home` (local disk); original preserved at
  `~/.codex.nfs-backup`; node-local, state won't follow to other machines.
  (2) its IPC socket dir `os.tmpdir()/codex-ipc` collided with ANOTHER USER's copy on
  this shared node (their `/tmp/codex-ipc` is 755 → our `listen EACCES`, panel never
  opens). Fix: per-user `TMPDIR=/tmp/tmp-$(id -u)` exported at the TOP of `~/.bashrc`
  (above the interactivity guard, so the VS Code remote server inherits it). Requires
  a VS Code server restart to take effect; keep this export even post-reboot — the
  collision is an extension bug, not an NFS symptom.
- Once `time ~/miniconda3/condabin/conda --version` is back to ~1 s (post-reboot),
  these workarounds are obsolete: move `/var/tmp/AdrienK/codex-home` back to `~/.codex`
  (replacing the symlink, after stopping the extension), delete
  `/var/tmp/AdrienK/{miniforge3,conda-local.sh}`, remove `~/.codex.nfs-backup`,
  and delete this section.

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
