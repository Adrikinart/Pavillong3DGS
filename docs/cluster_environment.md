# Cluster environment (isiacluster)

Findings from the environment audit (2026-07-15). Detected, not assumed —
re-run `scripts/inspect_cluster.sh` to refresh.

## Login node `isiacluster`
- CPU-only (no GPU, no `nvcc`, no `colmap`, no environment-modules).
- Python 3.10.12; `ffmpeg`/`ffprobe` present; Slurm client present.
- Has internet access (used for the one-time env build).
- **Conda on the NFS `/home` mount is pathologically slow here** (documented
  NFSv4 storm). Use the local-disk mamba at `/var/tmp/$USER/miniforge3` as the
  solver, installing the env into an NFS prefix the GPU nodes can read.

## Slurm GPU partitions
| partition | node | GPUs | VRAM | arch | sm |
|---|---|---|---|---|---|
| rtx30 | GPURACK1 | 2× RTX 3090 | 24 GB | Ampere | sm_86 |
| rtx40 | GPURACK3 | 2× RTX 4090 | 24 GB | Ada | sm_89 |
| rtx50 | GPURACK4 | 2× RTX 5080 | 16 GB | **Blackwell** | **sm_120** |
| rtxpro | GPURACK2 | 2× RTX PRO 4500 | ~20 GB | **Blackwell** | **sm_120** |
| rtxpro | GPURACK5 | 2× RTX PRO 6000 | **~96 GB** | **Blackwell** | **sm_120** |

- Driver **595.71.05** (very recent; supports CUDA 13). Max walltime **3 days**.
- GPURACK5 (RTX PRO 6000, ~96 GB) is the preferred smoke-test / heavy node.
- Slurm `RealMemory` is misconfigured (reports 1M) → `--mem` accounting is
  unreliable; rely on `--cpus-per-task` and observed `FreeMem`.
- **No `nvcc`/`colmap`/modules on the GPU nodes either** — everything is provided
  through the `v2gs` conda env.

## Blackwell (sm_120) software requirements
- PyTorch must be a **cu128+** build to expose `sm_120` kernels.
- `tiny-cuda-nn` is avoided (Blackwell build friction) → **direct gsplat** backend.
- gsplat compiles its CUDA extension on first import; the env ships `cuda-nvcc`
  12.8 so it can JIT-build for `sm_120` (`TORCH_CUDA_ARCH_LIST=12.0`). Extension
  builds are cached on node-local `TORCH_EXTENSIONS_DIR` to avoid recompiling.

## Storage
- `/home` NFS: 62T, ~87% full, SQLite-hostile → never run COLMAP DBs there.
- Local disk `/` on nodes: ~764 GB free → node-local scratch `/var/tmp/$USER`.
- Scratch resolution order: `$SLURM_TMPDIR`, `$TMPDIR`, `/scratch/$USER`,
  `/local_scratch/$USER`, `/var/tmp/$USER`.

## Datasets
`/home/AdrienK/Datasets/VideosForNVSpersonnal/`
- `Pavillon/` — 3 iPhone 14 Pro 4K HEVC clips (primary target, a scene).
- `casque saint georges doudou/` — 10 clips (camera + iPhone), an object.
- iPhone clips carry rotation metadata (e.g. `rotate=90`); frame extraction
  honors it via ffmpeg auto-rotate.
