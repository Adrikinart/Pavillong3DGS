#!/usr/bin/env bash
#SBATCH --job-name=v2gs-train
#SBATCH --partition=rtxpro
#SBATCH --nodelist=GPURACK5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=experiments/slurm_logs/%x_%j.out
#SBATCH --error=experiments/slurm_logs/%x_%j.err
# ---------------------------------------------------------------------------
# Train (+ evaluate + export + report) a 3DGS reconstruction.
# Usage:  sbatch scripts/slurm/train.sbatch configs/pipeline/scene_pavillon.yaml
# Override node/partition:  sbatch --partition=rtx50 --nodelist=GPURACK4 ... <config>
# ---------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p experiments/slurm_logs

CONFIG="${1:-${V2GS_CONFIG:-configs/pipeline/scene_pavillon.yaml}}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"

# node-local scratch for CUDA-extension builds and caches (avoid recompiling / NFS)
export TORCH_EXTENSIONS_DIR="/var/tmp/$USER/torch_ext"
export PIP_CACHE_DIR="/var/tmp/$USER/pip_cache"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-}"   # set by inspect-env if empty
mkdir -p "$TORCH_EXTENSIONS_DIR" "$PIP_CACHE_DIR"

export PATH="$ENV_PREFIX/bin:$PATH"
echo "[train.sbatch] host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG env=$ENV_PREFIX"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true

# detect arch for gsplat JIT if not preset
if [ -z "$TORCH_CUDA_ARCH_LIST" ]; then
  export TORCH_CUDA_ARCH_LIST="$("$ENV_PREFIX/bin/python" -c "import torch;p=torch.cuda.get_device_properties(0);print(f'{p.major}.{p.minor}')" 2>/dev/null || echo '')"
fi
echo "[train.sbatch] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# run-all skips already-completed preprocessing and proceeds through train/eval/export.
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli run-all --config "$CONFIG" --verbose
rc=$?
echo "[train.sbatch] run-all exit=$rc"
exit $rc
