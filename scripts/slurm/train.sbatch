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

# Prepare env (PATH + CUDA_HOME/CPATH so gsplat's CUDA ext can JIT-build, plus
# node-local TORCH_EXTENSIONS_DIR/scratch). Single source: scripts/_activate_env.sh
source scripts/_activate_env.sh
export PIP_CACHE_DIR="/var/tmp/$USER/pip_cache"; mkdir -p "$PIP_CACHE_DIR"
echo "[train.sbatch] host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG env=$ENV_PREFIX"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
echo "[train.sbatch] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST CUDA_HOME=$CUDA_HOME"

# run-all skips already-completed preprocessing and proceeds through train/eval/export.
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli run-all --config "$CONFIG" --verbose
rc=$?
echo "[train.sbatch] run-all exit=$rc"
exit $rc
