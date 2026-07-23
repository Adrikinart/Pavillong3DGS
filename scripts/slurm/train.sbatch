#!/usr/bin/env bash
#SBATCH --job-name=v2gs-train
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --output=experiments/slurm_logs/%x_%j.out
#SBATCH --error=experiments/slurm_logs/%x_%j.err
# ---------------------------------------------------------------------------
# Train (+ evaluate + export + report) a 3DGS reconstruction.
# Usage:  sbatch scripts/slurm/train.sbatch configs/pipeline/pavillon/scene_pavillon.yaml
# Pin a node if you need one:  sbatch --nodelist=GPURACK2 ... <config>
#
# Deliberately NO default --nodelist. This used to pin GPURACK5, which went DOWN on
# 2026-07-16; every job submitted without an explicit override then queued forever against
# a dead node while other nodes in the partition sat free, and Slurm reports that only as
# "ReqNodeNotAvail" -- easy to read as ordinary queueing. The partition is the right level
# of constraint: rtxpro holds the only GPUs whose driver satisfies our cu130 torch build
# (the rtx30/rtx40/rtx50 nodes expose CUDA 12.8 and fail with "driver too old"), so letting
# the scheduler choose within it is both safe and self-healing when a node returns.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p experiments/slurm_logs

CONFIG="${1:-${V2GS_CONFIG:-configs/pipeline/pavillon/scene_pavillon.yaml}}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"

# Prepare env (PATH + CUDA_HOME/CPATH so gsplat's CUDA ext can JIT-build, plus
# node-local TORCH_EXTENSIONS_DIR/scratch). Single source: scripts/_activate_env.sh
source scripts/_activate_env.sh
export PIP_CACHE_DIR="/var/tmp/$USER/pip_cache"; mkdir -p "$PIP_CACHE_DIR"
# Depth-prior model (DepthAnything-v2) is pre-cached on NFS ~/.cache/huggingface;
# force offline so a GPU node without internet uses the cache instead of hanging.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[train.sbatch] host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG env=$ENV_PREFIX"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true
echo "[train.sbatch] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST CUDA_HOME=$CUDA_HOME"

# run-all skips already-completed preprocessing and proceeds through train/eval/export.
# Extra args after the config are forwarded (e.g. --force --from-stage train to
# re-freeze an edited config and re-train a sibling run without re-running COLMAP).
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli run-all --config "$CONFIG" --verbose "${@:2}"
rc=$?
echo "[train.sbatch] run-all exit=$rc"
exit $rc
