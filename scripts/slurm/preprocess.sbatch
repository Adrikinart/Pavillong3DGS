#!/usr/bin/env bash
#SBATCH --job-name=v2gs-preprocess
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm_logs/%x_%j.out
#SBATCH --error=experiments/slurm_logs/%x_%j.err
# Deliberately NO default --nodelist: it used to pin GPURACK5, which went DOWN on
# 2026-07-16, so jobs queued forever against a dead node while others sat free.
# Pin explicitly (--nodelist=GPURACK2) only when you need a specific node.
# ---------------------------------------------------------------------------
# Preprocess: inspect -> extract -> filter -> masks -> COLMAP -> validate ->
# normalize -> split. Runs on a GPU node so masking + node-local scratch (COLMAP
# SQLite) are available. COLMAP feature extraction defaults to CPU SIFT.
# Usage:  sbatch scripts/slurm/preprocess.sbatch configs/pipeline/pavillon/scene_pavillon.yaml
# ---------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p experiments/slurm_logs

CONFIG="${1:-${V2GS_CONFIG:-configs/pipeline/pavillon/scene_pavillon.yaml}}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
source scripts/_activate_env.sh

echo "[preprocess.sbatch] host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG"
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli run-all --config "$CONFIG" \
     --to-stage split_dataset --verbose
exit $?
