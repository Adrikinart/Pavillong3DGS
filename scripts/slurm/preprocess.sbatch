#!/usr/bin/env bash
#SBATCH --job-name=v2gs-preprocess
#SBATCH --partition=rtxpro
#SBATCH --nodelist=GPURACK5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=experiments/slurm_logs/%x_%j.out
#SBATCH --error=experiments/slurm_logs/%x_%j.err
# ---------------------------------------------------------------------------
# Preprocess: inspect -> extract -> filter -> masks -> COLMAP -> validate ->
# normalize -> split. Runs on a GPU node so masking + node-local scratch (COLMAP
# SQLite) are available. COLMAP feature extraction defaults to CPU SIFT.
# Usage:  sbatch scripts/slurm/preprocess.sbatch configs/pipeline/scene_pavillon.yaml
# ---------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p experiments/slurm_logs

CONFIG="${1:-${V2GS_CONFIG:-configs/pipeline/scene_pavillon.yaml}}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
export PATH="$ENV_PREFIX/bin:$PATH"
export TMPDIR="/var/tmp/$USER"; mkdir -p "$TMPDIR"

echo "[preprocess.sbatch] host=$(hostname) job=$SLURM_JOB_ID config=$CONFIG"
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli run-all --config "$CONFIG" \
     --to-stage split_dataset --verbose
exit $?
