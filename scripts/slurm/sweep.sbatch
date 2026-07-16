#!/usr/bin/env bash
#SBATCH --job-name=v2gs-sweep
#SBATCH --partition=rtxpro
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120
#SBATCH --array=0-3
#SBATCH --output=experiments/slurm_logs/%x_%A_%a.out
#SBATCH --error=experiments/slurm_logs/%x_%A_%a.err
# ---------------------------------------------------------------------------
# Parallel experiment sweep via a Slurm job array. Each array task trains one
# variant (different seeds / configs) against the SAME shared dataset build, so
# COLMAP/masks are computed once by preprocess.sbatch beforehand.
#
# Usage:
#   sbatch scripts/slurm/preprocess.sbatch configs/pipeline/scene_pavillon.yaml   # once
#   sbatch scripts/slurm/sweep.sbatch      configs/pipeline/scene_pavillon.yaml
# Edit VARIANTS below to define the sweep axis.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p experiments/slurm_logs
CONFIG="${1:-${V2GS_CONFIG:-configs/pipeline/scene_pavillon.yaml}}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
source scripts/_activate_env.sh

# one --set override per array index (seed sweep example)
VARIANTS=(
  "train.seed=0"
  "train.seed=1"
  "train.seed=2"
  "train.seed=3"
)
OVERRIDE="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
RUN_ID="sweep_${SLURM_ARRAY_TASK_ID}_$(echo "$OVERRIDE" | tr '=.' '__')"
echo "[sweep] task=$SLURM_ARRAY_TASK_ID override=$OVERRIDE run_id=$RUN_ID"

srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli run-all --config "$CONFIG" \
     --from-stage train --set "$OVERRIDE" --train-run-id "$RUN_ID" --verbose
exit $?
