#!/usr/bin/env bash
#SBATCH --job-name=v2gs-eval
#SBATCH --partition=rtxpro
#SBATCH --nodelist=GPURACK5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=experiments/slurm_logs/%x_%j.out
#SBATCH --error=experiments/slurm_logs/%x_%j.err
# ---------------------------------------------------------------------------
# Evaluate + export + report for an already-trained run.
# Usage:  sbatch scripts/slurm/evaluate.sbatch <config> [train_run_id]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p experiments/slurm_logs
CONFIG="${1:-${V2GS_CONFIG:-configs/pipeline/scene_pavillon.yaml}}"
TR_ID="${2:-}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
source scripts/_activate_env.sh
TR_ARG=""; [ -n "$TR_ID" ] && TR_ARG="--train-run-id $TR_ID"
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli evaluate --config "$CONFIG" $TR_ARG --verbose
srun "$ENV_PREFIX/bin/python" -m video_to_3dgs.cli export   --config "$CONFIG" $TR_ARG --verbose
exit $?
