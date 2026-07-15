#!/usr/bin/env bash
# Convenience wrapper to run the full pipeline for a config, locally (inside an
# existing GPU allocation) or by submitting Slurm jobs.
#
# Usage:
#   scripts/run_pipeline.sh local    configs/pipeline/scene_pavillon.yaml
#   scripts/run_pipeline.sh slurm    configs/pipeline/scene_pavillon.yaml
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODE="${1:-local}"
CONFIG="${2:-configs/pipeline/scene_pavillon.yaml}"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
PY="${ENV_PREFIX}/bin/python"

case "$MODE" in
  local)
    export TORCH_EXTENSIONS_DIR="/var/tmp/$USER/torch_ext"; mkdir -p "$TORCH_EXTENSIONS_DIR"
    "$PY" -m video_to_3dgs.cli run-all --config "$CONFIG" --verbose
    ;;
  slurm)
    echo "submitting preprocess..."
    PRE=$(sbatch --parsable scripts/slurm/preprocess.sbatch "$CONFIG")
    echo "preprocess job: $PRE"
    echo "submitting train (after preprocess)..."
    TRAIN=$(sbatch --parsable --dependency=afterok:$PRE scripts/slurm/train.sbatch "$CONFIG")
    echo "train job: $TRAIN"
    echo "monitor with: squeue -u $USER ; tail -f experiments/slurm_logs/*_${TRAIN}.out"
    ;;
  *)
    echo "unknown mode: $MODE (use 'local' or 'slurm')"; exit 1 ;;
esac
