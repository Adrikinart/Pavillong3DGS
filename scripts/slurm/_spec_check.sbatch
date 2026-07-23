#!/usr/bin/env bash
#SBATCH --job-name=v2gs-speccheck
#SBATCH --output=/home/AdrienK/my3DGS/experiments/slurm_logs/%x_%j.out
#SBATCH --error=/home/AdrienK/my3DGS/experiments/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
set -uo pipefail
cd /home/AdrienK/my3DGS
export V2GS_ENV_PREFIX=/home/AdrienK/envs/v2gs-cu128
source scripts/_activate_env.sh
python scripts/_specular_check.py
