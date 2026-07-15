#!/usr/bin/env bash
# Inspect the cluster environment (login node + one GPU node via srun).
# Writes JSON reports under experiments/.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
PY="${ENV_PREFIX}/bin/python"
[ -x "$PY" ] || PY="python3"
mkdir -p experiments

echo "===== login node ====="
hostname; uname -a
echo "--- scheduler ---"; sinfo -s 2>/dev/null || echo "no slurm"
echo "--- storage ---"; df -h /home /var/tmp 2>/dev/null
"$PY" -m video_to_3dgs.cli inspect-env --out experiments/env_login.json 2>/dev/null || \
  echo "(framework env inspection needs the v2gs env / package installed)"

PART="${1:-rtxpro}"
NODE="${2:-GPURACK5}"
echo "===== GPU node ${NODE} (${PART}) via srun ====="
srun --partition="$PART" --nodelist="$NODE" --gres=gpu:1 --time=00:15:00 --cpus-per-task=4 \
  bash -lc "cd $(pwd); ${ENV_PREFIX}/bin/python -m video_to_3dgs.cli inspect-env --gpu-check \
    --out experiments/env_${NODE}.json" \
  || echo "GPU srun inspection failed (env not ready?)"
echo "reports: experiments/env_login.json experiments/env_${NODE}.json"
