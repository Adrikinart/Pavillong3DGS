#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Smallest meaningful end-to-end run. Verifies env -> CUDA -> extract -> filter
# -> COLMAP -> normalize -> split -> short train -> checkpoint -> validation
# render -> metrics -> evaluate -> export -> report.
#
# Runs the GPU stages on a Blackwell node (GPURACK5) via srun. On success writes
# a completion marker and exits 0; the exit code + marker are machine-verifiable.
#
# Usage: scripts/run_smoke_test.sh [PARTITION] [NODE]
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PART="${1:-rtxpro}"
NODE="${2:-GPURACK5}"
CONFIG="configs/pipeline/smoke_test.yaml"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
PY="${ENV_PREFIX}/bin/python"
MARKER="experiments/SMOKE_TEST_OK"
rm -f "$MARKER"
mkdir -p experiments/slurm_logs

if [ ! -x "$PY" ]; then
  echo "ERROR: env python not found at $PY. Run scripts/bootstrap_environment.sh first." >&2
  exit 2
fi

echo "[smoke] 1/3 GPU environment check on ${NODE}"
srun --partition="$PART" --nodelist="$NODE" --gres=gpu:1 --time=00:20:00 --cpus-per-task=4 \
  bash -lc "cd $(pwd); export TORCH_EXTENSIONS_DIR=/var/tmp/$USER/torch_ext; mkdir -p \$TORCH_EXTENSIONS_DIR; \
            $PY -m video_to_3dgs.cli inspect-env --gpu-check --out experiments/env_${NODE}.json" \
  || { echo "[smoke] GPU env check FAILED"; exit 3; }

echo "[smoke] 2/3 end-to-end pipeline (run-all) on ${NODE}"
srun --partition="$PART" --nodelist="$NODE" --gres=gpu:1 --time=02:00:00 --cpus-per-task=8 \
  bash -lc "cd $(pwd); \
            export TORCH_EXTENSIONS_DIR=/var/tmp/$USER/torch_ext; mkdir -p \$TORCH_EXTENSIONS_DIR; \
            export TMPDIR=/var/tmp/$USER; mkdir -p \$TMPDIR; \
            export TORCH_CUDA_ARCH_LIST=\$($PY -c 'import torch;p=torch.cuda.get_device_properties(0);print(f\"{p.major}.{p.minor}\")'); \
            $PY -m video_to_3dgs.cli run-all --config $CONFIG --verbose" \
  || { echo "[smoke] pipeline FAILED"; exit 4; }

echo "[smoke] 3/3 verify outputs"
DID=$("$PY" -m video_to_3dgs.cli status --config "$CONFIG" 2>/dev/null | head -1 | awk '{print $2}')
RUN="experiments/runs/${DID}"
FAIL=0
"$PY" -m video_to_3dgs.cli status --config "$CONFIG" | tail -n +2
# require key artifacts
for pat in "$RUN"/trainings/*/checkpoints/ckpt_latest.pt \
           "$RUN"/trainings/*/eval.json \
           "$RUN"/exports/*/point_cloud.ply \
           "$RUN"/trainings/*/report/report.md; do
  if ! ls $pat >/dev/null 2>&1; then echo "[smoke] MISSING: $pat"; FAIL=1; fi
done
if [ "$FAIL" -eq 0 ]; then
  echo "SMOKE_TEST_OK $(date -u +%FT%TZ) dataset=$DID" > "$MARKER"
  echo "[smoke] SUCCESS -> $MARKER"
  exit 0
fi
echo "[smoke] FAILED: required artifacts missing"
exit 5
