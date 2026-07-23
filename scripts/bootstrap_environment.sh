#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Bootstrap the `v2gs` GPU environment for the video_to_3dgs framework.
#
# Design constraints (isiacluster, see CLAUDE.md):
#   * Conda on the NFS /home mount is pathologically slow on the LOGIN node, but
#     a fast local-disk mamba lives at /var/tmp/$USER/miniforge3.
#   * The env must live on /home so the healthy-NFS GPU nodes can import it.
#   * Blackwell (sm_120) needs a cu128 PyTorch build; gsplat compiles CUDA on
#     first import using the in-env nvcc.
#
# This script runs on the LOGIN node (has internet). It is idempotent: re-running
# updates the env in place. All output is teed to a build log.
# ------------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
MAMBA_ROOT="${V2GS_MAMBA_ROOT:-/var/tmp/$USER/miniforge3}"
LOG_DIR="${REPO_ROOT}/experiments/bootstrap_logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/bootstrap_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[bootstrap $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "repo=$REPO_ROOT env_prefix=$ENV_PREFIX mamba_root=$MAMBA_ROOT"
log "log file: $LOG"

# --- locate a fast mamba/conda -------------------------------------------------
MAMBA_BIN=""
if [[ -x "${MAMBA_ROOT}/bin/mamba" ]]; then
    MAMBA_BIN="${MAMBA_ROOT}/bin/mamba"
elif [[ -x "${MAMBA_ROOT}/condabin/mamba" ]]; then
    MAMBA_BIN="${MAMBA_ROOT}/condabin/mamba"
elif command -v mamba >/dev/null 2>&1; then
    MAMBA_BIN="$(command -v mamba)"
elif [[ -x "${MAMBA_ROOT}/bin/conda" ]]; then
    MAMBA_BIN="${MAMBA_ROOT}/bin/conda"
elif [[ -x "${MAMBA_ROOT}/condabin/conda" ]]; then
    MAMBA_BIN="${MAMBA_ROOT}/condabin/conda"
elif command -v conda >/dev/null 2>&1; then
    # Last resort, and worth having: the default MAMBA_ROOT lives under /var/tmp, which is
    # node-local and periodically cleaned, so the miniforge install this script was written
    # against can simply vanish. A conda anywhere on PATH is slower to solve but correct,
    # and /home conda works again since the 2026-07-15 NFS fix.
    MAMBA_BIN="$(command -v conda)"
fi
if [[ -z "$MAMBA_BIN" ]]; then
    log "ERROR: no mamba/conda found. Set V2GS_MAMBA_ROOT to a miniforge/miniconda install."
    log "  looked in: ${MAMBA_ROOT}/{bin,condabin}/{mamba,conda} and \$PATH"
    exit 2
fi
log "using solver: $MAMBA_BIN"
log "using solver: $MAMBA_BIN"

# --- create or update the env --------------------------------------------------
if [[ -d "$ENV_PREFIX" && -x "$ENV_PREFIX/bin/python" ]]; then
    log "env already exists at $ENV_PREFIX -> updating"
    "$MAMBA_BIN" env update -p "$ENV_PREFIX" -f "${REPO_ROOT}/environment.yml" 2>&1 | tee -a "$LOG"
else
    log "creating env at $ENV_PREFIX (this writes many files to NFS; may be slow)"
    "$MAMBA_BIN" env create -p "$ENV_PREFIX" -f "${REPO_ROOT}/environment.yml" 2>&1 | tee -a "$LOG"
fi
rc=${PIPESTATUS[0]:-$?}
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    log "ERROR: env creation failed (rc=$rc); no python at $ENV_PREFIX"
    exit 3
fi

PY="$ENV_PREFIX/bin/python"
log "python: $($PY --version 2>&1)"

# --- pip layer (torch cu128 + gsplat + ...) ------------------------------------
# Heavy /home writes are pathologically slow on the storming login node, so run
# this on a GPU node (healthy NFS + internet + nvcc) when Slurm is available and
# we have no local GPU. Single source of truth for the pip package set:
V2GS_PIP_PKGS="${V2GS_PIP_PKGS:-torch torchvision gsplat lpips rembg pillow-heif imageio imageio-ffmpeg}"
V2GS_GPU_PARTITION="${V2GS_GPU_PARTITION:-rtxpro}"
V2GS_GPU_NODE="${V2GS_GPU_NODE:-GPURACK5}"
# cu128 as the PRIMARY index so torch resolves to the cu128 build that MATCHES
# the in-env nvcc 12.8 (a cu130 torch + nvcc 12.8 mismatch breaks the gsplat build).
CU_INDEX="https://download.pytorch.org/whl/cu128"

pip_layer_cmd="source '$REPO_ROOT/scripts/_activate_env.sh'; \
'$ENV_PREFIX/bin/pip' install --index-url $CU_INDEX --extra-index-url https://pypi.org/simple $V2GS_PIP_PKGS"

if command -v nvidia-smi >/dev/null 2>&1; then
    log "local GPU detected -> installing pip layer locally"
    bash -lc "$pip_layer_cmd" 2>&1 | tee -a "$LOG"
elif command -v srun >/dev/null 2>&1; then
    log "no local GPU + Slurm present -> installing pip layer on $V2GS_GPU_NODE ($V2GS_GPU_PARTITION)"
    log "(this avoids the login-node NFS storm; gsplat builds for sm_120 on the GPU node)"
    srun --partition="$V2GS_GPU_PARTITION" --nodelist="$V2GS_GPU_NODE" --gres=gpu:1 \
         --time=01:00:00 --cpus-per-task=8 bash -lc "$pip_layer_cmd" 2>&1 | tee -a "$LOG"
else
    log "WARNING: no GPU and no Slurm -> installing pip layer locally (may be very slow on NFS)"
    bash -lc "$pip_layer_cmd" 2>&1 | tee -a "$LOG"
fi

# --- install the framework itself (editable, pure-python core) -----------------
log "installing video_to_3dgs (editable)"
"$ENV_PREFIX/bin/pip" install -e "$REPO_ROOT" --no-deps 2>&1 | tee -a "$LOG"

# --- record an exact lockfile --------------------------------------------------
log "writing requirements-lock.txt"
"$ENV_PREFIX/bin/pip" freeze > "${REPO_ROOT}/requirements-lock.txt" 2>>"$LOG" || true
"$MAMBA_BIN" list -p "$ENV_PREFIX" --explicit > "${LOG_DIR}/conda_explicit.txt" 2>>"$LOG" || true

# --- CPU-side torch sanity (login node has no GPU; just check import) ----------
log "torch import check (CPU login node — CUDA availability is False here):"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
try:
    import torch
    print("  torch", torch.__version__, "cuda_build", torch.version.cuda)
    # Blackwell kernels must be present in the wheel's arch list:
    try:
        archs = torch.cuda.get_arch_list()
        print("  arch_list", archs)
        print("  sm_120_present", any("120" in a for a in archs))
    except Exception as e:
        print("  arch_list check failed:", e)
except Exception as e:
    print("  torch import FAILED:", e)
try:
    import gsplat
    print("  gsplat", getattr(gsplat, "__version__", "?"), "(CUDA kernels JIT-compile on first GPU import)")
except Exception as e:
    print("  gsplat import note:", e)
PYEOF

log "DONE. Validate on a GPU node with:"
log "  srun --partition=rtxpro --nodelist=GPURACK5 --gres=gpu:1 --time=00:15:00 \\"
log "    $PY -m video_to_3dgs.cli inspect-env --gpu-check --out experiments/env_gpurack5.json"
log "bootstrap log saved to $LOG"
echo "BOOTSTRAP_EXIT_OK" | tee -a "$LOG"
