# Source this to prepare the v2gs env for GPU work (Blackwell sm_120).
#   source scripts/_activate_env.sh
#
# Ensures gsplat's CUDA extension can JIT-build: the conda CUDA 12.8 toolkit
# keeps its headers/libs under targets/x86_64-linux/{include,lib}, which is NOT
# on nvcc's default search path — hence the classic
#   "fatal error: cuda_runtime.h: No such file or directory".
# We add them via CUDA_HOME + CPATH + LIBRARY_PATH. The torch build must match
# this toolkit's CUDA major/minor (install torch from the cu128 index).
ENV_PREFIX="${V2GS_ENV_PREFIX:-/home/$USER/envs/v2gs}"
export PATH="$ENV_PREFIX/bin:$PATH"
export CUDA_HOME="$ENV_PREFIX"
_CUDA_T="$ENV_PREFIX/targets/x86_64-linux"
if [ -d "$_CUDA_T/include" ]; then
  export CPATH="$_CUDA_T/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$_CUDA_T/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$_CUDA_T/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/var/tmp/$USER/torch_ext}"
export TMPDIR="${TMPDIR:-/var/tmp/$USER}"
export MAX_JOBS="${MAX_JOBS:-8}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
# Blackwell arch for gsplat JIT (auto-detect when a GPU is visible)
if command -v nvidia-smi >/dev/null 2>&1 && [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="$("$ENV_PREFIX/bin/python" -c 'import torch;p=torch.cuda.get_device_properties(0);print(f"{p.major}.{p.minor}")' 2>/dev/null || echo '')"
fi
