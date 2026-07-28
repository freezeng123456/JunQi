#!/bin/bash
# Bootstrap H20 training env (torch + junqi_cuda). Run once per machine.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3.11}"
VENV="${VENV:-$REPO_ROOT/.venv}"

if ! command -v "$PYTHON" >/dev/null; then
  echo "Need $PYTHON (dnf install python3.11 python3.11-devel)"
  exit 1
fi

if ! command -v nvcc >/dev/null && [[ -x /usr/local/cuda-12.2/bin/nvcc ]]; then
  export PATH="/usr/local/cuda-12.2/bin:$PATH"
fi
if ! command -v nvcc >/dev/null; then
  echo "Installing cuda-nvcc-12-2 ..."
  dnf install -y cuda-nvcc-12-2 cuda-cudart-devel-12-2
  export PATH="/usr/local/cuda-12.2/bin:$PATH"
fi

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel
"$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cu121
"$VENV/bin/pip" install numpy pydantic pyyaml tqdm tensorboard pybind11 cmake pytest

export CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/usr/local/cuda-12.2/bin/nvcc}"
export CUDA_ARCH="${CUDA_ARCH:-90}"
"$VENV/bin/python" build_cuda.py

echo "OK: $("$VENV/bin/python" -c "import torch,junqi_cuda; print(torch.cuda.get_device_name(0), 'junqi_cuda', junqi_cuda.get_gpu_count())")"
