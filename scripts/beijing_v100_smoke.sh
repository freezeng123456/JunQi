#!/usr/bin/env bash

# BSCC's login profile reads a few optional terminal variables. Load it before
# enabling nounset so a non-interactive Slurm batch shell starts reliably.
source /etc/profile
set -euo pipefail

# One-card Beijing BSCC acceptance run. The allocation is deliberately
# bounded by completed games, not by an assistant-invented wall-clock cap.
# The long campaign uses the same environment variables and build gate.

module load gcc/11.4.0 cuda/12.1
module load cmake/3.23.4

RUN_ROOT="${RUN_ROOT:-/data02/home/${USER}/JunQi_ataraxos_strict_20261024}"
VENV="${VENV:-/data02/home/${USER}/venvs/junqi-atx}"
REPO_ROOT="${RUN_ROOT}"
SMOKE_DIR="${SMOKE_DIR:-${REPO_ROOT}/exps/v100_ataraxos_strict_smoke}"
export SMOKE_DIR

export PATH="${VENV}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUDA_ARCH=70
export CMAKE_CUDA_COMPILER="$(command -v nvcc)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/junqi_rl${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${SMOKE_DIR}"
python - <<'PY'
import json
import torch

assert torch.cuda.is_available(), "CUDA is not available"
assert torch.cuda.device_count() == 1, "smoke expects one GPU"
x = torch.randn(128, 128, device="cuda")
y = x @ x
assert torch.isfinite(y).all(), "CUDA probe produced non-finite values"
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "capability": torch.cuda.get_device_capability(),
    "arches": torch.cuda.get_arch_list(),
}))
PY

python -m pip install --no-deps -e "${REPO_ROOT}"
python build_cuda.py > "${SMOKE_DIR}/build.log" 2>&1
python -c 'import junqi_cuda; assert junqi_cuda.get_gpu_count() == 1; print("CUDA_EXTENSION_PASS")'
python -m pytest -q \
  tests/test_game_progress.py \
  tests/test_gpu_obs_parity.py \
  tests/test_compact_rollout_history.py \
  tests/test_gpu_rollout.py \
  > "${SMOKE_DIR}/tests.log" 2>&1

python scripts/train.py \
  --config configs/v100_ataraxos_strict.yaml \
  --target_completed_games 128 \
  --total_rollouts 32 \
  --save_dir "${SMOKE_DIR}" \
  --set \
    env__num_envs=16 \
    env__steps_per_env=64 \
    env__max_num_moves=64 \
    ppo__minibatch_size=16 \
    ppo__value_minibatch_size=16 \
    ppo__act_chunk_size=8 \
    arr__n_arr=16 \
    arr__pool_size=64 \
    arr__refresh_every=2 \
    save_every_games=64 \
    eval_every_games=64 \
    eval_num_games=8 \
    eval_enabled=false \
  > "${SMOKE_DIR}/train.log" 2>&1

test -s "${SMOKE_DIR}/training_games.jsonl"
test -s "${SMOKE_DIR}/game_progress.jsonl"
python scripts/analyze_first_moves.py "${SMOKE_DIR}/training_games.jsonl" \
  > "${SMOKE_DIR}/first_moves.json"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["SMOKE_DIR"])
summary = json.loads((root / "first_moves.json").read_text())
assert summary["games_with_first_move"] > 0
assert summary["games_missing_first_move"] == 0
print(json.dumps({"status": "PASS", **summary}, ensure_ascii=False))
PY
