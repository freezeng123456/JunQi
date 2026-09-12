#!/bin/bash
# Single-GPU engineering validation; submit with --time=00:20:00.
set -eo pipefail
source /etc/profile
set -u
module load gcc/11.4.0 cuda/12.1
VALIDATION_ROOT=/data02/run01/scv7tsq/junqi-engineering-20260912
VENV_ROOT=/data02/run01/scv7tsq/junqi-b347-20260911
cd "$VALIDATION_ROOT/source"
test -f "$VENV_ROOT/environment.ready"
export PATH="$VENV_ROOT/venv/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/junqi_rl"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_ARCH=70
export CMAKE_CUDA_COMPILER="$(command -v nvcc)"
trap 'rc=$?; printf "%s\n" "$rc" > "$VALIDATION_ROOT/job.exit"' EXIT
nvidia-smi > "$VALIDATION_ROOT/device.txt"
python -m pip freeze > "$VALIDATION_ROOT/requirements.txt"
printf 'building\n' > "$VALIDATION_ROOT/stage.txt"
timeout 360 python build_cuda.py > "$VALIDATION_ROOT/build.log" 2>&1
printf 'testing\n' > "$VALIDATION_ROOT/stage.txt"
timeout 600 python -m pytest -q -p no:cacheprovider \
  tests/test_gpu_obs_parity.py tests/test_gpu_combat_memory_parity.py \
  tests/test_compact_rollout_history.py tests/test_gpu_rollout.py \
  tests/test_combat_outcome_features.py tests/test_belief_refresh_constraints.py \
  tests/test_gpu_evaluation_fixed_games.py tests/test_random_eval_pairing.py \
  tests/test_evaluation_protocol.py \
  --junitxml="$VALIDATION_ROOT/gpu-tests.xml" > "$VALIDATION_ROOT/gpu-tests.log" 2>&1
printf 'measuring\n' > "$VALIDATION_ROOT/stage.txt"
timeout 180 python experiments/engineering_validation_20260912/probe.py "$VALIDATION_ROOT"
printf 'complete\n' > "$VALIDATION_ROOT/stage.txt"
