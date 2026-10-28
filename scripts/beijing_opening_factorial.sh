#!/usr/bin/env bash
# One arm per allocation; no invented wall-clock limit or dependent job chain.
source /etc/profile
set -euo pipefail
module load gcc/11.4.0 cuda/12.1 cmake/3.23.4
ARM="${1:?Pass control, low_lr, all_samples, or all_samples_low_lr}"
case "$ARM" in control|low_lr|all_samples|all_samples_low_lr) ;; *) exit 2 ;; esac
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/data02/home/${USER}/venvs/junqi-atx}"
export PATH="${VENV}/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/junqi_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
CFG="configs/v100_opening_factorial_${ARM}.yaml"
ROOT="${REPO_ROOT}/exps/opening_factorial_20261028"
GATE="${ROOT}/acceptance_${ARM}"
RUN="${ROOT}/${ARM}"
# Refuse to overwrite any earlier training or test attempt.
mkdir -p "$ROOT"
mkdir "$GATE"
test ! -e "$RUN"
git rev-parse HEAD > "$GATE/source_commit.txt"
git archive HEAD | sha256sum > "$GATE/source_archive.sha256"
python - <<'PY' > "$GATE/gpu.json"
import hashlib, json, pathlib, torch, junqi_cuda
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability() == (7, 0), "Expected V100"
x = torch.randn(128, 128, device='cuda')
assert torch.isfinite(x @ x).all()
binary = pathlib.Path(junqi_cuda.__file__)
print(json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
    cuda=torch.version.cuda, binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())))
PY
python -m pytest -q tests/test_first_move_metrics.py tests/test_game_progress.py \
  tests/test_game_progress_gpu.py tests/test_timestep_minibatches.py \
  > "$GATE/tests.log" 2>&1

# Full production network and 128-env learner rows: catches all-sample OOMs.
# Short game length is restricted to this acceptance process.
python scripts/train.py --config "$CFG" --save_dir "$GATE/train" \
  --target_completed_games 128 --total_rollouts 4 --set \
  env__steps_per_env=64 env__max_num_moves=64 save_every_games=128 \
  arr__n_arr=16 arr__pool_size=64 arr__refresh_every=2 \
  > "$GATE/train.log" 2>&1
python - "$GATE/train" <<'PY' > "$GATE/acceptance.json"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
result = json.loads((root / 'training_result.json').read_text())
assert result['status'] == 'target_reached', result['status']
metrics = json.loads((root / 'first_move_000128.json').read_text())
assert metrics['cumulative']['games'] >= 128
assert metrics['cumulative']['missing_first_actions'] == 0
assert metrics['cumulative']['missing_first_policy'] == 0
print(json.dumps(dict(status='PASS', completed_games=result['completed_games'])))
PY
python scripts/train.py --config "$CFG" > "$ROOT/${ARM}.train.log" 2>&1
python - "$RUN" <<'PY' > "$ROOT/${ARM}.completion.json"
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
r = json.loads((root / 'training_result.json').read_text())
assert r['status'] == 'target_reached' and r['completed_games'] >= 20000
for milestone in (5000, 10000, 20000):
    m = json.loads((root / f'first_move_{milestone:06d}.json').read_text())
    assert pathlib.Path(m['checkpoint']).is_file()
    assert m['cumulative']['missing_first_policy'] == 0
names = ['training_games.jsonl', 'game_progress.jsonl', 'checkpoint_games.jsonl',
         'training_result.json', 'first_move_005000.json', 'first_move_010000.json',
         'first_move_020000.json']
print(json.dumps(dict(status='COMPLETE', completed_games=r['completed_games'],
    sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}), indent=2))
PY
