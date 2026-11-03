#!/usr/bin/env bash
# One allocation: build, numerical/resume acceptance, full-size smoke, fresh 20k.
source /etc/profile
set -euo pipefail
module load gcc/11.4.0 cuda/12.1 cmake/3.23.4
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
export PATH="/data02/home/${USER}/venvs/junqi-atx/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/junqi_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export CUDA_ARCH=70 CMAKE_CUDA_COMPILER="$(command -v nvcc)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ROOT="$REPO_ROOT/exps/ataraxos_reference_20261103"
GATE="$ROOT/acceptance"
mkdir "$GATE"
test ! -e "$ROOT/control"
git rev-parse HEAD > "$GATE/source_commit.txt"
git archive HEAD | sha256sum > "$GATE/source_archive.sha256"
python build_cuda.py > "$GATE/build.log" 2>&1
python - <<'PY' > "$GATE/gpu.json"
import hashlib,json,pathlib,torch,junqi_cuda
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert torch.cuda.get_device_capability()==(7,0)
binary=pathlib.Path(junqi_cuda.__file__)
print(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),
    cuda=torch.version.cuda,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())))
PY
python -m pytest -q tests/test_ataraxos_reference.py tests/test_ataraxos_reference_gpu.py \
  tests/test_outcome_td.py tests/test_seat_returns.py tests/test_seat_pending.py \
  tests/test_raw_advantage.py tests/test_behavior_policy_reference.py \
  tests/test_same_seat_buffer.py tests/test_same_seat_gpu.py tests/test_first_move_metrics.py \
  tests/test_game_progress.py tests/test_game_progress_gpu.py tests/test_timestep_minibatches.py \
  > "$GATE/tests.log" 2>&1
python - "$GATE" <<'PY'
import pathlib,sys,yaml
from junqi_rl.training.config import load_yaml_config
p=pathlib.Path(sys.argv[1]);cfg=load_yaml_config('configs/v100_ataraxos_reference.yaml')
# Same 1536 environments, 400 rows, both full networks, setup banks and batch
# sizes. Only game duration/target/output differ for this disposable smoke.
cfg.update(reference_profile='',save_dir=str(p/'train'),target_completed_games=1536,
    save_every_games=1536,total_rollouts=1,checkpoint_keep=1)
cfg['env']['max_num_moves']=64
(p/'smoke.yaml').write_text(yaml.safe_dump(cfg))
PY
python scripts/train.py --config "$GATE/smoke.yaml" > "$GATE/train.log" 2>&1
python - "$GATE" <<'PY' > "$GATE/acceptance.json"
import hashlib,json,pathlib,sys,torch
root=pathlib.Path(sys.argv[1]);run=root/'train'
r=json.loads((run/'training_result.json').read_text())
assert r['status']=='target_reached' and r['completed_games']>=1536
checkpoint=(run/'ckpt_latest.pt').resolve()
s=torch.load(checkpoint,map_location='cpu',weights_only=False)
def digest(weights):
 h=hashlib.sha256()
 for name,tensor in sorted(weights.items()):
  assert torch.isfinite(tensor).all(),name
  h.update(name.encode());h.update(tensor.contiguous().numpy().tobytes())
 return h.hexdigest()
raw=digest(s['policy']);shadow=digest(s['ema']['model'])
arr=s['arrangement']['trainer'];setup=digest(arr['net']);setup_shadow=digest(arr['ema']['model'])
assert raw!=json.loads((run/'policy_initialization.json').read_text())['parameter_sha256']
assert setup!=json.loads((run/'arrangement_initialization.json').read_text())['parameter_sha256']
assert s['ema']['updates']==s['num_rollout']==1 and arr['ema']['updates']==1
assert s['num_train_step']>0 and arr['total_batches']>0
assert s['nan_skip_count']==s['grad_skip_count']==arr['grad_nan_skip_count']==0
assert 'advantage_history' in s['runtime_state']
assert s['runtime_state']['world']['independent_setup_seats']
m=json.loads((run/'first_move_001536.json').read_text())
assert m['cumulative']['missing_first_actions']==m['cumulative']['missing_first_policy']==0
result=dict(status='PASS',games=r['completed_games'],num_envs=1536,rows=400,
    move_updates=s['num_train_step'],setup_updates=arr['total_batches'],
    move_raw_sha256=raw,move_ema_sha256=shadow,setup_raw_sha256=setup,setup_ema_sha256=setup_shadow,
    checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest())
# Reclaim only this script's disposable smoke checkpoints after verification;
# production models and all smoke evidence/config/logs are retained.
del s,arr
for p in run.glob('ckpt_*.pt'):p.unlink()
print(json.dumps(result,indent=2))
PY
python scripts/train.py --config configs/v100_ataraxos_reference.yaml > "$ROOT/control.train.log" 2>&1
python - "$ROOT/control" <<'PY' > "$ROOT/completion.json"
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);r=json.loads((root/'training_result.json').read_text())
assert r['status']=='target_reached' and r['completed_games']>=20000
milestones={}
for target in (5000,10000,20000):
 p=root/f'first_move_{target:06d}.json';m=json.loads(p.read_text())
 assert m['completed_games']>=target and pathlib.Path(m['checkpoint']).is_file()
 assert m['cumulative']['missing_first_actions']==m['cumulative']['missing_first_policy']==0
 milestones[target]=dict(actual_games=m['completed_games'],metrics_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(dict(status='COMPLETE',completed_games=r['completed_games'],milestones=milestones),indent=2))
PY
