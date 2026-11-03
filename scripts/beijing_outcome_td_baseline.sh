#!/usr/bin/env bash
# One V100 allocation: acceptance, fresh 20k games, then frozen stage matches.
source /etc/profile
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
CFG=configs/v100_outcome_td_baseline.yaml
bash scripts/beijing_same_seat_raw.sh "$CFG"
module load gcc/11.4.0 cuda/12.1
export PATH="/data02/home/${USER}/venvs/junqi-atx/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/junqi_rl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
ROOT="$REPO_ROOT/exps/outcome_td_baseline_20261103"
python - "$ROOT" <<'PY'
import hashlib, json, pathlib, subprocess, sys
root = pathlib.Path(sys.argv[1])
run = root / 'control'
result = json.loads((run / 'training_result.json').read_text())
assert result['status'] == 'target_reached' and result['completed_games'] >= 20000
checkpoints = {}
for n in (5000, 10000, 20000):
    metrics = json.loads((run / f'first_move_{n:06d}.json').read_text())
    path = pathlib.Path(metrics['checkpoint'])
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    assert path.is_file(), path
    checkpoints[f'g{n}'] = dict(path=str(path), completed_games=metrics['completed_games'],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
manifest = dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    lineage_id=result['lineage_id'], checkpoints=checkpoints)
(root / 'stages-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
python scripts/evaluate_checkpoint_stages.py \
  --manifest "$ROOT/stages-manifest.json" --output "$ROOT/stage-matches" \
  --pairs g10000:g5000 g20000:g5000 g20000:g10000 \
  --games 400 --num-envs 64 > "$ROOT/stage-matches.log" 2>&1
