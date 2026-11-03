#!/usr/bin/env bash
# Bounded checkpoint evaluation; training is never changed or resumed here.
source /etc/profile
set -euo pipefail
module load gcc/11.4.0 cuda/12.1
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the frozen evaluation checkout}"
export PATH="/data02/home/${USER}/venvs/junqi-atx/bin:${PATH}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/junqi_rl:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
cd "$REPO_ROOT"
PHASE="${1:?current or final}"
MANIFEST="${2:?absolute stage manifest}"
OUTPUT="${3:?new output directory}"
test -f "$MANIFEST"
test ! -e "$OUTPUT"
case "$PHASE" in
  current)
    python scripts/evaluate_checkpoint_stages.py --manifest "$MANIFEST" \
      --pairs 975k:660k --games 4 --num-envs 4 --output "${OUTPUT}-smoke"
    python scripts/evaluate_checkpoint_stages.py --manifest "$MANIFEST" \
      --pairs 975k:660k --output "$OUTPUT"
    ;;
  final)
    RUN_ROOT="${4:?original training checkout}"
    EXPERIMENT="${5:?training experiment relative path}"
    ARCHIVE="${6:?new archive directory}"
    SOURCE="${7:?original training source commit}"
    python scripts/archive_completed_run.py --run-root "$RUN_ROOT" \
      --experiment "$EXPERIMENT" --output "$ARCHIVE" --expected-source-commit "$SOURCE"
    python scripts/evaluate_checkpoint_stages.py --manifest "$MANIFEST" \
      --pairs 1m:660k 1m:975k --output "$OUTPUT"
    ;;
  *) printf '%s\n' 'Expected phase current or final' >&2; exit 2 ;;
esac
