#!/bin/bash
# Submit from the frozen source checkout. Scheduler owns CUDA_VISIBLE_DEVICES.
source /etc/profile
set -euo pipefail
module load gcc/11.4.0 cuda/12.1
source /data02/home/scv7tsq/venvs/junqi-atx/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="$PWD:$PWD/junqi_rl${PYTHONPATH:+:$PYTHONPATH}"
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((19000 + SLURM_JOB_ID % 20000))
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NODES=${SLURM_JOB_NUM_NODES}
MODE=${1:-acceptance}
ARTIFACTS="$PWD/exps/ddp8_acceptance_20261105"
mkdir -p "$ARTIFACTS"
git rev-parse HEAD > "$ARTIFACTS/source_commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$ARTIFACTS/allocation-$SLURM_JOB_ID.txt"
srun --ntasks="$NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c \
  'hostname; nvidia-smi --query-gpu=name,uuid,memory.total --format=csv; python -c "import torch; print(torch.__version__, torch.cuda.device_count())"' \
  > "$ARTIFACTS/hardware-$SLURM_JOB_ID.txt"
launch() {
  local per_node=$1
  shift
  srun --ntasks="$NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c \
    'per_node=$1; shift; exec python -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node="$per_node" --node-rank="$SLURM_PROCID" --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" "$@"' \
    _ "$per_node" "$@"
}
if [[ "$MODE" == acceptance || "$MODE" == acceptance2 || "$MODE" == campaign ]]; then
  test "$NODES" = 1  # BSCC-N26 account policy permits single-node jobs only.
  worlds="2 8"
  if [[ "$MODE" == acceptance2 ]]; then worlds=2; fi
  if [[ "$MODE" == campaign ]]; then worlds=8; fi
  for world in $worlds; do
    per_node=$((world / NODES))
    target="$ARTIFACTS/world$world"
    test ! -e "$target" || { echo "Refusing to overwrite acceptance $target"; exit 2; }
    for stage in continuous initial resume; do
      launch "$per_node" scripts/accept_distributed_reference.py --mode "$stage" --output "$target"
    done
    python scripts/audit_distributed_reference.py "$target" --world "$world"
  done
  if [[ "$MODE" == acceptance2 ]]; then
    python -m pytest tests/test_ataraxos_reference_gpu.py tests/test_optimizer_resume_devices.py -q > "$ARTIFACTS/single-gpu-regression.txt"
    exit 0
  fi
  launch "$GPUS_PER_NODE" scripts/accept_distributed_reference.py --mode benchmark --output "$ARTIFACTS/full"
  python scripts/estimate_reference_runtime.py "$ARTIFACTS/full/benchmark"
  python -m pytest tests/test_ataraxos_reference_gpu.py tests/test_optimizer_resume_devices.py -q > "$ARTIFACTS/single-gpu-regression.txt"
  python -c "import json,pathlib; p=pathlib.Path('$ARTIFACTS'); assert all(json.loads((p/f'world{n}/acceptance.json').read_text())['passed'] for n in (2,8)); (p/'ACCEPTED.json').write_text(json.dumps({'job_id':'$SLURM_JOB_ID','passed':True,'world_sizes':[2,8]}))"
  if [[ "$MODE" == campaign ]]; then
    test ! -e exps/ataraxos_reference_8gpu_20261105/control
    launch "$GPUS_PER_NODE" scripts/train.py --config configs/v100_ataraxos_reference_8gpu.yaml
  fi
elif [[ "$MODE" == production ]]; then
  test -f "$ARTIFACTS/ACCEPTED.json"
  test ! -e exps/ataraxos_reference_8gpu_20261105/control
  launch "$GPUS_PER_NODE" scripts/train.py --config configs/v100_ataraxos_reference_8gpu.yaml
else
  echo "Unknown mode: $MODE"; exit 2
fi
