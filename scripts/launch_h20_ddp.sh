#!/bin/bash
# launch_h20_ddp.sh — Launch a SINGLE DDP training run across N GPUs.
#
# Unlike launch_h20_8gpu.sh (which runs 8 INDEPENDENT experiments, one per
# GPU), this launcher runs ONE experiment whose gradients are synchronously
# all-reduced across all N ranks via NCCL.  Use this when you have a config
# you trust and want to scale data-parallel throughput linearly.
#
# Usage:
#     bash scripts/launch_h20_ddp.sh <cfg_path> [nproc] [extra args...]
#
# Examples:
#     # 8-GPU DDP run on h20_ddp_exp5_selfplay
#     bash scripts/launch_h20_ddp.sh exps/h20_ddp_exp5_selfplay/cfg.yaml 8
#
#     # 2-GPU DDP run, GPUs 4 and 5
#     CUDA_VISIBLE_DEVICES=4,5 bash scripts/launch_h20_ddp.sh \
#         exps/h20_ddp_exp4_klanchor/cfg.yaml 2
#
#     # Resume from a checkpoint (e.g. 2-card ckpt → 8-card scale-up)
#     bash scripts/launch_h20_ddp.sh exps/h20_ddp_winner/cfg.yaml 8 \
#         --resume_cli exps/h20_ddp_winner/ckpt_latest.pt
#
# Background notes:
#   * Each rank runs ``cfg.env.num_envs`` parallel envs locally; effective
#     env count is ``world_size × cfg.env.num_envs``. Gradients are
#     all-reduced after each backward via DistributedDataParallel.
#   * Per-rank seeds are offset by ``global_rank * 10_000`` inside
#     scripts/train.py so each rank explores a distinct trajectory slice.
#   * Checkpoints are written by rank 0 only; their state_dict is the
#     UNWRAPPED policy/EMA, so they can be reloaded by any world_size
#     (including single-process resume for diagnostics).
#   * ``train.log`` is rank 0's stdout; non-rank-0 ranks tee into
#     ``train_rank{N}.log`` to keep tracebacks separated.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    cat <<'EOF'
Usage: bash scripts/launch_h20_ddp.sh <cfg_path> [nproc=auto] [extra args...]

Required:
  cfg_path    Path to a YAML training config (e.g. exps/h20_ddp_exp5_selfplay/cfg.yaml)

Optional:
  nproc       Number of GPU ranks (default: detect from CUDA_VISIBLE_DEVICES,
              or fall back to all visible nvidia-smi GPUs)
  extra...    Additional flags forwarded to scripts/train.py
              (e.g. --resume_cli /path/to/ckpt.pt --total_rollouts 1000)
EOF
    exit 2
fi

CFG_PATH="$1"
shift

if [[ ! -f "$CFG_PATH" ]]; then
    echo "[launch] Config not found: $CFG_PATH"
    exit 1
fi

# ---- Resolve nproc ----------------------------------------------------------
NPROC="${1:-}"
if [[ -n "${NPROC:-}" && "$NPROC" =~ ^[0-9]+$ ]]; then
    shift
else
    NPROC=""
fi
if [[ -z "$NPROC" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
    elif command -v nvidia-smi >/dev/null 2>&1; then
        NPROC=$(nvidia-smi --list-gpus | wc -l)
    else
        echo "[launch] Cannot auto-detect GPU count; pass nproc explicitly."
        exit 1
    fi
fi
if (( NPROC < 1 )); then
    echo "[launch] Refusing to launch with nproc=$NPROC"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- Pick a master_port that doesn't collide with concurrent runs -----------
# torchrun's --standalone defaults to a fixed port (29500), which collides
# when launching multiple DDP groups on the same host (the B2 phase below
# launches 4 concurrent 2-rank groups — they MUST have different ports).
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

echo "[launch] Repo:        $REPO_ROOT"
echo "[launch] Config:      $CFG_PATH"
echo "[launch] nproc:       $NPROC"
echo "[launch] master_port: $MASTER_PORT"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
echo "[launch] extra args:  $*"

# ---- Run --------------------------------------------------------------------
# --standalone: torchrun spins up the rendezvous server itself; no etcd needed.
# --nproc_per_node: number of GPU processes for this node.
# --master_port: keeps multiple concurrent DDP groups from clobbering each other.
PYTHON_BIN="${PYTHON_EXECUTABLE:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[launch] Python interpreter not found: $PYTHON_BIN"
    exit 1
fi
echo "[launch] Python:      $PYTHON_BIN"

# Invoke the module through the selected interpreter instead of relying on a
# torchrun console-script shebang, which is often stale after a shared conda
# environment is moved or mounted on another host.
exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    scripts/train.py \
    --config "$CFG_PATH" \
    "$@"
