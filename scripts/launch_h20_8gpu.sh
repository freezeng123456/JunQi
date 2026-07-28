#!/bin/bash
# launch_h20_8gpu.sh — Launch 8 parallel independent training experiments,
# one per H20 GPU. No DDP, no cross-GPU comms — each experiment is fully
# self-contained. Rationale in docs/H20_DEPLOYMENT.md.
#
# Usage:
#     bash scripts/launch_h20_8gpu.sh              # full slate
#     bash scripts/launch_h20_8gpu.sh 0 3          # only GPUs 0-3 (first 4)
#     DRYRUN=1 bash scripts/launch_h20_8gpu.sh     # print what would run
#
# Each experiment logs to:
#     exps/<exp_name>/train.log       (detailed train output)
#     exps/h20_launch/<exp_name>.log  (stdout+stderr capture, for `tail -f`)
#
# To check progress later:
#     bash scripts/launch_h20_8gpu.sh --status
#     # or: cat exps/h20_launch/STATUS.md
#
# To kill everything launched by this script:
#     bash scripts/launch_h20_8gpu.sh --kill

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$REPO_ROOT/exps/h20_launch"
PID_FILE="$LAUNCH_DIR/pids.txt"
STATUS_FILE="$LAUNCH_DIR/STATUS.md"

# Map GPU index -> experiment config path.  Keep this table as the
# authoritative source; adding/removing experiments = editing this list.
declare -a EXPERIMENTS=(
    "h20_exp1_v32_bf16_base"
    "h20_exp2_belief_bigger"
    "h20_exp3_move_bigger"
    "h20_exp4_draw_penalty"
    "h20_exp5_self_play"
    "h20_exp6_arr_slow_refresh"
    "h20_exp7_longer_train"
    "h20_exp8_bigger_both"
)

mkdir -p "$LAUNCH_DIR"

# ---------- Subcommands -------------------------------------------------------

cmd_kill() {
    if [[ ! -f "$PID_FILE" ]]; then
        echo "[launch] No pid file at $PID_FILE; nothing to kill."
        exit 0
    fi
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[launch] Killing PID $pid"
            kill "$pid" || true
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    echo "[launch] Done. Check nvidia-smi to confirm GPUs released."
}

cmd_status() {
    if [[ ! -f "$STATUS_FILE" ]]; then
        echo "[launch] No status file yet. Launch something first."
        exit 1
    fi
    cat "$STATUS_FILE"
}

if [[ "${1:-}" == "--kill" ]]; then
    cmd_kill
    exit 0
fi
if [[ "${1:-}" == "--status" ]]; then
    cmd_status
    exit 0
fi

# ---------- Range selection ---------------------------------------------------

FIRST_GPU="${1:-0}"
LAST_GPU="${2:-7}"
if ! [[ "$FIRST_GPU" =~ ^[0-9]+$ && "$LAST_GPU" =~ ^[0-9]+$ ]]; then
    echo "[launch] Usage: $0 [first_gpu last_gpu]  (defaults: 0 7)"
    exit 2
fi
if (( LAST_GPU < FIRST_GPU )); then
    echo "[launch] last_gpu ($LAST_GPU) must be >= first_gpu ($FIRST_GPU)"
    exit 2
fi
if (( LAST_GPU >= ${#EXPERIMENTS[@]} )); then
    echo "[launch] Only ${#EXPERIMENTS[@]} experiments defined; cap last_gpu."
    LAST_GPU=$(( ${#EXPERIMENTS[@]} - 1 ))
fi

# ---------- Pre-flight checks -------------------------------------------------

DRYRUN="${DRYRUN:-0}"
if [[ "$DRYRUN" != "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[launch] nvidia-smi not found. Aborting."
        exit 1
    fi
    N_GPUS=$(nvidia-smi --list-gpus | wc -l)
    if (( N_GPUS < LAST_GPU + 1 )); then
        echo "[launch] Requested GPU $LAST_GPU but only $N_GPUS GPUs visible."
        echo "[launch] Check \`nvidia-smi\` and adjust the range."
        exit 1
    fi
    echo "[launch] Detected $N_GPUS GPUs. Using GPUs $FIRST_GPU..$LAST_GPU."
fi

# Check each config file exists before launching anything.
for i in $(seq "$FIRST_GPU" "$LAST_GPU"); do
    cfg_path="$REPO_ROOT/exps/${EXPERIMENTS[$i]}/cfg.yaml"
    if [[ ! -f "$cfg_path" ]]; then
        echo "[launch] MISSING: $cfg_path"
        echo "[launch] Aborting — all configs must exist before launch."
        exit 1
    fi
done

# Don't clobber a running batch.
if [[ -f "$PID_FILE" ]]; then
    stale=1
    while read -r pid; do
        if kill -0 "$pid" 2>/dev/null; then
            stale=0
            break
        fi
    done < "$PID_FILE"
    if [[ "$stale" == "0" ]]; then
        echo "[launch] A previous batch is still running (see $PID_FILE)."
        echo "[launch] Run `$0 --kill` first, or `$0 --status` to inspect."
        exit 1
    fi
    rm -f "$PID_FILE"
fi

# ---------- Launch loop -------------------------------------------------------

: > "$PID_FILE"
echo "[launch] Repo: $REPO_ROOT"
echo "[launch] Log dir: $LAUNCH_DIR"
date_stamp=$(date +"%Y-%m-%d_%H:%M:%S")
echo "[launch] Start time: $date_stamp"

for i in $(seq "$FIRST_GPU" "$LAST_GPU"); do
    exp="${EXPERIMENTS[$i]}"
    cfg_path="$REPO_ROOT/exps/$exp/cfg.yaml"
    log_path="$LAUNCH_DIR/$exp.log"
    echo "[launch] GPU $i -> $exp"
    echo "         cfg: $cfg_path"
    echo "         log: $log_path"

    if [[ "$DRYRUN" == "1" ]]; then
        echo "         (dryrun) skipping launch"
        continue
    fi

    # Launch, fully detached, redirected.  setsid so a shell close doesn't HUP
    # the job. CUDA_VISIBLE_DEVICES pins the GPU.  No other env vars — we
    # learned the hard way that PYTORCH_CUDA_ALLOC_CONF can perturb fp16/bf16
    # numerics (see commit f8d2dfb).
    (
        cd "$REPO_ROOT"
        CUDA_VISIBLE_DEVICES="$i" setsid python3 scripts/train.py \
            --config "$cfg_path" \
            > "$log_path" 2>&1 &
        echo "$!" >> "$PID_FILE"
    )
    # Small stagger so we don't hit cuBLAS / cuDNN autotune races when 8
    # processes all touch the CUDA driver in the same 10ms window.
    sleep 3
done

N_LAUNCHED=$(( LAST_GPU - FIRST_GPU + 1 ))
echo "[launch] Launched $N_LAUNCHED experiments. PIDs in $PID_FILE."
echo "[launch] Run \`bash $0 --status\` to see progress."
echo "[launch] Run \`bash $0 --kill\` to stop them all."

# Kick the status updater once so the file exists immediately.
bash "$REPO_ROOT/scripts/h20_update_status.sh" >/dev/null 2>&1 || true

# Optional: background a watchdog that refreshes STATUS.md every 5min.
# Commented off by default so we don't spawn a lingering daemon; user can
# enable by uncommenting or by running h20_update_status.sh manually.
# (
#     while true; do
#         sleep 300
#         bash "$REPO_ROOT/scripts/h20_update_status.sh" || break
#     done
# ) &
# echo "$!" >> "$PID_FILE"
