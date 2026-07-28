#!/bin/bash
# h20_update_status.sh — Write exps/h20_launch/STATUS.md with a snapshot
# of each experiment's progress: PID alive?, latest rollout, latest win_rate,
# GPU memory.  Invoked by launch_h20_8gpu.sh and can be rerun manually.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCH_DIR="$REPO_ROOT/exps/h20_launch"
PID_FILE="$LAUNCH_DIR/pids.txt"
STATUS_FILE="$LAUNCH_DIR/STATUS.md"

# Experiment list — must match launch_h20_8gpu.sh.
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

{
    echo "# H20 Launch Status"
    echo ""
    echo "Last updated: $(date +"%Y-%m-%d %H:%M:%S")"
    echo ""
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "## GPU overview"
        echo '```'
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
        echo '```'
        echo ""
    fi

    echo "## Experiments"
    echo ""
    echo "| # | GPU | Experiment | PID | Alive | Rollout | Last win_rate | Last avg_len |"
    echo "|---|----:|---|----:|:---:|---:|---:|---:|"

    # Map pid -> gpu index by position in pids.txt.
    if [[ -f "$PID_FILE" ]]; then
        mapfile -t pids < "$PID_FILE"
    else
        pids=()
    fi

    for i in "${!EXPERIMENTS[@]}"; do
        exp="${EXPERIMENTS[$i]}"
        log="$REPO_ROOT/exps/$exp/train.log"
        pid="${pids[$i]:-}"
        alive="—"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            alive="✅"
        elif [[ -n "$pid" ]]; then
            alive="💀"
        fi

        rollout="—"
        win_rate="—"
        avg_len="—"
        if [[ -f "$log" ]]; then
            # Last `[ rollout]` line gives us current R.
            last_r=$(grep -E "^\[\s*[0-9]+\]" "$log" 2>/dev/null | tail -1 || true)
            if [[ -n "$last_r" ]]; then
                rollout=$(echo "$last_r" | sed -E 's/^\[\s*([0-9]+)\].*/\1/')
            fi
            last_eval=$(grep "^\[eval\]" "$log" 2>/dev/null | tail -1 || true)
            if [[ -n "$last_eval" ]]; then
                win_rate=$(echo "$last_eval" | sed -E 's/.*win_rate=([0-9.]+).*/\1/')
                avg_len=$(echo "$last_eval" | sed -E 's/.*avg_len=([0-9.]+).*/\1/')
            fi
        fi

        echo "| $((i+1)) | $i | $exp | ${pid:-—} | $alive | $rollout | $win_rate | $avg_len |"
    done

    echo ""
    echo "## Quick commands"
    echo '```'
    echo "# Tail a specific experiment's log:"
    echo "tail -f $LAUNCH_DIR/<exp_name>.log"
    echo ""
    echo "# Kill everything:"
    echo "bash scripts/launch_h20_8gpu.sh --kill"
    echo ""
    echo "# Refresh this file:"
    echo "bash scripts/h20_update_status.sh"
    echo '```'
} > "$STATUS_FILE"

echo "[status] Wrote $STATUS_FILE"
