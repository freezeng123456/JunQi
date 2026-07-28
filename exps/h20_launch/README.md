# H20 Launch Directory

This directory is populated by `scripts/launch_h20_8gpu.sh` when you run
the 8-GPU experiment slate. On a fresh clone it's nearly empty — you get
the artifacts only after launching a batch.

## Files (after launch)

| File | Purpose |
|---|---|
| `pids.txt` | one PID per line, in GPU order (pid of GPU 0 on line 1, etc.) |
| `STATUS.md` | auto-generated live progress board (rollout, win_rate, alive?) |
| `<exp_name>.log` | stdout+stderr of each experiment (also tailable via `tail -f`) |

## Typical workflow on H20 host

```bash
# First time after git pull:
cd /path/to/JunQi
pytest tests/test_h20_configs.py -q   # sanity-check all 8 configs build

# Launch all 8:
bash scripts/launch_h20_8gpu.sh

# Check progress (refreshes STATUS.md):
bash scripts/launch_h20_8gpu.sh --status

# Tail a specific experiment:
tail -f exps/h20_launch/h20_exp2_belief_bigger.log

# Kill everything:
bash scripts/launch_h20_8gpu.sh --kill
```

## Running a subset

```bash
# Only GPUs 0-3 (first 4 experiments):
bash scripts/launch_h20_8gpu.sh 0 3

# Only GPU 5 (single experiment, for debugging):
bash scripts/launch_h20_8gpu.sh 5 5

# Print what would be launched without actually doing it:
DRYRUN=1 bash scripts/launch_h20_8gpu.sh
```

## Experiment slate

See `docs/H20_DEPLOYMENT.md` for the full hypothesis rationale. Quick index:

| # | GPU | Experiment | Hypothesis |
|---|---:|---|---|
| 1 | 0 | `h20_exp1_v32_bf16_base` | Baseline/control: does H20 give clean 4×speedup? |
| 2 | 1 | `h20_exp2_belief_bigger` | Bigger belief net breaks ceiling? |
| 3 | 2 | `h20_exp3_move_bigger` | Clean v26 re-test (no fp16 confound) |
| 4 | 3 | `h20_exp4_draw_penalty` | Higher kl_coef anchor fights stall? |
| 5 | 4 | `h20_exp5_self_play` | Self-play removes stall exploit? |
| 6 | 5 | `h20_exp6_arr_slow_refresh` | Stabler opponent pool helps? |
| 7 | 6 | `h20_exp7_longer_train` | Does 3000R data break 0.80 ceiling? |
| 8 | 7 | `h20_exp8_bigger_both` | Combined scaleup |

## Safety

The launcher:
- Uses `setsid` so closing your shell doesn't kill experiments.
- Writes every PID to `pids.txt` so `--kill` works even after a reconnect.
- Refuses to clobber a running batch (checks `pids.txt` for living PIDs).
- Sleeps 3s between launches to avoid cuBLAS/cuDNN autotune races.
- Stores no secrets; logs go under `exps/h20_launch/` only.
