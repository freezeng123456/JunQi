# H20 8-GPU Deployment Plan

**Target hardware**: 8 × NVIDIA H20 (96 GB HBM3 each, **6.4× T4 memory**, ~4.5× T4 FP16 throughput, native bfloat16).

**Strategy**: 8 parallel independent experiments, one per GPU (no DDP).

## Why not DDP?

Our current bottleneck is **algorithmic**, not compute:
- v25 at 500 rollouts: peak 0.86, end 0.625
- v32 at 1500 rollouts (3× data): peak ~0.80, end 0.711
- **More data ≠ better ceiling**. The 0.80 ceiling is algorithmic.

DDP would just reach the same 0.80 ceiling faster. Instead, we need to **test multiple algorithmic hypotheses** and pick the winner. 8 parallel independent runs is exactly that — it mirrors Ataraxos's Figure 13 ablation methodology (single-H100 per ablation, combined only at publication time).

**Quote from gap analysis §6**: *"Without distributed training → same shape of Elo trajectory, ~150 Elo lower final point"* — the paper itself confirms single-GPU runs are algorithmically sufficient.

## Resource budget per GPU

With 96 GB per H20 and bf16:

| Component | v32 (T4 use) | H20 target |
|---|---:|---:|
| ArrangementNet | 12.6M params | unchanged |
| BeliefNet | 3.89M params | **6× larger** possible |
| JunqiNet | 1.03M params | **4× larger** possible (4-5M) |
| num_envs | 128 | **512** (4×) |
| steps_per_env | 512 | 512 (same) |
| Env-steps per rollout | 65,536 | **262,144** (4×) |
| dtype | float16 | **bfloat16** (no scaler, more stable) |
| Estimated memory | ~14 GB | ~30 GB (headroom for bigger net/belief) |
| Estimated fps | ~4,500 | **~20,000** (4.5× hardware × similar util) |

## Runtime budget

- 500 rollouts × 262,144 env-steps = **131M env-steps** per experiment (**~4× v32's 98M**)
- At 20,000 fps: **~110 minutes** per 500R run
- 1000 rollouts: **~3.7 hours**
- **A single 24h window = ~6 sequential 1000R experiments per GPU = 48 experiments total across 8 GPUs.**

## The 8 experiments (initial slate)

The goal: isolate which algorithmic changes break the 0.80 ceiling.

| # | Name | Hypothesis | Delta from v32 |
|---|---|---|---|
| **1** | `h20_v32_bf16_base` | Baseline: v32 config on H20 with bf16 + 4× envs | dtype=bfloat16, num_envs=512 |
| **2** | `h20_belief_bigger` | Bigger belief net breaks ceiling | belief.depth=6, belief.embed=512 |
| **3** | `h20_move_bigger` | Bigger move net + more data (re-test v26 hypothesis without fp16/expandable_segments confound) | net.depth=6, embed=256 |
| **4** | `h20_draw_penalty` | Reward shaping: penalize draws/timeouts | env reward: draw=-0.5, timeout=-0.3 |
| **5** | `h20_self_play` | Self-play instead of vs-random | eval vs random; training vs (policy team) |
| **6** | `h20_arr_slower_refresh` | Slower arr-net refresh = stabler target | arr.refresh_every=4 (not 1) |
| **7** | `h20_longer_train` | 3000 rollouts single-run scale-up on v32 config | total_rollouts=3000 |
| **8** | `h20_bigger_both` | Combined scaleup: belief + move + data | exp 2 + exp 3 + exp 7 |

Experiments #4 and #5 are the highest expected value but most invasive. Exps #1 and #7 are cheap sanity controls.

## Launch mechanism

### Single-GPU launch (prep sanity check)
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train.py --config exps/h20_v32_bf16_base/cfg.yaml
```

### 8-GPU parallel launch
```bash
bash scripts/launch_h20_8gpu.sh
```
The launcher pins one experiment per GPU, tails each into `logs/h20_launch/<exp>.log`, and writes a live `STATUS.md` with PID + rollout + win_rate per experiment.

## Monitoring

- Per-exp log: `exps/<name>/train.log`
- Aggregated board: `exps/h20_launch/STATUS.md` (auto-refreshed every 5 min)
- Each experiment runs fully independently. Crash of one does not affect others.

## Reproducibility

- Each config pins `seed=X` (we use seeds 101-108 across the 8 experiments)
- bf16 removes the fp16 autocast non-determinism that crippled the T4 `expandable_segments` comparison
- `torch_deterministic=false` kept for throughput — use same-seed reruns if exact reproduction needed

## Known risks

1. **cuDNN kernel selection still non-deterministic across restarts** — even with same seed, expect ~±0.10 win_rate at any given rollout. Judge experiments on **final 50R average**, not single peak.
2. **Self-play (#5) is the biggest code change** — may need a day of work before it's runnable. The other 7 are config-only.
3. **Disk space**: 8 × 500R × ckpt_every=50 × ~30MB/ckpt ≈ 10 GB. Fine.

## Post-run analysis

After all 8 finish, produce `exps/h20_launch/RESULTS.md` with:
- Per-exp final-50R avg win_rate
- Peak win_rate + rollout
- Avg game length (stall-trap detector)
- Recommended winner for a 8×H20-DDP scale-up run

## Push & pull workflow

- Local prep done on the T4 machine, pushed to `origin/master`
- On H20 machine: `git pull` → `bash scripts/launch_h20_8gpu.sh`
- No GPU-specific state in git — all configs under `exps/`, scripts under `scripts/`
