# JunQi RL — Progress Log

## Current status

- **T4 single-GPU phase**: COMPLETE (v17 → v32, 16 runs).
- **H20 8-GPU phase**: PREP COMPLETE, not yet launched.
- **CombatMemory v4 (ADR-128)**: DARK-only, 50-channel v4 layout
  (OBS_CHANNELS = 306).  CPU reference + observation channels +
  BatchedGameState SoA + tests + v35 config — landed 2026-05-09.
  CUDA kernel side stubbed (header + plan); kernel implementation is
  the M-1f gate before the v35 training run.
- **Outstanding problem**: win_rate vs random opponent plateaus at ~0.80
  across all configurations tried on T4. We have strong evidence this is
  **algorithmic** (reward structure / self-play), not compute-limited.

## T4 result summary

| Run | Outcome | Peak | End | Key lesson |
|---|---|:---:|:---:|---|
| v17–v20 | 500R done | ~0.80 | 0.76 | baseline, 1.03M JunqiNet, Ataraxos hparams |
| v22 | OOM R54 | 0.72 | — | belief refresh memory leak |
| v23 | OOM R320 | 0.86 | — | BeliefNet FFN peak too big for T4 |
| v24 | collapse R89 | 0.80 | 0.22 | fp16 nondeterminism ⇒ unlucky trajectory |
| **v25** | 500R done | **0.77** | 0.625 | **chunked BeliefNet forward validated** |
| v26 | OOM R125 | 0.85 | — | 3× net scaleup + allocator fragmentation |
| v27 | OOM R134 | 0.84 | — | num_envs→64 just delayed the OOM |
| v28 | early-stop R240 | 0.82 | — | `expandable_segments` fixed OOM, exposed plateau |
| v29 | crash R51 | — | — | fp16 NaN in Categorical.sample() |
| v30 | early-stop R240 | 0.70 | — | NaN guard fixed crash, new "defensive stall" trap |
| v31 | early-stop R240 | 0.72 | — | 1M net + expandable_segments = same stall trap |
| **v32** | **1500R done** | **~0.80 (R80)** | **0.711** | **pure v25 replay**; 3× data did NOT break 0.80 ceiling |

## Major lessons learned

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments` perturbs fp16 numerics**
   enough to change cuDNN kernel dispatch and wreck training trajectories.
   Use only when the alternative is OOM.
2. **`torch.compile` + fp16 autocast is not bit-reproducible across runs**
   even with identical seeds. Expect ±0.10 win_rate across same-seed runs
   in the 0.5-0.8 regime.
3. **The 0.80 ceiling is NOT data-limited.** v32 (1500R = 3× v25's 500R)
   produced the same peak and a similar endpoint.
4. **"Defensive stall" failure mode**: v30 showed that a bigger net
   trained against a random opponent finds a "stall = +ret" local optimum
   (avg_len 1700 instead of 1000, win_rate stuck at 0.25).

## H20 8-GPU prep (ready to push)

See `docs/H20_DEPLOYMENT.md` for full plan. Summary:

- **Strategy**: 8 parallel independent experiments (no DDP), each on 1 GPU.
- **Rationale**: ceiling is algorithmic ⇒ need to test multiple hypotheses
  simultaneously, not scale one hypothesis faster. Mirrors Ataraxos's
  Figure 13 ablation methodology.
- **Precision**: `bfloat16` everywhere (no scaler needed, no fp16 NaN).
- **num_envs**: 512 per H20 (4× T4's 128, exploits 96GB HBM).

### The 8 experiments

| # | Name | Hypothesis |
|---|---|---|
| 1 | `h20_exp1_v32_bf16_base` | Control: v32 @ H20 with bf16 |
| 2 | `h20_exp2_belief_bigger` | 20M BeliefNet instead of 3.89M |
| 3 | `h20_exp3_move_bigger` | Clean v26 re-test (depth=6, embed=192) |
| 4 | `h20_exp4_draw_penalty` | kl_coef 0.1→0.4 fights stall |
| 5 | `h20_exp5_self_play` | Self-play removes stall exploit |
| 6 | `h20_exp6_arr_slow_refresh` | Stabler opponent pool |
| 7 | `h20_exp7_longer_train` | 3000R test of data-limit hypothesis |
| 8 | `h20_exp8_bigger_both` | Compound scaleup |

### Files added (ready to push)

- `docs/H20_DEPLOYMENT.md` — full plan, budget, hypothesis rationale
- `scripts/launch_h20_8gpu.sh` — parallel launcher with --kill/--status/DRYRUN
- `scripts/h20_update_status.sh` — writes live STATUS.md
- `exps/h20_exp{1..8}_*/cfg.yaml` — 8 experiment configs
- `exps/h20_launch/README.md` — launch-dir usage guide
- `tests/test_h20_configs.py` — 20 tests validating all 8 configs
- `scripts/train.py` — added `random_opponent` TrainConfig field for exp 5

### Test coverage

```
pytest tests/test_h20_configs.py -q    → 20 passed
pytest tests/test_train_integration.py → 9 passed (unchanged)
pytest tests/test_junqi_net_nan_guard  → 5 passed (unchanged)
```

## Repository state

### Full-content run dirs (cfg + ckpts + logs)
- `beat_random_v17_ataraxos_aligned` — no-belief reference
- `beat_random_v20_v17_noOOM` — stable no-belief baseline
- `beat_random_v23_belief_fixed` — belief peak 0.86
- `beat_random_v25_belief_seed123` — completed 500R with belief
- `beat_random_v28_expandseg` — memory-fix validation
- `beat_random_v29_slowlr` — lr_decay=0.6 first evidence
- `beat_random_v30_nanguard` — NaN guard validation
- `beat_random_v32_v25replay_1500r` — current endpoint; 1500R done
- `beat_random_v31_1m_1500r` — diagnostic reference (stall trap proof)

### Archive dirs (cfg.yaml only, git-tracked)
Pre-v16 (v2–v15), v16 (P0), v18, v19, v21a/b, v22, v24, v26, v27.

### H20 dirs (cfg only, no runs yet)
`h20_exp1..exp8`, `h20_launch/`.

## Key diagnostics / tools

- `scripts/diagnose_v24_belief.py` — belief metric cross-run
- `scripts/diagnose_zero_gradient.py` — loss=0 cross-run
- `scripts/analyse_ataraxos_gap.py` — paper vs us
- `scripts/train_expandseg.py` — launcher with PYTORCH_CUDA_ALLOC_CONF
  (kept for reference; default `train.py` is preferred)
- `tests/test_junqi_net_nan_guard.py` — NaN guard regression (5 tests)
- `tests/test_loss_zero_hypothesis.py` — loss=0 math proof (4 tests)
- `tests/test_h20_configs.py` — H20 config validation (20 tests)
- `tests/cleanup_stale_exps.py` — idempotent archive utility
