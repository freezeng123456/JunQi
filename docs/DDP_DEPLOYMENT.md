# 6-card DDP deployment plan (v33: break the 0.80 ceiling)

**Target**: 6 × NVIDIA H20 (96 GB each) on a single node. Time budget: hours,
not days — but we still want frequent checkpoints to rewind on collapses.

**Goal**: produce a checkpoint whose final-50R-avg `eval/win_rate` (vs
random opponent, the v32-comparable metric) is ≥ 0.85, breaking the 0.80
algorithmic ceiling that v17–v32 hit on T4 across 16 different
hyperparameter sweeps.

---

## Decision: jump straight to a single 6-card DDP self-play run

We deliberately skip the previously-planned A (v32 baseline regression) and
B (4-way 2-card DDP ablation) phases. Rationale:

1. **A is low-EV.** v32's exact behaviour is well documented in
   `docs/PROGRESS.md`; a 1-hour H20 baseline run only verifies "DDP code did
   not regress numerics", which is also covered by the unit/integration
   tests added in this branch (`tests/test_ddp_smoke.py`, plus the actual
   2-card and 6-card smoke runs in `exps/_smoke_*/`).
2. **B is hedge-against-uncertainty.** Self-play (exp5) is the most
   paper-aligned hypothesis and the only one that *structurally* attacks
   the "defensive stall trap" diagnosed in v30/v31 — bigger KL anchors and
   larger nets only treat the symptom (gradient drift) rather than the
   cause (vs-random's reward landscape rewards stalling). Per the Ataraxos
   gap analysis §6, self-play vs vs-random is ablation #2 in their Figure
   13 with the largest single-axis Elo gap.
3. **Time is abundant** but our compute window is also our debug window.
   Investing it all in the single highest-EV bet, then judging by win_rate
   trajectory, lets us collect more high-quality data than splitting 6 GPUs
   across 3 hypotheses for 2.5h each.

**If v33 fails** (win_rate flatlines below 0.7 by R500 or collapses below
0.45 post-R200): the early-stop trigger fires automatically. The fallback
is `kl_coef=0.4` self-play (treats the same trap with stronger anchor) on
the same 6-card DDP setup. We have not pre-staged that config; we'd write
it after observing v33's failure mode.

---

## The single config: `exps/h20_ddp_winner_v33_selfplay/cfg.yaml`

Differences from v32 (the most comparable baseline):

| Knob | v32 | v33 | Why |
|---|---|---|---|
| `random_opponent` | true | **false** | THE structural change: self-play |
| `env.num_envs` per rank | 128 | **512** | H20 has 6.4× T4 memory |
| `ppo.dtype` | float16 | **bfloat16** | H20 native; no GradScaler drama |
| `arr.ppo.autocast_dtype` | float16 | **bfloat16** | same |
| World size | 1 | **6 (DDP)** | 6× more transitions per rollout |
| `total_rollouts` | 1500 | 1500 | Same rollout count, 24× more env-steps |
| `save_every` | 100 | **25** | Granular ckpt for rewind |
| `eval_every` | 10 | 10 | Unchanged |
| `early_stop_win_rate` | 0.3 | **0.45** | Stricter floor; v32 typical R200+ ≥ 0.55 |
| `belief.warmup_rollouts` | 100 | 100 | Belief buffer fills in ~10 R per rank, well before warmup ends |
| `arr.kl_coef` | 0.01 (v17 era) → 0.1 (h20 era) | 0.1 | Paper-aligned (already true in h20 cfg) |
| `lr_decay` | 1.1 | 1.1 | Paper-aligned (already corrected post-v17) |
| `temperature_decay` | 0.3 | 0.3 | Paper-aligned magnet-KL schedule |
| `ppo.torch_compile` | true | **false** | torch 2.1 + DDP + BatchNorm dynamo bug |
| `arr.ppo.autocast_dtype` | bfloat16 | **float32** | torch 2.1 + DDP + bf16 ArrangementNet → CUDA SIGFPE |

---

## Launch

### Compact-history rollouts

`rollout.storage_mode: compact_history` is supported under DDP.  Each rank
owns an independent `GpuRolloutHistory` on its local CUDA device and rebuilds
only its own selected PPO minibatches; only gradients are synchronized.

Because `env.num_envs` is interpreted per rank, use
`configs/ataraxos_selfplay_value_all_ddp2.yaml` for a two-H20 continuation of
the current 10.8M run.  It uses 1024 envs per rank, preserving the previous
single-H20 global total of 2048 envs and therefore preserving rollout-indexed
learning-rate and temperature schedules.

Pre-flight:

```bash
# Sanity: dependencies and DDP unit tests pass
python3.11 -m pytest tests/test_ddp_smoke.py -q

# Verify the single-process path still works (5R smoke)
CUDA_VISIBLE_DEVICES=0 python3.11 scripts/train.py \
    --config exps/_smoke_single/cfg.yaml --total_rollouts 3
```

Long run:

```bash
cd /path/to/JunQi
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    nohup bash scripts/launch_h20_ddp.sh \
        exps/h20_ddp_winner_v33_selfplay/cfg.yaml 6 \
    > exps/h20_ddp_winner_v33_selfplay/launch.log 2>&1 &
echo $! > exps/h20_ddp_winner_v33_selfplay/launch.pid
```

`launch_h20_ddp.sh` wraps `torchrun --standalone --nproc_per_node=6` with
a randomised master_port (avoiding collision with concurrent runs). Per-
rank seeding (`global_rank * 10_000`) happens inside `scripts/train.py`.

---

## Monitoring

```bash
# Live tail of rank 0 output (this is the "main" log)
tail -f exps/h20_ddp_winner_v33_selfplay/train.log

# Per-rank tracebacks if something crashes (only non-rank-0 ranks tee here)
ls exps/h20_ddp_winner_v33_selfplay/train_rank*.log

# GPU utilisation should be 6/6 cards at >85% during collect, lower
# during PPO update / eval.
watch -n 5 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

What to look for in rank-0's `train.log`:

* `Per-rank env steps per rollout: 262,144 (global: 1,572,864 across 6 ranks)`
  on first rollout — confirms DDP is wired right.
* `[ N] loss_p=...  loss_v=... ret=... lr=... fps=... elapsed=...` every 5
  rollouts. fps is **per-rank**; cluster fps ~= fps × 6.
* `[eval] win_rate=... avg_len=...` every 10 rollouts.

Healthy v33 trajectory should look like (rough projection from v32 + self-play):

| Rollout | win_rate (eval) | avg_len | Notes |
|---|---|---|---|
| 0–50 | 0.25–0.45 | 3000+ | Pre-warmup; belief net random; setup pool fresh |
| 50–150 | 0.45–0.65 | 1500–2500 | Belief warmed, magnet KL still strong |
| 150–500 | 0.65–0.80 | 800–1500 | Mid-game learning; alpha annealing biting |
| 500–1500 | **0.80+** | 600–1000 | **Self-play breaking the ceiling — the goal** |

Red flags (kill manually if observed):

* `avg_len > 3500` after R200: defensive stall reasserting itself even in
  self-play. Pivot to `kl_coef=0.4` config.
* `loss_p` consistently > 1.0 after R100: PPO ratio explosion, possibly
  bad bf16 numerics. Lower `clip_range` to 0.15 and resume.
* `train/nan_skip_total` > 50 after R500: bf16 should not produce NaNs in
  this regime; investigate before continuing.

---

## Stopping & resuming

Graceful stop:

```bash
kill -SIGINT $(cat exps/h20_ddp_winner_v33_selfplay/launch.pid)
# All ranks save final ckpt and exit at next eval boundary.
```

Resume from the latest ckpt (works in any world_size — state dict is
unwrapped):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    bash scripts/launch_h20_ddp.sh \
        exps/h20_ddp_winner_v33_selfplay/cfg.yaml 6 \
        --resume_cli exps/h20_ddp_winner_v33_selfplay/ckpt_latest.pt
```

Single-card diagnosis from a DDP ckpt (e.g. attach pdb to investigate a
trajectory):

```bash
CUDA_VISIBLE_DEVICES=0 python3.11 scripts/train.py \
    --config exps/h20_ddp_winner_v33_selfplay/cfg.yaml \
    --resume_cli exps/h20_ddp_winner_v33_selfplay/ckpt_000500.pt \
    --total_rollouts 502  # just inspect 2 rollouts
```

This works because `PPOTrainer.state_dict()` saves the unwrapped
JunqiNet keys — verified by `tests/test_ddp_smoke.py::test_unwrapped_state_dict_loads_cross_world_size`.

---

## What v33a taught us (post-mortem)

The first attempt (`exps/h20_ddp_winner_v33_selfplay/`) ran for ~37 min
before going permanently into NaN at R107. Ckpts and the train.log are
preserved in `exps/_crash_diagnosis_v33a/` for forensic reference. The
failure chain was:

1. **R69**: `loss_v` spiked from 0.0072 → 0.2186 (~30× jump). Rollout's
   value head, trained with **hard 3-bin NLL** until then, encountered an
   outlier batch where the prediction was confidently wrong (a quirk of
   PPO + self-play under bf16 + 6-card DDP large-batch effects). The
   per-sample NLL was 5.3+ and the gradient through it ~200.
2. **R70**: `win_rate` dropped 0.641 → 0.422 — the value head briefly
   poisoned the policy.
3. **R100**: Policy recovered; eval `win_rate=0.664` (best yet, beating
   v32's same-rollout marker).
4. **R107**: A second value-loss spike triggered an actual NaN gradient.
   The **forward-only** NaN guard (the one inherited from v32 era) didn't
   help — by the time forward sees NaN, params are already poisoned. The
   `optimizer.step()` ran with NaN gradients and wrote NaN into the
   parameters.
5. **R108-114**: EMA shadow contaminated to NaN over a handful of steps
   (`shadow ← 0.999·shadow + 0.001·NaN_params`). `trainer.ema.model` is
   what eval and collect use, so the entire training pipeline collapsed.
6. **R114-180**: Permanent stuck state. `loss_p=loss_v=ret=lr=NaN`,
   `win_rate` oscillated around 0.5 (random policy noise).

**v33b code-level fixes** (committed alongside this branch):

* **Gradient-NaN guard** in `PPOTrainer._update_step` and
  `ArrangementPPOTrainer._step`. Before `optimizer.step()`, every rank
  checks `torch.isfinite(grad_norm)`, then `all_reduce(MAX)` on the bad
  flag — if ANY rank has a non-finite gradient, EVERY rank zeros its
  gradients and skips the step. Counter `train/grad_skip_total` (and
  `arr_train/grad_skip_total`) is surfaced in logs as an early warning.
* **Soft cross-entropy value loss** (`_value_loss` in `ppo.py`) replaces
  hard NLL. Scalar return ∈ [-1, +1] is encoded as a 3-bin soft target
  by linear interpolation between the two adjacent bin centres
  (matches Ataraxos `pyengine/core/rl.py` line 558-561). Bounds the
  per-sample value-loss gradient and eliminates the spike pattern that
  seeded the R69 explosion.

**v33b cfg-level conservatism** (vs v33a):

| Knob | v33a | v33b | Why |
|---|---|---|---|
| `lr_ceil` | 1e-4 | 5e-5 | Halve peak LR for self-play stability |
| `clip_range` | 0.20 | 0.15 | Tighter PPO trust region |
| `max_grad_norm` | 0.267 | 0.20 | More aggressive grad clipping |
| `kl_coef` | 0.10 | 0.20 | Stronger KL anchor to old policy |
| `arr.ppo.kl_coef` | 0.10 | 0.20 | Same, for arrangement net |
| `seed` | 105 | 205 | Different trajectory than v33a |

**Why we did NOT resume from v33a's R100 ckpt** (it was verified clean):
A 30R resume smoke showed every gradient was NaN immediately after
loading. Diagnosis: the saved Adam optimizer state held momentum and
variance estimates accumulated under the hard-NLL value-loss
gradient distribution. With the new soft-CE loss, gradient magnitudes
and directions differ, and the stale moments produced numerical
instability on the first few resumed steps. By contrast, a from-scratch
30R smoke under the v33b code path is perfectly healthy (win_rate
climbed 0.570 → 0.719, no NaN events). Cost of starting over: ~1.4h of
v33a's compute. Verdict: cheap to pay, not worth the resume diagnostics.

## torch 2.1 footguns we hit during smoke

While bringing the DDP path up on this 6×H20 box (torch 2.1.2+cu121 ⨯
NCCL ⨯ Python 3.11 ⨯ junqi_cuda built locally), we hit and worked
around three issues. Documented here so future readers can recognise
them quickly:

1. **`torch.amp.GradScaler` doesn't exist in torch 2.1.** Repo's
   `junqi_rl/training/ppo.py` originally imports `from torch.amp import
   GradScaler`, which only works on torch ≥ 2.3. Fixed with a try/except
   that falls back to `torch.cuda.amp.GradScaler()` for the older API.
2. **`gpu_collector.py` force-compiles `policy.act` every rollout** even
   when `cfg.ppo.torch_compile=false`. On torch 2.1 this trips a dynamo
   decomposition bug at the BatchNorm layer in JunqiNet's CNN stem
   (only under DDP + bf16 autocast simultaneously). We added a
   `use_compile` parameter and threaded `cfg.ppo.torch_compile` through.
3. **DDP + bf16 autocast on `ArrangementNet` triggers CUDA SIGFPE** at
   the first arr training step. Reproduced on every run with
   `arr.ppo.autocast_dtype: bfloat16`; vanishes with `float32`. We
   default to fp32 in the v33 cfg with a TODO to revisit on torch 2.4+.
   Note: `ppo.dtype: bfloat16` (the main JunqiNet path) is fine; only
   the arrangement-net forward exhibits this.

Independently, `dataclass(slots=True)` in `junqi_core` requires Python
3.10+ — we installed `python3.11-devel` plus a fresh torch 2.1.2 and
rebuilt `junqi_cuda` against cp311.

## What we explicitly did NOT change in this branch

To keep the experiment honest:

* **Belief net architecture** — still the stateless 3.89M variant. The
  Ataraxos gap analysis §1.3 calls out the temporal/AR variants as
  ~+5% belief CE accuracy each, but that's a separate experiment.
* **Move net size** — still 1M params (depth=4, embed=128). The
  upstream gap analysis §1.2 calls out scaling to ~10M as the second-
  highest-EV change after self-play, but compounding both in one
  experiment makes attribution impossible.
* **Test-time search** — completely absent (not even the rollout
  primitive). Worth +123 Elo per the paper's ablation, but only
  meaningful with a strong belief net first.
* **Observation features (threat / evade / protection / provenance)** —
  unchanged 256 channels. All four feature families are P2 work.

If v33 succeeds, the next branch should add **bigger move net** (single-
axis change, ~1 day of work). If v33 plateaus at <0.85, the next branch
should add **temporal belief net + bigger move net + 6-card DDP** as a
combined push.
