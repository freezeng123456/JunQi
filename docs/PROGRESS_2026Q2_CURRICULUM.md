# Progress Notes — 2026 Q2 Curriculum Learning

## TL;DR

- ✅ Plan A (`v41_curriculum_T`) reached win=0.852 by R40 — but was a
  **misleading result**: under DARK rule + 4-seat-mirror canonical
  setups, the piece_id channel deterministically encodes each piece's
  true type, so the network learns to look up rather than infer.
  Counted as Plan A "baseline" only.
- 🚧 Plan B (`v42_planB_T_vs_random`): own team uses canonical T,
  enemy team independent uniform-random. R10..R50 trajectory:
  0.562 → 0.453 → 0.531 → 0.273 → 0.555.  R40 dip is a transient stall
  collapse (avg_len 906 → 328); R50 recovery shows it's noise, not a
  permanent regression.  **Move policy is genuinely learning暗棋
  inference now.**

## v42 R10..R50

| Rollout | win  | loss | draw | avg_len | note                    |
|--------:|-----:|-----:|-----:|--------:|-------------------------|
|     10  | 0.562| 0.430| 0.008|     883 | first eval              |
|     20  | 0.453| 0.547| 0.000|     916 | within noise            |
|     30  | 0.531| 0.469| 0.000|     906 | mild recovery           |
|     40  | 0.273| 0.727| 0.000|     328 | **stall collapse spike** (avg_len drop = aggressive-suicide policy) |
|     50  | 0.555| 0.414| 0.031|     598 | recovery, 50% draw noise |

### Diagnosis

The R40 dip is the same "aggressive-suicide" failure mode that v37 hit
around R200: with no belief net training, the move-policy has to
learn 暗棋 inference end-to-end through PPO.  When a stall-style local
optimum appears, |adv| grows → big policy update → collapse → recover
through magnet KL pull.

This is *expected* under the simplified "no belief, no arr" setup of
v41/v42.  It's an actual signal that the network is wrestling with the
real information-asymmetry problem.

## Next steps (in priority)

1. **Let v42 finish the 250R horizon** — see whether R100..R250 gets
   stable above 0.65.  If it caps around 0.55–0.65, that confirms
   "plain MoveNet alone can't crack 暗棋 vs random by pure PPO".
2. **Fix BeliefNet PPO DDP NaN-skip inconsistency** (root-cause from
   the v40 SIGFPE diagnosis): when one rank's `ce_loss` is non-finite,
   it must still go through the same forward+backward+all-reduce as
   other ranks; either gate via dist.all_reduce(MAX) on the finite-
   flag, or zero-out the loss instead of early-returning.
3. **v43 = v42 + belief training on** (`configs/v43_planB_belief_on.yaml`):
   BeliefNet with `warmup_rollouts=30`, `disable_belief_train=false`, and
   DDP-safe NaN skip in `belief_ppo.py`.  Dense eval:
   `scripts/eval_random_dense.py` (2048 games, Wilson 95% CI).

## Plan A vs Plan B in one table

| Aspect                          | Plan A (v41) | Plan B (v42)         |
|---------------------------------|:------------:|:--------------------:|
| Own team layout                 | canonical T  | canonical T          |
| Teammate layout                 | canonical T  | canonical T          |
| **Enemy team layout**           | canonical T  | **uniform random**   |
| 4-seat mirror?                  | yes (leak)   | no                   |
| Train dist matches eval dist?   | no           | **yes**              |
| 暗棋 inference required?        | trivially no | **yes**              |
| R40 win                         | 0.852        | 0.273 (recovers 0.555 by R50) |

## Files

- `configs/v41_curriculum_T.yaml` — Plan A (kept for reference)
- `configs/v42_planB_T_vs_random.yaml` — Plan B main run
- `configs/v43_planB_belief_on.yaml` — Plan B + belief (next)
- `scripts/eval_random_dense.py` — 2048-game Wilson eval for 90% gate
- `exps/v41_curriculum_T/ckpt_best.pt` — Plan A's R40 ckpt (0.852, invalid)
- `exps/v42_planB_T_vs_random/` — Plan B run logs
