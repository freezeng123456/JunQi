# Ataraxos move/setup reference for four-player JunQi

This is the move/setup self-play baseline, before belief training or search.
The recipe is enforced by `training/ataraxos_contract.py`, not by its name.

Sources: [paper S3](https://arxiv.org/html/2511.07312v2) and
[official source at 92db29e8](https://github.com/AtaraxosAI/stratego/tree/92db29e8ffc323b1b8a2804b5c3f84695d036b05).

## Architecture and updates

- Move: linear cell projection, zero-padded value token, position std 0.1,
  8 pre-norm layers, width 384, 8 heads, ReLU FFN width 1536, final norm,
  biased 384-dimensional Q/K action head, categorical loss/draw/win value.
  Linear layers retain PyTorch defaults, as upstream. No GraphStem or
  learned CLS is used in this profile. Historical checkpoints keep their graph architecture.
- Setup: 4 causal layers, width 512, 8 heads, ReLU FFN width 2048,
  position std 0.1, policy/value/future-entropy heads; no absolute-seat embedding.
- Raw advantages, same-seat GAE 0.5 and full categorical TD 0.8, gamma 1.
  Intervening opponents or teammates never supply a seat's bootstrap value.
  Eliminated seats retain their pending decision until the team outcome.
- Retain the top quartile of raw absolute advantages with threshold at least
  0.01. Recompute the threshold over a bounded rolling history, including old
  behavior values and new outcomes. Train each newly ready action once;
  historical states affect the quantile only. Uninitialized buffer slots and
  dummy post-terminal actions are not samples in JunQi.
- Shuffle complete timestep rows; one AdamW step per filtered row, one epoch.
  Policy clip 0.2, reverse KL to behavior 0.1, value coefficient 1,
  gradient norm 0.267, weight decay 0. Adam defaults match upstream.
- Learning rate `clip(0.5/(iteration+1)^1.1, 5e-6, 1e-4)` is set before
  the first update of each iteration. Piece-then-destination magnet coefficient
  `clip(0.05/(iteration+1)^0.3, .001, .1)` uses the same iteration clock.
- EMA decay .999 updates once after each training phase, for both networks.
  Actors sample from current weights. Checkpoints retain raw weights,
  optimizer, EMA weights/counters, pending decisions and rolling history.
  Evaluation defaults to EMA and can explicitly select raw weights.
- Setup PPO: batch 1024, 5 epochs, LR 5e-5, clip .2, KL .1,
  value .5, entropy-prediction 1, gradient norm .5. Setup entropy coefficient
  `max(.001, .1/(iteration+1)^.3)`, normalization 10. Each collection interval
  starts a fresh reward-count window; in-flight setup behavior statistics remain.

## Explicit adaptations and source discrepancies

- The paper uses two-player Stratego, 16 learner GPUs and BF16. This run uses
  four-player team JunQi, one V100 and FP32. It matches the paper's **per-GPU**
  1536 environments, not the 16-GPU global batch or compute budget.
- Each iteration collects 400 global actions per environment: four times the
  official code's `train_every_per_player - 1 = 100`. Player elimination makes
  actual own-decision counts variable. The paper describes 202 global actions
  in two-player Stratego; the released code uses 200 after its initial boundary.
- The paper specifies 1000 setups per player per refresh; we generate 4000
  for four seats, refreshing every iteration. The released code's default
  `n_arr=1024` is a **total**, split across players; this profile follows the
  paper's per-player sampling count. CUDA samples a separate bank index per
  seat at reset, rather than selecting a pre-paired four-seat board.
- Inputs retain the authorized 317 known-fact channels and no 28 scalar
  summaries or hand-written probability beliefs. Legal actions, legal setup
  completions, board geometry and terminal rewards follow JunQi rules.
- The code's .001 entropy floors are explicit here; the paper formulas omit
  those floors. The piece-weighted magnet follows the paper, rather than the
  library's generic uniform-action default.
- Forward/backward chunks only bound memory: gradients use exact sample
  weights, clipping and Adam occur once per logical batch. Floating-point
  summation can differ. CPU float64 update tests and GPU restart tests check
  semantics; this is not bitwise reproduction of a different GPU's BF16 run.
- The rolling quantile spans 4404 real-action rows, the four-player counterpart
  of the official maximum-game plus collection window. Only occupied valid
  rows count during warm-up; no uninitialized zero entries are treated as data.

## Run and evidence

`configs/v100_ataraxos_reference.yaml` starts from fresh weights, with a 20,000
completed-game target and 5k/10k/20k evidence. `scripts/beijing_ataraxos_reference.sh`
builds the new CUDA reset kernel, runs numerical/GPU acceptance and full-size
smoke, then starts production only after acceptance succeeds. No wall-clock
limit or automatic monitoring task is added. Acceptance uses short games;
production restores max 4000 actions and the checked reference configuration.

This baseline establishes an auditable training recipe. Opening diversity and
playing strength remain experimental outcomes, not acceptance assertions.
