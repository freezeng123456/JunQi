# P1 BeliefNet — design & landing plan

## Goal

Replace JunQi's hand-coded deductive belief rules (R1, R4, R5/R7, R6, R9,
I5 — already implemented on GPU in `src/env/cuda/src/belief.cu`) with a
neural belief network that outputs a distribution over the 12 tracked
piece types for each enemy piece, given the observer's full history.

**Why**: deductive rules give a binary "known / uniform-over-remaining"
belief. A neural net can interpolate with prior regularities like
"ZHADAN rarely sits in front row", "JUNQI is always in a stronghold",
"a piece that moved 2 steps on turn 1 is likely a fast ranked combatant",
etc. In Ataraxos Stratego this lifts HALF_DARK → DARK win-rate by a
measurable margin (~+5–10% vs uniform belief).

---

## Ataraxos reference architectures

Ataraxos ships three variants of the belief network. All three share:

- **Input**: `(B, T, C_in, H, W)` stacked infostate tensors (T=time, the
  observer's sequence of observations across the whole game so far) +
  piece-id one-hot planes.
- **Encoder**: spatial attention per time step; the temporal axis is
  consumed via LSTM or per-layer temporal attention.
- **Decoder**: causal transformer over the (up to N_CLASSIC_PIECE) unknown
  enemy pieces. Each piece gets a positional embedding; the decoder
  predicts `(N_PIECE_TYPE,)` logits autoregressively (masked-softmax
  across remaining per-type inventory + has-moved constraint).
- **Loss**: cross-entropy between predicted log_prob and the truth-at-death
  labels (when a piece is revealed, its type is the label).
- **Training signal**: Ataraxos logs three losses:
  - `ce_loss` — main training signal.
  - `marginalized_uniform_kl` — diagnostic; should drop as the net beats
    a naive uniform-over-remaining baseline.
  - `uniform_kl` — same baseline but without position marginalization.

### Variant 1: `BeliefTransformer` (152 LOC)

- Stateless, "single-step" encoder (reads only the current board, no time).
- 6-layer encoder + 6-block decoder, 512 embed, 8 heads.
- Simplest; lowest training cost.
- Trade-off: can't infer "this piece moved fast in turn 2" because history
  is absent.

### Variant 2: `LSTMBeliefTransformer` (262 LOC)

- Interleaves spatial attention (per time step) with **one LSTM over
  time**, inserted after layer `lstm_before_layer=3`.
- 6 spatial layers + 1 temporal LSTM + 6-block decoder.
- Residual LSTM connection + layer norm.
- Better accuracy than stateless; cheap inference (LSTM is O(T)).

### Variant 3: `TemporalBeliefTransformer` (242 LOC)

- Interleaves spatial attention with **causal temporal attention per
  layer** (6 pairs of spatial + temporal).
- Most expressive; most compute.
- Their final production config.

Ataraxos's config picks: `use_lstm_model=False, use_temporal_model=True`.

---

## JunQi adaptation — key differences

| Axis | Ataraxos Stratego | JunQi 四国军棋 |
|------|-------------------|----------------|
| Players | 2 | 4 (2 teams × 2 seats) |
| Board | 10×10 with 2 lakes (N_OCCUPIABLE_CELL=92) | 17×17 with 5 camps/seat, 2 strongholds/seat (NUM_ON_BOARD_CELLS=129) |
| Piece types | 12 (R, S, 2..9, B, F, Bomb) | 12 (JUNQI..GONGB), vocab idx 0..11 |
| Pieces per player | 40 | 25 per seat × 4 seats = 100 total, but each OBSERVER tracks only **enemies** (2 seats × 25 = 50 pieces max) |
| "Unknown" pieces per observer | ~40 at game start | 50 (2 enemy seats × 25 pieces) |
| Belief shape | `(N_CLASSIC_PIECE, N_PIECE_TYPE) = (40, 12)` | `(50, 12)` OR `(4 seats, 12 types, 289 cells)` (current dense layout) |
| Handedness | Yes (left-right symmetry) | **No** (4-rotational symmetry instead) |
| Feature orchestrator inputs | threaten / evade / actadj / cemetery / battle / protect / piece_ids | We have the 256-channel observation (including dark_own/dark_teammate, belief planes, move_bucket, death_reason…) |

### Architecture choice for JunQi P1

**Start with Variant 1 (stateless `BeliefTransformer`)**. Rationale:
- Current observation already encodes **move_history** (32 channels) and
  **death_reason** (12 channels), which together capture much of what
  Ataraxos's temporal attention would learn. The 256-channel observation
  is the JunQi analog of Ataraxos's "infostate_tensor + piece_ids".
- Stateless means we can plug it straight into the existing per-step
  observation builder: feed the 256-channel obs through the encoder,
  decode into `(4 × 12 × 289)` to replace the fixed prior table at
  `_build_belief_prior_table`.
- Training-time: re-use the existing `GpuRollout` — no circular buffer
  needed.
- If accuracy plateaus, graduate to LSTMBeliefTransformer later.

### Decoder target shape

Two options, match Ataraxos contracts:

**Option A (per-piece)**: `(B, 2 * 25, 12)` — 2 enemy seats × 25 pieces,
one distribution per piece. Requires a permutation: sort enemy pieces by
seat×slot so positional embeddings are meaningful.

**Option B (per-cell)**: `(B, NUM_ENEMY_CELLS, 12)` — one distribution
per enemy-occupied cell on the board. Simpler to match the existing
`_beliefs` buffer shape `(N, 4, 12, 289)` — just fill cells where an
enemy piece sits.

**Recommendation: Option B.** JunQi's current observation/belief pipeline
is cell-indexed, not piece-indexed. Option B drops in place with zero
downstream changes. Cost: we don't benefit from piece-ordering, so the
decoder attends over **cells** instead of pieces. This is actually what
Stratego's encoder does too — they only restructure into piece order in
the decoder.

---

## Detailed architecture (P1 first cut)

### Config

```python
@dataclass
class BeliefNetConfig:
    n_encoder_layer: int = 4           # smaller than Ataraxos (6) for T4
    n_decoder_block: int = 4
    n_head: int = 8
    embed_dim: int = 256               # vs Ataraxos's 512 — T4 memory
    ff_factor: int = 4
    dropout: float = 0.0               # no dropout at inference
    pos_emb_std: float = 0.1
    use_temporal: bool = False         # start stateless
```

### Forward contract

```
Input:
  obs_spatial : (B, 256, 17, 17)   — same as JunqiNet
  obs_global  : (B, 28)            — same as JunqiNet
  enemy_mask  : (B, 289)           — bool; True where any enemy piece sits

Output:
  belief_logits : (B, 289, 12)     — per-cell 12-class logits
                                     (for cells without enemies, we'll
                                      ignore via enemy_mask downstream)
```

### Body

1. **Encoder**: CNN stem (reuse `junqi_net.CNNStem`, reduce 256 → 256)
   → flatten to (B, 289, 256) → add positional embedding → N layers of
   `SelfAttentionLayer` (reuse the causal-off version from `arrangement_net.py`).

2. **Decoder**: skip a separate decoder for P1. Just take the encoder
   output, project `(B, 289, 256) → (B, 289, 12)` via `nn.Linear`, and
   mask via `enemy_mask`. This gives us **non-autoregressive** beliefs —
   every cell independently predicts its own 12-way distribution.

   **Why drop the autoregressive decoder?** It's worth ~5% accuracy vs
   non-AR, but it requires implementing per-piece ordering (slot_idx) in
   the decoder — adds 50+ LOC and a new sampling loop. Save for P1.5 if
   win-rate gains don't appear.

### Constraint masking

Each cell c on seat s must mask logits to types that are:
1. Still in seat s's remaining inventory (use `belief_remaining_arr`).
2. Compatible with the slot's placement constraints (reuse
   `SLOT_TYPE_ALLOWED` from `arrangement_net.py` — JUNQI only in
   stronghold, DILEI only in back two rows, ZHADAN not in front row).

Both are static per-cell-per-seat; stash precomputed bool tables.

---

## Loss formulation

```python
def compute_belief_loss(
    logits_pred: Tensor,           # (B, 289, 12)
    true_type_idx: Tensor,         # (B, 289) long — ground truth when revealed,
                                   # -1 (sentinel) when still unknown
    enemy_mask: Tensor,            # (B, 289) bool
) -> dict[str, Tensor]:
    # Only contribute loss from REVEALED enemy cells (true_type_idx >= 0).
    revealed = (true_type_idx >= 0) & enemy_mask
    if not revealed.any():
        return {"ce_loss": torch.zeros((), device=logits_pred.device)}

    log_prob = F.log_softmax(logits_pred, dim=-1)   # (B, 289, 12)
    ce_per_cell = -log_prob.gather(
        -1, true_type_idx.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)                                    # (B, 289)
    ce_loss = (ce_per_cell * revealed).sum() / revealed.float().sum().clamp(min=1)

    # Diagnostics (uniform baseline)
    n_remaining = _build_remaining_vec(enemy_mask)   # per-seat remaining counts
    uniform_prob = n_remaining / n_remaining.sum(-1, keepdim=True)
    uniform_ce = -(uniform_prob * log_prob).sum(-1)
    uniform_kl = (uniform_ce - (-uniform_prob * uniform_prob.log()).sum(-1))

    return {
        "ce_loss": ce_loss,
        "uniform_kl": (uniform_kl * revealed).sum() / revealed.float().sum().clamp(min=1),
    }
```

The "true label" is recovered during rollout:
- On combat resolution, `Event.EAT` / `Event.KILLED` / `Event.BOMB`
  reveals the dead piece's type.
- Death events produce `state.deaths[pid].piece_type` (already tracked).
- Terminal labels: every surviving piece's type is revealed at game end.

So labels become available **reactively** — each `add_labels(env_state,
fired_envs)` call at termination writes 25+25=50 enemy-piece types per
finished env into a replay buffer.

---

## Integration with existing belief pipeline

Current flow:
```
GpuRollout.reset()
  → _build_belief_prior_table()  # fixed 30×12 prior
  → init_beliefs_for_reset_envs()  # uploads to d_belief
  → ... step_batch ...
  → update_beliefs_after_step()   # applies R1/R4/R5/R7/R6/R9/I5
```

Target flow:
```
GpuRollout.reset()
  → belief_net.forward(obs_batch, enemy_mask)   # (N, 289, 12)
  → replace d_belief[cell, type] with softmax of logits
  → ... step_batch ...
  → update_beliefs_after_step()   # keep deductive rules as corrections
```

The neural net **sits in parallel with** (not replaces) the deductive
rules. After each step we run the deductive updater first — R1 migration,
R4 flag capture — then **re-run belief_net.forward** to refresh the prior
for next step.

Why keep both? The deductive rules give bit-exact updates for
deterministic events (flag captured → that cell has JUNQI with
probability 1). The neural net interpolates for everything else. Layering
them means the net doesn't have to learn what the rules already know.

Implementation hook: `GpuRollout.update_beliefs_device()` already takes
a `step_result` and calls `_cuda.update_beliefs_after_step`. We add one
line: `if self._belief_net is not None: self._belief_net_refresh()`.

---

## Buffer and training loop

### Data model

Per-env rolling replay of `(obs_snapshot, revealed_labels_later)`. The
snapshot is captured **when a piece dies**; the obs that goes in is the
obs **one step before the death**. On `add_rewards`-style callback at
game end, we also stash surviving-piece types from the final state.

For P1 simplicity, we don't buffer full trajectories (Ataraxos's LSTM
variant needs that; ours doesn't). Just per-step `(obs_at_step_t,
type_labels_revealed_at_step_t)` tuples.

### Buffer

`BeliefBuffer` (CPU, maybe ~50k entries).
- `add(obs, enemy_mask, type_labels, mask_labels)` — called during
  rollout on every reveal.
- `sample(batch_size)` — uniform-random minibatch.
- `trim(max_age_steps)` — LRU drop when full.

### Training step

Analogous to `ArrangementPPOTrainer.train_epoch`:
```python
for batch in buffer.sample(batch_size):
    logits = belief_net(batch.obs_spatial, batch.obs_global, batch.enemy_mask)
    losses = compute_belief_loss(logits, batch.true_type_idx, batch.enemy_mask)
    losses["ce_loss"].backward()
    optim.step()
    ema.update()
```

Defaults: `lr=5e-5`, Adam (not AdamW), `max_grad_norm=0.5`, batch=64,
epochs=2/rollout.

---

## Landing plan (P1.1 → P1.7)

| Step | Deliverable | Est. LOC | Tests |
|------|-------------|----------|-------|
| **P1.1** | `junqi_rl/networks/belief_net.py` — BeliefNetConfig + BeliefNet (CNN + transformer encoder + per-cell linear head + slot-type mask) | ~280 | shape/grad/mask |
| **P1.2** | `junqi_rl/belief/buffer.py` — CPU BeliefBuffer with add/sample/trim | ~220 | round-trip, LRU |
| **P1.3** | `junqi_rl/belief/reveal_tracker.py` — hook into termination + combat events to emit labels | ~150 | event → label mapping |
| **P1.4** | `junqi_rl/training/belief_ppo.py` — trainer (just CE loss, EMA) | ~180 | loss decreases, EMA works |
| **P1.5** | `GpuRollout.refresh_beliefs_neural()` — upload BeliefNet output to `d_belief` every k steps | ~80 | parity with deductive-only baseline |
| **P1.6** | `scripts/train.py` integration (BeliefTrainConfig in TrainConfig.belief) | ~60 | smoke v16 unchanged, v17 with belief runs |
| **P1.7** | `exps/beat_random_v17_belief/cfg.yaml` + 500-rollout comparison | ~100 YAML | compare win_rate to v16 |

**Total: ~1,070 LOC + ~400 LOC tests.** Similar scale to P0.

---

## Success criteria

At the end of P1:

1. All unit tests pass (BeliefNet forward/grad, buffer LRU, loss > 0 on
   random init, loss decreases with gradient steps).
2. 500-rollout v17 smoke run: no NaN, no CUDA OOM, fps ≥ 70% of v16 fps
   (i.e. BeliefNet doesn't kill throughput).
3. Belief CE loss at end of training is **below** the uniform_kl baseline
   (means the net has learned something beyond "uniform over remaining
   types").
4. v17 win_rate vs random at rollout 500 is **≥ v16 win_rate + 3%**
   (weak signal, but real).

If any of these fail, we diagnose before moving on. If (4) fails but
(3) passes, we suspect the main RL loop isn't benefiting from the
improved belief; next step would be to check how BeliefNet output enters
the observation builder.

---

## Open questions (defer to P1.3-P1.4)

- **Trajectory replay or single-step?** P1 starts single-step. If
  accuracy is poor, promote to LSTM (Variant 2).
- **EMA or no EMA?** Ataraxos uses EMA with decay 0.99 for belief
  inference. Cheap, follow suit.
- **Per-seat or shared across seats?** Ataraxos's belief is
  per-observer (two observers for 2 players). We need 4 seats × per-seat
  belief. Use one shared net with a **seat embedding** (same trick as
  ArrangementNet) to keep parameter count flat.

---

## Timeline

Given P0 took ~4h of focused implementation and got us 7 commits + 62
tests, P1 should fit in ~6h:
- P1.1 (net)                     → 1h
- P1.2 (buffer)                  → 45min
- P1.3 (reveal tracker)          → 45min
- P1.4 (trainer)                 → 45min
- P1.5 (GpuRollout wire-in)      → 1h
- P1.6 (train.py integration)    → 45min
- P1.7 (validation run — 3h bg)  → parallel with P1.5/6

Call it **1 focused day** of implementation + 3h background training.
