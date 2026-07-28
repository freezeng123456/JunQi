# Ataraxos ↔ JunQi — Gap Analysis

**Paper**: *Superhuman AI for Stratego Using Self-Play Reinforcement Learning and Test-Time Search* (Ataraxos, 2025).
**Scope**: full 46-page paper (main text + all appendices) vs JunQi's current code state after P0 (ArrangementNet) landing and P1 (BeliefNet) planning.
**Purpose**: identify what we have, what we don't, and which of the missing pieces are genuinely load-bearing for playing strength vs. which are computational luxuries we can skip on our hardware budget.

We grade each axis on a **three-symbol scale**:
- ✅ parity or close-enough
- 🟡 partial / simplified version present
- ❌ not implemented

---

## 0. Headline differences

| Axis | Ataraxos (Stratego) | JunQi (四国军棋) | Grade |
|---|---|---|---|
| Hardware used for full training run | 16 × H100 for 1 week (RL) + 4 × H100 for 4 days (belief) ≈ $8k of compute | 1 × T4 (15 GB, no bfloat16), best-effort | n/a |
| Self-play throughput claimed | ~10 M env-steps / s over the cluster | ~1k–2k env-steps / s single-T4 (same order on a per-GPU basis vs paper's 1536 env × 202 moves every iter ≈ 310k steps/iter/GPU) | 🟡 |
| Total env-steps at end of RL run | **2.08 × 10¹¹** (208 billion transitions) | Largest run so far: v14b ≈ 5 × 10⁸ (~400× smaller) | ❌ at scale |
| Game | 2-player, 10×10 board, 40 pieces, lakes | 4-player (2v2), 17×17 board, 25 pieces per seat, camps/strongholds, railroads | different problem; most algorithms port |
| Mixed precision | **bfloat16** everywhere (3× speedup ablated in paper) | fp16 for policy rollout, fp32 for arr net | 🟡 (no bf16 support on T4 anyway) |

### The one-sentence takeaway

After P0 we have the **correct algorithmic skeleton** (autoregressive setup net with categorical value + entropy prediction + MC backup + advantage filtering) but none of the **scale-driven ingredients** (dynamic damping, 40-ply search, belief-guided sampling) that the paper claims give the last 100+ Elo. The gap analysis below breaks down where each piece sits.

---

## 1. Network architectures

### 1.1 Setup / arrangement net

Ataraxos's setup net (Table 23) is a *causal, autoregressive, row-major placement head*. This is literally what P0.1 of our `ArrangementNet` does.

| Hyperparameter | Ataraxos | JunQi v16 | Grade |
|---|---|---|---|
| Depth | 4 | 4 | ✅ |
| Embed dim | 512 | 512 | ✅ |
| Heads | 8 | 8 | ✅ |
| FFN inner | 2,048 (factor 4) | 2,048 (ff_factor=4) | ✅ |
| Pos-emb init std | 0.1 | 0.1 | ✅ |
| Total parameters | 12.6 M | ~12.6 M (we should measure; matches config) | ✅ |
| **Outputs** | (win/loss/draw probs, conditional entropy prediction, next-piece dist) | (N_VF_CAT=3 cat-vf, ent_pred, next-piece dist) | ✅ |
| Entropy target | H(σ̄ \| σ; θ_t) divided by 10 | ent_pred / reg_norm = ent_pred / 10 (coded as `1.0/reg_norm` in loss) | ✅ |

**Grade: ✅**. This is a very close copy of their setup net; we followed their architecture verbatim to within the constraints of JunQi's 4-seat board geometry (30 slots vs 40).

### 1.2 Move / policy net

Ataraxos's move net (Table 24) is a much bigger beast than anything we currently run.

| Hyperparameter | Ataraxos | JunQi v16 `JunqiNet` | Grade |
|---|---|---|---|
| Depth | **8** | 4 | 🟡 |
| Embed dim | **384** | 128 | 🟡 |
| Heads | 8 | 4 | 🟡 |
| FFN inner | 1,536 (factor 4) | 512 | 🟡 |
| Total parameters | **14.7 M** | ~1.5 M (rough) | ❌ 10× smaller |
| Value head | **3-category** (win/loss/draw) | 3-bin categorical (v15_3bin onwards) | ✅ |
| Positional embedding | learned, std 0.1 | learned | ✅ |
| Input channels | 455 (see §2 below) | 256 | 🟡 |

**Grade: 🟡**. The move net is the single biggest architectural gap. Ataraxos's move net is ~10× our parameter count and 2× our depth. On a T4 we cannot run their 14.7 M-parameter move net at their batch/throughput combo, but scaling our net toward their shape is probably the single most effective thing we could do (depth 4→6, embed 128→256 would already close ~half the gap).

### 1.3 Belief net

Ataraxos ships three variants of a belief net; their production config (Table 25) is the **large temporal one** with 57 M parameters.

| Hyperparameter | Ataraxos production | P1 JunQi plan (doc'd in `P1_BELIEF_NET_PLAN.md`) | Grade |
|---|---|---|---|
| Encoder depth | 6 | 4 | 🟡 |
| Decoder blocks | 4 (autoregressive, causal) | **0** (non-AR per-cell linear head) | ❌ (-5% acc acknowledged in plan) |
| Embed dim | 512 | 256 | 🟡 |
| Heads | 8 | 8 | ✅ |
| Dropout | 0.2 | 0.0 | 🟡 |
| Temporal attention | Yes, interleaved per layer (the "TemporalBeliefTransformer" variant) | Off for P1 (stateless "BeliefTransformer" variant) | ❌ |
| Parameters | 57.1 M | ~5 M (estimated) | ❌ 10× smaller |
| Loss | cross-entropy against ground-truth hidden pieces | cross-entropy against revealed enemies | ✅ (formulation matches) |

**Grade: ❌**. We explicitly picked the smallest of the three variants for P1 and dropped the AR decoder. That is a deliberate cost-of-labor decision, but worth labelling as a known shortfall. When P1 lands we can measure the belief CE against the uniform-over-remaining baseline; if our belief net barely beats uniform, the gap here is probably what's to blame.

### 1.4 Attention parameterization

Ataraxos: "Pre-layernorm was used for all of the networks" — they reference Xiong et al. [33]. Also, they use learned absolute positional embeddings everywhere, including in the belief decoder's "piece order" inputs.

JunQi: we use pre-norm (checked in `arrangement_net.py` and `junqi_net.py`), and F.scaled_dot_product_attention for flash attention. ✅ parity.

---

## 2. Observation / infostate representation (Appendix C)

This is where we have the **largest per-feature gap** that is quietly algorithmic.

Ataraxos's infostate is **455 channels** (!) for the move net:
- **Channels 0–42**: piece properties (ranked, unrevealed, revealed type, colour, lake, etc.) + k-move-rule fractional counter.
- **Channels 43–108**: **threat / evasion / active-adjacency** bookkeeping (piece has ever threatened a piece of type T, has ever evaded a piece of type T, has ever been actively adjacent to an unrevealed piece, etc.). Mirrored for the opponent.
- **Channels 109–130**: starting positions of captured pieces.
- **Channels 131–250**: death locations bucketed by 6 distinct **causes of death** (attack stronger, attack equal, attack unrevealed, got attacked stronger, got attacked equal, got attacked unrevealed).
- **Channels 251–354**: **protection-move bookkeeping** (piece C protected piece B against piece A; maintained for revealed types × both players × protector/protected).
- **Channels 355–455**: **101 channels** one-hot encoding the starting square of each currently-on-board piece (the "track of provenance" feature).
- **Plus**: 32 channels encoding the **most recent 32 moves** (+1 at dest, -1 at source).

JunQi's 256-channel obs:
- Board statics (piece/me/teammate/enemy/camp/stronghold/railroad) ✅
- Move history (32 channels) ✅ (matches Ataraxos)
- Death reason (12 channels) 🟡 — we track *that* pieces died, but not with their 6-cause taxonomy (attacker stronger/equal/unrevealed × attacker/defender side).
- Bucket counters (move/eat/survive) ✅ partial — they extract more granular features
- Belief planes (seat × 12 × 289) ✅
- **No threat/evasion/active-adjacency features** ❌
- **No protection-move features** ❌
- **No "starting square of each piece" provenance one-hot** ❌

### Why this matters

Each of the threat/evasion/protection families encodes **durable history**: "piece X was *ever* in situation Y" is a summary statistic of everything the observer has ever seen piece X do, without needing an LSTM. Ataraxos designed these features explicitly so their **stateless** move net could see history-derived information without paying for temporal attention (they say this explicitly in Appendix K: "Ataraxos accesses history through features rather than learning across time directly").

JunQi currently relies on raw move_history channels (32-step ring) and whatever the CNN can extract. This is **cheaper per feature** but **much less information-dense**. If we ever want JunQi's move net to play deceptively / recognise threat patterns without a big temporal stack, we need to land these feature families.

**Grade: 🟡**. We have the raw mechanics (move history, death reasons) but not the derived-feature engineering. Biggest single opportunity for observation-level improvement.

**Effort to close**: moderate — most of these are "piece X has ever …" booleans maintained incrementally on the GPU. One kernel per feature family, updated inside `record_move_history` / `update_state_after_step`. Budget: ~1–2 days per family, 3 families to match, call it 1 week.

---

## 3. Training loop & data collection

### 3.1 Self-play & data pipeline

| Aspect | Ataraxos | JunQi v16 | Grade |
|---|---|---|---|
| Parallel envs per GPU | **1,536** | 128 (v16) | 🟡 (memory-bound on T4) |
| Steps per env between train iters | 202 (= 101 per player) | 512 (env steps per rollout) | 🟡 (different cadence) |
| Transitions per training iter | ~310k / GPU × 16 GPU = 5M | ~65k / T4 | 🟡 |
| Pool-of-setups mixing | 1,000 setups per player per GPU, **regenerated after each training iteration** | n_arr=1024, refresh_every=1 rollout | ✅ matches |
| **Advantage filtering** (quantile+magnitude) | **top 25% by \|δ\|** AND \|δ\| ≥ 0.01 | adv_filt_rate=0.75, adv_filt_thresh=0.01 | ✅ exact match |
| Epochs per iter (move net) | **1** | 4 (v16 `num_epochs_per_rollout`) | 🟡 — we're over-training per rollout |
| Epochs per iter (setup net) | 5 (over completed-game setups only) | 4 | ✅ close |
| Data batch for move net | 202 batches of ~1536 (pre-filter) → ~380 / batch (post-filter) | minibatch_size=512 | 🟡 |
| Data batch for setup net | 1,024 | 256 | 🟡 |

**Grade: ✅ on the algorithms, 🟡 on the scale**. We match their advantage-filtering quantile and magnitude numbers to the third decimal. We've copied the pool-of-setups-per-iter pattern. We're just running ~100× fewer steps per iter than they are.

### 3.2 Loss formulations

**Setup net** (Appendix D.3):
```
δ = (o − E[v_θt(σ)]) + α·(H(σ̄|σ;θt) − h_θt(σ))
L_π = −min(rδ, clip(r, 0.8, 1.2)δ) + 0.1·KL(π_θ(σ), π_θt(σ))
L_v = −log v_θ(o|σ)
L_h = ((H(σ̄|σ;θt)/10) − h_θ(σ))²
L_setup = L_π + 0.5·L_v + L_h
```

JunQi v16 `ArrangementPPOConfig`:
```
clip_range=0.2, policy_coef=1.0, vf_coef=0.5, ent_pred_coef=0.5, kl_coef=0.01
```

**⚠️ kl_coef discrepancy**: Ataraxos uses `0.1` for the reverse-KL-to-sampling-policy term; we have `0.01`. Worth bumping to match — this is a single YAML edit.

**⚠️ ent_pred_coef discrepancy**: Ataraxos's `L_h` has *implicit* coefficient 1 (it's added directly, no scaling). Our `ent_pred_coef=0.5`. Probably fine; ent_pred is a diagnostic regression target, doubling its coef won't move much.

**Move net** (Appendix D.4):
```
L_π = −min(rδ, clip(r,0.8,1.2)δ) + 0.1·KL(π_θ, π_θt) + α·KL(π_θ, ρ)
L_v = cross_entropy(ξ, v_θ(x))
L_move = L_π + 1.0·L_v
λ (advantage) = 0.5
λ (outcome)   = 0.8
```

JunQi v16:
```
kl_coef=0.1 (policy vs sampled π_θt)  ← ✅ matches
td_lambda=0.8 ← maps to our *gae_lambda* name; Ataraxos calls it "outcome-λ"
gae_lambda=0.5 ← this is their *advantage-λ*, so ✅ exact match
vf_coef=1.0  ← ✅ matches
policy_coef=1.0  ← ✅ matches
```

**Grade: ✅**. v16's core PPO losses are close-to-verbatim ports of Ataraxos's. The `arr_trainer.kl_coef=0.01` vs paper's 0.1 is the one small delta worth fixing.

### 3.3 Magnet policy (move net reverse-KL to uniform mover)

Ataraxos's "magnet policy" ρ is deliberately silly: **"selects piece to move uniformly and moves it uniformly"** (Table 21). The reverse-KL-to-magnet coefficient α schedules as `0.05 / (iter)^0.3` — so at iter=1 it's 0.05, at iter=40k it's ~0.0015.

JunQi: we have `uniform_magnet: true` wired in v16 PPO config. I haven't audited the *schedule*; we might be using a constant coefficient. 🟡.

**Action item**: confirm in `junqi_rl/training/ppo.py` that the magnet-KL coefficient has an annealing schedule matching `0.05 / t^0.3`. If not, schedule it.

### 3.4 Regularization temperature α for setup net

Ataraxos schedules α = `0.1 / (iter)^0.3` for the setup net max-entropy coefficient. At iter=1 it's 0.1; at iter=40k it's ~0.003.

JunQi: we have `reg_temp` in arrangement buffer — **check whether this is annealed or constant**. From P0.3 memory, I believe we use a fixed `reg_temp`.

**Action item**: verify + add annealing schedule `reg_temp(t) = reg_temp_init / t^0.3`.

### 3.5 Dynamic damping (the paper's §2.4 flagship innovation)

This is the paper's named technique: **coordinated scheduling of regularization temperature + update size**. As training progresses:
- Early: **large regularization** (strong α), **large update steps** (high LR, large batch advantages) → "heavily regularized exploration".
- Late: **small regularization** (weak α), **small update steps** (low LR, small clipped steps) → "small unregularized exploitation".

The observable signatures (Figures 9, 11): as α↓, entropy↓, KL(θ, θt)↓, import-ratio-clip-rate↓, gradient-norm↓, LR↓ *in unison*.

In their LR schedule: `clip(0.5 / t^1.1, lr_floor=5e-6, lr_ceil=1e-4)`.

JunQi v16: we have *individual* annealing knobs (`lr_decay`, `temperature_decay`, `temperature_coef`…) but I don't believe we have the **coordinated** schedule that Ataraxos emphasises. Also, checking v16 config:

```yaml
lr_coef: 0.5
lr_decay: 0.6            # t^0.6, not t^1.1
lr_ceil: 0.0001          # matches
lr_floor: 5.0e-06        # matches
temperature_coef: 0.05
temperature_decay: 0.3   # t^0.3 — matches Ataraxos for α
```

So we already match the magnet-KL annealing exponent (0.3). **But our LR decay exponent is 0.6, theirs is 1.1**. With t^0.6 we decay slower, effectively doing more late-stage updating. Worth experimenting.

**Grade: 🟡**. Individual knobs exist and some match to 1 decimal; the *coordination* pitch of "dynamic damping" is partly implicit because we use the same monotone `t^exp` schedule family. Biggest literal discrepancy: `lr_decay` 0.6 → 1.1 to match paper.

### 3.6 Exponential moving average

Ataraxos: **EMA decay 0.999** for both setup and move networks. Used for evaluation.

JunQi v16 main: `ppo.ema_decay: 0.999` ✅.
JunQi v16 arr: `arr.ema_decay: 0.999` ✅.

**Grade: ✅ parity**.

---

## 4. Test-time search (their big fat §2.5 / D.7)

This is the **single largest algorithmic piece we are completely missing**.

Ataraxos's search procedure:
1. Sample ~`1000 / num_legal_actions` hidden-piece configurations from the **belief network**, given the current position.
2. Run 1,000 **depth-40 rollouts** (one per (legal move × sampled configuration) combination; after the initial forced move, subsequent 39 moves are from the move net).
3. Compute per-legal-move average value `q̂` from the value head of the move net.
4. Play `M ~ π_search` where:
   ```
   π_search = argmax_π  ⟨q̂, π⟩ − α·KL(π, ρ) − β·KL(π, π_θ)
            ∝ exp(q̂/β) · ρ^(α/β) · π_θ^(β/β)     (closed form)
   ```
   with α=0.002 (reverse KL to magnet policy) and β=0.02 (reverse KL to move policy).

**Ablation (Table 28)**:
- No search: Elo **2095**.
- 40-ply × 1000 rollouts: Elo **2218** — a **+123 Elo** lift.
- Drop β (the network-KL) → Elo crashes to **1733**: the search without policy anchor *overfits to the opponent model of the move net* and becomes exploitable. This is a crucial practical lesson.

Cost: ~1.26 H100-seconds per move.

**JunQi status**:
- Belief network: in planning (P1).
- Rollout search: ❌ not implemented at all.
- Policy-anchored mirror descent: ❌ not implemented.
- Magnet policy: 🟡 available as `ρ(x) = uniform over legal`, same as Ataraxos.

**Grade: ❌ entirely**. This is "P2 or P3 territory". It also depends on P1 belief quality being good enough for the sampled configurations to be diverse-but-plausible; a uniform belief will produce terrible search results.

**Estimated implementation effort**:
- Rollout primitive on GPU (replay env from a given state, with all pieces revealed to the searcher): moderate; piggybacks on GpuRollout. 3 days.
- Belief sampling (conditional, per-seat inventory-constrained — critically important for JunQi's much tighter inventory): 2 days.
- Search loop + mixture policy: 1 day.
- Tuning α/β: 2 days.

So ~2 weeks. But we expect 100+ Elo out of this.

---

## 5. Infrastructure / performance

| Item | Ataraxos | JunQi | Grade |
|---|---|---|---|
| GPU-native rollout buffer | StrategoRolloutBuffer, zero-copy to trainer | GpuRollout, zero-copy via DLPack ✅ | ✅ |
| Vectorized env throughput | 10 M states/s cluster-wide | ~2 k states/s single T4 | 🟡 (~1k× smaller cluster) |
| bfloat16 | **Yes everywhere** (3× speedup ablated) | No (T4 doesn't support it; we use fp16 for policy rollout) | ❌ (hardware constraint) |
| Torch compile | Presumably used (not stated) | `torch_compile: true` in v16 | ✅ |
| Mixed-precision grad scaling | implicit via bf16 | fp16 needs grad scaler | hardware-forced difference |
| PyTorch flash attention | Yes | Yes (`F.scaled_dot_product_attention`) | ✅ |

**Grade: ✅ on code design, ❌ on absolute scale** — but that gap is hardware, not software.

---

## 6. Self-play ablation findings (Figure 13)

The paper runs single-seed ablations on a single H100:
1. Without distributed training → same *shape* of Elo trajectory, ~150 Elo lower final point.
2. Above + **without setup learning (uniform setups)** → trajectory *flattens*: "both because of the weakness of uniformly distributed setups and because of the bad assumptions the associated self-play distribution causes the move network to make about the pieces of its opponents".
3. Above + **without bfloat16** → slower iterations.
4. Above + **without advantage filtering** → slower iterations + larger move entropy + sample inefficiency.

**Implications for JunQi**:
- Ablation (2) is direct confirmation that our P0 ArrangementNet is load-bearing, not cosmetic. The setup-learning-off curve is visibly below the setup-learning-on one even at 45h wall-clock.
- Ablation (4) tells us that our `adv_filt_rate=0.75, adv_filt_thresh=0.01` is in fact a crucial part of the recipe. Worth keeping even if it looks wasteful.

---

## 7. What we gained in P0 vs what's next (prioritised gap list)

**What we have after P0**:
- ✅ ArrangementNet with cat-VF + entropy prediction + AR piece placement head.
- ✅ Arrangement pool upload, per-iter refresh, hash-dedup buffer.
- ✅ Advantage filtering matched to paper (quantile 0.75, thresh 0.01).
- ✅ λ-returns with λ=0.5 advantage, λ=0.8 outcome.
- ✅ EMA 0.999 on both setup and move nets.
- ✅ Uniform magnet policy wired for move net.

**Ranked gap list — do in this order**:

| # | Gap | Effort | Expected lift | Can we run it on T4? |
|---|---|---|---|---|
| 1 | **Belief net (P1)** as planned | 1 day | +3–8% win-rate alone; gates search | Yes (5M params is fine) |
| 2 | **Confirm `kl_coef`=0.1 for arr net** (currently 0.01) | 15 min | 🤷 small | Yes |
| 3 | **Anneal `reg_temp` = 0.1/t^0.3** for arr net | 1 h | measurable — they use it explicitly | Yes |
| 4 | **Anneal magnet-KL α = 0.05/t^0.3** for move net | 1 h | measurable | Yes |
| 5 | **Bump `lr_decay` 0.6 → 1.1** to match paper | 15 min | small, possibly negative early | Yes |
| 6 | **Reduce move-net `num_epochs_per_rollout` 4 → 1** to match paper | 15 min | 4× more data volume per wall-clock, better sample efficiency | Yes |
| 7 | **Scale move net: depth 4→6, embed 128→256** toward their 14.7M | 1 day to implement; may OOM on T4 with `num_envs=128` | big | **maybe** — will need `num_envs=64` |
| 8 | **Death-cause taxonomy (6-way)** in obs | 2 days (CUDA kernel) | small-to-medium | Yes |
| 9 | **Threat/evasion/active-adjacency feature families** | 1 week | medium (stateless way to add history) | Yes |
| 10 | **Protection-move feature families** | 3 days | small-to-medium | Yes |
| 11 | **Starting-square provenance channels** | 3 days | small | Yes |
| 12 | **Promote belief net to LSTMBeliefTransformer (P1.5)** | 2 days | +5% belief acc | Yes |
| 13 | **Promote belief net AR decoder** | 2 days | +5% belief acc | Yes |
| 14 | **Test-time search** (rollout + mixture policy) | 2 weeks | +100 Elo per paper | Yes, at inference |
| 15 | **bfloat16** | — | ~3× throughput | ❌ needs A100/H100 |

Items 2–6 are a single afternoon's worth of config tweaks. Items 7–11 are the "architectural" round. Item 14 is the moonshot.

---

## 8. Explicit NON-gaps (we are in better shape than paper in these aspects)

- **Game generality**: JunQi natively supports 4 players × 2 teams while Ataraxos is single-game-only. Our belief net must handle 4 seats via shared-weights + seat embedding, which we've already designed (P1 plan §2).
- **4-rotational symmetry data augmentation** (item in our ROADMAP): Stratego uses a left-right augmentation post-hoc on setups (Appendix H intro). We get 4× free symmetries instead.
- **Compact action space** (129² = 16,641 vs 289²): we already adopted this; it halves our move-head parameter count.
- **Categorical value function**: we adopted their 3-bin scheme in v15_3bin and maintained it; no regression.

---

## 9. Computational budget sanity check

Ataraxos consumed ~**208 × 10⁹ env-steps** and reached 2218 Elo vs strong humans.

Our v16 smoke gets ~2000 env-steps/s on T4. To match their 208G steps, we'd need ~**3.3 years** of continuous single-T4. Clearly not happening.

A more reasonable v17-v20 target: **1 × 10⁹ env-steps** (1 week of T4, ~500×10⁶ on-policy transitions). That is the scale at which we should expect to see the *shape* of the paper's training curves, not their absolute magnitudes. Their ablation (Figure 13) shows the single-H100 run reaches ~1900 Elo vs a "perfect-play" reference in ~45h of wall-clock — a single T4 is ~4× slower than a single H100 for fp16 matmul, so **~200h of T4** ≈ 8 days is our realistic rough match to their single-H100 ablation.

---

## 10. Summary

- **Algorithm parity: surprisingly good** — after P0, the *shape* of our training loop matches theirs down to λ, filtering quantile, and EMA decay.
- **Biggest algorithmic gap: test-time search** — explicitly +120 Elo per their ablation. Worth a dedicated P2 when P1 belief lands.
- **Biggest "quiet" gap: observation feature engineering** — threat/evasion/protection families give ~50% more information density than JunQi's current obs.
- **Biggest "ambient" gap: compute** — 16×H100×1-week vs 1×T4, i.e. ~600× less. Most of this is not closeable; some is (bf16 on A100 if we ever get one; scale move net to 14.7M params once the GPU allows).
- **Six near-trivial fixes** (§7 rows 2–6) should land in v17 before we bother with anything else.

**Next concrete moves** (in order):
1. Land P1 BeliefNet (already planned).
2. Apply the 6 afternoon-scale config fixes in §7 as v17 baseline.
3. After v17 stabilises, attack the observation feature families (threat/evasion/protection) as v18.
4. Belief-quality permitting, prototype test-time search as P2 standalone experiment.
