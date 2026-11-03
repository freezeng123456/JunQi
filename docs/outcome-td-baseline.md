# Ataraxos-aligned outcome TD baseline

This experiment repairs one mismatch in the same-seat raw-advantage lineage:
categorical critic targets now retain the complete loss/draw/win distribution
instead of projecting a scalar TD return back into adjacent outcome bins

For a player's consecutive decisions i and i+1, with p_i ordered loss/draw/win:

- Policy value: V_i = p_i(win) - p_i(loss)
- Raw GAE: A_i = V_(i+1) - V_i + 0.5 A_(i+1)
- Value target: q_i = 0.2 p_(i+1) + 0.8 q_(i+1)
- Terminal successor: one-hot actual team result; no further GAE continuation
- Critic loss: -sum(q_i * log p_i)

At an unfinished rollout boundary the next same-seat prediction bootstraps
the earlier action, while that successor's own unresolved action is carried
once, retaining its original prediction and full behavior policy
An eliminated seat is closed by the final team result, not by another player's
value estimate

The collector, pending cache, minibatches and exact-resume contract carry all
three probabilities; categorical same-seat PPO rejects scalar targets
Historical next-actor training retains its explicitly legacy scalar projection
for reproducibility and is not part of this baseline
Old checkpoint inference is unchanged; old scalar-cache exact training resumes
are rejected, rather than silently relabelled as distributional TD

## Controlled protocol

Use `configs/v100_outcome_td_baseline.yaml` and
`scripts/beijing_outcome_td_baseline.sh`: one V100, no artificial time limit,
fresh initialization, 20,000 completed games, checkpoints and per-game first
moves at 5k/10k/20k (actual counts can overshoot at rollout boundaries)
Keep the same 8-layer 320-wide move network, learned ArrangementNet, 128 parallel
environments, 256 simulator steps, seed 20261024, raw GAE, top-25% filtering,
terminal rewards, known-only 317-channel input, no global summary or BeliefNet
The unchanged exploration schedule is alpha=max(0.015, 0.05/t^0.1)

Acceptance covers probability semantics, pending transitions, deterministic
CUDA save/resume in both storage modes, and a full-size 128-game training smoke
After training, three 400-game paired matches compare 10k:5k, 20k:5k and 20k:10k
on shared random setups with team swaps and seeded stochastic policy sampling
These matches isolate move-policy progress; they do not measure ArrangementNet
quality or establish the causal gain over the earlier target formulation
No automatic monitor is created

## Remaining differences from Ataraxos

This is an alignment baseline adapted to four-player JunQi, not an exact
Stratego reproduction: GraphStem/317-channel inputs and action geometry differ;
action-key width remains 64 rather than 384; rollout parallelism is 128 rather
than 1,536 per GPU; retained timestep batches are much smaller; timestep order
remains chronological; exploration decay/floor differ; evaluation uses current
weights rather than an EMA policy
Changing these together would prevent attribution of the outcome-target repair
Entropy and first-move diversity are diagnostics, not evidence of playing strength

Sources: [Nature paper](https://www.nature.com/articles/s41586-026-11036-y),
[paper and supplement](https://arxiv.org/html/2511.07312v2)
Official code audit: 92db29e8ffc323b1b8a2804b5c3f84695d036b05
