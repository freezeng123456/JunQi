# Strict Ataraxos-style JunQi run

This run is a fresh self-play experiment. It does not resume a checkpoint
whose opening behavior has already collapsed.

The move policy keeps the existing source–destination bilinear action head.
The change is the exploration reference distribution and its schedule:

\[
\rho(a\mid s)=
\frac{1}{M(s)}\frac{1}{D_i(s)},
\]

where \(i\) is the source piece, \(M(s)\) is the number of movable source
pieces, and \(D_i(s)\) is the number of legal destinations for source \(i\).
The policy is trained with the reverse KL to this distribution, using
\(\alpha_t=0.05/t^{0.3}\), together with the old-policy reverse KL, PPO
clipping, gradient clipping, and the annealed learning-rate schedule from
Ataraxos.

The opening metric is collected from the action actually applied in every
completed self-play game. The old synthetic random-board opening probe is
disabled for this run. Each record in `training_games.jsonl` contains
`first_action`, `first_source`, and `first_destination`. To summarize it:

```text
python scripts/analyze_first_moves.py <run-dir>/training_games.jsonl
```

The first gates are 5,000, 10,000, 20,000, and 50,000 completed games. At
each gate, inspect first-source entropy, top-1 mass, move entropy,
`KL(policy, magnet)`, old-policy KL, clip fraction, and returns. These gates
diagnose early collapse; they are not claims of final playing strength.

Belief training is disabled in this phase. It will be trained later on
trajectories from the stable final policy and evaluated separately from the
blueprint policy.
