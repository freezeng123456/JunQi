# Learned arrangements with counted self-play

The previous facts-only campaign trained moves with independent random legal
arrangements. ArrangementNet was disabled. Its final preserved checkpoint on
2026-10-22 contains 5,713,909 completed games; these are move-training games,
not prior arrangement-training experience.

## New campaign

- Initialize move weights from the preserved checkpoint, with a fresh move
  optimizer and a randomly initialized arrangement policy
- Keep `known_only` observations, 317 spatial features, zero global inputs,
  no legacy belief probabilities, no reward shaping, and full self-play
- Use `configs/v100_facts_arrangement.yaml`: a two-layer, 128-dimensional
  autoregressive setup transformer, conditioned on seat
- Generate 128 seat arrangements every eight rollouts, independently combine
  them into 4,096 boards, and train setup PPO from terminal team rewards
- Count each completed four-player game once in a new lineage; inherited move
  experience is recorded separately in the run protocol

The October 22 run targets the remaining 4,286,091 completed games of the
previously authorized ten-million-game campaign. The 5,713,909 inherited games
and new arrangement games must remain separately reported. Acceptance and
resume games count toward the new target; they are not discarded.

## Correctness and efficiency

The legal mask reserves enough remaining headquarters, rear-row and non-front
slots to finish placing the flag, mines and bombs. These are exact setup
constraints. The mask permits every complete legal lineup, including a
commander in an unused headquarters; tactical preference must be learned.
Counted training rejects any random fallback rather than crediting it as a
neural-policy sample.

GameProgress records the reset seed and setup-pool generation for every live
game. Refreshing the pool affects new games only. Terminal events are copied
once per rollout, and their original setups receive rewards in one batch.
This path adds no per-move arrangement callbacks or engine readbacks. A setup
receives +1 for a team win, -1 for a loss, and 0 for a draw. All four seats get
their respective team rewards. Duplicate terminal outcomes are aggregated.

Checkpoints retain both networks and optimizers, in-flight CUDA games, RNG,
active setup generations, and setup replay statistics. The setup refresh
schedule uses absolute rollout indices, so resume does not insert an extra
refresh. The buffer covers the maximum game lifetime plus a refresh interval.
Each new terminal batch is trained once with the configured PPO epochs;
ready rows are not repeatedly trained without receiving a new outcome.

## Acceptance and evaluation

One V100 allocation runs native compilation, actual CUDA regression tests,
1,000 normal-length training games, a resumed segment of at least 200 games,
then the remaining formal campaign. Formal training has no assistant-imposed
elapsed-time cap. It saves every 10,000 completed games and evaluates every
100,000 games.

Two comparisons are saved against the frozen 5,713,909-game move policy:

1. Identical fixed random setups with swapped team assignments, isolating
   move-policy differences
2. Each agent uses its own setup mechanism: learned setups for the candidate,
   random setups for the old baseline, with swapped team assignments across
   fixed held-out scenario seeds

The second comparison bootstraps scenario pairs, since the combined boards
change when team assignments swap. It must not be reported as identical-board
pairing. Records include setup source, board hash and assignment. Evaluation
does not replace the live training pool and restores training RNG.

Required launch evidence: finite model and optimizer tensors, nonzero setup
updates, changed setup policy head, all four terminal credits per completed
game, successful continuation, and actual formal training progress. These
checks establish implementation correctness, not improved playing strength.
