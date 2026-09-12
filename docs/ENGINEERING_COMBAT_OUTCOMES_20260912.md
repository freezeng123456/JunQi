# Evaluation, neural belief constraints, and combat outcome features

The candidate keeps the 317 spatial channels, 28 global values, and 16,641
canonical actions. `configs/iteration_combat_outcomes.yaml` enables the new
policy residual. Existing configs leave it disabled, with identical parameter
keys and baseline initialization. This is an engineering candidate; it does
not establish better playing strength.

## Evaluation protocol

GPU evaluation assigns the requested games fixed IDs and completes bounded
waves without replacing short games while long games are still in progress.
Every game receives a uniform lineup from `setup_seed + game_id` (or
`seed + game_id` when no separate setup seed is supplied). First and later
waves follow exactly the same setup source. The training reset pool is not
used or replaced. Opponent randomness and stochastic policy sampling are
keyed by game seed, ply, and stream; batch scheduling cannot advance another
game's random stream. Floating-point network execution across batch sizes is
not promised bit exact.

Both team assignments use paired game IDs and setups. `num_envs` is honored;
completed results, ongoing games, completed-game lengths, and total work are
counted separately. Each evaluation step updates deductive beliefs. Training
writes `eval_<rollout>_random.jsonl` and `_h2h.jsonl`; dense evaluation writes
per-game JSONL beside the checkpoint by default or to `--records`. Records
include ID, seeds, lineup hash, side, outcome, length, and termination reason.
The legacy `eval_fixed_setup_pool` flag selects the fixed lineup seed; its
name remains for configuration compatibility.

Old evaluation scores are not directly interchangeable with the corrected
suite. Re-evaluate comparison checkpoints under this protocol.

## Neural belief integration

Each CUDA game batch owns its soft belief buffer and observer mapping.
Creating or evaluating another world cannot overwrite a training world's
beliefs. Full reset clears combat memory and the move-history ring.

Before the first neural refresh, the rollout allocates an independent
rule-belief buffer initialized from the deductive state. Both buffers then
receive rule events and resets. Neural refresh reweights only live enemies,
preserves excluded types and known one-hot identities, and preserves other
cells. Nonfinite/negative mass is excluded; zero mass over allowed types falls
back to the normalized rule prior. Post-step soft beliefs are projected onto
the independent rule support. Numerical certainty from the network never
becomes a permanent rule fact. With neural refresh unused, the additional
rule buffer and second rule update are absent.

BeliefNet's existing training labels and outputs are in world coordinates;
its inputs are canonical observations with observer identity. Only the live
enemy occupancy mask is rotated back before applying these constraints.

## Three action features

For each legal attack, sum the target's observer-visible type probabilities
under the exact combat rules into:

1. attacker eats and survives;
2. attacker alone dies;
3. both pieces die.

Moves receive three zeros. The feature is a derived representation of existing
information, not calibrated game-win probability and not new private knowledge.
It reads only own-piece planes, enemy occupancy, and enemy belief planes from
the acting observation. It never queries hidden types or enemy true legal moves.

The policy adds a 3 -> 16 -> 1 residual (81 trainable parameters), whose last
layer starts at zero. Computation is shared by attacker type and target cell:
12 x 129 triples suffice before gathering source-cell scores. No action-feature
history tensor is stored. The common policy head covers collection, selected
PPO action evaluation, shared policy/value encoding, EMA copies, and greedy
assessment. The value head is unchanged. Checkpoint metadata records
`combat_feature_version`; enabled and disabled models cannot silently exchange
checkpoints. Enabling this candidate on old weights requires an explicit
migration, rather than silently loading incomplete weights.

Collectors log chosen learner attack fraction/count and mean of the three
outcomes over chosen attacks. Random-opponent moves and terminated lanes are
excluded from the GPU learner metrics. These are feature-exposure diagnostics,
not calibration or strength measurements.

## Lossless device history compression

Observer-specific deductions and type masks retain all four views. Only the
public chain PID relation is shared across observers. Store its low/high
bitmaps once instead of four times; derive reverse-capture bitmaps by taking
the transpose for each observer's own victim IDs during minibatch gathering.
All reconstructed fields are written, including zeros, when scratch is reused.
The live game-state layout and existing observation kernel interfaces stay
compatible. CPU state/replay serialization is unchanged.

Persistent native history decreases from 47,243 to 33,803 bytes per transition,
a 13,440-byte saving. At N=128, T=512, this removes 840 MiB from history; the
combat-memory portion decreases by 43.75%. Scalar rollout storage is separate.
These numbers describe capacity, not whole-run throughput. Reconstructed
observations, legal masks, and selected combat features must match exactly.

The all-seat GPU observation buffer is allocated on first use. Actor-only
collection/evaluation avoids that buffer; BeliefNet or another all-seat consumer
allocates it when required. At N=128 its spatial/global payload is 178.986 MiB.
This conditional saving is separate from the history saving.

## Validation entry point

`experiments/engineering_validation_20260912/run.sh` builds for the assigned
V100 (`sm_70`), runs bounded CUDA parity/lifecycle/history/policy checks, and
records a small paired forward-timing probe. It has a single-card 20-minute
scheduler limit; it performs no strength-training campaign. Source commit,
archive digest, scheduler result, tests, and measurements belong in the run's
evidence directory. Successful submission/build alone is not validation.
