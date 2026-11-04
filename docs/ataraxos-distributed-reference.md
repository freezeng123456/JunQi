# Distributed Ataraxos reference

The eight-GPU profile preserves the single-GPU recipe's **global** batch and
rollout cadence: 1,536 environments, 400 plies per environment, 4,000 generated
lineups, and setup minibatches of 1,024. Each of eight ranks owns 192 environments,
500 generated lineups, and a nominal setup minibatch share of 128. Architecture,
raw same-seat TD/GAE, filtering rate, learning schedules, and EMA remain unchanged.
The stopping target is 1,000,000 completed games summed across ranks.

The reference chunk learner broadcasts initial parameters before constructing
EMA/collection snapshots. It sums gradients weighted by the global selected
sample count, then clips and steps once per global logical row. Empty local rows
contribute zero; globally empty rows are skipped on every rank. The history
quantile and shuffled row order are shared. Setup-ready rows are gathered,
shuffled globally and partitioned without truncating to the smallest rank.
Collection, unfinished games, and layout reward ownership remain rank local.

Global game IDs namespace rank-local IDs. A single writer publishes game records
and global progress. Each checkpoint contains shared learner/Adam/EMA state plus
checksummed per-rank runtime shards (world, pending decisions, advantage history,
setup buffer, counters and RNG). CUDA RNG capture touches the current rank's
GPU only. Exact resume requires the same world size and local environment count.
Restart truncates uncheckpointed journal tails before replaying them. Two full
checkpoints are retained; compact move/setup raw+EMA exports remain at every
100k milestone for subsequent match comparisons. Parallel collection can cross
a requested milestone by up to the completions in one global rollout; artifacts
report the actual count.

Validation is gated: two-process CPU union-batch equivalence (uneven/empty ranks),
then real CUDA collection/training and bit-exact restart on two and eight ranks,
then three production-size rollouts for throughput. `scripts/slurm_ataraxos_reference.sh`
separates acceptance from production. The acceptance harness intentionally uses
a bounded rollout ceiling; production always uses the completed-game target.
No acceptance result is evidence of playing strength or long-run non-collapse.

Live BSCC-N26 submission policy rejected multi-node jobs during this task;
only single-node allocations of 1–8 GPUs are allowed. Use one eight-GPU node
with `--nodes=1 --gpus=8 --ntasks=1 --cpus-per-task=80`. The launcher preserves
scheduler device visibility and derives ranks from Slurm/torchrun. An available
two-GPU allocation may run `acceptance2` first. The single eight-GPU `campaign`
job then runs eight-rank restart acceptance and production-size benchmarking,
and starts the million-game run only after both acceptance receipts pass.
Resource queue time is separate from measured training time.
