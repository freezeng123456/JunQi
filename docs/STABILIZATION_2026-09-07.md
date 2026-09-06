# Main-line stabilization — 2026-09-07

This note records the release gates after the observation compaction and
GraphStem migration.  `main` is the canonical branch for new work.

## Completed by this stabilization change

- CUDA schema checks compare the compiled extension against the Python runtime
  constants instead of pinning the obsolete 412-channel value.
- CUDA parity runs on relevant pushes to `main`, nightly, and on manual
  dispatch.  It includes observation, step, CombatMemory, engineer, CSR legal
  mask, legal-action and compact-history checks.
- New PPO checkpoints record observation/action/policy-stem schema metadata.
  Legacy checkpoints remain loadable when their state-dict keys and shapes are
  actually compatible.
- Resume and baseline loads fail before mutation with a targeted message when a
  checkpoint crosses a schema boundary such as 412→317 channels or
  CNNStem→GraphStem.
- Tracked H20 configs no longer contain a machine-local evaluation checkpoint.
  An optional compatible yardstick is supplied at launch time.
- The historical architecture document is explicitly marked stale where its
  old tensor counts no longer match the runtime implementation.

## Canonical branch and `master`

`main` and `master` have diverged.  Do not merge `master` wholesale into
`main`: its DDP/compact-history work predates the 317-channel and GraphStem
changes.  Audit each `master`-only training commit against current
observation/network semantics, then port only the still-needed behavior with
current parity tests.

After the required pieces are ported, freeze or delete `master` so a feature
cannot be "fixed" on a second product line without reaching the default branch.

## External repository-admin actions

These cannot be implemented by a code PR:

1. The README classifies the project as proprietary/internal while GitHub
   currently exposes the repository publicly.  Decide which statement is
   authoritative.  If the internal classification is correct, make the
   repository private and audit history for secrets/private artifacts.  If
   public visibility is intentional, fix the licensing/classification text.
2. Protect `main` and require the normal CPU/legacy CI checks before merge.
   Keep the self-hosted GPU workflow off untrusted pull-request events; it runs
   on trusted `main` pushes, schedule and manual dispatch instead.

## Experiment gate after merge

GraphStem injects **static road/rail adjacency**, not occupancy-conditioned
long-range rail reachability.  Its correctness tests establish topology and
gradient flow, not playing strength.

Before promoting it as stronger than the previous CNN stem, run a controlled
paired evaluation with the same training budget, setup/opponent distribution,
seeds and optimizer.  Report paired H2H/Elo (or paired score interval),
environment steps, samples/sec, peak GPU memory and learning curves.  Do not
start a long production run solely from the CPU microbenchmark.
