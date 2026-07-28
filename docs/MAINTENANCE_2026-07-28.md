# Maintenance baseline — 2026-07-28

This pass turns the imported project into a maintainable private repository
without changing the original GTK piece bitmaps.

## Delivered

- Portable Python packaging, core/RL test profiles, strict core typing and
  GitHub Actions jobs for Python 3.10/3.12, RL CPU and C/GTK builds.
- CombatMemory v6 `eaten_by_pid` state on CPU and CUDA, including allocation,
  reset, upload/download, kernel updates, observation channels and 16-field
  parity coverage.
- Count-based balanced evaluation on both teams with paired seeds, greedy
  policy actions and Wilson 95% confidence intervals.
- A bounded SHA-256 checkpoint league with deterministic sampling, historical
  cross-play and Elo updates.
- Automatic policy-augmented evaluation replays, including whether each action
  came from policy sampling, greedy policy or the random opponent.
- A local visual replay viewer for both ordinary and policy trajectories.
- Legacy client hardening: persistent app mute, active audio-process cleanup,
  checked thread creation, bounded protocol parsing and graceful rejection of
  malformed replay moves instead of process-wide assertions.
- Restored the 12 deterministic observation golden hashes for the stable
  v6/412-channel layout; the regression gate is no longer permanently skipped.

## Validation boundary

The CPU/core suite, static checks, wheel installation, live local replay HTTP
service, legacy builds and malformed-UDP smoke test run on macOS and in normal
GitHub Actions.

CUDA source and parity contracts are complete, but actual kernel execution must
run on a Linux NVIDIA worker. Use the manual `CUDA parity` workflow or:

```bash
python build_cuda.py
python -m pytest -q \
  tests/test_gpu_step_batch.py \
  tests/test_gpu_obs_parity.py \
  tests/test_gpu_combat_memory_parity.py
```

H20 training quality is not a code-completion claim. A model should only be
promoted when paired multi-seed evaluation is complete and the required Wilson
lower bound is met. Inspect saved progress games with:

```bash
junqi-replay exps/<run>/replays/eval_<rollout>_00_team0.npz
```

## Recommended next work

1. Attach a self-hosted NVIDIA runner to the manual CUDA workflow and retain
   parity logs as release evidence.
2. Run a short H20 smoke experiment before any long training job, then compare
   current EMA, random opponent and league checkpoints with identical seeds.
3. Gradually replace the remaining legacy GTK/C global state with explicit
   lifecycle ownership. The input boundaries are now hardened, but the client
   remains a 2018-era single-process architecture.
4. Keep checkpoints, replay datasets and credentials out of Git. Revoke any
   token that has been pasted into chat or terminal output.
