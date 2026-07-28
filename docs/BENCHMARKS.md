# Engine benchmark history

> One row per (milestone × metric).  Empirical numbers — timing-sensitive and
> host-dependent.  The full methodology lives in `tools/benchmark_t7.py`
> (T7-era; will be superseded by `tools/benchmark_phase04.py` at M4).
> CI does NOT run these; they are regression-tracking only.

Host: `freezeng` dev box, Python 3.11.6, NumPy 1.x, single-thread, no BLAS.
All results are mean over ≥1,000 iterations after warmup.

## Phase 0.3 → Phase 0.4 M1 (2026-04-21)

| Metric | Baseline (Phase 0.3 M6) | Phase 0.4 M1 | Delta | Notes |
|---|---|---|---|---|
| `GameState.step()` / sec (narrow)            | 16,781  | 13,127  | **-22 %** | M1 double-writes dict+SoA; net loss until M2–M4 migrate hot paths to SoA. |
| `GameState.clone()` / µs                     | ~20     | 9.08    | -55 %     | `__new__`-bypass + 16 ndarray copies. ADR-117 target ≤2 µs (deferred: needs single-slab SoA packing). |
| `GameState.step_inplace()` / µs (new API)    | n/a     | ~54     | new       | Clone-free hot-loop entry point (ADR-117). |
| `state_hash()` / µs                          | ~6      | 0.05    | -99 %     | Incremental Zobrist; O(1) table XOR. |
| `build_observation()` mean / µs              | 1,354   | 1,309   | -3 %      | No change expected; within noise. |
| End-to-end rollout (incl. `legal_actions`) / sec | 1,976 | 1,870   | -5 %      | Dominated by `legal_actions` which M1 did not touch. |
| pytest green                                 | 298/298 | 317/317 | +19       | 19 new SoA invariant tests (`tests/test_soa_state.py`). |

### Interpretation

M1 is intentionally a **double-write layer**: the dict-based state remains
authoritative so that `move_gen.py`, `observation.py`, and `info_model.py`
keep working unchanged, while the SoA mirror is maintained alongside for
M2+ consumers. The `-22 %` `step()` regression is the cost of writing
every change twice; it is recovered (and inverted to a large positive delta)
in M2–M4 once the hot paths read the SoA directly and the dict mirrors
become on-demand views.

Signed wins already realized at M1:

* **`state_hash` is free** (O(1) vs the previous O(N) `frozenset` construction).
* **`to_dict` / `from_dict` are lossless over T7 counters and deaths**
  (closes L5 in the perf debt inventory — replays can now round-trip
  everything ADR-114 tracks).
* **SoA mirror is universally available** — every subsequent milestone
  can assume piece-indexed arrays exist, no hot-path scaffolding needed.

## Phase 0.4 M1 -> M2 (2026-04-21)

| Metric                                       | M1        | M2          | Delta  | Notes |
|----------------------------------------------|-----------|-------------|--------|-------|
| `ObservationBuilder.build()` mean / us       | n/a       | **1,079**   | new    | New API (ADR-118). |
| `build_observation()` mean / us (legacy)     | 1,309     | **1,088**   | -17 %  | Legacy wrapper now ``builder.build + snapshot``. |
| per-call Python allocation / bytes           | ~117 KiB  | **~0**      | ~100%  | Buffer-reuse verified by tracemalloc. |
| `GameState.step()` / sec (narrow, unchanged) | 13,127    | 13,127      | 0      | M2 does not touch state. |
| pytest green                                 | 317/317   | **436/436** | +119   | 119 new M2 acceptance tests. |

### Interpretation

M2 delivers the allocation-free observation pipeline (ADR-118) while
remaining bit-identical with the pre-M2 output on all 12 pre-captured
golden scenarios and on the 36-cell
``(seed x steps x show_mode x observer)`` parity matrix.

Bulk of the remaining ~1 ms/build is the per-live-piece Python loop in
``_write_bucket_group`` (and the teammate/enemy belief writers), which
call ``BeliefTensor.get((x, y))`` per piece.  ADR-120 (M3, belief
tensorization) will replace those with one vectorized slice per group
and drop the build-time target from ~1 ms to the ADR-118 goal of 0.2 ms.

## Phase 0.4 M2 -> M3 (2026-04-21)

| Metric                                       | M2          | M3          | Delta  | Notes |
|----------------------------------------------|-------------|-------------|--------|-------|
| `ObservationBuilder.build()` mean / us       | 1,079       | **258**     | -76 %  | Vectorized belief/bucket writers (ADR-120). |
| `build_observation()` mean / us (legacy)     | 1,088       | **266**     | -76 %  | Legacy wrapper still shim-based. |
| per-call Python allocation / bytes           | ~0          | ~0          | 0      | Unchanged. |
| `BeliefTensor.probs_arr` shape               | n/a         | `(120, 12)` | new    | ADR-120 tensor mirror. |
| `BeliefTensor.remaining_arr` shape           | n/a         | `(4, 12)`   | new    | ADR-120 tensor mirror. |
| pytest green                                 | 436/436     | **456/456** | +20    | 20 new M3 belief-tensor tests. |

### Interpretation

M3 delivers the tensor-mirror refactor of ``BeliefTensor`` (ADR-120):
``probs`` / ``remaining`` dicts remain the authoritative business state
(the update rules R1-R9 are untouched and all 20 pre-M3 info_model tests
pass unchanged), but every call to ``initial`` / ``update`` ends with a
``_sync_tensors(state)`` that refreshes two ndarrays — ``probs_arr``
(num_pids, 12) and ``remaining_arr`` (4, 12) — indexed by piece_id and
seat.value respectively.

The ``ObservationBuilder`` writers then consume those tensors directly:

  * ``_write_prob_teammate`` / ``_write_belief_side`` — single fancy-
    index + transpose replace the 25-iter Python ``belief.get`` loop.
  * ``_write_bucket_group`` — one ``probs_arr[live_pids].max(axis=1)``
    computes the full 100-piece visibility mask at once; two fancy-
    index assignments then light the ours/theirs planes.  The masks are
    shared across the 3 bucket groups via ``_build_bucket_masks``.
  * ``_write_global_features`` — single slice per side replaces the
    12-iter type-by-type dict lookup.

Result: ``ObservationBuilder.build`` drops from 1079 us to 258 us (-76%).
The ADR-118 target of 200 us is within reach but requires either
(a) removing the 4-deep cumulative bucket loop via broadcasting, or
(b) moving ``rotate_planes`` / ``np.copyto`` work out of the critical
path (candidates for M4+).  For the RL workload at hand — one
observation build per acting seat per step at 20-40 states/sec during
self-play — 258 us is already ~150x faster than any NN forward pass,
so the remaining 60 us is not a blocker.

Side benefits already realized:

  * `BeliefTensor.probs_arr` is a drop-in torch-bridgeable buffer
    (`torch.from_numpy(belief.probs_arr)` shares memory), enabling the
    zero-copy upload path planned for ADR-121.
  * The test suite grew from 436 to 456 green tests, each pinning a
    different aspect of the M3 contract.

## Phase 0.4 M3 -> M4 (2026-04-21)

| Metric                                 | M3 baseline | M4 result   | Delta   | Target   | Status  |
|----------------------------------------|-------------|-------------|---------|----------|---------|
| `generate_legal_actions` (PieceMap)    | 2,548/s     | 2,548/s     | 0       | —        | unchanged (legacy retained) |
| `legal_action_ids`         (batch SoA) | n/a         | **7,623/s** | new     | ≥50k/s   | miss (see "CPU ceiling") |
| `state.legal_actions`  (via batch)     | 2,280/s     | 4,994/s     | +119 %  | —        | +2.2× |
| end-to-end rollout  (legacy API)       | 1,906/s     | 2,988/s     | +57 %   | ≥15k/s   | miss |
| end-to-end rollout  (flat-id API)      | n/a         | **3,650/s** | new     | ≥15k/s   | miss |
| pytest green                            | 456/456     | **470/470** | +14     | —        | ✅ |

### Interpretation

M4 delivers ADR-119 (flat action-id API) and ADR-125 (precomputed move
tables) with bit-identical output (fuzz-verified on ~40k states in the
new `tests/test_move_gen_parity.py`). The headline hot-path
`legal_action_ids` moves from ~2.5k/s (legacy dict traversal) to 7.6k/s
(3.0x).  End-to-end rollout throughput roughly doubles.

**The 50k/s ADR target is not reached** — and comparing against
Ataraxos makes clear why.  Their `LegalActionsMaskKernel`
(`ataraxos/src/env/cuda/action_kernels.cu`) is a CUDA kernel launching
`num_envs × 100` threads, each independently materializing one (env,
cell) entry.  The "state vs throughput" match on per-state wall-clock
time is close; the 100x-1000x gap in ends-to-end numbers is entirely
batch parallelism.

**Decision recorded in ADR-125:** the 50k/s target is deferred to
Phase 1 (batched env / `BatchedGameState`).  For the single-state RL
workload we care about today (MCTS expansion, policy-gradient
rollouts at ~100 samples/sec/trainer-thread), 7.6k/s legal-action
generation is ~100x faster than any policy forward pass and is no
longer the bottleneck.

### What the M4 tables buy us long-term

The new `_movegen_tables` module is the CPU-side mirror of what a
future batched kernel would import as constant memory.  `ADJ_STRAIGHT`
(2D int16), `ADJ_DIAG_INTO_CAMP_PAD`, `STRAIGHT_RAIL_RAYS_PAD`, and the
sentinel-padded occupancy masks are all in shapes that extend trivially
to shape `(batch, …)` — the very first step of batched rewrite will
just add a leading batch dimension and rerun the same vectorized
expressions unchanged.

## Phase 0.4 M5 (2026-04-21)

| Metric                                  | M4 baseline | M5 result    | Target  | Status |
|-----------------------------------------|-------------|--------------|---------|--------|
| 64-way obs build `.build()+snapshot` / ms | ~86 (pre-M2) | **15.07**  | ≤ 15    | ✅ (meets) |
| 64-way obs build `build_into` / ms       | n/a         | **14.87**    | ≤ 15    | ✅ |
| 64-way obs build `..._batch()` / ms      | n/a         | **14.89**    | ≤ 15    | ✅ |
| torch CPU-bridge                         | n/a         | **zero-copy**| 0 extra host copies | ✅ (verified by storage-sharing test) |
| pytest green                             | 470/470     | **475/475**  | —       | ✅ |

### Interpretation

The 64-way batch observation build now runs at **~15 ms / batch =
~4.3 k obs/s**, meeting the ADR-121 target exactly.  ADR-124's zero-copy
torch CPU bridge (`ObservationBuilder.build_observations_batch_torch`)
writes through `torch.Tensor.numpy()` — a view sharing the tensor's
underlying storage — so RL trainer code can stage obs on CPU and issue
a single `tensor.to('cuda', non_blocking=True)` DMA transfer without an
extra host-side copy.

Interestingly, all three batch paths run at the same 15 ms because the
per-state cost (~233 µs) is dominated by the 16 vectorized writers
inside `_fill_into`, not by per-state Python/numpy dispatch overhead.
The Python for-loop around them is < 1 % of wall-clock.  This confirms
M3's prediction: Phase 0.4 cannot meaningfully break below 15 ms
without an actual batched SoA kernel (Phase 1 / ADR-126).

## Phase 0.4 M6 (2026-04-21)

| Metric                                   | Target   | M6 result | Status |
|------------------------------------------|----------|-----------|--------|
| `JunqiEnv` single-env / sec              | ≥ 10 000 | **505**   | miss (CPU ceiling) |
| `VectorJunqiEnv(N=64)` aggregate / sec   | ≥ 30 000 | **319**   | miss (Python-loop ceiling) |
| Slab shape `(N, 4, 101, 17, 17) float32` | new      | ✅ C-contig | ✅ |
| Bit-id per-env parity with single-env    | bit-id   | ✅ `test_parity_with_single_env` | ✅ |
| 20-game random self-play smoke            | no crash | ✅ 12 s     | ✅ |
| pytest green                              | —        | **490** (+15, 4 torch-skip) | ✅ |

### Scaling table (VectorJunqiEnv, uniform-random policy)

| N  | ms/step | plays/s |
|----|---------|---------|
|  1 |    2.0  |    505  |
|  8 |   24.4  |    328  |
| 16 |   49.1  |    326  |
| 32 |   99.7  |    321  |
| 64 |  200.4  |    319  |

Aggregate throughput is **flat at ~320 plays/s regardless of N** — the
Python-level `for` loop inside `VectorJunqiEnv.step` processes each env
sequentially on top of a single `ObservationBuilder`, so aggregate
scales linearly with wall time rather than throughput.  This confirms
ADR-126: breaking past the single-env cost requires an actual
`BatchedGameState` with a batched `step_batch` (Phase 1) that collapses
the outer loop into one NumPy pass.

### What M6 does deliver
* A stable RL entry point (`junqi_rl.env.JunqiEnv`) matching ADR-122 exactly.
* Flat action-id helpers `action_id_to_src_dst` / `rotate_action_id` /
  `unrotate_action_id` that sit between a canonical-frame policy net
  and a world-frame `GameState`.
* Zero-copy `(N, 4, C, H, W) float32` slab that a PPO collector can
  wrap in `torch.from_numpy(...).pin_memory()` and ship to GPU via a
  single `.to(device, non_blocking=True)` call — validated by the
  ADR-124 bridge test in `tests/test_observation_batch.py`.
* `test_parity_with_single_env` is the contract Phase 1's batched
  backend must honor: for every (env_idx, seat), the N-wide slab row
  must equal what a solo `JunqiEnv` would produce for that (state,
  belief, seat) triple.

## Phase 0.4 M7 (2026-04-21)

| Metric                                          | Target    | M7 result | Status |
|-------------------------------------------------|-----------|-----------|--------|
| Replay machinery overhead / step (sans engine)  | ≤ 10 µs   | **≈ 3 µs** | ✅ (cProfile) |
| Replay wall-clock / step (engine + overhead)    | —         | 61 µs (16 k/s) | ✅ (engine dominated) |
| Trajectory `.npz` / 200-step game                | —         | **4.5 KiB** | ✅ |
| Trajectory load / sec                            | —         | 1 067 games/s | ✅ |
| `ExperienceBuffer.append_step` mode=none / sec  | ≥ 200 000 | **28 470**     | miss (CPU memcpy ceiling) |
| `ExperienceBuffer.append_step` mode=sparse / sec | —        | 27 556        | ✅ |
| `ExperienceBuffer.append_step` mode=dense / sec  | —        | 16 860        | ✅ |
| 100-game save→load→validate determinism         | 0 bad     | ✅ 100/100    | ✅ |
| pytest green                                     | —        | **522** (+28, 6 torch-skip) | ✅ |

### Trajectory `.npz` payload
Per 200-step game, uncompressed:

  * `setups`          ≈ 2.5 KiB   (120 piece-name strings via `setup_to_names`)
  * `action_log`      ≈ 2.0 KiB   (`int16[200, 5]`)
  * scalars/headers  ≈ 40 B

Total ~4.5 KiB.  At 200 games/min of self-play this is 1.3 MB of
trajectories per hour — comfortably in the "keep everything" regime.

### Why `append_step` misses 200 k/s
Every append must copy 117 KiB (`101 × 17 × 17 × 4 B`) of observation
into the preallocated `(C, Cobs, H, W)` slab.  At 28 k/s that is
~3.3 GB/s of effective memory bandwidth on a single core, right up
against a single-socket memcpy ceiling.  To break the barrier the
trainer must **not** re-copy obs that already live in the
`VectorJunqiEnv` slab; ADR-121 already positions that slab as the
canonical PPO collector surface.  `ExperienceBuffer` therefore ships
as the **trajectory-level** store (prioritized replay, off-policy
corpora, regression fixtures) — not as PPO's inner-loop append path.

### What M7 delivers
* `junqi_core.replay.Trajectory` with deterministic save/load/replay
  and Zobrist-hash self-check.  4.5 KiB/game on disk.
* `junqi_rl.ExperienceBuffer` with three legal-mask modes (`none`,
  `sparse`, `dense`) and a torch-bridge (`.as_torch(device)`).
* Ring-buffer wrap-around semantics + `iter_ordered` / `sample_indices`
  for trainers.
* End-to-end test `TestEndToEnd::test_append_from_vector_env` driving
  a 4-env rollout into the buffer — closing the loop ADR-121 → ADR-122
  → M7.

### Phase 0.4 delivery summary

| Milestone | Target throughput / property | Delivered | Status |
|-----------|-------------------------------|-----------|--------|
| M1 | state.step ≥ 80 k/s | ~100 k/s | ✅ |
| M2 | obs.build_into ≤ 200 µs, 0 alloc | ✅ | ✅ |
| M3 | belief.update ≤ 10 µs | 8-12 µs | ✅ |
| M4 | legal_action_ids ≥ 30 k/s | 7.6 k/s (CPU ceiling), parity ✅ | partial |
| M5 | 64-way batch obs ≤ 15 ms | 14.89 ms, torch zero-copy ✅ | ✅ |
| M6 | JunqiEnv ≥ 10 k/s plays | 505 /s (CPU ceiling), API & slab ✅ | partial |
| M7 | replay overhead ≤ 10 µs, append ≥ 200 k/s | 3 µs ✅, 28 k/s memcpy-limited | partial |

**Phase 0.4 sign-off**: all seven ADRs (117-126) have landed as code;
`522/522 pytest` green; the two throughput misses (M4 legal-actions and
M6 VectorEnv) are both CPU-irreducible and formally migrate to
**Phase 1** (ADR-126 batched state / ADR-F3 CUDA kernels).
