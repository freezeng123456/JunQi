# JunQi Phase 1 M1/M2 — Engineering Optimization Report

**Date**: 2026-04-23  
**Scope**: Benchmark, optimize and harden the completed M1 (CPU batched) / M2
(GPU `legal_action_ids_batch`) milestones against the ataraxos reference system.

## 1. Baseline (before optimization)

| Test suite              | Result            |
|-------------------------|-------------------|
| `tests/` (613 tests)    | 613 pass, 9 skip  |
| `test_legal_actions_gpu`| 10/10 pass        |

Throughput on an NVIDIA A100-ish CUDA device (`junqi_cuda` built with
CMAKE_CUDA_ARCHITECTURES=75;80;86;89;90):

| N     | kernel-only | end-to-end | bottleneck |
|-------|-------------|------------|------------|
| 32    | 435 k/s     | 42 k/s     | pack 62%   |
| 128   | 829 k/s     | 54 k/s     | pack 80%   |
| 512   | 1105 k/s    | 56 k/s     | pack 83%   |
| 1024  | 1257 k/s    | 50 k/s     | pack 87%   |
| 2048  | 1519 k/s    | 47 k/s     | pack 90%   |

Observation: the CUDA kernel already exceeds the Phase 1b target (≥500 k/s)
by 2.5× even at modest batch sizes, but the Python→GPU hand-off dominated
the end-to-end path.  This mirrors the classical ataraxos finding:
*"game simulation is highly optimized, but training pipeline wastes 20–50%
of GPU capacity through unnecessary CPU synchronization and redundant kernel
launches."*

## 2. Optimizations applied

### 2.1 Vectorize `_pack_state_arrays` (`junqi_rl/env_gpu.py`)

**Before** — pre-allocated `(N, K)` buffers filled via a Python `for i, env in
enumerate(envs)` loop with 17 field assignments each.

**After** — one `states` list resolution, then `np.concatenate` per field.
`np.concatenate` is ~3× faster than the equivalent `np.stack` + reshape
because it's written in C and skips the intermediate 2-D allocation.

**Impact @ N=1024**: pack 18.0 ms → 7.6 ms (2.4× faster).

### 2.2 Lite H2D upload path (CUDA + pybind11)

The `legal_action_kernel` only reads 6 of the 20 SoA fields
(`piece_seat_arr`, `piece_type_arr`, `alive`, `pos_x`, `pos_y`,
`cell_piece_id`).  The previous code uploaded all 20 on every call —
burning ~70% of PCIe bandwidth on unused data.

Added `DeviceGameStateBatch::copy_from_host_legal_lite` that:

1. Packs the 6 fields into a single pinned host staging buffer
   (allocated once, size scales with largest batch seen).
2. Transfers the whole blob with one `cudaMemcpy`.
3. Runs a small scatter kernel on device to split the blob into the
   individual SoA arrays.

Accompanying Python helper `_pack_state_arrays_lite` produces just the 6
arrays the kernel needs.

**Impact @ N=1024**: end-to-end 10.0 ms → 3.9 ms (**2.67× speedup**),
throughput 106 k/s → **272 k envs/sec**.

## 3. After optimization

| N     | kernel-only | old e2e | new e2e (lite) | speedup |
|-------|-------------|---------|----------------|---------|
| 32    | 432 k/s     | 42 k/s  | **206 k/s**    | 4.9×    |
| 128   | 831 k/s     | 54 k/s  | **268 k/s**    | 5.0×    |
| 512   | 1167 k/s    | 56 k/s  | **284 k/s**    | 5.1×    |
| 1024  | 1289 k/s    | 50 k/s  | **272 k/s**    | 5.5×    |
| 2048  | 1523 k/s    | 47 k/s  | **211 k/s**    | 4.5×    |

The kernel itself is ~1.3 M envs/sec — 2.5× the Phase 1b target.  The
practical Python-callable throughput @ N=1024 is **272 k envs/sec**, a
**5.5× improvement** over the baseline end-to-end path.

## 4. Correctness — strict delivery testing

### 4.1 Regression coverage

| Test suite                             | Pass | Skip | Notes                             |
|----------------------------------------|------|------|-----------------------------------|
| `tests/` (full regression)             | 624  | 9    | 11 new tests added                |
| `test_legal_actions_gpu.py`            | 10   | 0    | M2 parity (32 envs × 50 steps)    |
| `test_legal_actions_gpu_stress.py`    | 8    | 0    | 50 games × 5 depths + N=256 + full trajectory |
| `test_gpu_legal_lite.py`               | 3    | 0    | Lite vs full upload parity        |

Skipped tests are torch/DARK-mode only and unrelated to GPU functionality.

### 4.2 Rules compliance audit

GPU kernel reviewed against `docs/RULES.md` §2.1-2.6 and `LEGACY_PARITY.md`:

| Rule section                                  | GPU behaviour                       | OK |
|-----------------------------------------------|-------------------------------------|----|
| §2.3 legal-move gate (seat, alive, mobile)    | lines 250–268 of `game_state.cu`    | ✅ |
| §2.3.1 orthogonal/diagonal 1-step             | ADJACENT_CELLS slots 0–3 / 4–7      | ✅ |
| §2.3.2 non-engineer rail ray                  | STRAIGHT_RAIL_RAYS walk             | ✅ |
| §2.3.3 engineer BFS                           | ENG_BRANCHES 2-chain walk           | ✅ |
| §2.4 path blocking (empty intermediates)      | ray breaks on first occupied cell   | ✅ |
| §2.5 camp occupation (cannot attack in-camp)  | `!CAMP_FLAT[nb]` enemy guard        | ✅ |
| §1.5 curve rails                              | 0 curves in current board geometry  | N/A |

### 4.3 Bug fixes during audit (from previous session)

1. **Section 3c rail ray k=0 blocking** — ray started at k=1 instead of
   k=0, skipping the blocking check on the immediate neighbour.  Fix: walk
   from k=0, check for blocking but do not emit (3a handles that cell).
2. **Engineer BFS k=0 double-emit** — the first entry in each BFS branch
   is always the immediate rail ortho neighbour, which §3a already emits.
   Same fix applied.

## 5. Remaining opportunities (future work)

The end-to-end path is now kernel-compute-bound for N≤512 and pack-bound
for N≥1024.  Next wins, in order of ROI:

1. **Persistent pack buffers in Python** — currently each `np.concatenate`
   re-allocates the output.  Carrying a per-size buffer pool could save
   ~1 ms at N=1024.
2. **GPU-side game step (M3)** — eliminates the state repack entirely:
   the CPU never touches SoA except at episode boundaries.  This is the
   ataraxos model (state lives on GPU end-to-end) and is scheduled for
   M3.  After M3 the pack cost disappears.
3. **CUDA streams + async H2D** — overlap the lite upload with the kernel
   of the previous batch.  Requires double-buffered staging, but doubles
   sustained throughput for long rollouts.

## 6. Files changed

- `src/env/cuda/src/game_state.cu` — added `copy_from_host_legal_lite` + scatter kernel + header include.
- `src/env/cuda/include/junqi_cuda.h` — declare `copy_from_host_legal_lite`.
- `src/env/cuda/src/bindings.cpp` — bind `copy_from_host_legal_lite` to Python.
- `junqi_rl/env_gpu.py` — vectorize `_pack_state_arrays`; add `_pack_state_arrays_lite`.
- `tools/bench_gpu_legal_actions.py` — new benchmark harness.
- `tools/profile_pack.py` — new profiler for pack breakdown.
- `tests/test_legal_actions_gpu_stress.py` — new stress parity suite (8 tests).
- `tests/test_gpu_legal_lite.py` — new lite-path parity suite (3 tests).
