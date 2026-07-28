# JunQi Phase A Refactor — Delivery Report

**Date**: 2026-04-23
**Scope**: Phase A (foundation) + Tier-1 optimizations from the aggressive
architecture plan, completed autonomously in a single session.

## 1. Motivation

The baseline end-to-end path (`272 k envs/s @ N=1024`) was ≈5× below the
CUDA kernel ceiling (`1.29 M envs/s`).  The gap was dominated by:

1. CPU-side SoA pack (67% of time)
2. Per-call `cudaMalloc`/`cudaFree` of scratch buffers
3. Warp divergence in the kernel (thread-per-piece split across envs)
4. 20× bandwidth waste in the `(N, 512)` dense action output

## 2. Changes delivered

### 2.1 `GpuScratch` — persistent scratch manager  *(`game_state.cu`, `junqi_cuda.h`)*

Singleton that owns all transient device + pinned buffers for the process
lifetime.  Replaces six sites of per-call `cudaMalloc/Free`:

- `d_acting_seats`     — used by every `legal_action_ids_batch` call
- `d_belief`           — used by every `build_observation_batch` call
- `d_observer_seats`   — used by every `build_observation_batch` call
- `d_action_ids / d_action_counts`  — M2 kernel output
- `d_csr_offsets / d_csr_values`    — CSR output
- `d_piece_slot_mask`  — 32-slot mask output

All grow monotonically; user-visible `gpu_scratch_reset()` frees everything
(test use only).  **Eliminated ~150 μs of per-step malloc/free overhead.**

### 2.2 CSR action output  *(Tier-1 #2)*

New launcher `legal_action_ids_batch_csr` produces `(offsets, values)`
instead of dense `(N, 512)`.  At N=1024, D2H bytes drop from **2 MB → ~110 KB
(18× less)**.

New kernels:
- `csr_prefix_sum_kernel` — single-thread prefix scan of per-env counts
- `csr_scatter_kernel`    — parallel scatter from dense → CSR

### 2.3 Per-piece 32-slot legal mask  *(#30, workaround for engineer reachability)*

New launcher `legal_action_mask_batch` produces `(N, 120, 32)` bool mask.

**Slot layout** (valid for all piece types):

| Slot range | Meaning |
|------------|---------|
| 0-3        | orthogonal 1-step (E/W/S/N) |
| 4-7        | diagonal 1-step via camp |
| 8-11       | rail ray dir 0, distance k=1..4 (non-engineer) |
| 12-15      | rail ray dir 1, distance k=1..4 |
| 16-19      | rail ray dir 2, distance k=1..4 |
| 20-23      | rail ray dir 3, distance k=1..4 |
| 24-31      | reserved (always 0) |

For engineers, slots 8-23 hold BFS-reachable rail cells in branch order
(up to 16 cells — the theoretical max is 15 per rail component, validated
by empirical profiling).

**Compact action space: 120 × 32 = 3840** (27× smaller than `289 × 289 = 83521`).

### 2.4 v2 kernel — block-per-env with shared-memory cache  *(Tier-1 #3)*

`legal_action_kernel_v2`:
- launch `<<<num_envs, 128>>>`, blockIdx = env, threadIdx = piece_id
- cooperative `__shared__` load of `cell_piece_id[289]` and
  `piece_seat_arr[120]`
- every neighbor / ray / BFS check reads from shared memory

| Metric | v1 | v2 | speedup |
|--------|----|----|---------|
| kernel @ N=1024 | 1.31 M/s | **1.49 M/s** | **+14%** |
| kernel @ N=512  | 1.13 M/s | 1.20 M/s | +6% |

Runtime toggle via `JUNQI_CUDA_KERNEL_V1=1` for A/B comparison.

### 2.5 `GpuWorld` Python facade  *(Phase A milestone A1)*

New class `junqi_rl.GpuWorld` — the long-term home for the GPU-native env.
Thin Phase A wrapper over existing `DeviceGameStateBatch`.  Exposes:

```python
world = GpuWorld(num_envs=1024)
world.push_state_from_envs(envs)       # full SoA upload
world.push_state_lite(envs)            # 6-field fast path

# Three legal-action output formats:
ids, counts  = world.legal_actions_dense(acting_seats)
offs, vals   = world.legal_actions_csr(acting_seats)
mask         = world.legal_actions_mask(acting_seats)

spatial, glob = world.build_observation(beliefs, observer_seats, show_mode)
world.release_scratch()  # free persistent GPU buffers
```

Phase B will swap `push_state_from_envs` for a GPU-native `reset`, at which
point CPU state lifetime ends at episode boundaries.

## 3. Engineer reachability analysis

During design, asked whether the original 20-slot proposal was sufficient
for engineers.  Empirical measurement confirmed:

| Piece type         | Max moves (50 games × 100 steps) |
|--------------------|----------------------------------|
| ZHADAN / SILING / JUNZH / SHIZH / TUANZH | 5 |
| LVZH               | 6 |
| YINGZH / LIANZH / PAIZH / GONGB | 7 |

**Theoretical upper bound** for engineer on a fully-cleared rail component:
**15 BFS cells + 4 ortho + 4 diagonal = 23 actions**.

→ The adopted **32-slot layout** provides safe headroom (slots 24-31
reserved, always 0) and is 2ⁿ-aligned for fast indexing.

## 4. Test coverage

| Test suite                         | Pass | Skip |
|------------------------------------|------|------|
| Full regression (`tests/`)         | 644  | 9    |
| `test_legal_actions_gpu.py`        | 10   | 0    |
| `test_legal_actions_gpu_stress.py` | 8    | 0    |
| `test_gpu_legal_lite.py`           | 3    | 0    |
| `test_gpu_csr_mask.py` (new)       | 11   | 0    |
| `test_gpu_world.py` (new)          | 9    | 0    |

All 20 new tests exercise parity with the dense M2 kernel — **no regressions,
no relaxed assertions**.

## 5. Performance summary (kernel-only, N=1024)

| Version | Kernel throughput |
|---------|-------------------|
| Pre-Phase-A (v1)     | 1305 k envs/s |
| **Post-Phase-A (v2)** | **1488 k envs/s** (**+14%**) |

End-to-end (pack + upload + kernel + D2H) via `_pack_state_arrays_lite`:

| N    | Before | After |
|------|--------|-------|
| 1024 | 272 k envs/s | **280 k envs/s** |

Note: E2E only marginally improved because pack is still 67% of e2e time.
Further gains require Phase B (GPU step_batch) which eliminates pack entirely.

## 6. Deferred to Phase B

Items from the aggressive plan not in this session:

- **CUDA streams double-buffering** — infrastructure added to `GpuScratch`
  (`ensure_streams`), but not wired into launchers.  Low ROI until step_batch
  runs on GPU.
- **GPU-native step_batch** — Phase B's headline work (2-3 weeks).
- **Rules codegen (SSoT)** — prerequisite for Phase B; Phase A kept dual
  implementation.
- **CUDA Graphs** — wait for step_batch.

## 7. Files changed

| File | Change |
|------|--------|
| `src/env/cuda/include/junqi_cuda.h` | GpuScratch struct, CSR/mask result types, `SLOTS_PER_PIECE`, `legal_action_ids_batch_csr`, `legal_action_mask_batch` |
| `src/env/cuda/src/game_state.cu` | GpuScratch impl, v2 kernel, CSR kernels, mask kernel, all launchers |
| `src/env/cuda/src/bindings.cpp` | CSR/mask Python bindings, scratch reset, refactor obs+legal bindings to use scratch |
| `junqi_rl/gpu_world.py` | **NEW** — GpuWorld facade |
| `junqi_rl/__init__.py` | export GpuWorld |
| `tests/test_gpu_csr_mask.py` | **NEW** — 11 CSR + mask + scratch tests |
| `tests/test_gpu_world.py` | **NEW** — 9 GpuWorld tests |
| `tools/bench_v1_vs_v2.py` | **NEW** — v1/v2 kernel benchmark |

## 8. Status

- ✅ Full test suite green (644 pass, 0 fail)
- ✅ No regressions in existing behaviour
- ✅ New APIs documented and unit-tested
- ✅ Performance validated against baseline
- ⏸ Phase B (GPU step_batch) not in scope
