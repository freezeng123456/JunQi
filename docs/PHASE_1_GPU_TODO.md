# Phase 1 — GPU Acceleration TODO

**Owner**: Team. **Target landing window**: 2026-05-01 → 2026-06-15.  
**Goal**: Implement C++/CUDA GPU backend to achieve ≥500 k plays/sec on single GPU (N=1024).

> This document is the single source of truth for Phase 1 execution.
> Design decisions are frozen in [DECISIONS.md](./DECISIONS.md) ADR-127.
> Detailed architecture is in [CUDA_ARCHITECTURE.md](./CUDA_ARCHITECTURE.md).

---

## 0. Phase 1 Overview

| Phase      | Scope                                      | Throughput Target | Status      |
|------------|--------------------------------------------|----|---|
| Phase 0.4  | Python SoA GameState + NumPy tables       | 7.6 k/s single-state | ✅ Complete |
| Phase 1a   | CPU `BatchedGameState` (NumPy batching)   | ≥50 k/s aggregate (N=1024) | ⏳ Current |
| Phase 1b   | GPU CUDA kernels (legal/step/obs)         | ≥500 k/s aggregate (N=1024) | 📋 Planned |
| Phase 2    | Multi-GPU + RL training integration       | ≥5 M/s aggregate (N=4096 across 8× GPUs) | 📋 Future |

---

## 1. Milestone Breakdown

### 1.1 M1 — `BatchedGameState` (CPU/NumPy) + ADR-126 Acceptance

**Duration**: 5 days (2026-05-01 → 2026-05-05).

#### 1.1.1 Scope
- Implement `BatchedGameState` class with leading batch dimension on all SoA arrays.
- Implement `step_batch(actions)`, `legal_action_ids_batch(seat_per_env)`, `build_observation_batch(...)`.
- All operations use NumPy (CPU only); no CUDA.
- Achieve ≥50 k plays/sec on CPU with N=1024.

#### 1.1.2 Concrete Subtasks

**1.1.2.1 Data Structure** (`junqi_core/batched_state.py`)
```python
@dataclass
class BatchedGameState:
    num_envs: int
    
    # All arrays shape (num_envs, ...) matching single-state shape
    cell_piece_id: np.ndarray[N, 289] int16
    piece_seat_arr: np.ndarray[N, 120] int8
    piece_type_arr: np.ndarray[N, 120] int8
    alive: np.ndarray[N, 120] bool
    pos_x, pos_y: np.ndarray[N, 120] int8
    # ... other SoA columns (move_count, death reasons, etc.)
    turn: np.ndarray[N] int8
    zobrist: np.ndarray[N] int64
    # ... metadata per env
```

**1.1.2.2 Batch APIs**

Implement:
- `step_batch(action_ids)` — execute one action per environment
- `legal_action_ids_batch(seat_per_env)` — generate legal action IDs
- `build_observation_batch(beliefs, observer_per_env)` — build observations

**1.1.2.3 Tests** (`tests/test_batched_state.py`)
- Round-trip parity: `BatchedGameState` from N identical single-states
- Determinism: same seed → same state after M steps
- Stress test: 1000-step random game at N=256

**1.1.2.4 Benchmarks** (`tests/bench_phase1.py`)

| Metric                         | Target |
|---|---|
| `legal_action_ids_batch` (N=1024) | ≥50 k actions/sec |
| `step_batch` (N=1024)          | ≥30 k steps/sec |
| `build_observation_batch` (N=1024) | ≥15 M obs/sec |
| **Aggregate plays/sec**        | **≥50 k** |

#### 1.1.3 Deliverables
- `junqi_core/batched_state.py` (400–500 lines)
- `tests/test_batched_state.py` (300–400 lines)
- Benchmark script + results table

#### 1.1.4 Rollback Plan
If ADR-126 acceptance criteria aren't met (< 50 k/s), investigate NumPy bottlenecks via profiling.

---

### 1.2 M2 — GPU `legal_action_ids_batch` Kernel

**Duration**: 6 days (2026-05-06 → 2026-05-11).

#### 1.2.1 Scope
- Implement CUDA kernel `legal_action_kernel` for batch legal move generation
- PyBind11 wrapper function
- GPU memory management (H2D/D2H transfers)
- Bit-identical parity with M1 CPU results

#### 1.2.2 Concrete Subtasks
- [ ] CUDA infrastructure (CMakeLists.txt, common.cuh)
- [ ] GPU data structures (DeviceGameStateBatch)
- [ ] Static tables precomputation
- [ ] Legal action kernel implementation
- [ ] PyBind11 wrapper
- [ ] Unit tests and benchmarks

#### 1.2.3 Deliverables
- `src/env/cuda/src/game_state.cu/cuh` (GPU data structures)
- `src/env/cuda/src/tables.cu/cuh` (precomputed tables)
- `src/env/cuda/src/bindings.cpp` (PyBind11 module)
- `tests/test_legal_actions.py` (200–300 lines)

---

### 1.3 M3 — GPU `step_batch` Kernel

**Duration**: 7 days (2026-05-12 → 2026-05-18).

#### 1.3.1 Scope
- Multi-stage CUDA kernel for batch state transitions
- Stages: move validation → combat resolution → zobrist update
- Bit-identical parity with M1 CPU `step_batch`

#### 1.3.2 Key Implementations
- Move validation stage kernel
- Combat resolution stage kernel
- Zobrist update stage kernel
- Device utilities for move legality checking

#### 1.3.3 Deliverables
- Extended `src/env/cuda/src/game_state.cu` (+1000 lines)
- `tests/test_step_batch.py` (200–300 lines)

---

### 1.4 M4 — GPU `build_observation_batch` Kernel

**Duration**: 8 days (2026-05-19 → 2026-05-26).

#### 1.4.1 Scope
- CUDA kernel for all 101 spatial + 28 global observation features
- Process all 4 seats per environment in parallel
- Canonical frame coordinate transformations

#### 1.4.2 Key Implementations
- Vectorized channel writers (16 groups)
- Coordinate frame rotation device functions
- Occupancy mask computation and reuse

#### 1.4.3 Deliverables
- `src/env/cuda/src/observation.cu/cuh` (1000–1500 lines)
- `tests/test_observation_batch.py` (200–300 lines)

---

### 1.5 M5 — End-to-End Integration & Training Loop

**Duration**: 5 days (2026-05-27 → 2026-05-31).

#### 1.5.1 Scope
- Wire all three kernels into `VectorJunqiEnv`
- GPU backend flag in constructor (`use_cuda=True`)
- PPO training loop integration
- Documentation and examples

#### 1.5.2 Concrete Subtasks
- [ ] VectorJunqiEnv GPU support
- [ ] GPU memory management (pinned buffers, unified memory)
- [ ] Training loop integration
- [ ] CI/CD setup (GPU testing in GitHub Actions)
- [ ] Documentation and examples
- [ ] Pre-built wheels

#### 1.5.3 Deliverables
- Modified `junqi_rl/env.py` (~100 lines)
- New `junqi_rl/train.py` (~300 lines example)
- `examples/train_junqi_ppo_gpu.py` (~200 lines)
- `scripts/bench_gpu_throughput.py` (~150 lines)
- `docs/GPU_BACKEND_GUIDE.md` (~200 lines)
- GitHub Actions workflow for GPU CI

---

## 2. Performance Targets

### 2.1 Phase 1a (CPU / NumPy) — ADR-126 Acceptance

| Metric                       | Target      |
|---|---|
| `legal_action_ids_batch` (N=1024) | ≥50 k/s     |
| `step_batch` (N=1024)        | ≥30 k/s     |
| Aggregate plays/sec (N=1024) | **≥50 k/s** |
| Per-env parity @ 10k steps   | Bit-identical |

### 2.2 Phase 1b (GPU) — GPU Kernel Acceptance

| Metric                       | Target      |
|---|---|
| `legal_action_ids_batch_cuda` (N=1024) | ≥500 k/s    |
| `step_batch_cuda` (N=1024)   | ≥600 k/s    |
| Aggregate plays/sec (N=1024) | **≥500 k/s** |
| Per-env parity @ 10k steps   | Bit-identical |
| GPU memory usage @ N=1024    | ≤32 GB      |

---

## 3. Dependency Chain

```
M1 (CPU Batching) ─┐
                   ├─→ M2 (GPU Legal) ─┐
                   ├─→ M3 (GPU Step)  ─┤
                   ├─→ M4 (GPU Obs)   ─┤
                                      └→ M5 (Integration)
```

M2, M3, M4 can run in parallel after M1 completes.

---

## 4. Success Criteria

Phase 1 is complete when:

1. ✅ M1 achieves ≥50 k plays/sec on CPU (ADR-126 acceptance)
2. ✅ M2–M4 kernels hit target throughputs
3. ✅ GPU results bit-identical with CPU on 50 random games @ N=32
4. ✅ All 52 golden replay JSON files execute successfully
5. ✅ Training loop runs end-to-end without errors
6. ✅ No GPU memory leaks over 1M training steps
7. ✅ Full documentation and examples
8. ✅ All tests green (500+ GPU tests)
9. ✅ Pre-built wheels available

---

**Phase 1 Target Completion**: 2026-06-15

See [CUDA_ARCHITECTURE.md](./CUDA_ARCHITECTURE.md) for detailed 12-section design document.
