# C++/CUDA Backend Architecture for JunQi

> **Status**: DRAFT (2026-04-22, Phase 1 planning)
> **Scope**: Engineering-level design for C++/CUDA GPU backend to accelerate GameState, move generation, and observation building.
> **Audience**: Core team and C++/CUDA specialists.
> **Companion**: `ARCHITECTURE.md` (Python frontend), `DECISIONS.md` (ADR-114..ADR-126), `PHASE_0.4_PERF_TODO.md` (Phase 0.4 baseline).

---

## 1. Executive Summary

Phase 0.4 achieved 7.6 k plays/sec single-state throughput on CPU using NumPy tables (a 3× improvement from 2.5 k). However, achieving the ≥50 k plays/sec target requires **batch parallelism** (ADR-126), not further per-state optimization. The Python layer has hit the dispatch-ceiling limit where each NumPy operation carries a 2–5 µs overhead.

This document outlines a **C++/CUDA backend** (Phase 1–2 scope) that:

1. **Preserves the Python API** — all existing code continues to work via PyBind11 wrappers.
2. **Maintains SoA (Structure-of-Arrays) memory layout** — data flows seamlessly between Python numpy arrays and GPU device memory.
3. **Implements GPU kernels** for:
   - Batched legal move generation (`legal_action_ids_batch`)
   - Batched step execution (`step_batch`)
   - Batched observation generation (all 101 channels in one kernel)
4. **Targets**:
   - **ADR-126 acceptance**: ≥50 k plays/sec aggregated throughput at N=1024 (CPU implementation first, then GPU).
   - **Phase 1 final** (post-GPU): ≥500 k plays/sec aggregated (128-GPU cluster scaling).

---

## 2. Architecture Overview

### 2.1 Two-Phase Implementation Strategy

**Phase 1a (CPU / NumPy)**: Introduce `BatchedGameState` and matching APIs (`step_batch`, `legal_action_ids_batch`, `build_observation_batch`). These run on CPU with NumPy and pass ADR-126 acceptance (≥50 k/sec).

**Phase 1b (GPU / CUDA)**: Reimplement the three hotspots in CUDA:
- `legal_action_ids_batch` → GPU kernel with one thread per (env, cell) pair.
- `step_batch` → GPU kernel(s) for state transitions + combat resolution.
- `build_observation_batch` → GPU kernel(s) for all 101 observation channels.

Each phase is independently shippable and testable.

### 2.2 Data Residency Strategy

```
┌─ Python Layer ────────────────────────────────────────────┐
│                                                            │
│  JunqiEnv / VectorJunqiEnv                               │
│      ↓                                                     │
│  numpy arrays: obs_spatial (N, 4, 101, 17, 17) float32  │
│                obs_global (N, 4, 28) float32             │
│                rewards (N, 4) int32                      │
│                                                            │
│  Pybind11 wrappers:                                      │
│    • reset_batch(n_envs, seeds) → obs                    │
│    • step_batch(actions) → (obs, rewards, done, info)    │
│    • legal_action_ids_batch(seat_per_env) → action_ids   │
│                                                            │
└──────────────────────────────────┬──────────────────────────┘
                                   │ PyBind11 bindings
                                   │
┌─ GPU Memory Layer ─────────────────────────────────────────┐
│                                                             │
│  DeviceGameStateBatch:                                    │
│    • d_cell_piece_id (N, 289) int16                      │
│    • d_piece_seat_arr (N, 120) int8                      │
│    • d_piece_type_arr (N, 120) int8                      │
│    • d_alive (N, 120) bool                               │
│    • d_pos_x, d_pos_y (N, 120) int8                      │
│    • d_move_count_arr (N, 120) int16                     │
│    • d_turn (N,) int8                                    │
│    • d_zobrist (N,) int64                                │
│    • ... (other SoA columns)                             │
│                                                             │
│  DeviceObservationBatch:                                 │
│    • d_spatial (N, 4, 101, 17, 17) float32              │
│    • d_global (N, 4, 28) float32                         │
│                                                             │
│  Precomputed Static Tables (GPU constant memory):        │
│    • STRAIGHT_RAIL_DESTS (289, 4, 4) int16              │
│    • ENGINEER_REACHABLE_STATIC (289, K_max) int16        │
│    • ADJACENT_CELLS (289, 8) int16                       │
│    • CAMP_FLAT, STRONGHOLD_FLAT, etc. (289,) bool       │
│                                                             │
│  Kernels:                                                 │
│    • legal_action_kernel <<<(N, 289, H), T>>>            │
│    • step_batch_kernel <<<(N, 32), 256>>>                │
│    • observation_kernel <<<(N, 4, 101), 256>>>           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Memory Layout and Indexing

**Piece Indexing** (0–119, 30 pieces per seat × 4 seats):
```c++
// Given a piece_id (0–119):
int seat = piece_id / 30;        // 0–3
int slot = piece_id % 30;        // 0–29
```

**Cell Indexing** (0–288, row-major on 17×17 board):
```c++
// (x, y) ↔ flat index:
int flat = y * 17 + x;           // x, y ∈ [0, 16]
int x = flat % 17;
int y = flat / 17;
```

**Batch Index Order**:
- Host numpy arrays: `(N, ...)`  where N is batch size
- GPU device arrays: same layout with unified memory or explicit H2D transfer
- Action IDs (flat): `action_id = src_flat * 289 + dst_flat`

---

## 3. C++ Data Structures

### 3.1 `DeviceGameStateBatch` (GPU-resident)

```cpp
// In cuda/src/game_state.cuh
#pragma once
#include <cuda_runtime.h>

namespace junqi_cuda {

struct DeviceGameStateBatch {
  // Dimensions
  int num_envs;
  int max_pieces = 120;
  int num_cells = 289;
  int board_size = 17;
  
  // SoA arrays (device memory)
  // Shape: (num_envs, max_pieces) unless otherwise noted
  int16_t* d_cell_piece_id;      // (num_envs, num_cells)
  int8_t*  d_piece_seat_arr;     // (num_envs, max_pieces)
  int8_t*  d_piece_type_arr;     // (num_envs, max_pieces)
  bool*    d_alive;              // (num_envs, max_pieces)
  int8_t*  d_pos_x;              // (num_envs, max_pieces)
  int8_t*  d_pos_y;              // (num_envs, max_pieces)
  int8_t*  d_zero_x;             // (num_envs, max_pieces)
  int8_t*  d_zero_y;             // (num_envs, max_pieces)
  int16_t* d_move_count_arr;     // (num_envs, max_pieces)
  int16_t* d_active_eat_arr;     // (num_envs, max_pieces)
  int16_t* d_passive_surv_arr;   // (num_envs, max_pieces)
  int8_t*  d_death_reason_arr;   // (num_envs, max_pieces)
  int16_t* d_death_step_arr;     // (num_envs, max_pieces)
  int16_t* d_death_loc_flat_arr; // (num_envs, max_pieces)
  int8_t*  d_turn;               // (num_envs,)
  int64_t* d_zobrist;            // (num_envs,)
  int32_t* d_move_counter;       // (num_envs,)
  int32_t* d_moves_since_last_combat; // (num_envs,)
  bool*    d_seat_dead_arr;      // (num_envs, 4)
  bool*    d_seat_flag_revealed_arr; // (num_envs, 4)
  
  // Methods
  DeviceGameStateBatch(int num_envs);
  ~DeviceGameStateBatch();
  
  // Copy from host numpy arrays (via PyBind11)
  void copy_from_host(
    const int16_t* h_cell_piece_id,
    const int8_t*  h_piece_seat_arr,
    // ... other arrays
    cudaStream_t stream = 0
  );
  
  // Copy to host numpy arrays
  void copy_to_host(
    int16_t* h_cell_piece_id,
    int8_t*  h_piece_seat_arr,
    // ... other arrays
    cudaStream_t stream = 0
  ) const;
};

} // namespace junqi_cuda
```

### 3.2 `DeviceObservationBatch` (GPU-resident)

```cpp
// In cuda/src/observation.cuh
namespace junqi_cuda {

struct DeviceObservationBatch {
  int num_envs;
  int num_seats = 4;
  int num_channels = 101;
  int board_height = 17;
  int board_width = 17;
  int num_global_dims = 28;
  
  // Shape: (num_envs, num_seats, num_channels, board_height, board_width)
  float* d_spatial;
  
  // Shape: (num_envs, num_seats, num_global_dims)
  float* d_global;
  
  DeviceObservationBatch(int num_envs);
  ~DeviceObservationBatch();
  
  void copy_to_host(
    float* h_spatial,
    float* h_global,
    cudaStream_t stream = 0
  ) const;
};

} // namespace junqi_cuda
```

### 3.3 Precomputed Static Tables

All tables are stored in GPU constant memory and precomputed at initialization:

```cpp
// In cuda/src/tables.cuh
namespace junqi_cuda {

// Static tables (constant memory or global, pinned on GPU)
extern __constant__ int16_t STRAIGHT_RAIL_DESTS[289 * 4 * 4];
extern __constant__ int16_t ADJACENT_CELLS[289 * 8];
extern __constant__ bool CAMP_FLAT[289];
extern __constant__ bool STRONGHOLD_FLAT[289];
extern __constant__ bool RAIL_FLAT[289];
extern __constant__ bool NINE_GRID_FLAT[289];
extern __constant__ bool CURVE_RAIL_CELLS[NUM_CURVES * 8];  // NUM_CURVES = 2

// Zobrist tables (may be too large for constant memory, use global)
__device__ int64_t* d_zobrist_piece;  // [120 * 14 * 289]
__device__ int64_t* d_zobrist_turn;   // [4]
__device__ int64_t* d_zobrist_dead;   // [4]
__device__ int64_t* d_zobrist_flag;   // [4]

} // namespace junqi_cuda
```

---

## 4. CUDA Kernel Design

### 4.1 Legal Action Generation (`legal_action_ids_batch`)

**Goal**: For each environment, compute the set of legal action IDs for a given acting seat.

**Kernel Launch Config**:
```cpp
dim3 grid(num_envs, 1, 1);
dim3 block(256, 1, 1);
// Kernel: legal_action_kernel <<<grid, block>>>
```

**Pseudocode**:
```cpp
__global__ void legal_action_kernel(
  int num_envs,
  const int8_t* d_piece_seat_arr,      // (num_envs, 120)
  const int8_t* d_piece_type_arr,      // (num_envs, 120)
  const bool* d_alive,                 // (num_envs, 120)
  const int8_t* d_pos_x,               // (num_envs, 120)
  const int8_t* d_pos_y,               // (num_envs, 120)
  const int16_t* d_cell_piece_id,      // (num_envs, 289)
  const int8_t* d_acting_seat,         // (num_envs,)
  
  // Output: action_ids for this env
  int32_t* d_action_ids,               // (num_envs, 83521)
  int16_t* d_action_count,             // (num_envs,)
  
  // Static tables
  const int16_t* STRAIGHT_RAIL_DESTS,  // [289, 4, 4]
  const int16_t* ADJACENT_CELLS,       // [289, 8]
  // ... other tables
) {
  int env_idx = blockIdx.x;
  int tid = threadIdx.x;
  
  int offset = env_idx * 120;
  
  // Each thread processes one piece
  for (int pid = tid; pid < 120; pid += blockDim.x) {
    if (d_alive[offset + pid] && d_piece_seat_arr[offset + pid] == d_acting_seat[env_idx]) {
      // This is a legal piece for this seat; compute reachable destinations
      int8_t x = d_pos_x[offset + pid];
      int8_t y = d_pos_y[offset + pid];
      int16_t src_flat = y * 17 + x;
      
      // For each legal destination, append action_id = src_flat * 289 + dst_flat
      // Use shared memory or atomic add to write to d_action_ids[env_idx, :]
      
      compute_reachable_dests(
        pid, src_flat,
        d_piece_type_arr[offset + pid],
        d_cell_piece_id + env_idx * 289,
        d_piece_seat_arr + offset,
        d_acting_seat[env_idx],
        STRAIGHT_RAIL_DESTS, ADJACENT_CELLS,
        CAMP_FLAT, STRONGHOLD_FLAT,
        // ... output to shared action buffer
      );
    }
  }
  
  // Synchronize and write out action count per env
  __syncthreads();
  if (tid == 0) {
    // Compute total unique destinations for this env
    // Write to d_action_count[env_idx]
  }
}
```

**Implementation Strategy**:
1. Each thread computes one piece's reachable cells.
2. Use shared memory (8 KB per block) for intermediate results.
3. Atomic operations or prefix-sum for output serialization.
4. Return a **flat array of action IDs** + a **per-env count**.

**Performance Target**:
- Latency: <100 µs per env at N=1024 (amortized).
- Throughput: ≥500 k action IDs/sec aggregate.

---

### 4.2 Batch Step Execution (`step_batch`)

**Goal**: Execute one action per environment in parallel; update all SoA arrays + combat resolution + zobrist.

**Kernel Launch Config**:
```cpp
dim3 grid(num_envs, 1, 1);
dim3 block(256, 1, 1);  // Per-env threads
// or multi-stage kernels if combat resolution is complex
```

**Stages**:

**Stage 1: Move Validation & Placement**
```cpp
__global__ void step_batch_move_stage(
  int num_envs,
  const int32_t* d_action_ids,   // (num_envs,) — one action per env
  int16_t* d_cell_piece_id,      // (num_envs, 289)
  int8_t* d_pos_x, int8_t* d_pos_y, // (num_envs, 120)
  int16_t* d_move_count_arr,
  // ... etc
) {
  int env_idx = blockIdx.x;
  
  // Unpack action_id
  int32_t action_id = d_action_ids[env_idx];
  int16_t src_flat = action_id / 289;
  int16_t dst_flat = action_id % 289;
  
  // Look up source piece
  int16_t piece_id = d_cell_piece_id[env_idx * 289 + src_flat];
  
  // Check if destination is occupied
  int16_t target_piece_id = d_cell_piece_id[env_idx * 289 + dst_flat];
  
  if (target_piece_id >= 0) {
    // Combat! (deferred to Stage 2)
    // Mark for resolution
  } else {
    // Empty destination: simple move
    d_pos_x[env_idx * 120 + piece_id] = dst_flat % 17;
    d_pos_y[env_idx * 120 + piece_id] = dst_flat / 17;
    d_cell_piece_id[env_idx * 289 + src_flat] = -1;
    d_cell_piece_id[env_idx * 289 + dst_flat] = piece_id;
  }
  
  // Increment move counter
  d_move_count_arr[env_idx * 120 + piece_id]++;
}
```

**Stage 2: Combat Resolution** (if needed)
```cpp
__global__ void step_batch_combat_stage(
  // ... similar to above, but resolve battles
  // Uses d_piece_type_arr to compare strengths
  // Updates alive, death reasons, zobrist
) {
  // ...
}
```

**Stage 3: Zobrist Update**
```cpp
__global__ void step_batch_zobrist_stage(
  // Update zobrist hashes incrementally via XOR
) {
  // ...
}
```

**Performance Target**:
- Latency: <50 µs per env at N=1024 (amortized).
- Throughput: ≥600 k steps/sec aggregate.

---

### 4.3 Batched Observation Generation (`build_observation_batch`)

**Goal**: Generate all 101 observation channels for all 4 seats in every environment.

**Kernel Launch Config**:
```cpp
// One thread per (env, channel, cell)
dim3 grid(num_envs, 4, 1);    // (num_envs, num_seats)
dim3 block(256, 1, 1);         // Threads per channel
// Total: ~1M+ threads for N=1024

// or, schedule one thread per env + use shared memory
dim3 grid(num_envs * 4, 1, 1);
dim3 block(128, 1, 1);
// One block processes all 101 channels for one env+seat pair
```

**Observation Channel Groups** (from Python `observation.py`):

| Group                     | Channels | Source Data        |
|---------------------------|----------|-------------------|
| `piece_own`               | 12       | piece type bitmap |
| `prob_teammate`           | 12       | belief tensor     |
| `dark_teammate`           | 1        | belief tensor     |
| `piece_side_enemy`        | 2        | cell occupancy    |
| `belief_left_right`       | 24       | belief tensor     |
| `dead_flags`              | 4        | seat dead flags   |
| `flag_revealed`           | 4        | flag reveal state |
| `board_static`            | 6        | topology (const)  |
| `turn_history`            | 4        | turn, move count  |
| `move_bucket`             | 8        | move counters     |
| `active_eat_bucket`       | 8        | combat stats      |
| `passive_survive_bucket`  | 8        | survival stats    |
| `death_reason`            | 6        | death info        |
| `dead_at_zero`            | 2        | started dead?     |

**Pseudocode**:
```cpp
__global__ void observation_kernel(
  int num_envs, int num_seats,
  const int8_t* d_piece_seat_arr,
  const int8_t* d_piece_type_arr,
  const bool* d_alive,
  const int8_t* d_pos_x, int8_t* d_pos_y,
  const int16_t* d_cell_piece_id,
  const int16_t* d_move_count_arr,
  const int8_t* d_death_reason_arr,
  const float* d_belief_tensor,  // Precomputed or passed in
  const int8_t* d_acting_seat,   // Per env
  
  // Output: (num_envs, 4, 101, 17, 17)
  float* d_obs_spatial,
  float* d_obs_global     // (num_envs, 4, 28)
) {
  int env_idx = blockIdx.x;
  int seat_idx = blockIdx.y;
  int tid = threadIdx.x;
  int num_threads = blockDim.x;
  
  int num_channels = 101;
  int board_height = 17;
  int board_width = 17;
  
  // Precompute rotations and coordinate frame conversions for this (env, seat)
  // ...
  
  // Process all channels in parallel (grid stride)
  for (int c = tid; c < num_channels; c += num_threads) {
    for (int y = 0; y < board_height; y++) {
      for (int x = 0; x < board_width; x++) {
        int flat = y * 17 + x;
        
        // Determine which channel group and write value
        float value = 0.0f;
        
        if (c < 12) {
          // piece_own: bitmap of own pieces at this cell
          value = (d_cell_piece_id[env_idx * 289 + flat] >= 0 &&
                   d_piece_seat_arr[env_idx * 120 + d_cell_piece_id[env_idx * 289 + flat]] == seat_idx)
                  ? 1.0f : 0.0f;
        } else if (c < 24) {
          // prob_teammate: belief state
          value = d_belief_tensor[/* ... */];
        } 
        // ... other channels
        
        // Write to output
        int out_idx = ((env_idx * 4 + seat_idx) * 101 + c) * 289 + flat;
        d_obs_spatial[out_idx] = value;
      }
    }
  }
  
  // Global features computed once per (env, seat)
  if (tid < 28) {
    float value = 0.0f;
    int g = tid;
    
    // Compute global feature g for this env and seat
    // ...
    
    d_obs_global[(env_idx * 4 + seat_idx) * 28 + g] = value;
  }
}
```

**Performance Target**:
- Latency: <200 µs per (env, seat) at N=1024.
- Throughput: ≥200 k observations/sec aggregate.

---

## 5. PyBind11 Bindings

All CUDA kernels are exposed to Python via PyBind11. Example:

```cpp
// In cuda/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "game_state.cuh"
#include "observation.cuh"

namespace py = pybind11;

// Wrapper function for legal_action_ids_batch
py::array_t<int32_t> legal_action_ids_batch_cuda(
  int num_envs,
  py::array_t<int8_t> piece_seat_arr,      // (num_envs, 120)
  py::array_t<int8_t> piece_type_arr,      // (num_envs, 120)
  py::array_t<bool> alive,                 // (num_envs, 120)
  py::array_t<int8_t> pos_x,               // (num_envs, 120)
  py::array_t<int8_t> pos_y,               // (num_envs, 120)
  py::array_t<int16_t> cell_piece_id,      // (num_envs, 289)
  py::array_t<int8_t> acting_seat          // (num_envs,)
) {
  // Allocate device memory
  DeviceGameStateBatch d_state(num_envs);
  
  // Copy from host to device
  d_state.copy_from_host(
    piece_seat_arr.data(),
    // ... etc
  );
  
  // Launch kernel
  dim3 grid(num_envs, 1, 1);
  dim3 block(256, 1, 1);
  legal_action_kernel <<<grid, block>>>(
    num_envs,
    d_state.d_piece_seat_arr,
    // ... etc
  );
  
  // Copy results back to host
  py::array_t<int32_t> result({num_envs, 83521});
  // ...
  d_state.copy_to_host(/* ... */);
  
  return result;
}

PYBIND11_MODULE(junqi_cuda, m) {
  m.def("legal_action_ids_batch", &legal_action_ids_batch_cuda);
  m.def("step_batch", &step_batch_cuda);
  m.def("build_observation_batch", &build_observation_batch_cuda);
}
```

---

## 6. Integration with Python RL Stack

### 6.1 VectorJunqiEnv with GPU Backend

```python
# In junqi_rl/env.py (modified for Phase 1b GPU support)

class VectorJunqiEnv:
  def __init__(self, num_envs, use_cuda=False):
    self.num_envs = num_envs
    self.use_cuda = use_cuda
    
    if use_cuda:
      import junqi_cuda
      self._cuda_module = junqi_cuda
    else:
      self._cuda_module = None
    
    self._states = [GameState.new_game(setup) for _ in range(num_envs)]
    # ... init obs buffers, etc.
  
  def step(self, action_ids):
    """Execute one step across all environments."""
    
    if self.use_cuda:
      # Gather state arrays into batch form
      batch_state = self._prepare_batch_state()
      
      # Call CUDA kernel
      (obs_spatial, obs_global, rewards, done, info) = \
        self._cuda_module.step_batch(batch_state, action_ids)
      
      # Scatter results back to individual states
      self._scatter_batch_state(batch_state)
    else:
      # CPU path (Phase 1a)
      for env_idx in range(self.num_envs):
        state, result = self._states[env_idx].step(action_ids[env_idx])
        self._states[env_idx] = state
        # ... update obs, rewards, etc.
    
    return (obs_spatial, obs_global, rewards, done, info)
```

### 6.2 Training Loop (PPO) Integration

```python
# Phase 1 / Phase 2 RL loop remains unchanged
# The GPU backend is swappable via a constructor flag

def train_junqi_ppo():
  num_envs = 1024
  env = VectorJunqiEnv(num_envs, use_cuda=torch.cuda.is_available())
  
  policy = JunqiPolicyNetwork()
  optimizer = torch.optim.Adam(policy.parameters())
  
  for epoch in range(num_epochs):
    obs_spatial, obs_global = env.reset()
    
    for step in range(steps_per_epoch):
      # Policy inference (already on GPU if using torch)
      with torch.no_grad():
        logits = policy(obs_spatial, obs_global)
      
      # Env step (now GPU-accelerated if use_cuda=True)
      obs_spatial, obs_global, rewards, done, info = env.step(actions)
      
      # ... collect experience, compute loss, backprop
```

---

## 7. File Organization

```
src/env/cuda/
├── CMakeLists.txt
├── src/
│   ├── game_state.cu          # DeviceGameStateBatch, kernels stage 1–3
│   ├── game_state.cuh
│   ├── observation.cu         # Observation building kernel
│   ├── observation.cuh
│   ├── tables.cu              # Precomputed tables (zobrist, move tables)
│   ├── tables.cuh
│   ├── common.cuh             # Shared macros, device utilities
│   ├── bindings.cpp           # PyBind11 module
│   └── bindings.cuh
├── include/
│   └── junqi_cuda.h           # Public API
└── tests/
    ├── test_legal_actions.cu
    ├── test_step_batch.cu
    └── test_observation.cu
```

---

## 8. Performance Model

### 8.1 Kernel Execution Times (estimated)

| Operation                | Single Env | N=1024   | Latency Target |
|--------------------------|-----------|----------|---|
| `legal_action_ids_batch` | ~10 µs    | <100 µs  | ✓ |
| `step_batch`            | ~5 µs     | <50 µs   | ✓ |
| `build_observation_batch`| ~200 µs   | <200 µs  | ✓ |
| **Total per step**      | **~215 µs**| **<350 µs** | ✓ |
| **Throughput**          | **4.6 k/s**| **≥2.9 M/s (2.9×10⁶ plays/sec)**| ✓ |

(Numbers are rough estimates; actual measurements will refine these.)

### 8.2 Memory Usage

| Structure                      | Size per Env | N=1024  |
|--------------------------------|--|--|
| `DeviceGameStateBatch` columns | ~8 KB        | 8 MB    |
| `DeviceObservationBatch`       | ~22 MB       | 22 GB   |
| **Static tables (constant)**   | ~1 MB (once) | 1 MB    |
| **Total GPU memory needed**    | -            | **~32 GB** |

(Modern A100/H100 have 40–80 GB, so this is feasible for N=1024.)

---

## 9. Milestone Breakdown (Phase 1)

### 9.1 M1 — CPU `BatchedGameState` + ADR-126 acceptance

- Implement `BatchedGameState` class in Python.
- Implement `step_batch`, `legal_action_ids_batch`, `build_observation_batch` using NumPy on CPU.
- Achieve ≥50 k plays/sec aggregate.
- All tests pass; bit-identical parity with single-state `GameState`.

**Deliverable**: `junqi_core/batched_state.py` + `junqi_rl/env.py` (modified for batch API).

### 9.2 M2 — GPU `legal_action_ids_batch` kernel

- Port M1's legal action generation to CUDA.
- PyBind11 wrapper.
- Benchmark: target ≥500 k actions/sec aggregate at N=1024.

**Deliverable**: `src/env/cuda/src/game_state.cu`, `bindings.cu` (partial).

### 9.3 M3 — GPU `step_batch` kernel

- Port move execution, combat resolution, zobrist to CUDA.
- Multi-stage kernel (move → combat → zobrist).
- Benchmark: target ≥600 k steps/sec aggregate.

**Deliverable**: `src/env/cuda/src/game_state.cu` (extended).

### 9.4 M4 — GPU `build_observation_batch` kernel

- Implement all 101 observation channels in one kernel.
- Integration with belief tensor (may run on CPU initially).
- Benchmark: target ≥200 k observations/sec aggregate.

**Deliverable**: `src/env/cuda/src/observation.cu`, `bindings.cpp` (complete).

### 9.5 M5 — End-to-end GPU backend integration

- Full PyBind11 module (`junqi_cuda.so`).
- Integration with `VectorJunqiEnv`.
- Benchmark full training loop.
- Documentation + examples.

**Deliverable**: Complete `src/env/cuda/` module + RL training script.

---

## 10. Risk Mitigation

### 10.1 Correctness

- **Continuous parity testing**: Every GPU kernel is tested against Python reference implementation.
- **Determinism**: Use same random seed; compare results bit-by-bit.
- **Golden test suite**: Reuse existing `tests/golden/*.json` for replay verification.

### 10.2 Performance

- **Profiling**: Use `nvidia-smi`, `nsys`, `ncu` to identify bottlenecks.
- **Kernel tuning**: Vary block size, grid size, shared memory usage based on measured data.
- **Memory transfer overhead**: Minimize H2D copies; prefer GPU-resident data structures.

### 10.3 Maintenance

- **Code organization**: Clear separation between CPU and GPU implementations.
- **Documentation**: Every kernel has a detailed docstring explaining its algorithm and assumptions.
- **CI/CD**: GPU tests run in GitHub Actions (or cloud runner) on every commit.

---

## 11. Future Extensions (Phase 2+)

### 11.1 BeliefTensor GPU Kernels

Observation generation currently assumes belief tensors are precomputed on CPU. Phase 2 can move belief updates to GPU via a matching kernel.

### 11.2 Multi-GPU Support

Scale observation generation across multiple GPUs using NCCL and ring-allreduce patterns.

### 11.3 Reinforcement Learning Integration

- **In-place rollout collection**: RL collectors access GPU observation buffers directly.
- **Reward computation**: Implement team reward calculation on GPU (currently CPU-bound).
- **Action sampling**: GPU-side action masking + categorical sampling for policy inference.

### 11.4 Competitive Scaling

- **128-GPU cluster**: expected ≥500 M plays/sec (scaling factor ~1000×).
- **Ataraxos parity**: target matching or exceeding Ataraxos SOTA throughput.

---

## 12. References

- **ADR-117**: Structure-of-Arrays GameState (Python Phase 0.4).
- **ADR-118**: Pre-allocated observation buffers.
- **ADR-119**: Flat action space encoding.
- **ADR-121**: Batched observation construction.
- **ADR-123**: SoA API batch-axis forward compatibility.
- **ADR-125**: Move generation via precomputed static tables.
- **ADR-126**: Batched environment for ≥50 k plays/sec (Phase 1 scope).
- **Ataraxos**: https://github.com/google/... (SOTA Stratego RL)
  - Reference: `src/env/cuda/action_kernels.cu` for batch parallelism patterns.

---

## Appendix: Coordinate Frame Transformations

The observation tensor is built in **canonical (observer-centric) frame**, regardless of which seat is acting. All CUDA kernels must apply rotations consistently.

```cpp
// Example: convert world-frame cell to canonical-frame cell for observer_seat
__device__ int16_t rotate_cell_to_canonical(
  int16_t world_flat,
  int8_t observer_seat
) {
  // Rotation matrix depends on observer_seat ∈ {0, 1, 2, 3}
  // Seat 0 (SOUTH): no rotation
  // Seat 1 (WEST): 90° rotation
  // Seat 2 (NORTH): 180° rotation
  // Seat 3 (EAST): 270° rotation
  
  int x = world_flat % 17;
  int y = world_flat / 17;
  
  int canonical_x, canonical_y;
  if (observer_seat == 0) {
    canonical_x = x; canonical_y = y;
  } else if (observer_seat == 1) {
    canonical_x = y; canonical_y = 16 - x;
  } else if (observer_seat == 2) {
    canonical_x = 16 - x; canonical_y = 16 - y;
  } else {  // observer_seat == 3
    canonical_x = 16 - y; canonical_y = x;
  }
  
  return canonical_y * 17 + canonical_x;
}
```

All 16 vectorized observation writers (ADR-118, ADR-121) apply this transformation implicitly via pre-computed rotation matrices (or equivalently, via `np.rot90` on the host for CPU path; GPU kernels compute it directly).

---

**End of Document**
