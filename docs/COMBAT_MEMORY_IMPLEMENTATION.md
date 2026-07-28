# CombatMemory v4 — Implementation Plan & Status

> Companion to `docs/COMBAT_MEMORY_DESIGN.md` (the *what*).
> This document tracks the *how*: which files change, which milestones
> land, what the acceptance gates are.

Status: **CPU reference complete (v4) ✅ ; CUDA kernels complete (M5) ✅**.

DARK-mode-only.  No HALF_DARK / BRIGHT support in v4.

---

## 1. Channel layout (frozen)

50 channels added (256 → **306**).  Indices `[0, 256)` unchanged
(pre-CM checkpoints can resume by zero-padding the input conv).

### Layer 1 — projected to enemy alive pieces (45 ch)

| Group               |  N | Binary semantics |
|---------------------|---:|---|
| `cm_kill_mine_type` | 12 | multi-hot of victim-types (only when victim_seat == observer) |
| `cm_kill_mine_ge`   |  3 | ≥1 / ≥2 / ≥3 (popcount of direct pid bitmap) |
| `cm_kill_other_ge`  |  3 | ≥1 / ≥2 / ≥3 (DARK: queue counter of non-my victims) |
| `cm_chain_type`     | 12 | multi-hot of chain-link my-piece types |
| `cm_chain_ge`       |  3 | ≥1 / ≥2 / ≥3 (chain pid bitmap popcount) |
| `cm_floor_ge`       |  9 | cumulative ≥GONGB ... ≥SILING |
| `cm_is_gongb`       |  1 | A1 (ate observer's DILEI) ∨ A2 (path-revealed GONGB) |
| `cm_not_gongb`      |  1 | B1 (ate observer's non-DILEI) ∨ chain inferred |
| `cm_dilei_candidate`|  1 | runtime: alive ∧ at_zero_pos ∧ in_back_two_rows ∧ ¬attacked_by_known_GONGB |

### Layer 2 — theory-of-mind on my own alive pieces (5 ch)

Aggregates the two opponents' (left + right) views via `AND` (path-revealed only):

| Group                    |  N | Binary semantics |
|--------------------------|---:|---|
| `cm_my_kill_count_ge`    |  3 | min(left.count, right.count) ≥ 1 / ≥2 / ≥3 |
| `cm_my_is_gongb`         |  1 | left.is_gongb ∧ right.is_gongb (= path-revealed; consistent both opponents) |
| `cm_my_dilei_candidate`  |  1 | runtime: alive ∧ at_zero ∧ in_back ∧ ¬(any opponent saw a known-GONGB attack) |

The AND ensures we only project information that *both* opponents
already know — i.e. truly public info derived from path-reveal or
chain.  Single-opponent facts (e.g. "the team that lost a piece knows
the killer's victim-type") stay private.

---

## 2. Information boundary (frozen, DARK-only)

| Update                                          | When triggered |
|-------------------------------------------------|---|
| `direct_ate_my_*` (pid, type, last_step)        | victim_seat == observer (only the loser's owner can use victim_type) |
| `kill_mine_*` channels (Layer 1)                | derived from `direct_ate_my_*`; same gate |
| `kill_other_*` channels (Layer 1)               | victim_seat ≠ observer; only counts |
| `floor_ge` lift                                 | victim_seat == observer ∧ victim is ordinary-rank |
| `is_gongb` (A1)                                 | victim_seat == observer ∧ victim_type == DILEI ∧ Event.EAT |
| `is_gongb` (A2 — path-revealed)                 | move_requires_gongb(src, dst) returns True; broadcast all 4 observers |
| `not_gongb`                                     | victim_seat == observer ∧ victim_type ≠ DILEI ∧ Event.EAT  ⇒ killer can't be GONGB |
| `attacked_by_known_gongb` (defender flag)       | KILLED + attacker known GONGB at time of attack |
| `dilei_candidate` (runtime)                     | computed at obs-build time from zero_pos + move_count + attacked_by_known_gongb |
| Chain propagation                                | All 4 observers; operates on public piece_ids |

BOMB events: **no CombatMemory write** (both pieces dead, no live
target).  Implicit ZHADAN-vs-DILEI deduction is left to BeliefTensor's
remaining-inventory mechanism.

---

## 3. Files touched (v4)

### Core CPU path

| File | Change |
|---|---|
| `junqi_core/combat_memory.py` | **rewritten** — v4 data shape, two entry points (`apply_combat_event`, `apply_path_revealed_gongb`) |
| `junqi_core/move_gen.py` | added `move_requires_gongb(pieces, src, dst)` helper |
| `junqi_core/state.py` | `step_inplace` calls v4 API in EAT/KILLED branches; BOMB skipped; engineer move triggers `apply_path_revealed_gongb` |
| `junqi_core/observation.py` | 50-ch v4 layout, `_write_combat_memory` rewritten with two-layer projector |
| `junqi_core/batched_state.py` | SoA fields renamed to v4; `_step_single` calls v4 API; `_build_pieces_view` helper for path-reveal |

### Tests

| File | Change |
|---|---|
| `tests/test_combat_memory.py` | rewritten for v4 rule set |
| `tests/test_observation_combat_memory.py` | rewritten for v4 channel layout |
| `tests/test_gpu_combat_memory_parity.py` | **new** — bit-parity between GPU step_batch and CPU BatchedGameState across 14 cm_* fields |

### Docs / configs

| File | Change |
|---|---|
| `docs/COMBAT_MEMORY_DESIGN.md` | v4 design narrative |
| `docs/COMBAT_MEMORY_IMPLEMENTATION.md` | this document |
| `configs/v35_combat_memory_full.yaml` | OBS_CHANNELS=306; num_envs=320 |

### CUDA — kernels complete

| File | Change |
|---|---|
| `src/env/cuda/include/junqi_cuda.h` | `NUM_OBS_CHANNELS=306`; v4 device pointers `d_cm_*` (14 arrays) |
| `src/env/cuda/src/combat_memory.cuh` | **new** — `CMEnvPtrs`, `cm_apply_event_dev`, `cm_apply_path_revealed_gongb_dev`, `cm_move_requires_gongb_dev`, `cm_write_channels_device` |
| `src/env/cuda/src/combat_memory.cu` | **new** — full implementation: rail-graph BFS for path-reveal (reuses `STRAIGHT_RAIL_RAYS`/`CURVE_RAIL_OF`/`ENG_RAIL_*` constant memory), event update, observation projector |
| `src/env/cuda/src/game_state.cu` | DeviceGameStateBatch ctor: 14 cudaMalloc + cudaMemset; dtor: 14 cudaFree.  `step_batch_kernel` extended with 14 cm pointers; calls `cm_apply_event_dev` in EAT/KILLED branches; calls `cm_apply_path_revealed_gongb_dev` when src is GONGB and the move path is engineer-only.  BOMB explicitly skips. |
| `src/env/cuda/src/observation.cu` | `observation_kernel` extended with 11 const cm pointers; one thread per (env, observer) calls `cm_write_channels_device` to fill channels [256..306). |
| `src/env/cuda/src/bindings.cpp` | `cm_copy_from_host` / `cm_copy_to_host` (parity-test only). |
| `src/env/cuda/CMakeLists.txt` | `combat_memory.cu` added to kernel sources, `combat_memory.cuh` to headers. |

---

## 4. CPU↔GPU interaction audit

The user's hard requirement on CombatMemory v4: **no host transfers
in the training hot path**.  Audit result:

| Site | Direction | Frequency | Notes |
|---|---|---|---|
| `DeviceGameStateBatch` ctor — 14 × `cudaMalloc` + `cudaMemset` | none (allocation) | once per `DeviceGameStateBatch` lifetime | The `cudaMemset` writes 0 (and 0xff for `int16` *_step fields, encoding -1).  No host buffer involved. |
| `step_batch_kernel` (CombatMemory updates) | none | every step | Pure device kernel.  Reads/writes only `d_cm_*` and `d_*` SoA arrays already on GPU. |
| `observation_kernel` (Layer 1 & 2 channel writes) | none | every observation build | Pure device kernel.  Reads `d_cm_*` and projects to `d_spatial`. |
| `cm_move_requires_gongb_dev` (path-revealed BFS) | none | every step where src_type == GONGB | Reuses the device-resident `STRAIGHT_RAIL_RAYS`, `CURVE_RAIL_OF`, `ENG_RAIL_TO_IDX/CELLS/ADJ` constant memory tables (uploaded once at `init_tables`). |
| `cm_copy_from_host` (binding) | H2D | **PARITY TEST ONLY** | Never called from training code; used by `tests/test_gpu_combat_memory_parity.py`. |
| `cm_copy_to_host` (binding) | D2H | **PARITY TEST ONLY** | Same. |

There is no place in the production training pipeline (`junqi_rl/env_gpu.py`,
`junqi_rl/gpu_rollout.py`, `junqi_rl/gpu_world.py`) that reads or writes
the CombatMemory `d_cm_*` arrays from the host.  The only legal training
transfers (already pre-existing for the rest of the SoA state) are unchanged.

---

## 5. CUDA kernel sketches (now implemented in `combat_memory.cu`)

### `cm_apply_event_dev`
```cpp
// One env's CMEnvPtrs view (slice of (N,4,120) SoA).
// is_eat=true → defender dies (V=defender, K=attacker survives).
// is_eat=false → KILLED, attacker dies (V=attacker, K=defender survives).
//
// Order of operations matches CPU exactly:
//   1. KILLED preflight: attacker is V; check if V was a "known GONGB"
//      for each observer (own seat + GONGB type, or previous is_gongb).
//      If yes, set attacked_by_known_gongb[obs, K] = true.
//   2. Chain propagation (all 4 observers):
//      chain[K] |= direct[V] | chain[V] | bit(V); chain_type same.
//      rank_floor[K] = max(rank_floor[K], rank_floor[V] + 1) if V has floor.
//   3. Per-observer dispatch on V_visible = (V_seat == obs):
//      - !v_visible (DARK rule): direct_other_count[K]++ ; last_direct_step[K] = step.
//      - v_visible: full type-aware update (direct bitmap + type mask
//        + floor lift + is_gongb / not_gongb flags).
```

### `cm_move_requires_gongb_dev`
```cpp
// Returns true iff the path src→dst can ONLY be taken by a GONGB.
// Order:
//   1. 1-step orthogonal → return false.
//   2. 1-step diagonal via camp → return false.
//   3. Off-rail endpoint → return false.
//   4. Same row/col + straight_rail_clear (forward-scan STRAIGHT_RAIL_RAYS)
//      → return false (non-engineer can do it).
//   5. Different row+col, same curve, curve_rail_clear (BFS in
//      ENG_RAIL_ADJ restricted to CURVE_RAIL_OF) → return false.
//   6. Otherwise return engineer_can_reach (full ENG_RAIL_ADJ BFS).
```

### `cm_write_channels_device`
```cpp
// Per (env, observer): walks pid 0..119, projects:
//   Layer 1 (ch 256..300, 45 ch) onto enemy alive pieces in the
//     observer's canonical frame.
//   Layer 2 (ch 301..305, 5 ch) onto observer's own alive pieces,
//     using AND of opponents' (left + right) state for is_gongb /
//     OR for attacked_by_known_gongb (so dilei_candidate goes false
//     as soon as either opponent saw a known-GONGB attack).
```

---

## 6. Acceptance gates

| Gate | Target | Status |
|---|---|---|
| `OBS_CHANNELS` | 306 (asserted in `observation._self_check`) | ✅ |
| `pytest tests/test_combat_memory.py` | green | ✅ |
| `pytest tests/test_observation_combat_memory.py` | green | ✅ |
| `pytest tests/test_gpu_combat_memory_parity.py` | green (requires CUDA) | ⏳ awaiting GPU run |
| `pytest -q tests/` | green (no regression on T7 / Phase-0.4 suite) | ⏳ awaiting GPU run |
| `step()` throughput | ≥ 14 k plays/s (baseline 16.8 k) | ⏳ awaiting bench |
| CPU↔GPU CombatMemory parity | bit-identical over 50 random games × N=32 × 200 steps | ⏳ awaiting GPU run |
| `v35_combat_memory_full` | win_rate vs random ≥ 0.85 (current ceiling 0.80) | ⏳ awaiting training |
