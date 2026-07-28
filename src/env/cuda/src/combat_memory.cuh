/*
 * combat_memory.cuh
 *
 * GPU-side CombatMemory v6.  Mirror of junqi_core/combat_memory.py: same
 * data layout, same update rules, same observation projection.
 *
 * NO HOST INTERACTION in the hot path.  The only legal CPU↔GPU traffic
 * for this subsystem is:
 *   - one-time cudaMalloc/cudaMemset in DeviceGameStateBatch ctor;
 *   - parity-test copy_from_host / copy_to_host (NOT used during training).
 *
 * All per-step updates run inside step_batch_kernel as device functions;
 * all per-build channel writes run inside the observation kernel via
 * cm_write_channels_device().  No cudaMemcpy, no cudaMemset, no host
 * lambda.  Rail topology and zobrist tables are already device-resident
 * (see tables.cuh) — we reuse them directly.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace junqi_cuda {

// Constants (must match Python combat_memory.py).
constexpr int CM_NUM_PIDS_DEV       = 120;
constexpr int CM_NUM_OBSERVERS_DEV  = 4;
constexpr int CM_NUM_TRACKED_TYPES  = 12;
constexpr int CM_NUM_RANK_FLOORS_DEV = 9;

// PieceType.value → tracked-type idx in [0, 12); -1 if not tracked.
//   JUNQI=2 → 0 ; DILEI=3 → 1 ; ZHADAN=4 → 2 ; SILING=5 → 3 ; JUNZH=6 → 4
//   SHIZH=7 → 5 ; LVZH=8 → 6 ; TUANZH=9 → 7 ; YINGZH=10 → 8 ; LIANZH=11 → 9
//   PAIZH=12 → 10 ; GONGB=13 → 11
__device__ inline int8_t cm_type_to_idx(int8_t pt_val) {
    if (pt_val < 2 || pt_val > 13) return -1;
    return (int8_t)(pt_val - 2);
}

// PieceType.value → ordinary rank in [1, 9]; 0 if not ordinary.
//   GONGB=13 → 1, PAIZH=12 → 2, LIANZH=11 → 3, YINGZH=10 → 4,
//   TUANZH=9 → 5, LVZH=8 → 6, SHIZH=7 → 7, JUNZH=6 → 8, SILING=5 → 9.
__device__ inline int8_t cm_rank_of_type(int8_t pt_val) {
    switch (pt_val) {
        case 13: return 1;
        case 12: return 2;
        case 11: return 3;
        case 10: return 4;
        case  9: return 5;
        case  8: return 6;
        case  7: return 7;
        case  6: return 8;
        case  5: return 9;
        default: return 0;
    }
}

__device__ inline int8_t cm_next_floor_after_eat(int8_t victim_type_val) {
    int8_t r = cm_rank_of_type(victim_type_val);
    if (r == 0) return 0;
    int8_t p = (int8_t)(r + 1);
    return (p > 9) ? (int8_t)9 : p;
}

// Geometry: is `pos_flat` in `seat`'s own back-two-rows?
//   SOUTH y∈{15,16}; NORTH y∈{0,1}; WEST x∈{0,1}; EAST x∈{15,16}.
__device__ inline bool cm_in_back_two_rows(int16_t pos_flat, int8_t owner_seat) {
    int y = pos_flat / 17;
    int x = pos_flat % 17;
    switch (owner_seat) {
        case 0: return y >= 15;   // SOUTH
        case 1: return x <= 1;    // WEST
        case 2: return y <= 1;    // NORTH
        case 3: return x >= 15;   // EAST
        default: return false;
    }
}

// ---------------------------------------------------------------------------
// Path-revealed-GONGB device check.
//
// Returns true iff src→dst (with the move geometrically legal for some
// piece type) is a route that ONLY a GONGB can take.  Mirrors
// junqi_core/move_gen.py::move_requires_gongb.
//
// All inputs come from device-resident SoA + constant memory; no host
// traffic.  Caller must already have validated that the move is legal
// for the moving piece.
// ---------------------------------------------------------------------------
__device__ bool cm_move_requires_gongb_dev(
    const int16_t* cpid_env,   // (289,) int16 — cell_piece_id for this env
    int16_t src_flat,
    int16_t dst_flat
);

// ---------------------------------------------------------------------------
// Update API (called from step_batch_kernel).
//
// Each device function takes a (env, observer)-agnostic CombatMemory pointer
// bundle for one env (slice of the (N, 4, 120) SoA).  These run inside the
// per-env step kernel and never touch host.
// ---------------------------------------------------------------------------

struct CMEnvPtrs {
    uint64_t* direct_lo;             // (4, 120) uint64
    uint64_t* direct_hi;             // (4, 120) uint64
    uint16_t* direct_type;           // (4, 120) uint16
    int16_t*  last_direct_step;      // (4, 120) int16
    int16_t*  direct_other_count;    // (4, 120) int16
    uint64_t* chain_lo;              // (4, 120) uint64
    uint64_t* chain_hi;              // (4, 120) uint64
    uint16_t* chain_type;            // (4, 120) uint16
    int16_t*  last_chain_step;       // (4, 120) int16
    uint64_t* eaten_by_pid_lo;       // (4, 120) uint64
    uint64_t* eaten_by_pid_hi;       // (4, 120) uint64
    int8_t*   rank_floor;            // (4, 120) int8
    int16_t*  rank_floor_step;       // (4, 120) int16
    bool*     is_gongb;              // (4, 120) bool
    bool*     not_gongb;             // (4, 120) bool
    bool*     attacked_by_known_gongb; // (4, 120) bool
};

// Apply one EAT or KILLED combat event to all 4 observers' memory.
// (BOMB events do NOT call this — both pieces die, no live target.)
__device__ void cm_apply_event_dev(
    CMEnvPtrs cm,
    bool is_eat,                    // true: EAT (defender dies); false: KILLED (attacker dies)
    int   attacker_pid,
    int   defender_pid,
    int   attacker_seat,
    int   defender_seat,
    int8_t attacker_type_val,
    int8_t defender_type_val,
    int16_t defender_pos_flat,
    int   death_step
);

// Mark a piece as a publicly-revealed GONGB across all 4 observers.
// Called from step_batch_kernel when src is a GONGB and the move path
// is GONGB-only (cm_move_requires_gongb_dev returns true).
__device__ inline void cm_apply_path_revealed_gongb_dev(
    CMEnvPtrs cm,
    int pid)
{
    // is_gongb[obs, pid] = true for all 4 observers.
    cm.is_gongb[0 * CM_NUM_PIDS_DEV + pid] = true;
    cm.is_gongb[1 * CM_NUM_PIDS_DEV + pid] = true;
    cm.is_gongb[2 * CM_NUM_PIDS_DEV + pid] = true;
    cm.is_gongb[3 * CM_NUM_PIDS_DEV + pid] = true;
}

// ---------------------------------------------------------------------------
// CombatMemory v4 observation channels.
//
// Writes the 50 v4 channels into d_spatial[(env*num_seats+slot) * stride
// + (256 + ...) * 17 * 17].  Called from within observation_kernel after
// the base 256-channel writes.
//
// Performs canonical-frame rotation per-observer using the same formulas
// as the existing observation_kernel.
// ---------------------------------------------------------------------------
__device__ void cm_write_channels_device(
    int   env_id,
    int   observer_seat,             // 0..3
    int   spatial_base,              // canonical-frame base offset for this (env, observer)
    float* spatial,                  // base = d_obs_spatial; we write [base + 256*PLANE ..]
    // Per-(env, observer, pid) CombatMemory state
    const uint64_t* cm_direct_lo_env,         // (4, 120)
    const uint64_t* cm_direct_hi_env,
    const uint16_t* cm_direct_type_env,
    const int16_t*  cm_direct_other_count_env,
    const uint64_t* cm_chain_lo_env,
    const uint64_t* cm_chain_hi_env,
    const uint16_t* cm_chain_type_env,
    const int8_t*   cm_rank_floor_env,
    const bool*     cm_is_gongb_env,
    const bool*     cm_not_gongb_env,
    const bool*     cm_attacked_by_known_gongb_env,
    // Layer-3 (v5): step trackers, used for recency channels.
    const int16_t*  cm_last_direct_step_env,
    const int16_t*  cm_last_chain_step_env,
    const int16_t*  cm_rank_floor_step_env,
    // Layer-4 (v6): victim-anchored reverse projection.
    const uint64_t* cm_eaten_by_pid_lo_env,
    const uint64_t* cm_eaten_by_pid_hi_env,
    int             move_counter,
    // Per-(env, pid) game state (for dilei_candidate runtime check)
    const int8_t*   piece_seat_env,            // (120,)
    const int8_t*   piece_type_env,            // (120,) — used for kill_mine_count masks
    const bool*     alive_env,                 // (120,)
    const int8_t*   pos_x_env,                 // (120,)
    const int8_t*   pos_y_env,                 // (120,)
    const int8_t*   zero_x_env,                // (120,)
    const int8_t*   zero_y_env,                // (120,)
    const int16_t*  move_count_env             // (120,)
);

}  // namespace junqi_cuda
