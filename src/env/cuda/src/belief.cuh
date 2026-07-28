/*
 * belief.cuh
 * GPU-side belief tensor update kernels for imperfect-information JunQi.
 *
 * Phase 1: Deductive belief updates only (R1, R4, R5/R7, R6, R9, I5).
 * No remaining-inventory tracking (deferred to Phase 2).
 *
 * Belief tensor layout: [N, 4, 12, 289] float32 (C-contiguous)
 *   N   = num_envs
 *   4   = observer seats (SOUTH, WEST, NORTH, EAST)
 *   12  = tracked piece types (JUNQI..GONGB)
 *   289 = flat board cells (17×17)
 *
 * The belief at [env, obs, type, cell] is the observer's posterior probability
 * that the piece at `cell` is of `type`.  Own/teammate pieces are always
 * one-hot (known); enemy pieces carry a non-degenerate distribution.
 */

#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

namespace junqi_cuda {

// Forward declaration — full definition lives in junqi_cuda.h / game_state.cu.
struct DeviceGameStateBatch;

// Per-slot prior table for enemy piece initialization (HALF_DARK mode).
// Indexed [slot_in_seat * 12 + type_idx].  Camp slots are all-zero.
// Uploaded once at startup via upload_belief_prior_table().
extern __constant__ float BELIEF_PRIOR_TABLE[30 * 12];

// Stronghold world-frame flat positions for each seat, precomputed.
// SEAT_STRONGHOLDS[seat * 2 + k] = flat cell of k-th stronghold.
extern __constant__ int16_t SEAT_STRONGHOLDS[4 * 2];

// ---------------------------------------------------------------------------
// belief_init_kernel — initialise beliefs for newly-reset environments.
//
// Called AFTER reset_terminated_envs_kernel has written fresh piece arrays.
// For each terminated env (just reset):
//   - For each observer seat, iterate over all 120 pieces:
//     * Own/teammate pieces → one-hot (type known under HALF_DARK)
//     * Enemy pieces → per-slot prior from BELIEF_PRIOR_TABLE
//     * Dead/camp pieces → zero
//
// Grid: (num_envs,)  Block: 128 threads
// ---------------------------------------------------------------------------
__global__ void belief_init_kernel(
    int num_envs,
    const int8_t*  d_piece_seat_arr,   // (N, 120)
    const int8_t*  d_piece_type_arr,   // (N, 120)
    const bool*    d_alive,            // (N, 120)
    const int8_t*  d_pos_x,           // (N, 120)
    const int8_t*  d_pos_y,           // (N, 120)
    const bool*    d_terminated,       // (N,) — True for just-reset envs
    float*         d_belief            // (N, 4, 12, 289) — output
);

// ---------------------------------------------------------------------------
// belief_update_kernel — incremental belief update after one step.
//
// Called AFTER step_device() has written events/state, BEFORE next obs build.
// For each non-terminated env, applies deductive rules R1–R9 + I5 for all
// 4 observer seats.
//
// Grid: (num_envs,)  Block: 128 threads
// ---------------------------------------------------------------------------
__global__ void belief_update_kernel(
    int num_envs,
    // Post-step game state (already mutated by step_batch_kernel)
    const int8_t*  d_piece_seat_arr,      // (N, 120)
    const int8_t*  d_piece_type_arr,      // (N, 120)
    const bool*    d_alive,               // (N, 120)
    const int8_t*  d_pos_x,              // (N, 120)
    const int8_t*  d_pos_y,              // (N, 120)
    const int16_t* d_cell_piece_id,       // (N, 289)
    const bool*    d_seat_dead_arr,       // (N, 4)
    const bool*    d_seat_flag_revealed,  // (N, 4)
    const bool*    d_terminated,          // (N,)
    // Step result (from StepDeviceBuffers, persistent)
    const int8_t*  d_event,               // (N,) — EV_MOVE/EAT/KILLED/BOMB
    const bool*    d_flag_captured,        // (N,)
    const int32_t* d_world_actions,        // (N,) — src_flat * 289 + dst_flat
    // Pre-step snapshot (saved before step)
    const bool*    d_prev_seat_flag_revealed, // (N, 4) — for detecting new reveals
    const bool*    d_prev_seat_dead,          // (N, 4) — for detecting new deaths
    // Belief tensor (in-place update)
    float*         d_belief               // (N, 4, 12, 289)
);

// ---------------------------------------------------------------------------
// Host-callable launcher functions
// ---------------------------------------------------------------------------

// Upload the 30×12 prior table to constant memory.  Call once at init.
void upload_belief_prior_table(const float* h_table);

// Upload the 4×2 stronghold positions.  Call once at init.
void upload_seat_strongholds(const int16_t* h_strongholds);

// Initialise beliefs for all envs whose d_terminated == true (just reset).
// Must be called AFTER reset_terminated_envs has written fresh game state.
// NOTE: default arguments live in junqi_cuda.h (public header) only.
void init_beliefs_for_reset_envs(
    const DeviceGameStateBatch& d_state,
    float* d_belief,
    int stream_id
);

// Snapshot seat_flag_revealed and seat_dead before the step.
// Must be called BEFORE step_device().
void snapshot_pre_step_flags(
    const DeviceGameStateBatch& d_state,
    int stream_id
);

// Incremental belief update after a step.
// Must be called AFTER step_device() and BEFORE the next obs build.
void update_beliefs_after_step(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_event,
    const bool* d_flag_captured,
    const int32_t* d_world_actions,
    float* d_belief,
    int stream_id
);

}  // namespace junqi_cuda
