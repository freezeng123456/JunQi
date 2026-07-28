/*
 * observation.cuh
 * Observation tensor generation kernel declarations.
 */

#pragma once

#include <cuda_runtime.h>

namespace junqi_cuda {

// Kernel: Build observation tensors for all (env, seat) pairs.
// Each thread handles one (env, slot) pair where slot ∈ [0, num_seats).
// The actual observer seat for slot i is d_observer_seats[env * num_seats + i].
//
// Output layout (canonical frame, observer always at SOUTH):
//   d_obs_spatial[(env*num_seats + slot) * NUM_OBS_CHANNELS * 17 * 17
//                 + ch * 17 * 17 + canonical_flat]   float32
//   d_obs_global[(env*num_seats + slot) * NUM_GLOBAL_DIMS + dim]  float32
__global__ void observation_kernel(
    int num_envs, int num_seats,
    // Per-piece SoA arrays (each length = num_envs * 120)
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const int8_t*  d_zero_x,
    const int8_t*  d_zero_y,
    const int16_t* d_move_count_arr,
    const int16_t* d_active_eat_arr,
    const int16_t* d_passive_surv_arr,
    const int8_t*  d_death_reason_arr,
    const int16_t* d_death_loc_flat_arr,
    // Per-seat arrays (each length = num_envs * 4)
    const bool*    d_seat_dead_arr,
    const bool*    d_seat_flag_revealed_arr,
    // Per-env scalars (each length = num_envs)
    const int8_t*  d_turn,
    const int32_t* d_move_counter,
    const int32_t* d_moves_since_last_combat,
    // Belief tensor [num_envs, NUM_SEATS, NUM_TRACKED_TYPES, NUM_CELLS] float32
    const float*   d_belief_tensor,
    // Observer assignment [num_envs * num_seats] int8
    const int8_t*  d_observer_seats,
    // Move history ring buffer
    const int16_t* d_move_history,       // (N, MOVE_HISTORY_LEN, 2)
    const int32_t* d_history_write_idx,  // (N,)
    const int32_t* d_history_count,      // (N,)
    // Outputs
    float*         d_obs_spatial,
    float*         d_obs_global,
    // Show mode: 0=BRIGHT, 1=DARK, 2=HALF_DARK (controls dark_teammate channel)
    int8_t         show_mode
);

}  // namespace junqi_cuda
