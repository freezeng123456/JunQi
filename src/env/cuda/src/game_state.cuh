/*
 * game_state.cuh
 * GPU-resident game state data structures and kernel declarations.
 */

#pragma once

#include <cuda_runtime.h>
#include "common.cuh"

namespace junqi_cuda {

// Forward declarations
struct DeviceGameStateBatch;

// Kernel: Generate legal action IDs for a batch of game states.
// Each thread handles one (env, piece) candidate.
// Output: d_action_ids[env * MAX_ACTIONS_PER_ENV + k] = action_id (src*289+dst)
//         d_action_counts[env] = number of legal actions for that env.
// MAX_ACTIONS_PER_ENV is allocated by the caller; 512 is a safe upper bound.
// NOTE: d_action_counts uses int32_t (not int16_t) because atomicAdd requires
// at least 32-bit types on all CUDA compute capabilities.
__global__ void legal_action_kernel(
  int num_envs,
  const int8_t* d_piece_seat_arr,
  const int8_t* d_piece_type_arr,
  const bool* d_alive,
  const int8_t* d_pos_x, const int8_t* d_pos_y,
  const int16_t* d_cell_piece_id,
  const int8_t* d_acting_seats,
  int32_t* d_action_ids,
  int32_t* d_action_counts
);

// Kernel: Move pieces (first stage of step — no combat yet).
__global__ void step_batch_move_stage(
  int num_envs,
  const int32_t* d_action_ids,
  int16_t* d_cell_piece_id,
  int8_t* d_pos_x, int8_t* d_pos_y
);

// Kernel: Resolve combat at destination cells.
__global__ void step_batch_combat_stage(
  int num_envs,
  const int8_t* d_piece_type_arr,
  bool* d_alive,
  int8_t* d_death_reason_arr
);

// Kernel: Update Zobrist hash incrementally after a move.
__global__ void step_batch_zobrist_stage(
  int num_envs,
  int8_t* d_turn,
  int64_t* d_zobrist
);

}  // namespace junqi_cuda
