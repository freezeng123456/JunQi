/*
 * rollout_history.cu
 * Compact device-resident rollout history and minibatch reconstruction.
 */

#include "junqi_cuda.h"

#include <cstdlib>
#ifdef __noinline__
#undef __noinline__
#endif
#include <stdexcept>

#include <cuda_runtime.h>

#include "common.cuh"

namespace junqi_cuda {

namespace {

constexpr int OBSERVER_BELIEF_STRIDE = NUM_TRACKED_TYPES * NUM_CELLS;
constexpr int CM_OBSERVER_STRIDE = CM_NUM_PIDS;
// Public/theory-of-mind combat-memory channels read the adjacent opponents'
// observer slices.  Retain all four slices so reconstruction is input-exact.
constexpr int CM_HISTORY_STRIDE = NUM_SEATS * CM_OBSERVER_STRIDE;

template <typename T>
__global__ void snapshot_observer_slice_kernel(
    const T* __restrict__ source,
    const int8_t* __restrict__ acting_seats,
    T* __restrict__ destination,
    int num_envs,
    int step,
    int observer_stride) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = num_envs * observer_stride;
  if (linear >= total) return;
  const int env = linear / observer_stride;
  const int offset = linear % observer_stride;
  const int seat = static_cast<int>(acting_seats[env]);
  const size_t source_index =
      (static_cast<size_t>(env) * NUM_SEATS + seat) * observer_stride + offset;
  const size_t destination_index =
      (static_cast<size_t>(step) * num_envs + env) * observer_stride + offset;
  destination[destination_index] = source[source_index];
}

template <typename T>
__global__ void gather_rows_kernel(
    const T* __restrict__ source,
    const int64_t* __restrict__ flat_indices,
    T* __restrict__ destination,
    int batch_size,
    int stride) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * stride;
  if (linear >= total) return;
  const int batch = linear / stride;
  const int offset = linear % stride;
  const int64_t source_row = flat_indices[batch];
  destination[linear] = source[static_cast<size_t>(source_row) * stride + offset];
}

template <typename T>
__global__ void gather_observer_slice_kernel(
    const T* __restrict__ source,
    const int64_t* __restrict__ flat_indices,
    const int8_t* __restrict__ acting_seats,
    T* __restrict__ destination,
    int batch_size,
    int observer_stride) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * observer_stride;
  if (linear >= total) return;
  const int batch = linear / observer_stride;
  const int offset = linear % observer_stride;
  const int seat = static_cast<int>(acting_seats[batch]);
  const int64_t source_row = flat_indices[batch];
  const size_t destination_index =
      (static_cast<size_t>(batch) * NUM_SEATS + seat) * observer_stride + offset;
  destination[destination_index] =
      source[static_cast<size_t>(source_row) * observer_stride + offset];
}

template <typename T>
void launch_snapshot_observer_slice(
    const T* source,
    const int8_t* acting_seats,
    T* destination,
    int num_envs,
    int step,
    int observer_stride) {
  const int total = num_envs * observer_stride;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  snapshot_observer_slice_kernel<T><<<grid, block>>>(
      source, acting_seats, destination, num_envs, step, observer_stride);
}

template <typename T>
void launch_gather_rows(
    const T* source,
    const int64_t* flat_indices,
    T* destination,
    int batch_size,
    int stride) {
  const int total = batch_size * stride;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  gather_rows_kernel<T><<<grid, block>>>(
      source, flat_indices, destination, batch_size, stride);
}

template <typename T>
void launch_gather_observer_slice(
    const T* source,
    const int64_t* flat_indices,
    const int8_t* acting_seats,
    T* destination,
    int batch_size,
    int observer_stride) {
  const int total = batch_size * observer_stride;
  const int block = 256;
  const int grid = (total + block - 1) / block;
  gather_observer_slice_kernel<T><<<grid, block>>>(
      source,
      flat_indices,
      acting_seats,
      destination,
      batch_size,
      observer_stride);
}

void ensure_replay_capacity(DeviceRolloutHistory& history, int batch_size) {
  if (batch_size <= history.replay_capacity) {
    history.replay_state->num_envs = batch_size;
    history.replay_obs->num_envs = batch_size;
    return;
  }

  delete history.replay_state;
  delete history.replay_obs;
  history.replay_state = nullptr;
  history.replay_obs = nullptr;
  if (history.d_replay_belief != nullptr) {
    CUDA_CHECK(cudaFree(history.d_replay_belief));
    history.d_replay_belief = nullptr;
  }

  history.replay_state = new DeviceGameStateBatch(batch_size);
  history.replay_obs = new DeviceObservationSingleBatch(batch_size);
  CUDA_CHECK(cudaMalloc(
      &history.d_replay_belief,
      static_cast<size_t>(batch_size) * BELIEF_STRIDE * sizeof(float)));
  history.replay_capacity = batch_size;
}

}  // namespace

DeviceRolloutHistory::DeviceRolloutHistory(int steps, int envs)
    : num_steps(steps), num_envs(envs) {
  if (steps <= 0 || envs <= 0) {
    throw std::invalid_argument(
        "DeviceRolloutHistory requires positive num_steps and num_envs");
  }
  const size_t rows = static_cast<size_t>(steps) * envs;

#define ALLOC_HISTORY(field, stride, T)                                        \
  do {                                                                          \
    const size_t bytes = rows * static_cast<size_t>(stride) * sizeof(T);        \
    CUDA_CHECK(cudaMalloc(&field, bytes));                                       \
    history_bytes += static_cast<uint64_t>(bytes);                               \
  } while (0)

  ALLOC_HISTORY(d_piece_seat_arr, 120, int8_t);
  ALLOC_HISTORY(d_piece_type_arr, 120, int8_t);
  ALLOC_HISTORY(d_alive, 120, bool);
  ALLOC_HISTORY(d_pos_x, 120, int8_t);
  ALLOC_HISTORY(d_pos_y, 120, int8_t);
  ALLOC_HISTORY(d_zero_x, 120, int8_t);
  ALLOC_HISTORY(d_zero_y, 120, int8_t);
  ALLOC_HISTORY(d_move_count_arr, 120, int16_t);
  ALLOC_HISTORY(d_active_eat_arr, 120, int16_t);
  ALLOC_HISTORY(d_passive_surv_arr, 120, int16_t);
  ALLOC_HISTORY(d_death_reason_arr, 120, int8_t);
  ALLOC_HISTORY(d_death_loc_flat_arr, 120, int16_t);
  ALLOC_HISTORY(d_cell_piece_id, NUM_CELLS, int16_t);
  ALLOC_HISTORY(d_seat_dead_arr, NUM_SEATS, bool);
  ALLOC_HISTORY(d_seat_flag_revealed_arr, NUM_SEATS, bool);
  ALLOC_HISTORY(d_turn, 1, int8_t);
  ALLOC_HISTORY(d_move_counter, 1, int32_t);
  ALLOC_HISTORY(d_moves_since_last_combat, 1, int32_t);
  ALLOC_HISTORY(d_move_history, MOVE_HISTORY_LEN * 2, int16_t);
  ALLOC_HISTORY(d_history_write_idx, 1, int32_t);
  ALLOC_HISTORY(d_history_count, 1, int32_t);
  ALLOC_HISTORY(d_observer_belief, OBSERVER_BELIEF_STRIDE, float);

  ALLOC_HISTORY(d_cm_direct_lo, CM_HISTORY_STRIDE, uint64_t);
  ALLOC_HISTORY(d_cm_direct_hi, CM_HISTORY_STRIDE, uint64_t);
  ALLOC_HISTORY(d_cm_direct_type, CM_HISTORY_STRIDE, uint16_t);
  ALLOC_HISTORY(d_cm_last_direct_step, CM_HISTORY_STRIDE, int16_t);
  ALLOC_HISTORY(d_cm_direct_other_count, CM_HISTORY_STRIDE, int16_t);
  ALLOC_HISTORY(d_cm_chain_lo, CM_HISTORY_STRIDE, uint64_t);
  ALLOC_HISTORY(d_cm_chain_hi, CM_HISTORY_STRIDE, uint64_t);
  ALLOC_HISTORY(d_cm_chain_type, CM_HISTORY_STRIDE, uint16_t);
  ALLOC_HISTORY(d_cm_last_chain_step, CM_HISTORY_STRIDE, int16_t);
  ALLOC_HISTORY(d_cm_eaten_by_pid_lo, CM_HISTORY_STRIDE, uint64_t);
  ALLOC_HISTORY(d_cm_eaten_by_pid_hi, CM_HISTORY_STRIDE, uint64_t);
  ALLOC_HISTORY(d_cm_rank_floor, CM_HISTORY_STRIDE, int8_t);
  ALLOC_HISTORY(d_cm_rank_floor_step, CM_HISTORY_STRIDE, int16_t);
  ALLOC_HISTORY(d_cm_is_gongb, CM_HISTORY_STRIDE, bool);
  ALLOC_HISTORY(d_cm_not_gongb, CM_HISTORY_STRIDE, bool);
  ALLOC_HISTORY(d_cm_attacked_by_known_gongb, CM_HISTORY_STRIDE, bool);

#undef ALLOC_HISTORY
}

DeviceRolloutHistory::~DeviceRolloutHistory() {
  auto free_device = [](void* pointer) {
    if (pointer != nullptr) cudaFree(pointer);
  };

  free_device(d_piece_seat_arr);
  free_device(d_piece_type_arr);
  free_device(d_alive);
  free_device(d_pos_x);
  free_device(d_pos_y);
  free_device(d_zero_x);
  free_device(d_zero_y);
  free_device(d_move_count_arr);
  free_device(d_active_eat_arr);
  free_device(d_passive_surv_arr);
  free_device(d_death_reason_arr);
  free_device(d_death_loc_flat_arr);
  free_device(d_cell_piece_id);
  free_device(d_seat_dead_arr);
  free_device(d_seat_flag_revealed_arr);
  free_device(d_turn);
  free_device(d_move_counter);
  free_device(d_moves_since_last_combat);
  free_device(d_move_history);
  free_device(d_history_write_idx);
  free_device(d_history_count);
  free_device(d_observer_belief);

  free_device(d_cm_direct_lo);
  free_device(d_cm_direct_hi);
  free_device(d_cm_direct_type);
  free_device(d_cm_last_direct_step);
  free_device(d_cm_direct_other_count);
  free_device(d_cm_chain_lo);
  free_device(d_cm_chain_hi);
  free_device(d_cm_chain_type);
  free_device(d_cm_last_chain_step);
  free_device(d_cm_eaten_by_pid_lo);
  free_device(d_cm_eaten_by_pid_hi);
  free_device(d_cm_rank_floor);
  free_device(d_cm_rank_floor_step);
  free_device(d_cm_is_gongb);
  free_device(d_cm_not_gongb);
  free_device(d_cm_attacked_by_known_gongb);
  free_device(d_replay_belief);

  delete replay_state;
  delete replay_obs;
}

void DeviceRolloutHistory::snapshot(
    const DeviceGameStateBatch& state,
    const float* belief,
    const int8_t* acting_seats,
    int step,
    int /*stream_id*/) {
  if (state.num_envs != num_envs) {
    throw std::invalid_argument("history/state num_envs mismatch");
  }
  if (step < 0 || step >= num_steps) {
    throw std::out_of_range("history snapshot step out of range");
  }
  if (belief == nullptr || acting_seats == nullptr) {
    throw std::invalid_argument("history snapshot received null device pointer");
  }
  const size_t row_offset = static_cast<size_t>(step) * num_envs;

#define SNAPSHOT_ROWS(destination, source, stride, T)                          \
  CUDA_CHECK(cudaMemcpyAsync(                                                   \
      destination + row_offset * static_cast<size_t>(stride),                  \
      source,                                                                   \
      static_cast<size_t>(num_envs) * (stride) * sizeof(T),                    \
      cudaMemcpyDeviceToDevice))

  SNAPSHOT_ROWS(d_piece_seat_arr, state.d_piece_seat_arr, 120, int8_t);
  SNAPSHOT_ROWS(d_piece_type_arr, state.d_piece_type_arr, 120, int8_t);
  SNAPSHOT_ROWS(d_alive, state.d_alive, 120, bool);
  SNAPSHOT_ROWS(d_pos_x, state.d_pos_x, 120, int8_t);
  SNAPSHOT_ROWS(d_pos_y, state.d_pos_y, 120, int8_t);
  SNAPSHOT_ROWS(d_zero_x, state.d_zero_x, 120, int8_t);
  SNAPSHOT_ROWS(d_zero_y, state.d_zero_y, 120, int8_t);
  SNAPSHOT_ROWS(d_move_count_arr, state.d_move_count_arr, 120, int16_t);
  SNAPSHOT_ROWS(d_active_eat_arr, state.d_active_eat_arr, 120, int16_t);
  SNAPSHOT_ROWS(d_passive_surv_arr, state.d_passive_surv_arr, 120, int16_t);
  SNAPSHOT_ROWS(d_death_reason_arr, state.d_death_reason_arr, 120, int8_t);
  SNAPSHOT_ROWS(d_death_loc_flat_arr, state.d_death_loc_flat_arr, 120, int16_t);
  SNAPSHOT_ROWS(d_cell_piece_id, state.d_cell_piece_id, NUM_CELLS, int16_t);
  SNAPSHOT_ROWS(d_seat_dead_arr, state.d_seat_dead_arr, NUM_SEATS, bool);
  SNAPSHOT_ROWS(
      d_seat_flag_revealed_arr,
      state.d_seat_flag_revealed_arr,
      NUM_SEATS,
      bool);
  SNAPSHOT_ROWS(d_turn, state.d_turn, 1, int8_t);
  SNAPSHOT_ROWS(d_move_counter, state.d_move_counter, 1, int32_t);
  SNAPSHOT_ROWS(
      d_moves_since_last_combat,
      state.d_moves_since_last_combat,
      1,
      int32_t);
  SNAPSHOT_ROWS(
      d_move_history,
      state.d_move_history,
      MOVE_HISTORY_LEN * 2,
      int16_t);
  SNAPSHOT_ROWS(d_history_write_idx, state.d_history_write_idx, 1, int32_t);
  SNAPSHOT_ROWS(d_history_count, state.d_history_count, 1, int32_t);

#define SNAPSHOT_CM(destination, source, T)                                    \
  SNAPSHOT_ROWS(destination, source, CM_HISTORY_STRIDE, T)

  SNAPSHOT_CM(d_cm_direct_lo, state.d_cm_direct_lo, uint64_t);
  SNAPSHOT_CM(d_cm_direct_hi, state.d_cm_direct_hi, uint64_t);
  SNAPSHOT_CM(d_cm_direct_type, state.d_cm_direct_type, uint16_t);
  SNAPSHOT_CM(d_cm_last_direct_step, state.d_cm_last_direct_step, int16_t);
  SNAPSHOT_CM(
      d_cm_direct_other_count,
      state.d_cm_direct_other_count,
      int16_t);
  SNAPSHOT_CM(d_cm_chain_lo, state.d_cm_chain_lo, uint64_t);
  SNAPSHOT_CM(d_cm_chain_hi, state.d_cm_chain_hi, uint64_t);
  SNAPSHOT_CM(d_cm_chain_type, state.d_cm_chain_type, uint16_t);
  SNAPSHOT_CM(d_cm_last_chain_step, state.d_cm_last_chain_step, int16_t);
  SNAPSHOT_CM(d_cm_eaten_by_pid_lo, state.d_cm_eaten_by_pid_lo, uint64_t);
  SNAPSHOT_CM(d_cm_eaten_by_pid_hi, state.d_cm_eaten_by_pid_hi, uint64_t);
  SNAPSHOT_CM(d_cm_rank_floor, state.d_cm_rank_floor, int8_t);
  SNAPSHOT_CM(d_cm_rank_floor_step, state.d_cm_rank_floor_step, int16_t);
  SNAPSHOT_CM(d_cm_is_gongb, state.d_cm_is_gongb, bool);
  SNAPSHOT_CM(d_cm_not_gongb, state.d_cm_not_gongb, bool);
  SNAPSHOT_CM(
      d_cm_attacked_by_known_gongb,
      state.d_cm_attacked_by_known_gongb,
      bool);

#undef SNAPSHOT_CM

#undef SNAPSHOT_ROWS

  launch_snapshot_observer_slice(
      belief,
      acting_seats,
      d_observer_belief,
      num_envs,
      step,
      OBSERVER_BELIEF_STRIDE);

  KERNEL_CHECK();
}

RolloutHistoryReconstruction DeviceRolloutHistory::reconstruct(
    const int64_t* flat_indices,
    const int8_t* acting_seats,
    int batch_size,
    int8_t show_mode,
    int /*stream_id*/) {
  if (batch_size <= 0) {
    throw std::invalid_argument("history reconstruction batch must be positive");
  }
  if (flat_indices == nullptr || acting_seats == nullptr) {
    throw std::invalid_argument(
        "history reconstruction received null device pointer");
  }
  ensure_replay_capacity(*this, batch_size);

#define GATHER_ROWS(source, destination, stride, T)                            \
  launch_gather_rows(                                                           \
      source, flat_indices, destination, batch_size, stride)

  GATHER_ROWS(d_piece_seat_arr, replay_state->d_piece_seat_arr, 120, int8_t);
  GATHER_ROWS(d_piece_type_arr, replay_state->d_piece_type_arr, 120, int8_t);
  GATHER_ROWS(d_alive, replay_state->d_alive, 120, bool);
  GATHER_ROWS(d_pos_x, replay_state->d_pos_x, 120, int8_t);
  GATHER_ROWS(d_pos_y, replay_state->d_pos_y, 120, int8_t);
  GATHER_ROWS(d_zero_x, replay_state->d_zero_x, 120, int8_t);
  GATHER_ROWS(d_zero_y, replay_state->d_zero_y, 120, int8_t);
  GATHER_ROWS(
      d_move_count_arr,
      replay_state->d_move_count_arr,
      120,
      int16_t);
  GATHER_ROWS(
      d_active_eat_arr,
      replay_state->d_active_eat_arr,
      120,
      int16_t);
  GATHER_ROWS(
      d_passive_surv_arr,
      replay_state->d_passive_surv_arr,
      120,
      int16_t);
  GATHER_ROWS(
      d_death_reason_arr,
      replay_state->d_death_reason_arr,
      120,
      int8_t);
  GATHER_ROWS(
      d_death_loc_flat_arr,
      replay_state->d_death_loc_flat_arr,
      120,
      int16_t);
  GATHER_ROWS(
      d_cell_piece_id,
      replay_state->d_cell_piece_id,
      NUM_CELLS,
      int16_t);
  GATHER_ROWS(
      d_seat_dead_arr,
      replay_state->d_seat_dead_arr,
      NUM_SEATS,
      bool);
  GATHER_ROWS(
      d_seat_flag_revealed_arr,
      replay_state->d_seat_flag_revealed_arr,
      NUM_SEATS,
      bool);
  GATHER_ROWS(d_turn, replay_state->d_turn, 1, int8_t);
  GATHER_ROWS(d_move_counter, replay_state->d_move_counter, 1, int32_t);
  GATHER_ROWS(
      d_moves_since_last_combat,
      replay_state->d_moves_since_last_combat,
      1,
      int32_t);
  GATHER_ROWS(
      d_move_history,
      replay_state->d_move_history,
      MOVE_HISTORY_LEN * 2,
      int16_t);
  GATHER_ROWS(
      d_history_write_idx,
      replay_state->d_history_write_idx,
      1,
      int32_t);
  GATHER_ROWS(d_history_count, replay_state->d_history_count, 1, int32_t);

#define GATHER_CM(source, destination, T)                                      \
  GATHER_ROWS(source, destination, CM_HISTORY_STRIDE, T)

  GATHER_CM(d_cm_direct_lo, replay_state->d_cm_direct_lo, uint64_t);
  GATHER_CM(d_cm_direct_hi, replay_state->d_cm_direct_hi, uint64_t);
  GATHER_CM(d_cm_direct_type, replay_state->d_cm_direct_type, uint16_t);
  GATHER_CM(
      d_cm_last_direct_step,
      replay_state->d_cm_last_direct_step,
      int16_t);
  GATHER_CM(
      d_cm_direct_other_count,
      replay_state->d_cm_direct_other_count,
      int16_t);
  GATHER_CM(d_cm_chain_lo, replay_state->d_cm_chain_lo, uint64_t);
  GATHER_CM(d_cm_chain_hi, replay_state->d_cm_chain_hi, uint64_t);
  GATHER_CM(d_cm_chain_type, replay_state->d_cm_chain_type, uint16_t);
  GATHER_CM(
      d_cm_last_chain_step,
      replay_state->d_cm_last_chain_step,
      int16_t);
  GATHER_CM(
      d_cm_eaten_by_pid_lo,
      replay_state->d_cm_eaten_by_pid_lo,
      uint64_t);
  GATHER_CM(
      d_cm_eaten_by_pid_hi,
      replay_state->d_cm_eaten_by_pid_hi,
      uint64_t);
  GATHER_CM(d_cm_rank_floor, replay_state->d_cm_rank_floor, int8_t);
  GATHER_CM(
      d_cm_rank_floor_step,
      replay_state->d_cm_rank_floor_step,
      int16_t);
  GATHER_CM(d_cm_is_gongb, replay_state->d_cm_is_gongb, bool);
  GATHER_CM(d_cm_not_gongb, replay_state->d_cm_not_gongb, bool);
  GATHER_CM(
      d_cm_attacked_by_known_gongb,
      replay_state->d_cm_attacked_by_known_gongb,
      bool);

#undef GATHER_CM

#undef GATHER_ROWS

  launch_gather_observer_slice(
      d_observer_belief,
      flat_indices,
      acting_seats,
      d_replay_belief,
      batch_size,
      OBSERVER_BELIEF_STRIDE);

  KERNEL_CHECK();

  build_observation_single_seat(
      *replay_state,
      d_replay_belief,
      acting_seats,
      *replay_obs,
      show_mode);
  bool* legal_mask =
      legal_mask_canonical_batch(*replay_state, acting_seats);

  RolloutHistoryReconstruction result;
  result.d_spatial = replay_obs->d_spatial;
  result.d_global = replay_obs->d_global;
  result.d_legal_mask = legal_mask;
  result.batch_size = batch_size;
  return result;
}

}  // namespace junqi_cuda
