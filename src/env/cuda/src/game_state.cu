/*
 * game_state.cu
 * DeviceGameStateBatch memory management + step kernel stubs.
 *
 * Array sizes (all indexed [env * stride + ...]):
 *   piece-indexed (120 per env): cell_piece_id_per_piece, piece_seat_arr,
 *     piece_type_arr, alive, pos_x, pos_y, zero_x, zero_y, move_count_arr,
 *     active_eat_arr, passive_surv_arr, death_reason_arr, death_step_arr,
 *     death_loc_flat_arr
 *   cell-indexed  (289 per env): cell_piece_id
 *   seat-indexed  (4 per env):   seat_dead_arr, seat_flag_revealed_arr
 *   scalar (1 per env):          turn, zobrist, move_counter,
 *                                moves_since_last_combat
 */

#include "junqi_cuda.h"
#include "game_state.cuh"
#include "tables.cuh"
#include "common.cuh"
#include "combat_memory.cuh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

namespace junqi_cuda {

// ---------------------------------------------------------------------------
// DeviceGameStateBatch — constructor
// ---------------------------------------------------------------------------
DeviceGameStateBatch::DeviceGameStateBatch(int n) : num_envs(n) {
    CUDA_CHECK(cudaMalloc(&d_cell_piece_id_per_piece, (size_t)n * 120 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_piece_seat_arr,          (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_piece_type_arr,          (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_alive,                   (size_t)n * 120 * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_pos_x,                   (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_pos_y,                   (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_zero_x,                  (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_zero_y,                  (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_move_count_arr,          (size_t)n * 120 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_active_eat_arr,          (size_t)n * 120 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_passive_surv_arr,        (size_t)n * 120 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_death_reason_arr,        (size_t)n * 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_death_step_arr,          (size_t)n * 120 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_death_loc_flat_arr,      (size_t)n * 120 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_cell_piece_id,           (size_t)n * 289 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_seat_dead_arr,           (size_t)n * 4   * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_seat_flag_revealed_arr,  (size_t)n * 4   * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_turn,                    (size_t)n * 1   * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_zobrist,                 (size_t)n * 1   * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_move_counter,            (size_t)n * 1   * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_moves_since_last_combat, (size_t)n * 1   * sizeof(int32_t)));
    // Phase 1b — per-env termination state (init zero).
    CUDA_CHECK(cudaMalloc(&d_terminated,              (size_t)n * 1   * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_winner_team,             (size_t)n * 1   * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_draw,                    (size_t)n * 1   * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_terminated,  0, (size_t)n * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_winner_team, 0xff, (size_t)n * sizeof(int8_t)));  // -1
    CUDA_CHECK(cudaMemset(d_draw,        0, (size_t)n * sizeof(bool)));
    // Move history ring buffer — zeroed on init.
    CUDA_CHECK(cudaMalloc(&d_move_history,
        (size_t)n * MOVE_HISTORY_LEN * 2 * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_history_write_idx, (size_t)n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_history_count,     (size_t)n * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_move_history, 0,
        (size_t)n * MOVE_HISTORY_LEN * 2 * sizeof(int16_t)));
    CUDA_CHECK(cudaMemset(d_history_write_idx, 0, (size_t)n * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(d_history_count,     0, (size_t)n * sizeof(int32_t)));

    // CombatMemory v6 — per-(env, observer, pid) SoA, shape (n, 4, 120).
    // Allocated once and ZEROED here; updated entirely on-device by
    // step_batch_kernel and observation_kernel.  Only copy_from_host_v4 /
    // copy_to_host_v4 (parity-test path) ever transfer these to/from
    // host — they are NOT touched in the training hot path.
    const size_t CM_N    = (size_t)n * 4 * 120;
    CUDA_CHECK(cudaMalloc(&d_cm_direct_lo,                 CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_direct_hi,                 CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_direct_type,               CM_N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_last_direct_step,          CM_N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_direct_other_count,        CM_N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_chain_lo,                  CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_chain_hi,                  CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_chain_type,                CM_N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_last_chain_step,           CM_N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_eaten_by_pid_lo,           CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_eaten_by_pid_hi,           CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_rank_floor,                CM_N * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_rank_floor_step,           CM_N * sizeof(int16_t)));
    CUDA_CHECK(cudaMalloc(&d_cm_is_gongb,                  CM_N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_cm_not_gongb,                 CM_N * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_cm_attacked_by_known_gongb,   CM_N * sizeof(bool)));

    CUDA_CHECK(cudaMemset(d_cm_direct_lo,                 0, CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_cm_direct_hi,                 0, CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_cm_direct_type,               0, CM_N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d_cm_last_direct_step,       0xff, CM_N * sizeof(int16_t)));   // -1
    CUDA_CHECK(cudaMemset(d_cm_direct_other_count,        0, CM_N * sizeof(int16_t)));
    CUDA_CHECK(cudaMemset(d_cm_chain_lo,                  0, CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_cm_chain_hi,                  0, CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_cm_chain_type,                0, CM_N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(d_cm_last_chain_step,        0xff, CM_N * sizeof(int16_t)));   // -1
    CUDA_CHECK(cudaMemset(d_cm_eaten_by_pid_lo,          0, CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_cm_eaten_by_pid_hi,          0, CM_N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_cm_rank_floor,                0, CM_N * sizeof(int8_t)));
    CUDA_CHECK(cudaMemset(d_cm_rank_floor_step,        0xff, CM_N * sizeof(int16_t)));   // -1
    CUDA_CHECK(cudaMemset(d_cm_is_gongb,                  0, CM_N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_cm_not_gongb,                 0, CM_N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_cm_attacked_by_known_gongb,   0, CM_N * sizeof(bool)));
}

// ---------------------------------------------------------------------------
// DeviceGameStateBatch — destructor
// ---------------------------------------------------------------------------
DeviceGameStateBatch::~DeviceGameStateBatch() {
    auto f = [](void* p) { if (p) cudaFree(p); };
    f(d_cell_piece_id_per_piece);
    f(d_piece_seat_arr);
    f(d_piece_type_arr);
    f(d_alive);
    f(d_pos_x);
    f(d_pos_y);
    f(d_zero_x);
    f(d_zero_y);
    f(d_move_count_arr);
    f(d_active_eat_arr);
    f(d_passive_surv_arr);
    f(d_death_reason_arr);
    f(d_death_step_arr);
    f(d_death_loc_flat_arr);
    f(d_cell_piece_id);
    f(d_seat_dead_arr);
    f(d_seat_flag_revealed_arr);
    f(d_turn);
    f(d_zobrist);
    f(d_move_counter);
    f(d_moves_since_last_combat);
    f(d_terminated);
    f(d_winner_team);
    f(d_draw);
    f(d_move_history);
    f(d_history_write_idx);
    f(d_history_count);
    // CombatMemory v6
    f(d_cm_direct_lo);
    f(d_cm_direct_hi);
    f(d_cm_direct_type);
    f(d_cm_last_direct_step);
    f(d_cm_direct_other_count);
    f(d_cm_chain_lo);
    f(d_cm_chain_hi);
    f(d_cm_chain_type);
    f(d_cm_last_chain_step);
    f(d_cm_eaten_by_pid_lo);
    f(d_cm_eaten_by_pid_hi);
    f(d_cm_rank_floor);
    f(d_cm_rank_floor_step);
    f(d_cm_is_gongb);
    f(d_cm_not_gongb);
    f(d_cm_attacked_by_known_gongb);
}

// ---------------------------------------------------------------------------
// copy_from_host: H→D transfer for all arrays (one cudaMemcpy per array)
// ---------------------------------------------------------------------------
void DeviceGameStateBatch::copy_from_host(
    const int16_t* h_cell_piece_id_per_piece,
    const int8_t*  h_piece_seat_arr,
    const int8_t*  h_piece_type_arr,
    const bool*    h_alive,
    const int8_t*  h_pos_x,
    const int8_t*  h_pos_y,
    const int8_t*  h_zero_x,
    const int8_t*  h_zero_y,
    const int16_t* h_move_count_arr,
    const int16_t* h_active_eat_arr,
    const int16_t* h_passive_surv_arr,
    const int8_t*  h_death_reason_arr,
    const int16_t* h_death_step_arr,
    const int16_t* h_death_loc_flat_arr,
    const int16_t* h_cell_piece_id,
    const bool*    h_seat_dead_arr,
    const bool*    h_seat_flag_revealed_arr,
    const int8_t*  h_turn,
    const int64_t* h_zobrist,
    const int32_t* h_move_counter,
    const int32_t* h_moves_since_last_combat,
    int /*stream_id*/)
{
    // For simplicity stream_id is ignored in this iteration (synchronous H2D).
    // A future version can pass a cudaStream_t array.
    const int n = num_envs;
#define H2D(ptr, src, count, T) \
    CUDA_CHECK(cudaMemcpy(ptr, src, (size_t)(n) * (count) * sizeof(T), cudaMemcpyHostToDevice))

    H2D(d_cell_piece_id_per_piece, h_cell_piece_id_per_piece, 120, int16_t);
    H2D(d_piece_seat_arr,          h_piece_seat_arr,          120, int8_t);
    H2D(d_piece_type_arr,          h_piece_type_arr,          120, int8_t);
    H2D(d_alive,                   h_alive,                   120, bool);
    H2D(d_pos_x,                   h_pos_x,                   120, int8_t);
    H2D(d_pos_y,                   h_pos_y,                   120, int8_t);
    H2D(d_zero_x,                  h_zero_x,                  120, int8_t);
    H2D(d_zero_y,                  h_zero_y,                  120, int8_t);
    H2D(d_move_count_arr,          h_move_count_arr,          120, int16_t);
    H2D(d_active_eat_arr,          h_active_eat_arr,          120, int16_t);
    H2D(d_passive_surv_arr,        h_passive_surv_arr,        120, int16_t);
    H2D(d_death_reason_arr,        h_death_reason_arr,        120, int8_t);
    H2D(d_death_step_arr,          h_death_step_arr,          120, int16_t);
    H2D(d_death_loc_flat_arr,      h_death_loc_flat_arr,      120, int16_t);
    H2D(d_cell_piece_id,           h_cell_piece_id,           289, int16_t);
    H2D(d_seat_dead_arr,           h_seat_dead_arr,           4,   bool);
    H2D(d_seat_flag_revealed_arr,  h_seat_flag_revealed_arr,  4,   bool);
    H2D(d_turn,                    h_turn,                    1,   int8_t);
    H2D(d_zobrist,                 h_zobrist,                 1,   int64_t);
    H2D(d_move_counter,            h_move_counter,            1,   int32_t);
    H2D(d_moves_since_last_combat, h_moves_since_last_combat, 1,   int32_t);
#undef H2D
}

// ---------------------------------------------------------------------------
// copy_from_host_legal_lite
//
// Fused single-transfer H2D path for the 6 fields the legal_action_ids_batch
// kernel reads.  We stage them into one pinned host buffer (allocated once),
// then memcpy the whole blob to a single device buffer, then a tiny scatter
// kernel splits it back into the 6 per-field device arrays.
//
// Rationale:
//   * One large transfer saturates PCIe much better than 6 small ones.
//   * Pinned host memory doubles the achievable bandwidth (~12 GB/s → 24 GB/s).
//   * The scatter kernel is cheaper than 5 extra cudaMemcpy launches.
//
// The staging buffers are owned by a process-level static manager so they
// outlive individual DeviceGameStateBatch instances but auto-scale with the
// largest batch size encountered.
// ---------------------------------------------------------------------------

// Field layout in the staging blob (offset in bytes, per batch):
//   [0          .. 120N     ): piece_seat_arr        (int8)
//   [120N       .. 240N     ): piece_type_arr        (int8)
//   [240N       .. 360N     ): alive                 (bool/int8)
//   [360N       .. 480N     ): pos_x                 (int8)
//   [480N       .. 600N     ): pos_y                 (int8)
//   [600N       .. 600N+578N): cell_piece_id         (int16, 2 bytes/elem)
// Total bytes per env: 120*5 + 289*2 = 600 + 578 = 1178.
constexpr int LITE_BYTES_PER_ENV = 1178;

// Per-process staging buffers.  Allocated on first call to size for N envs,
// and reallocated if the caller grows beyond s_lite_envs.
static void*    s_lite_host = nullptr;       // pinned host buffer
static uint8_t* s_lite_device = nullptr;     // device-side staging buffer
static int      s_lite_envs = 0;

static void ensure_lite_buffers(int N) {
    if (N <= s_lite_envs) return;
    if (s_lite_host)   cudaFreeHost(s_lite_host);
    if (s_lite_device) cudaFree(s_lite_device);
    size_t bytes = (size_t)N * LITE_BYTES_PER_ENV;
    CUDA_CHECK(cudaMallocHost(&s_lite_host,   bytes));
    CUDA_CHECK(cudaMalloc    (&s_lite_device, bytes));
    s_lite_envs = N;
}

// Scatter kernel: walk the big staging blob and copy each field into the
// DeviceGameStateBatch's individual arrays.  One thread per (env, index)
// pair; simple 1-D grid.  The reads are fully coalesced by field layout.
__global__ void scatter_lite_kernel(
    const uint8_t* __restrict__ blob,
    int     N,
    int8_t* __restrict__ d_piece_seat_arr,
    int8_t* __restrict__ d_piece_type_arr,
    bool*   __restrict__ d_alive,
    int8_t* __restrict__ d_pos_x,
    int8_t* __restrict__ d_pos_y,
    int16_t*__restrict__ d_cell_piece_id)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 120 piece-indexed elements per field × 5 fields + 289 cell-indexed (int16).
    // Total per-env threads = 120*5 + 289 = 889.
    // We let each thread move exactly one element; threads beyond budget return.
    int per_env = 120 * 5 + 289;
    int total   = N * per_env;
    if (tid >= total) return;

    int env = tid / per_env;
    int off = tid % per_env;

    // The 6 fields share one blob.  Lay them out compactly so each field's
    // per-env offset is predictable:
    //   env_base_bytes = env * LITE_BYTES_PER_ENV
    const uint8_t* base = blob + (size_t)env * LITE_BYTES_PER_ENV;

    if (off < 120) {
        // piece_seat_arr
        int i = off;
        d_piece_seat_arr[env * 120 + i] = (int8_t)base[i];
    } else if (off < 240) {
        int i = off - 120;
        d_piece_type_arr[env * 120 + i] = (int8_t)base[120 + i];
    } else if (off < 360) {
        int i = off - 240;
        d_alive[env * 120 + i] = (bool)base[240 + i];
    } else if (off < 480) {
        int i = off - 360;
        d_pos_x[env * 120 + i] = (int8_t)base[360 + i];
    } else if (off < 600) {
        int i = off - 480;
        d_pos_y[env * 120 + i] = (int8_t)base[480 + i];
    } else {
        int i = off - 600;                // 0..288
        // cell_piece_id is int16 starting at byte 600 in env slot.
        const int16_t* src16 = reinterpret_cast<const int16_t*>(base + 600);
        d_cell_piece_id[env * 289 + i] = src16[i];
    }
}

void DeviceGameStateBatch::copy_from_host_legal_lite(
    const int8_t*  h_piece_seat_arr,
    const int8_t*  h_piece_type_arr,
    const bool*    h_alive,
    const int8_t*  h_pos_x,
    const int8_t*  h_pos_y,
    const int16_t* h_cell_piece_id,
    int /*stream_id*/)
{
    const int N = num_envs;
    ensure_lite_buffers(N);

    // Stage into the pinned host blob.  Field offsets per env are:
    //   byte 0    : piece_seat_arr[120]
    //   byte 120  : piece_type_arr[120]
    //   byte 240  : alive[120]
    //   byte 360  : pos_x[120]
    //   byte 480  : pos_y[120]
    //   byte 600  : cell_piece_id[289] (int16 -> 578 bytes)
    // We pack all envs consecutively so each env occupies LITE_BYTES_PER_ENV
    // bytes contiguously in the staging blob.
    uint8_t* host_blob = (uint8_t*)s_lite_host;
    for (int e = 0; e < N; ++e) {
        uint8_t* dst = host_blob + (size_t)e * LITE_BYTES_PER_ENV;
        std::memcpy(dst +   0, h_piece_seat_arr + (size_t)e * 120, 120);
        std::memcpy(dst + 120, h_piece_type_arr + (size_t)e * 120, 120);
        std::memcpy(dst + 240, h_alive          + (size_t)e * 120, 120);
        std::memcpy(dst + 360, h_pos_x          + (size_t)e * 120, 120);
        std::memcpy(dst + 480, h_pos_y          + (size_t)e * 120, 120);
        std::memcpy(dst + 600, h_cell_piece_id  + (size_t)e * 289, 289 * sizeof(int16_t));
    }

    // Single large H2D transfer.
    size_t bytes = (size_t)N * LITE_BYTES_PER_ENV;
    CUDA_CHECK(cudaMemcpy(s_lite_device, s_lite_host, bytes, cudaMemcpyHostToDevice));

    // Scatter on device.
    int per_env = 120 * 5 + 289;
    int total   = N * per_env;
    int block   = 256;
    int grid    = (total + block - 1) / block;
    scatter_lite_kernel<<<grid, block>>>(
        s_lite_device, N,
        d_piece_seat_arr, d_piece_type_arr,
        d_alive, d_pos_x, d_pos_y,
        d_cell_piece_id
    );
    KERNEL_CHECK();
}

// ---------------------------------------------------------------------------
// copy_to_host: D→H transfer for all arrays
// ---------------------------------------------------------------------------
void DeviceGameStateBatch::copy_to_host(
    int16_t* h_cell_piece_id_per_piece,
    int8_t*  h_piece_seat_arr,
    int8_t*  h_piece_type_arr,
    bool*    h_alive,
    int8_t*  h_pos_x,
    int8_t*  h_pos_y,
    int8_t*  h_zero_x,
    int8_t*  h_zero_y,
    int16_t* h_move_count_arr,
    int16_t* h_active_eat_arr,
    int16_t* h_passive_surv_arr,
    int8_t*  h_death_reason_arr,
    int16_t* h_death_step_arr,
    int16_t* h_death_loc_flat_arr,
    int16_t* h_cell_piece_id,
    bool*    h_seat_dead_arr,
    bool*    h_seat_flag_revealed_arr,
    int8_t*  h_turn,
    int64_t* h_zobrist,
    int32_t* h_move_counter,
    int32_t* h_moves_since_last_combat,
    int /*stream_id*/) const
{
    const int n = num_envs;
#define D2H(dst, src, count, T) \
    CUDA_CHECK(cudaMemcpy(dst, src, (size_t)(n) * (count) * sizeof(T), cudaMemcpyDeviceToHost))

    D2H(h_cell_piece_id_per_piece, d_cell_piece_id_per_piece, 120, int16_t);
    D2H(h_piece_seat_arr,          d_piece_seat_arr,          120, int8_t);
    D2H(h_piece_type_arr,          d_piece_type_arr,          120, int8_t);
    D2H(h_alive,                   d_alive,                   120, bool);
    D2H(h_pos_x,                   d_pos_x,                   120, int8_t);
    D2H(h_pos_y,                   d_pos_y,                   120, int8_t);
    D2H(h_zero_x,                  d_zero_x,                  120, int8_t);
    D2H(h_zero_y,                  d_zero_y,                  120, int8_t);
    D2H(h_move_count_arr,          d_move_count_arr,          120, int16_t);
    D2H(h_active_eat_arr,          d_active_eat_arr,          120, int16_t);
    D2H(h_passive_surv_arr,        d_passive_surv_arr,        120, int16_t);
    D2H(h_death_reason_arr,        d_death_reason_arr,        120, int8_t);
    D2H(h_death_step_arr,          d_death_step_arr,          120, int16_t);
    D2H(h_death_loc_flat_arr,      d_death_loc_flat_arr,      120, int16_t);
    D2H(h_cell_piece_id,           d_cell_piece_id,           289, int16_t);
    D2H(h_seat_dead_arr,           d_seat_dead_arr,           4,   bool);
    D2H(h_seat_flag_revealed_arr,  d_seat_flag_revealed_arr,  4,   bool);
    D2H(h_turn,                    d_turn,                    1,   int8_t);
    D2H(h_zobrist,                 d_zobrist,                 1,   int64_t);
    D2H(h_move_counter,            d_move_counter,            1,   int32_t);
    D2H(h_moves_since_last_combat, d_moves_since_last_combat, 1,   int32_t);
#undef D2H
}

// ---------------------------------------------------------------------------
// Constants for action generation
// ---------------------------------------------------------------------------
static constexpr int MAX_ACTIONS_PER_ENV = 512;
// Piece type constants (match PieceType enum in junqi_cuda.h)
static constexpr int8_t PT_JUNQI  = 2;  // immobile
static constexpr int8_t PT_DILEI  = 3;  // immobile
static constexpr int8_t PT_GONGB  = 13; // engineer

// ---------------------------------------------------------------------------
// Device helper: try to append one action id to the output array.
// Returns false if the buffer is full (silent overflow guard).
// Uses an atomic increment on d_action_counts[env] to reserve a slot.
// ---------------------------------------------------------------------------
__device__ inline void emit_action(
    int32_t*  d_action_ids,
    int32_t*  d_action_counts,
    int       env,
    int32_t   action_id)
{
    int slot = (int)atomicAdd(&d_action_counts[env], 1);
    if (slot < MAX_ACTIONS_PER_ENV) {
        d_action_ids[env * MAX_ACTIONS_PER_ENV + slot] = action_id;
    }
    // If overflow: count was already incremented — caller can detect via
    // d_action_counts[env] > MAX_ACTIONS_PER_ENV but we do not assert here.
}

// ---------------------------------------------------------------------------
// legal_action_kernel
//
// Thread assignment: tid = blockIdx.x * blockDim.x + threadIdx.x
//   env = tid / 120
//   pid = tid % 120
//
// Each thread processes one (env, piece) pair for the acting seat of that env.
// ---------------------------------------------------------------------------
__global__ void legal_action_kernel(
    int            num_envs,
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const int16_t* d_cell_piece_id,
    const int8_t*  d_acting_seats,
    int32_t*       d_action_ids,
    int32_t*       d_action_counts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = tid / 120;
    int pid = tid % 120;
    if (env >= num_envs) return;

    // -----------------------------------------------------------------------
    // 1. Gate: piece must belong to the acting seat, be alive, be mobile, and
    //    sit on a valid non-stronghold cell.
    // -----------------------------------------------------------------------
    int8_t acting_seat = d_acting_seats[env];
    int base120 = env * 120;
    int8_t pseat = d_piece_seat_arr[base120 + pid];
    if (pseat != acting_seat) return;

    bool alive = d_alive[base120 + pid];
    if (!alive) return;

    int8_t ptype = d_piece_type_arr[base120 + pid];
    // Immobile types: JUNQI (2) and DILEI (3).
    if (ptype == PT_JUNQI || ptype == PT_DILEI) return;

    int8_t sx = d_pos_x[base120 + pid];
    int8_t sy = d_pos_y[base120 + pid];
    // Dead pieces can have negative pos; skip if off-board.
    if (sx < 0 || sy < 0) return;
    int16_t src_flat = (int16_t)(sy * 17 + sx);
    if (!ON_BOARD_FLAT[src_flat]) return;
    if (STRONGHOLD_FLAT[src_flat]) return;

    // -----------------------------------------------------------------------
    // 2. Build occupancy for this env.
    //    empty[cell]: cell_piece_id[cell] < 0
    //    acting_team: acting_seat & 1
    //    For enemy cells: landable = enemy piece AND not in camp
    // -----------------------------------------------------------------------
    const int16_t* cpid = d_cell_piece_id + env * 289;
    // We read piece_seat_arr indexed per-cell to get team.
    // team_of_cell[cell] = piece_seat_arr[ cpid[cell] ] & 1, if occupied.
    int8_t acting_team = (int8_t)(acting_seat & 1);

    int32_t src_col = (int32_t)src_flat * 289;

    // -----------------------------------------------------------------------
    // 3a. Orthogonal 1-step (slots 0..3 in ADJACENT_CELLS)
    // -----------------------------------------------------------------------
    for (int k = 0; k < 4; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
        if (nb < 0) continue;
        // Evaluate landability:
        int16_t occ_pid = cpid[nb];
        bool is_empty = (occ_pid < 0);
        bool landable;
        if (is_empty) {
            landable = true;
        } else {
            // Enemy? enemy = same-team piece = same (team) from seat.
            int8_t occ_seat = d_piece_seat_arr[base120 + occ_pid];  // base120 correct
            // Note: occ_pid is the global piece id index within this env.
            // d_piece_seat_arr[env * 120 + occ_pid]
            int8_t occ_team = (int8_t)(occ_seat & 1);
            bool is_enemy = (occ_team != acting_team);
            // Enemy attackable = enemy AND not in camp.
            landable = is_enemy && !CAMP_FLAT[nb];
        }
        if (landable) {
            emit_action(d_action_ids, d_action_counts, env, src_col + nb);
        }
    }

    // -----------------------------------------------------------------------
    // 3b. Diagonal 1-step into/out-of camp (slots 4..7 in ADJACENT_CELLS,
    //     -1 when not applicable)
    // -----------------------------------------------------------------------
    for (int k = 4; k < 8; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
        if (nb < 0) continue;
        int16_t occ_pid = cpid[nb];
        bool is_empty = (occ_pid < 0);
        bool landable;
        if (is_empty) {
            landable = true;
        } else {
            int8_t occ_seat = d_piece_seat_arr[base120 + occ_pid];
            int8_t occ_team = (int8_t)(occ_seat & 1);
            bool is_enemy = (occ_team != acting_team);
            landable = is_enemy && !CAMP_FLAT[nb];
        }
        if (landable) {
            emit_action(d_action_ids, d_action_counts, env, src_col + nb);
        }
    }

    // -----------------------------------------------------------------------
    // 3c. Rail long-range (only if src is a rail cell)
    // -----------------------------------------------------------------------
    if (!RAIL_FLAT[src_flat]) return;  // no more moves possible for off-rail pieces

    bool is_engineer = (ptype == PT_GONGB);

    if (!is_engineer) {
        // Non-engineer: straight rail rays.
        // STRAIGHT_RAIL_RAYS[src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN + k]
        // is the ORDERED list of rail cells reached from src walking the rail
        // adjacency graph in direction dir; -1 = end of ray.
        //
        // Under the legacy-correct topology each "next step" along a ray may
        // be 2 cells away (NineGrid jumps), so k=0 is NOT guaranteed to equal
        // the ortho 1-step neighbour.  We suppress duplicates with section 3a
        // by comparing the emitted cell to the 4 ortho neighbours of src.
        int16_t ortho0 = ADJACENT_CELLS[src_flat * 8 + 0];
        int16_t ortho1 = ADJACENT_CELLS[src_flat * 8 + 1];
        int16_t ortho2 = ADJACENT_CELLS[src_flat * 8 + 2];
        int16_t ortho3 = ADJACENT_CELLS[src_flat * 8 + 3];

        for (int dir = 0; dir < 4; ++dir) {
            int ray_base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
            for (int k = 0; k < STRAIGHT_RAY_LEN; ++k) {
                int16_t nb = STRAIGHT_RAIL_RAYS[ray_base + k];
                if (nb < 0) break;   // end of ray
                bool is_ortho_of_src = (nb == ortho0) || (nb == ortho1)
                                    || (nb == ortho2) || (nb == ortho3);
                int16_t occ_pid = cpid[nb];
                bool is_empty = (occ_pid < 0);
                if (is_empty) {
                    if (!is_ortho_of_src) {
                        emit_action(d_action_ids, d_action_counts, env, src_col + nb);
                    }
                    // continue ray through empty cell
                } else {
                    // Occupied: enemy-attackable (if not already handled by 3a).
                    if (!is_ortho_of_src) {
                        int8_t occ_seat = d_piece_seat_arr[base120 + occ_pid];
                        int8_t occ_team = (int8_t)(occ_seat & 1);
                        bool is_enemy = (occ_team != acting_team);
                        if (is_enemy && !CAMP_FLAT[nb]) {
                            emit_action(d_action_ids, d_action_counts, env, src_col + nb);
                        }
                    }
                    break;  // ray blocked regardless (stop here)
                }
            }
        }

        // -----------------------------------------------------------------
        // 3d. Curve-rail BFS (non-engineer) — walks rail-graph neighbours
        //     restricted to cells sharing the same CURVE_RAIL_OF id as src.
        //     Emits empty destinations and enemy-attackable terminals.
        // -----------------------------------------------------------------
        int8_t src_curve = CURVE_RAIL_OF[src_flat];
        if (src_curve > 0) {
            int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
            if (src_ri >= 0) {
                // Two-word visited bitmask covers ENG_NUM_RAIL=73 entries.
                uint64_t v0 = 0, v1 = 0;
                if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
                else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

                int8_t queue[ENG_NUM_RAIL];
                int qhead = 0, qtail = 0;
                queue[qtail++] = src_ri;

                while (qhead < qtail) {
                    int8_t cur_ri = queue[qhead++];
                    for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                        int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                        if (nb_ri < 0) continue;
                        // Test/mark visited.
                        if (nb_ri < 64) {
                            uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                            if (v0 & b) continue;
                            v0 |= b;
                        } else {
                            uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                            if (v1 & b) continue;
                            v1 |= b;
                        }
                        int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                        // Must stay on the same curve rail.
                        if (CURVE_RAIL_OF[nb_flat] != src_curve) continue;

                        // Suppress cells already covered by section 3a (ortho)
                        // or by the straight-rail walk (same row or column as
                        // src).  Curve-BFS is strictly the "diagonal" curve
                        // case on CPU (`_curve_rail_clear`) — straight-axis
                        // cells are reached via `_straight_rail_clear` first.
                        int nb_x = nb_flat % 17;
                        int nb_y = nb_flat / 17;
                        bool same_axis = (nb_x == sx) || (nb_y == sy);

                        int16_t occ_pid = cpid[nb_flat];
                        bool is_empty = (occ_pid < 0);
                        if (is_empty) {
                            if (!same_axis) {
                                emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                            }
                            queue[qtail++] = nb_ri;
                        } else {
                            if (!same_axis) {
                                int8_t occ_seat = d_piece_seat_arr[base120 + occ_pid];
                                int8_t occ_team = (int8_t)(occ_seat & 1);
                                bool is_enemy = (occ_team != acting_team);
                                if (is_enemy && !CAMP_FLAT[nb_flat]) {
                                    emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                                }
                            }
                            // Do NOT enqueue — blocked by piece.
                        }
                    }
                }
            }
        }
    } else {
        // Engineer BFS over the rail graph using ENG_RAIL_ADJ.
        // Walks empty rail cells, stopping at enemy-attackable terminals.
        int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
        if (src_ri < 0) return;

        // Two-word visited bitmask — 73 rail cells fit in 64+9 bits.
        uint64_t v0 = 0, v1 = 0;
        if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
        else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

        // Cache ortho neighbours so BFS does not re-emit cells already covered
        // by section 3a.
        int16_t ortho0 = ADJACENT_CELLS[src_flat * 8 + 0];
        int16_t ortho1 = ADJACENT_CELLS[src_flat * 8 + 1];
        int16_t ortho2 = ADJACENT_CELLS[src_flat * 8 + 2];
        int16_t ortho3 = ADJACENT_CELLS[src_flat * 8 + 3];

        int8_t queue[ENG_NUM_RAIL];
        int qhead = 0, qtail = 0;
        queue[qtail++] = src_ri;

        while (qhead < qtail) {
            int8_t cur_ri = queue[qhead++];
            for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                if (nb_ri < 0) continue;
                if (nb_ri < 64) {
                    uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                    if (v0 & b) continue;
                    v0 |= b;
                } else {
                    uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                    if (v1 & b) continue;
                    v1 |= b;
                }
                int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                bool is_ortho_of_src = (nb_flat == ortho0) || (nb_flat == ortho1)
                                    || (nb_flat == ortho2) || (nb_flat == ortho3);

                int16_t occ_pid = cpid[nb_flat];
                bool is_empty = (occ_pid < 0);
                if (is_empty) {
                    if (!is_ortho_of_src) {
                        emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                    }
                    queue[qtail++] = nb_ri;
                } else {
                    if (!is_ortho_of_src) {
                        int8_t occ_seat = d_piece_seat_arr[base120 + occ_pid];
                        int8_t occ_team = (int8_t)(occ_seat & 1);
                        bool is_enemy = (occ_team != acting_team);
                        if (is_enemy && !CAMP_FLAT[nb_flat]) {
                            emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                        }
                    }
                    // Blocked: do NOT enqueue.
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// legal_action_kernel_v2 — block-per-env with shared-memory occupancy cache.
//
// Launch config: <<<num_envs, 128>>>
//   blockIdx.x  = env index
//   threadIdx.x = piece id (0..119); threads 120..127 idle.
//
// Each block cooperatively loads the 289-entry cell_piece_id array into shared
// memory once, then every piece-thread in the block does its 8-neighbour /
// ray / BFS lookups against the shared copy.
//
// The v1 kernel (2D tid → (env, pid) split across block boundary) caused
// warps to straddle envs, preventing coalesced global loads of cpid.  v2
// localises all cpid accesses to one env per block → L1/L2 traffic drops by
// ~8× and warp divergence caused by per-env branches disappears.
//
// Bit-identical output with v1 (validated by test_gpu_kernel_v2_parity.py).
// ---------------------------------------------------------------------------
__global__ void legal_action_kernel_v2(
    int            num_envs,
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const int16_t* d_cell_piece_id,
    const int8_t*  d_acting_seats,
    int32_t*       d_action_ids,
    int32_t*       d_action_counts)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    int pid = threadIdx.x;

    // Shared-memory occupancy cache for this env's 289 cells.
    __shared__ int16_t s_cpid[289];
    // Cooperative load: each thread loads ~289/128 ≈ 2-3 cells.
    const int16_t* g_cpid = d_cell_piece_id + env * 289;
    for (int i = threadIdx.x; i < 289; i += blockDim.x) {
        s_cpid[i] = g_cpid[i];
    }

    // Also cache piece_seat_arr for this env — used for team checks on
    // occupied cells (up to 120 reads per thread × 120 threads).
    __shared__ int8_t s_pseat[120];
    if (threadIdx.x < 120) {
        s_pseat[threadIdx.x] = d_piece_seat_arr[env * 120 + threadIdx.x];
    }
    __syncthreads();

    // Inactive threads beyond 120 return after the cooperative loads.
    if (pid >= 120) return;

    // ---- Gate checks ----
    int8_t acting_seat = d_acting_seats[env];
    int8_t pseat = s_pseat[pid];
    if (pseat != acting_seat) return;

    int base120 = env * 120;
    if (!d_alive[base120 + pid]) return;

    int8_t ptype = d_piece_type_arr[base120 + pid];
    if (ptype == PT_JUNQI || ptype == PT_DILEI) return;

    int8_t sx = d_pos_x[base120 + pid];
    int8_t sy = d_pos_y[base120 + pid];
    if (sx < 0 || sy < 0) return;
    int16_t src_flat = (int16_t)(sy * 17 + sx);
    if (!ON_BOARD_FLAT[src_flat]) return;
    if (STRONGHOLD_FLAT[src_flat]) return;

    int8_t acting_team = (int8_t)(acting_seat & 1);
    int32_t src_col = (int32_t)src_flat * 289;

    // Helper lambda: check whether dst nb is landable.
    // Reads shared cpid + shared pseat — no global memory after gate.
    auto landable = [&](int16_t nb) -> bool {
        int16_t occ = s_cpid[nb];
        if (occ < 0) return true;                 // empty
        int8_t occ_team = (int8_t)(s_pseat[occ] & 1);
        bool is_enemy = (occ_team != acting_team);
        return is_enemy && !CAMP_FLAT[nb];
    };

    // ---- 3a. Orthogonal 1-step ----
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
        if (nb >= 0 && landable(nb)) {
            emit_action(d_action_ids, d_action_counts, env, src_col + nb);
        }
    }

    // ---- 3b. Diagonal 1-step via camp ----
    #pragma unroll
    for (int k = 4; k < 8; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
        if (nb >= 0 && landable(nb)) {
            emit_action(d_action_ids, d_action_counts, env, src_col + nb);
        }
    }

    // ---- 3c. Rail long-range ----
    if (!RAIL_FLAT[src_flat]) return;
    bool is_engineer = (ptype == PT_GONGB);

    if (!is_engineer) {
        // Non-engineer: straight rail rays (same semantics as v1 kernel).
        int16_t ortho0 = ADJACENT_CELLS[src_flat * 8 + 0];
        int16_t ortho1 = ADJACENT_CELLS[src_flat * 8 + 1];
        int16_t ortho2 = ADJACENT_CELLS[src_flat * 8 + 2];
        int16_t ortho3 = ADJACENT_CELLS[src_flat * 8 + 3];

        #pragma unroll
        for (int dir = 0; dir < 4; ++dir) {
            int ray_base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
            for (int k = 0; k < STRAIGHT_RAY_LEN; ++k) {
                int16_t nb = STRAIGHT_RAIL_RAYS[ray_base + k];
                if (nb < 0) break;
                bool is_ortho_of_src = (nb == ortho0) || (nb == ortho1)
                                    || (nb == ortho2) || (nb == ortho3);
                int16_t occ = s_cpid[nb];
                bool is_empty = (occ < 0);
                if (is_empty) {
                    if (!is_ortho_of_src) {
                        emit_action(d_action_ids, d_action_counts, env, src_col + nb);
                    }
                } else {
                    if (!is_ortho_of_src) {
                        int8_t occ_team = (int8_t)(s_pseat[occ] & 1);
                        bool is_enemy = (occ_team != acting_team);
                        if (is_enemy && !CAMP_FLAT[nb]) {
                            emit_action(d_action_ids, d_action_counts, env, src_col + nb);
                        }
                    }
                    break;
                }
            }
        }

        // Curve-rail BFS (non-engineer) — see v1 for details.
        int8_t src_curve = CURVE_RAIL_OF[src_flat];
        if (src_curve > 0) {
            int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
            if (src_ri >= 0) {
                uint64_t v0 = 0, v1 = 0;
                if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
                else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

                int8_t queue[ENG_NUM_RAIL];
                int qhead = 0, qtail = 0;
                queue[qtail++] = src_ri;

                while (qhead < qtail) {
                    int8_t cur_ri = queue[qhead++];
                    for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                        int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                        if (nb_ri < 0) continue;
                        if (nb_ri < 64) {
                            uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                            if (v0 & b) continue;
                            v0 |= b;
                        } else {
                            uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                            if (v1 & b) continue;
                            v1 |= b;
                        }
                        int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                        if (CURVE_RAIL_OF[nb_flat] != src_curve) continue;

                        // Same-axis cells handled by straight-rail above; see v1.
                        int nb_x = nb_flat % 17;
                        int nb_y = nb_flat / 17;
                        bool same_axis = (nb_x == sx) || (nb_y == sy);
                        int16_t occ = s_cpid[nb_flat];
                        bool is_empty = (occ < 0);
                        if (is_empty) {
                            if (!same_axis) {
                                emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                            }
                            queue[qtail++] = nb_ri;
                        } else {
                            if (!same_axis) {
                                int8_t occ_team = (int8_t)(s_pseat[occ] & 1);
                                bool is_enemy = (occ_team != acting_team);
                                if (is_enemy && !CAMP_FLAT[nb_flat]) {
                                    emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
        if (src_ri < 0) return;

        uint64_t v0 = 0, v1 = 0;
        if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
        else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

        int16_t ortho0 = ADJACENT_CELLS[src_flat * 8 + 0];
        int16_t ortho1 = ADJACENT_CELLS[src_flat * 8 + 1];
        int16_t ortho2 = ADJACENT_CELLS[src_flat * 8 + 2];
        int16_t ortho3 = ADJACENT_CELLS[src_flat * 8 + 3];

        int8_t queue[ENG_NUM_RAIL];
        int qhead = 0, qtail = 0;
        queue[qtail++] = src_ri;

        while (qhead < qtail) {
            int8_t cur_ri = queue[qhead++];
            for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                if (nb_ri < 0) continue;
                if (nb_ri < 64) {
                    uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                    if (v0 & b) continue;
                    v0 |= b;
                } else {
                    uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                    if (v1 & b) continue;
                    v1 |= b;
                }
                int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                bool is_ortho_of_src = (nb_flat == ortho0) || (nb_flat == ortho1)
                                    || (nb_flat == ortho2) || (nb_flat == ortho3);
                int16_t occ = s_cpid[nb_flat];
                bool is_empty = (occ < 0);
                if (is_empty) {
                    if (!is_ortho_of_src) {
                        emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                    }
                    queue[qtail++] = nb_ri;
                } else {
                    if (!is_ortho_of_src) {
                        int8_t occ_team = (int8_t)(s_pseat[occ] & 1);
                        bool is_enemy = (occ_team != acting_team);
                        if (is_enemy && !CAMP_FLAT[nb_flat]) {
                            emit_action(d_action_ids, d_action_counts, env, src_col + nb_flat);
                        }
                    }
                }
            }
        }
    }
}

__global__ void step_batch_move_stage(
    int /*num_envs*/,
    const int32_t* /*d_action_ids*/,
    int16_t*       /*d_cell_piece_id*/,
    int8_t*        /*d_pos_x*/, int8_t* /*d_pos_y*/)
{
    // Legacy three-stage scaffold; superseded by step_batch_kernel below.
}

__global__ void step_batch_combat_stage(
    int /*num_envs*/,
    const int8_t* /*d_piece_type_arr*/,
    bool*         /*d_alive*/,
    int8_t*       /*d_death_reason_arr*/)
{
    // Legacy three-stage scaffold; superseded by step_batch_kernel below.
}

__global__ void step_batch_zobrist_stage(
    int /*num_envs*/,
    int8_t*  /*d_turn*/,
    int64_t* /*d_zobrist*/)
{
    // Legacy three-stage scaffold; superseded by step_batch_kernel below.
}

// ===========================================================================
// step_batch_kernel — monolithic per-env step (Phase 1b).
//
// One thread per env.  Mirrors ``BatchedGameState.step_batch`` EXACTLY,
// including:
//   * Plain-move vs combat dispatch on dst occupancy.
//   * Combat resolution (ranked / mine / bomb / flag).
//   * Q7 SILING flag reveal.
//   * Flag-capture → seat surrender (kill every piece of that seat).
//   * Post-combat dead-sweep: seats with 0 alive pieces marked dead.
//   * Q12 turn-advance with "no legal moves ⇒ kill seat" repeat.
//   * _check_victory (checked twice: pre- and post-turn-advance).
//   * Incremental Zobrist XOR updates (matches _zobrist.py layout).
//
// Terminated envs are skipped.  Invalid actions (no src piece) silently skip.
// Output buffers (valid, event, terminated, winner_team, draw, flag_captured)
// are written at (env).
//
// NOTE on Zobrist: ``d_zobrist_*`` tables must be CPU-seeded via
// ``upload_zobrist_tables_from_host`` before any step kernel is launched;
// otherwise GPU hashes will diverge from CPU.
// ===========================================================================

// Piece-type constants shared with movegen kernels.
static constexpr int8_t PT_ZHADAN = 4;
static constexpr int8_t PT_SILING = 5;
// Death-reason constants (mirror junqi_core.rules.DeathReason).
static constexpr int8_t DR_KILLED_BY_ENEMY  = 0;
static constexpr int8_t DR_HIT_MINE_OR_BOMB = 1;
static constexpr int8_t DR_MUTUAL           = 2;
// Event constants (mirror junqi_core.rules.Event).
static constexpr int8_t EV_MOVE   = 1;
static constexpr int8_t EV_EAT    = 2;
static constexpr int8_t EV_BOMB   = 3;
static constexpr int8_t EV_KILLED = 4;
// Draw thresholds (mirror rules.MAX_NUM_MOVES*).
static constexpr int32_t MAX_NUM_MOVES_CONST              = 4000;
static constexpr int32_t MAX_NUM_MOVES_BETWEEN_ATTACKS_CONST = 200;

// ---------------------------------------------------------------------------
// Combat resolver — pure function, mirrors rules.resolve_combat.
// Inputs are PieceType.value ∈ [0, 13].
// Output is Event.value ∈ {1, 2, 3, 4}.
// Caller must guarantee attacker is mobile and defender != NONE/DARK.
// ---------------------------------------------------------------------------
__device__ inline int8_t gpu_resolve_combat(int8_t a, int8_t d) {
    // Bomb (either side) — mutual death against anything, mine and flag
    // included. Must precede the flag and mine branches; the ordering is the
    // rule itself. Mirrors rules.resolve_combat and legacy CompareChess.
    if (a == PT_ZHADAN || d == PT_ZHADAN) return EV_BOMB;
    // Flag capture.
    if (d == PT_JUNQI) return EV_EAT;
    // Mine: only GONGB eats; everyone else dies.
    if (d == PT_DILEI) return (a == PT_GONGB) ? EV_EAT : EV_KILLED;
    // Both ranked combatants — compare by rank (lower = stronger).
    if (a == d) return EV_BOMB;
    if (a <  d) return EV_EAT;
    return EV_KILLED;
}

// SILING reveal predicates (mirror rules.siling_reveals_{src,dst}).
__device__ inline bool gpu_siling_reveals_src(int8_t a, int8_t /*d*/, int8_t ev) {
    return (a == PT_SILING) && (ev == EV_KILLED || ev == EV_BOMB);
}
__device__ inline bool gpu_siling_reveals_dst(int8_t /*a*/, int8_t d, int8_t ev) {
    return (d == PT_SILING) && (ev == EV_EAT || ev == EV_BOMB);
}

// Death-reason classifier (mirror rules.classify_death_reason, attacker POV
// only — the caller only needs the src-side reason; BOMB callers pass MUTUAL
// directly, and EAT callers always hit KILLED_BY_ENEMY for the dst side).
__device__ inline int8_t gpu_attacker_death_reason(int8_t defender_type) {
    if (defender_type == PT_DILEI || defender_type == PT_ZHADAN)
        return DR_HIT_MINE_OR_BOMB;
    return DR_KILLED_BY_ENEMY;
}

// ---------------------------------------------------------------------------
// Device helper: has_legal_moves for a given seat in env `env`.
// Used inside Q12 chain.  Mirrors junqi_core.move_gen.has_legal_moves_soa.
//
// Returns true on FIRST legal move found; no enumeration needed.
// ---------------------------------------------------------------------------
__device__ bool gpu_has_legal_moves(
    int env,
    int8_t seat_val,
    const int16_t* cpid,                // (289,) this env's cell_piece_id
    const int8_t*  piece_seat_arr_env,  // (120,) this env's piece_seat_arr
    const int8_t*  piece_type_arr_env,  // (120,)
    const bool*    alive_env,           // (120,)
    const int8_t*  pos_x_env,           // (120,)
    const int8_t*  pos_y_env            // (120,)
) {
    int8_t seat_team = seat_val & 1;

    // Per-cell "landable" lookup: empty OR (enemy AND not camp).
    auto is_landable = [&](int nb_flat) -> bool {
        int16_t occ = cpid[nb_flat];
        if (occ < 0) return true;
        int8_t occ_seat = piece_seat_arr_env[occ];
        int8_t occ_team = occ_seat & 1;
        if (occ_team == seat_team) return false;
        return !CAMP_FLAT[nb_flat];
    };
    auto is_empty = [&](int nb_flat) -> bool {
        return cpid[nb_flat] < 0;
    };

    for (int pid = 0; pid < 120; ++pid) {
        if (!alive_env[pid]) continue;
        if (piece_seat_arr_env[pid] != seat_val) continue;
        int8_t ptype = piece_type_arr_env[pid];
        if (ptype == PT_JUNQI || ptype == PT_DILEI) continue;
        int8_t sx = pos_x_env[pid];
        int8_t sy = pos_y_env[pid];
        if (sx < 0 || sy < 0) continue;
        int src_flat = sy * 17 + sx;
        if (!ON_BOARD_FLAT[src_flat] || STRONGHOLD_FLAT[src_flat]) continue;

        // (a) ortho 1-step
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
            if (nb >= 0 && is_landable(nb)) return true;
        }
        // (b) diag into/out camp
        #pragma unroll
        for (int k = 4; k < 8; ++k) {
            int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
            if (nb >= 0 && is_landable(nb)) return true;
        }
        // (c) rail long-range
        if (!RAIL_FLAT[src_flat]) continue;
        bool is_eng = (ptype == PT_GONGB);
        if (!is_eng) {
            // Straight rails.
            #pragma unroll
            for (int dir = 0; dir < 4; ++dir) {
                int ray_base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
                for (int k = 0; k < STRAIGHT_RAY_LEN; ++k) {
                    int16_t nb = STRAIGHT_RAIL_RAYS[ray_base + k];
                    if (nb < 0) break;
                    if (is_landable(nb)) return true;
                    if (!is_empty(nb)) break;  // blocked (same-team or camp-enemy)
                }
            }
            // Curve rail BFS (only on curve cells).
            int8_t cid = CURVE_RAIL_OF[src_flat];
            if (cid > 0) {
                int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
                if (src_ri >= 0) {
                    uint64_t vis0 = 0, vis1 = 0;
                    auto visit = [&](int ri) -> bool {
                        if (ri < 64) {
                            uint64_t b = (uint64_t)1 << ri;
                            if (vis0 & b) return false;
                            vis0 |= b; return true;
                        } else {
                            uint64_t b = (uint64_t)1 << (ri - 64);
                            if (vis1 & b) return false;
                            vis1 |= b; return true;
                        }
                    };
                    int8_t queue[ENG_NUM_RAIL]; int qh = 0, qt = 0;
                    visit(src_ri); queue[qt++] = src_ri;
                    while (qh < qt) {
                        int8_t cur = queue[qh++];
                        #pragma unroll
                        for (int k = 0; k < ENG_ADJ_WIDTH; ++k) {
                            int8_t nb_ri = ENG_RAIL_ADJ[cur * ENG_ADJ_WIDTH + k];
                            if (nb_ri < 0) continue;
                            int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                            if (CURVE_RAIL_OF[nb_flat] != cid) continue;
                            if (!visit(nb_ri)) continue;
                            if (is_landable(nb_flat)) return true;
                            if (is_empty(nb_flat)) queue[qt++] = nb_ri;
                        }
                    }
                }
            }
        } else {
            // Engineer BFS.
            int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
            if (src_ri < 0) continue;
            uint64_t vis0 = 0, vis1 = 0;
            auto visit = [&](int ri) -> bool {
                if (ri < 64) {
                    uint64_t b = (uint64_t)1 << ri;
                    if (vis0 & b) return false;
                    vis0 |= b; return true;
                } else {
                    uint64_t b = (uint64_t)1 << (ri - 64);
                    if (vis1 & b) return false;
                    vis1 |= b; return true;
                }
            };
            int8_t queue[ENG_NUM_RAIL]; int qh = 0, qt = 0;
            visit(src_ri); queue[qt++] = src_ri;
            while (qh < qt) {
                int8_t cur = queue[qh++];
                #pragma unroll
                for (int k = 0; k < ENG_ADJ_WIDTH; ++k) {
                    int8_t nb_ri = ENG_RAIL_ADJ[cur * ENG_ADJ_WIDTH + k];
                    if (nb_ri < 0) continue;
                    if (!visit(nb_ri)) continue;
                    int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                    if (is_landable(nb_flat)) return true;
                    if (is_empty(nb_flat)) queue[qt++] = nb_ri;
                }
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Helper: Zobrist contribution for (winner_team).  winner ∈ {-1, 0, 1}
// maps to ZOB_WINNER[0..2].
// ---------------------------------------------------------------------------
__device__ inline int64_t zob_winner(int8_t winner_team) {
    int idx = (winner_team < 0) ? 0 : (winner_team + 1);
    return d_zobrist_winner[idx];
}

// ---------------------------------------------------------------------------
// Helper: kill every alive piece of `seat_val` in env `env`, clearing the
// board cell and accumulating XOR into ``zob``.  Returns number of pieces
// killed (for death-step bookkeeping).  Sets seat_dead[seat_val] = true.
// Does NOT XOR in ZOB_SEAT_DEAD — caller handles that for flag-capture vs Q12.
// If reveal_flag=true, also XORs in ZOB_SEAT_FLAG_REVEALED (for flag-capture
// path; legacy behaviour omits this for Q12).
// ---------------------------------------------------------------------------
__device__ int gpu_surrender_seat(
    int env,
    int8_t seat_val,
    int32_t death_step,
    bool reveal_flag,
    int16_t* cpid_env,
    bool* alive_env,
    int8_t* pos_x_env,
    int8_t* pos_y_env,
    const int8_t* piece_seat_arr_env,
    const int8_t* piece_type_arr_env,
    int8_t* death_reason_arr_env,
    int16_t* death_step_arr_env,
    int16_t* death_loc_flat_arr_env,
    bool* seat_dead_env,
    bool* seat_flag_revealed_env,
    int64_t& zob)
{
    if (seat_dead_env[seat_val]) return 0;
    seat_dead_env[seat_val] = true;
    zob ^= d_zobrist_seat_dead[seat_val];

    int killed = 0;
    for (int pid = 0; pid < 120; ++pid) {
        if (!alive_env[pid]) continue;
        if (piece_seat_arr_env[pid] != seat_val) continue;
        int8_t px = pos_x_env[pid];
        int8_t py = pos_y_env[pid];
        int8_t ptype = piece_type_arr_env[pid];
        if (px >= 0 && py >= 0) {
            int flat = py * 17 + px;
            zob ^= d_zobrist_piece[(int64_t)pid * 14 * 289
                                   + (int64_t)ptype * 289 + flat];
            cpid_env[flat] = -1;
        }
        alive_env[pid] = false;
        pos_x_env[pid] = -1;
        pos_y_env[pid] = -1;
        death_reason_arr_env[pid]   = DR_KILLED_BY_ENEMY;
        death_step_arr_env[pid]     = (int16_t)death_step;
        death_loc_flat_arr_env[pid] = -1;
        ++killed;
    }

    if (reveal_flag && !seat_flag_revealed_env[seat_val]) {
        seat_flag_revealed_env[seat_val] = true;
        zob ^= d_zobrist_seat_flag_revealed[seat_val];
    }
    return killed;
}

// ---------------------------------------------------------------------------
// The monolithic per-env step kernel.
// ---------------------------------------------------------------------------
__global__ void step_batch_kernel(
    int num_envs,
    const int32_t* d_action_ids,
    // Piece-indexed SoA (N × 120)
    int16_t* d_cell_piece_id_all,      // <-- cell-indexed (N × 289)
    const int8_t* d_piece_seat_arr_all,
    const int8_t* d_piece_type_arr_all,
    bool*   d_alive_all,
    int8_t* d_pos_x_all,
    int8_t* d_pos_y_all,
    int16_t* d_move_count_arr_all,
    int16_t* d_active_eat_arr_all,
    int16_t* d_passive_surv_arr_all,
    int8_t*  d_death_reason_arr_all,
    int16_t* d_death_step_arr_all,
    int16_t* d_death_loc_flat_arr_all,
    // Seat-indexed SoA (N × 4)
    bool* d_seat_dead_arr_all,
    bool* d_seat_flag_revealed_arr_all,
    // Scalars (N,)
    int8_t*  d_turn,
    int64_t* d_zobrist,
    int32_t* d_move_counter,
    int32_t* d_moves_since_last_combat,
    bool*    d_terminated,
    int8_t*  d_winner_team,
    bool*    d_draw,
    // CombatMemory v6 (N × 4 observers × 120 pids)
    uint64_t* d_cm_direct_lo_all,
    uint64_t* d_cm_direct_hi_all,
    uint16_t* d_cm_direct_type_all,
    int16_t*  d_cm_last_direct_step_all,
    int16_t*  d_cm_direct_other_count_all,
    uint64_t* d_cm_chain_lo_all,
    uint64_t* d_cm_chain_hi_all,
    uint16_t* d_cm_chain_type_all,
    int16_t*  d_cm_last_chain_step_all,
    uint64_t* d_cm_eaten_by_pid_lo_all,
    uint64_t* d_cm_eaten_by_pid_hi_all,
    int8_t*   d_cm_rank_floor_all,
    int16_t*  d_cm_rank_floor_step_all,
    bool*     d_cm_is_gongb_all,
    bool*     d_cm_not_gongb_all,
    bool*     d_cm_attacked_by_known_gongb_all,
    // Outputs (N,)
    bool*   d_valid_out,
    int8_t* d_event_out,
    bool*   d_terminated_out,
    int8_t* d_winner_out,
    bool*   d_draw_out,
    bool*   d_flag_captured_out)
{
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;

    // Per-env output defaults (safe for skipped envs).
    d_valid_out[env]         = false;
    d_event_out[env]         = 0;
    d_terminated_out[env]    = d_terminated[env];
    d_winner_out[env]        = d_winner_team[env];
    d_draw_out[env]          = d_draw[env];
    d_flag_captured_out[env] = false;

    // Skip terminated envs entirely.
    if (d_terminated[env]) return;

    // ------------------- Per-env pointer views -------------------
    int16_t* cpid     = d_cell_piece_id_all     + env * 289;
    const int8_t* piece_seat_arr = d_piece_seat_arr_all + env * 120;
    const int8_t* piece_type_arr = d_piece_type_arr_all + env * 120;
    bool*   alive_env = d_alive_all             + env * 120;
    int8_t* pos_x_env = d_pos_x_all             + env * 120;
    int8_t* pos_y_env = d_pos_y_all             + env * 120;
    int16_t* move_count_arr   = d_move_count_arr_all   + env * 120;
    int16_t* active_eat_arr   = d_active_eat_arr_all   + env * 120;
    int16_t* passive_surv_arr = d_passive_surv_arr_all + env * 120;
    int8_t*  death_reason_arr = d_death_reason_arr_all + env * 120;
    int16_t* death_step_arr   = d_death_step_arr_all   + env * 120;
    int16_t* death_loc_flat_arr = d_death_loc_flat_arr_all + env * 120;
    bool* seat_dead_env  = d_seat_dead_arr_all         + env * 4;
    bool* seat_flag_rev  = d_seat_flag_revealed_arr_all+ env * 4;

    // CombatMemory v6 — per-env slice (4×120 each).
    CMEnvPtrs cm;
    {
        const size_t off = (size_t)env * 4 * 120;
        cm.direct_lo               = d_cm_direct_lo_all                 + off;
        cm.direct_hi               = d_cm_direct_hi_all                 + off;
        cm.direct_type             = d_cm_direct_type_all               + off;
        cm.last_direct_step        = d_cm_last_direct_step_all          + off;
        cm.direct_other_count      = d_cm_direct_other_count_all        + off;
        cm.chain_lo                = d_cm_chain_lo_all                  + off;
        cm.chain_hi                = d_cm_chain_hi_all                  + off;
        cm.chain_type              = d_cm_chain_type_all                + off;
        cm.last_chain_step         = d_cm_last_chain_step_all           + off;
        cm.eaten_by_pid_lo         = d_cm_eaten_by_pid_lo_all           + off;
        cm.eaten_by_pid_hi         = d_cm_eaten_by_pid_hi_all           + off;
        cm.rank_floor              = d_cm_rank_floor_all                + off;
        cm.rank_floor_step         = d_cm_rank_floor_step_all           + off;
        cm.is_gongb                = d_cm_is_gongb_all                  + off;
        cm.not_gongb               = d_cm_not_gongb_all                 + off;
        cm.attacked_by_known_gongb = d_cm_attacked_by_known_gongb_all   + off;
    }

    // ------------------- Parse action -------------------
    int32_t action_id = d_action_ids[env];
    int src_flat = action_id / 289;
    int dst_flat = action_id % 289;

    int16_t src_pid = cpid[src_flat];
    if (src_pid < 0) return;  // invalid — no piece at src; output stays defaulted.
    int16_t dst_pid = cpid[dst_flat];
    bool has_dst = (dst_pid >= 0) && alive_env[dst_pid];
    int8_t src_type = piece_type_arr[src_pid];
    int8_t dst_type = has_dst ? piece_type_arr[dst_pid] : (int8_t)0;

    int8_t acting_seat = d_turn[env];
    int32_t old_mc = d_move_counter[env];
    int32_t old_mslc = d_moves_since_last_combat[env];
    int32_t death_step = old_mc + 1;

    // ------------------- Zobrist: XOR-out current scalars -------------------
    int64_t zob = d_zobrist[env];
    zob ^= d_zobrist_turn[acting_seat];
    zob ^= d_zobrist_move_counter[old_mc & ZOB_MOVE_COUNTER_MASK];
    zob ^= d_zobrist_moves_since_combat[old_mslc & ZOB_MOVES_SINCE_COMBAT_MASK];

    int8_t event_val = 0;
    bool flag_captured = false;
    int32_t new_mslc = old_mslc;

    // CombatMemory v4: path-revealed GONGB.  Engineer-only check.  Must
    // run BEFORE we mutate cpid (blockers along the path are still in
    // place).  Cheap fast-path filter on src_type avoids the BFS for
    // 99% of moves.
    if (src_type == PT_GONGB) {
        if (cm_move_requires_gongb_dev(cpid, (int16_t)src_flat, (int16_t)dst_flat)) {
            cm_apply_path_revealed_gongb_dev(cm, src_pid);
        }
    }

    if (!has_dst) {
        // ------- Plain move -------
        event_val = EV_MOVE;
        zob ^= d_zobrist_piece[(int64_t)src_pid * 14 * 289
                               + (int64_t)src_type * 289 + src_flat];
        zob ^= d_zobrist_piece[(int64_t)src_pid * 14 * 289
                               + (int64_t)src_type * 289 + dst_flat];
        cpid[src_flat] = -1;
        cpid[dst_flat] = src_pid;
        pos_x_env[src_pid] = (int8_t)(dst_flat % 17);
        pos_y_env[src_pid] = (int8_t)(dst_flat / 17);
        move_count_arr[src_pid] += 1;
        new_mslc = old_mslc + 1;
    } else {
        // ------- Combat -------
        int8_t ev = gpu_resolve_combat(src_type, dst_type);
        event_val = ev;

        // SILING reveal for src seat
        if (gpu_siling_reveals_src(src_type, dst_type, ev)) {
            int8_t s = piece_seat_arr[src_pid];
            if (!seat_flag_rev[s]) {
                seat_flag_rev[s] = true;
                zob ^= d_zobrist_seat_flag_revealed[s];
            }
        }
        if (gpu_siling_reveals_dst(src_type, dst_type, ev)) {
            int8_t s = piece_seat_arr[dst_pid];
            if (!seat_flag_rev[s]) {
                seat_flag_rev[s] = true;
                zob ^= d_zobrist_seat_flag_revealed[s];
            }
        }

        flag_captured = (dst_type == PT_JUNQI);

        int64_t pzob_src = (int64_t)src_pid * 14 * 289 + (int64_t)src_type * 289;
        int64_t pzob_dst = (int64_t)dst_pid * 14 * 289 + (int64_t)dst_type * 289;

        if (ev == EV_EAT) {
            // dst dies; src moves onto dst
            zob ^= d_zobrist_piece[pzob_dst + dst_flat];
            // kill dst piece
            alive_env[dst_pid]              = false;
            pos_x_env[dst_pid]              = -1;
            pos_y_env[dst_pid]              = -1;
            death_reason_arr[dst_pid]       = DR_KILLED_BY_ENEMY;
            death_step_arr[dst_pid]         = (int16_t)death_step;
            death_loc_flat_arr[dst_pid]     = (int16_t)dst_flat;
            // move src onto dst
            zob ^= d_zobrist_piece[pzob_src + src_flat];
            zob ^= d_zobrist_piece[pzob_src + dst_flat];
            cpid[src_flat] = -1;
            cpid[dst_flat] = src_pid;
            pos_x_env[src_pid] = (int8_t)(dst_flat % 17);
            pos_y_env[src_pid] = (int8_t)(dst_flat / 17);
            active_eat_arr[src_pid] += 1;
            new_mslc = 0;
            // CombatMemory v4: src ate dst.
            cm_apply_event_dev(
                cm,
                /*is_eat=*/true,
                /*attacker_pid=*/src_pid,
                /*defender_pid=*/dst_pid,
                /*attacker_seat=*/piece_seat_arr[src_pid],
                /*defender_seat=*/piece_seat_arr[dst_pid],
                /*attacker_type_val=*/src_type,
                /*defender_type_val=*/dst_type,
                /*defender_pos_flat=*/(int16_t)dst_flat,
                /*death_step=*/death_step);
        } else if (ev == EV_KILLED) {
            // src dies, dst survives
            zob ^= d_zobrist_piece[pzob_src + src_flat];
            alive_env[src_pid]          = false;
            cpid[src_flat]              = -1;
            pos_x_env[src_pid]          = -1;
            pos_y_env[src_pid]          = -1;
            death_reason_arr[src_pid]   = gpu_attacker_death_reason(dst_type);
            death_step_arr[src_pid]     = (int16_t)death_step;
            death_loc_flat_arr[src_pid] = (int16_t)dst_flat;
            passive_surv_arr[dst_pid]  += 1;
            new_mslc = 0;
            // CombatMemory v4: defender survived.
            cm_apply_event_dev(
                cm,
                /*is_eat=*/false,
                /*attacker_pid=*/src_pid,
                /*defender_pid=*/dst_pid,
                /*attacker_seat=*/piece_seat_arr[src_pid],
                /*defender_seat=*/piece_seat_arr[dst_pid],
                /*attacker_type_val=*/src_type,
                /*defender_type_val=*/dst_type,
                /*defender_pos_flat=*/(int16_t)dst_flat,
                /*death_step=*/death_step);
        } else { // EV_BOMB — both die; CombatMemory not updated (no live target).
            zob ^= d_zobrist_piece[pzob_src + src_flat];
            zob ^= d_zobrist_piece[pzob_dst + dst_flat];
            alive_env[src_pid] = false;
            alive_env[dst_pid] = false;
            cpid[src_flat] = -1;
            cpid[dst_flat] = -1;
            pos_x_env[src_pid] = -1; pos_y_env[src_pid] = -1;
            pos_x_env[dst_pid] = -1; pos_y_env[dst_pid] = -1;
            death_reason_arr[src_pid] = DR_MUTUAL;
            death_step_arr[src_pid]   = (int16_t)death_step;
            death_loc_flat_arr[src_pid] = (int16_t)dst_flat;
            death_reason_arr[dst_pid] = DR_MUTUAL;
            death_step_arr[dst_pid]   = (int16_t)death_step;
            death_loc_flat_arr[dst_pid] = (int16_t)dst_flat;
            new_mslc = 0;
        }
    }

    // ------- Flag capture → seat surrenders -------
    if (flag_captured) {
        int8_t surr = piece_seat_arr[dst_pid];
        if (!seat_dead_env[surr]) {
            // Reveal flag (matches CPU: _flag_reveal_delta called first, THEN
            // _surrender_seat_delta with reveal_flag=False).
            if (!seat_flag_rev[surr]) {
                seat_flag_rev[surr] = true;
                zob ^= d_zobrist_seat_flag_revealed[surr];
            }
            gpu_surrender_seat(
                env, surr, death_step, /*reveal_flag=*/false,
                cpid, alive_env, pos_x_env, pos_y_env,
                piece_seat_arr, piece_type_arr,
                death_reason_arr, death_step_arr, death_loc_flat_arr,
                seat_dead_env, seat_flag_rev, zob);
        }
    }

    // ------- Post-combat dead sweep -------
    #pragma unroll
    for (int sv = 0; sv < 4; ++sv) {
        if (seat_dead_env[sv]) continue;
        bool any = false;
        for (int pid = 0; pid < 120; ++pid) {
            if (piece_seat_arr[pid] == sv && alive_env[pid]) { any = true; break; }
        }
        if (!any) {
            seat_dead_env[sv] = true;
            zob ^= d_zobrist_seat_dead[sv];
        }
    }

    int32_t new_mc = old_mc + 1;

    // ------- Check termination (first pass) -------
    auto check_victory = [&](bool& term_out, int8_t& winner_out, bool& draw_out) {
        bool red_alive  = !seat_dead_env[0] || !seat_dead_env[2];
        bool blue_alive = !seat_dead_env[1] || !seat_dead_env[3];
        term_out = false; winner_out = -1; draw_out = false;
        if (!red_alive && !blue_alive) {
            term_out = true;
            winner_out = (int8_t)(acting_seat & 1);
            return;
        }
        if (!red_alive)  { term_out = true; winner_out = 1; return; }
        if (!blue_alive) { term_out = true; winner_out = 0; return; }
        if (new_mc >= MAX_NUM_MOVES_CONST ||
            new_mslc >= MAX_NUM_MOVES_BETWEEN_ATTACKS_CONST) {
            term_out = true; draw_out = true;
        }
    };

    bool terminated = false;
    int8_t winner_team = -1;
    bool draw = false;
    check_victory(terminated, winner_team, draw);

    // ------- Turn advance + Q12 (only if not terminated) -------
    int8_t new_turn = acting_seat;
    if (!terminated) {
        int8_t candidate = (acting_seat + 1) % 4;
        int chain = 0;
        while (chain < 4) {
            if (seat_dead_env[candidate]) {
                candidate = (candidate + 1) & 3;
                ++chain;
                continue;
            }
            bool moves = gpu_has_legal_moves(
                env, candidate, cpid,
                piece_seat_arr, piece_type_arr, alive_env,
                pos_x_env, pos_y_env);
            if (moves) break;
            // Q12: kill this seat (no flag reveal)
            gpu_surrender_seat(
                env, candidate, death_step, /*reveal_flag=*/false,
                cpid, alive_env, pos_x_env, pos_y_env,
                piece_seat_arr, piece_type_arr,
                death_reason_arr, death_step_arr, death_loc_flat_arr,
                seat_dead_env, seat_flag_rev, zob);
            candidate = (candidate + 1) & 3;
            ++chain;
        }
        new_turn = candidate;
        // Re-check victory (Q12 may have killed a whole team).
        check_victory(terminated, winner_team, draw);
    }

    // ------- Commit scalars + Zobrist XOR-in -------
    d_move_counter[env]            = new_mc;
    d_moves_since_last_combat[env] = new_mslc;
    zob ^= d_zobrist_move_counter[new_mc & ZOB_MOVE_COUNTER_MASK];
    zob ^= d_zobrist_moves_since_combat[new_mslc & ZOB_MOVES_SINCE_COMBAT_MASK];
    zob ^= d_zobrist_turn[new_turn];
    if (terminated) zob ^= d_zobrist_terminated;
    if (draw)       zob ^= d_zobrist_draw;
    // Winner slot swap (0=None was initial state because we only step
    // non-terminated envs; new idx depends on winner_team ≥ 0).
    if (winner_team >= 0) {
        zob ^= d_zobrist_winner[0];
        zob ^= d_zobrist_winner[winner_team + 1];
    }

    d_turn[env]        = new_turn;
    d_zobrist[env]     = zob;
    d_terminated[env]  = terminated;
    d_winner_team[env] = winner_team;
    d_draw[env]        = draw;

    d_valid_out[env]         = true;
    d_event_out[env]         = event_val;
    d_terminated_out[env]    = terminated;
    d_winner_out[env]        = winner_team;
    d_draw_out[env]          = draw;
    d_flag_captured_out[env] = flag_captured;
}


// ---------------------------------------------------------------------------
// GpuScratch — persistent device / pinned buffers.
// Lives for process lifetime; grows on demand; never reallocated smaller.
// ---------------------------------------------------------------------------
GpuScratch::~GpuScratch() {
    if (d_acting_seats)    cudaFree(d_acting_seats);
    if (d_belief)          cudaFree(d_belief);
    if (d_observer_seats)  cudaFree(d_observer_seats);
    if (d_action_ids)      cudaFree(d_action_ids);
    if (d_action_counts)   cudaFree(d_action_counts);
    if (d_csr_offsets)     cudaFree(d_csr_offsets);
    if (d_csr_values)      cudaFree(d_csr_values);
    if (d_piece_slot_mask) cudaFree(d_piece_slot_mask);
    if (stream_a) { cudaStreamDestroy((cudaStream_t)stream_a); }
    if (stream_b) { cudaStreamDestroy((cudaStream_t)stream_b); }
}

GpuScratch& GpuScratch::instance() {
    static GpuScratch g;
    return g;
}

void GpuScratch::reset() {
    if (d_acting_seats)   { cudaFree(d_acting_seats);   d_acting_seats   = nullptr; }
    if (d_belief)         { cudaFree(d_belief);         d_belief         = nullptr; }
    if (d_observer_seats) { cudaFree(d_observer_seats); d_observer_seats = nullptr; }
    if (d_action_ids)     { cudaFree(d_action_ids);     d_action_ids     = nullptr; }
    if (d_action_counts)  { cudaFree(d_action_counts);  d_action_counts  = nullptr; }
    if (d_csr_offsets)    { cudaFree(d_csr_offsets);    d_csr_offsets    = nullptr; }
    if (d_csr_values)     { cudaFree(d_csr_values);     d_csr_values     = nullptr; }
    if (d_piece_slot_mask){ cudaFree(d_piece_slot_mask);d_piece_slot_mask= nullptr; }
    if (stream_a) { cudaStreamDestroy((cudaStream_t)stream_a); stream_a = nullptr; }
    if (stream_b) { cudaStreamDestroy((cudaStream_t)stream_b); stream_b = nullptr; }
    acting_seats_cap = 0;
    belief_cap = 0;
    observer_seats_cap = 0;
    action_ids_cap = 0;
    csr_values_cap = 0;
    mask_cap = 0;
}

void GpuScratch::ensure_acting_seats(int n) {
    if (n <= acting_seats_cap) return;
    if (d_acting_seats) cudaFree(d_acting_seats);
    CUDA_CHECK(cudaMalloc(&d_acting_seats, (size_t)n * sizeof(int8_t)));
    acting_seats_cap = n;
}

void GpuScratch::ensure_belief(int n) {
    if (n <= belief_cap) return;
    if (d_belief) cudaFree(d_belief);
    size_t sz = (size_t)n * BELIEF_STRIDE * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_belief, sz));
    belief_cap = n;
}

void GpuScratch::ensure_observer_seats(int n) {
    if (n <= observer_seats_cap) return;
    if (d_observer_seats) cudaFree(d_observer_seats);
    size_t sz = (size_t)n * NUM_SEATS * sizeof(int8_t);
    CUDA_CHECK(cudaMalloc(&d_observer_seats, sz));
    observer_seats_cap = n;
}

void GpuScratch::ensure_action_ids(int n) {
    if (n <= action_ids_cap) return;
    if (d_action_ids)    cudaFree(d_action_ids);
    if (d_action_counts) cudaFree(d_action_counts);
    if (d_csr_offsets)   cudaFree(d_csr_offsets);
    CUDA_CHECK(cudaMalloc(&d_action_ids,    (size_t)n * MAX_ACTIONS_PER_ENV * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_action_counts, (size_t)n * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_csr_offsets,   (size_t)(n + 1) * sizeof(int32_t)));
    action_ids_cap = n;
}

void GpuScratch::ensure_csr_values(int t) {
    if (t <= csr_values_cap) return;
    if (d_csr_values) cudaFree(d_csr_values);
    CUDA_CHECK(cudaMalloc(&d_csr_values, (size_t)t * sizeof(int32_t)));
    csr_values_cap = t;
}

void GpuScratch::ensure_mask(int n) {
    if (n <= mask_cap) return;
    if (d_piece_slot_mask) cudaFree(d_piece_slot_mask);
    size_t sz = (size_t)n * NUM_PIECES * SLOTS_PER_PIECE * sizeof(bool);
    CUDA_CHECK(cudaMalloc(&d_piece_slot_mask, sz));
    mask_cap = n;
}

void GpuScratch::ensure_streams() {
    if (stream_a && stream_b) return;
    cudaStream_t sa, sb;
    CUDA_CHECK(cudaStreamCreate(&sa));
    CUDA_CHECK(cudaStreamCreate(&sb));
    stream_a = (void*)sa;
    stream_b = (void*)sb;
}

// ---------------------------------------------------------------------------
// legal_action_ids_batch launcher  (dense [N, 512] + counts)
//
// Uses v2 (block-per-env with shmem cache) by default.
// Set JUNQI_CUDA_KERNEL_V1=1 environment variable to force v1 for benchmarking.
// ---------------------------------------------------------------------------
static bool s_use_v1 = []() {
    const char* env = std::getenv("JUNQI_CUDA_KERNEL_V1");
    return env && env[0] == '1';
}();

std::pair<int32_t*, int32_t*> legal_action_ids_batch(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_acting_seats,
    int /*stream_id*/)
{
    const int N = d_state.num_envs;
    GpuScratch& s = GpuScratch::instance();
    s.ensure_action_ids(N);

    // Zero the counts so atomicAdd starts from 0.
    CUDA_CHECK(cudaMemset(s.d_action_counts, 0, (size_t)N * sizeof(int32_t)));

    if (s_use_v1) {
        int total_threads = N * 120;
        int block = 256;
        int grid  = (total_threads + block - 1) / block;
        legal_action_kernel<<<grid, block>>>(
            N,
            d_state.d_piece_seat_arr,
            d_state.d_piece_type_arr,
            d_state.d_alive,
            d_state.d_pos_x,
            d_state.d_pos_y,
            d_state.d_cell_piece_id,
            d_acting_seats,
            s.d_action_ids,
            s.d_action_counts
        );
    } else {
        // v2: one block per env, 128 threads, shared-memory cache.
        legal_action_kernel_v2<<<N, 128>>>(
            N,
            d_state.d_piece_seat_arr,
            d_state.d_piece_type_arr,
            d_state.d_alive,
            d_state.d_pos_x,
            d_state.d_pos_y,
            d_state.d_cell_piece_id,
            d_acting_seats,
            s.d_action_ids,
            s.d_action_counts
        );
    }
    KERNEL_CHECK();

    return {s.d_action_ids, s.d_action_counts};
}

// ---------------------------------------------------------------------------
// CSR helper kernels
// ---------------------------------------------------------------------------

// Simple single-block prefix scan over (N+1) counts → offsets.
// N is small (≤ NUM_ENVS_MAX = 4096); one block with 1024 threads handles
// up to 1024 envs via a Kogge-Stone style scan.  For larger N we fall back
// to two-level scan.  For now we use a plain sequential scan in one block —
// at N=1024 this is ~1μs and fully hidden by the pack/upload cost.
__global__ void csr_prefix_sum_kernel(
    const int32_t* __restrict__ counts,
    int32_t*       __restrict__ offsets,
    int N)
{
    // Single-thread sequential scan.  Good enough for N ≤ 4096.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int32_t acc = 0;
    offsets[0] = 0;
    for (int i = 0; i < N; ++i) {
        acc += counts[i];
        offsets[i + 1] = acc;
    }
}

// Scatter dense [N, MAX_ACTIONS_PER_ENV] action_ids into CSR values[Σcounts].
// One thread per (env, action_slot) pair; active iff slot < counts[env].
__global__ void csr_scatter_kernel(
    const int32_t* __restrict__ dense_ids,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int32_t*       __restrict__ csr_values,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = tid / MAX_ACTIONS_PER_ENV;
    int slot = tid % MAX_ACTIONS_PER_ENV;
    if (env >= N) return;
    int cnt = counts[env];
    if (slot >= cnt) return;
    int base = offsets[env];
    csr_values[base + slot] = dense_ids[env * MAX_ACTIONS_PER_ENV + slot];
}

// ---------------------------------------------------------------------------
// legal_action_ids_batch_csr launcher
// ---------------------------------------------------------------------------
LegalActionCsrResult legal_action_ids_batch_csr(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_acting_seats,
    int /*stream_id*/)
{
    const int N = d_state.num_envs;
    GpuScratch& s = GpuScratch::instance();
    s.ensure_action_ids(N);

    // Step 1: run the dense kernel (populates d_action_ids + d_action_counts).
    CUDA_CHECK(cudaMemset(s.d_action_counts, 0, (size_t)N * sizeof(int32_t)));
    if (s_use_v1) {
        int block1 = 256;
        int grid1  = (N * 120 + block1 - 1) / block1;
        legal_action_kernel<<<grid1, block1>>>(
            N,
            d_state.d_piece_seat_arr,
            d_state.d_piece_type_arr,
            d_state.d_alive,
            d_state.d_pos_x,
            d_state.d_pos_y,
            d_state.d_cell_piece_id,
            d_acting_seats,
            s.d_action_ids,
            s.d_action_counts
        );
    } else {
        legal_action_kernel_v2<<<N, 128>>>(
            N,
            d_state.d_piece_seat_arr,
            d_state.d_piece_type_arr,
            d_state.d_alive,
            d_state.d_pos_x,
            d_state.d_pos_y,
            d_state.d_cell_piece_id,
            d_acting_seats,
            s.d_action_ids,
            s.d_action_counts
        );
    }
    // No sync yet — prefix sum depends on d_action_counts but CUDA kernels
    // on the same stream are serialised, so this is safe.

    // Step 2: prefix sum counts → offsets.
    csr_prefix_sum_kernel<<<1, 1>>>(s.d_action_counts, s.d_csr_offsets, N);

    // Step 3: read total from device (offsets[N]) into host — need sync here.
    int32_t total = 0;
    CUDA_CHECK(cudaMemcpy(&total, s.d_csr_offsets + N, sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // Step 4: ensure csr_values buffer is big enough, then scatter.
    s.ensure_csr_values(total > 0 ? total : 1);
    if (total > 0) {
        int block2 = 256;
        int grid2  = (N * MAX_ACTIONS_PER_ENV + block2 - 1) / block2;
        csr_scatter_kernel<<<grid2, block2>>>(
            s.d_action_ids, s.d_action_counts, s.d_csr_offsets,
            s.d_csr_values, N
        );
    }
    KERNEL_CHECK();

    LegalActionCsrResult r;
    r.d_offsets      = s.d_csr_offsets;
    r.d_values       = s.d_csr_values;
    r.d_counts       = s.d_action_counts;
    r.N              = N;
    r.total_actions  = total;
    return r;
}

// ---------------------------------------------------------------------------
// Per-piece slot legal-action mask kernel.
//
// Output: mask[env, pid, slot] ∈ {0, 1}, flat shape (N, 120, SLOTS_PER_PIECE).
//
// Slot layout (SLOTS_PER_PIECE == 80):
//   slot 0-3   : ortho 1-step                   (ADJACENT_CELLS slots 0-3)
//   slot 4-7   : diag 1-step via camp           (ADJACENT_CELLS slots 4-7)
//   slot 8-55  : non-engineer straight rail     (4 dirs × 12 cells each,
//                 k=1..12 of STRAIGHT_RAIL_RAYS; k=0 skipped iff it matches
//                 an ortho neighbour of src)
//   slot 56-67 : non-engineer curve-rail BFS    (up to 12 cells on curve)
//   slot 68-79 : non-engineer reserved (always 0)
//
//   For engineers slots 8..79 hold BFS-reachable rail cells in traversal
//   order (up to 72 destinations); slot 0..7 still cover the ortho/diag
//   1-steps.  Reaching >72 is impossible (single connected rail component
//   of 73 cells minus the source).
//
// All unused slots are 0.
// ---------------------------------------------------------------------------
__global__ void legal_action_mask_kernel(
    int            num_envs,
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const int16_t* d_cell_piece_id,
    const int8_t*  d_acting_seats,
    bool*          d_mask)   // (N, 120, SLOTS_PER_PIECE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = tid / 120;
    int pid = tid % 120;
    if (env >= num_envs) return;

    bool* out_row = d_mask + (size_t)env * 120 * SLOTS_PER_PIECE
                           + (size_t)pid * SLOTS_PER_PIECE;
    // Zero all slots first (simple, predictable).
    for (int s = 0; s < SLOTS_PER_PIECE; ++s) out_row[s] = false;

    // Gate checks (same as dense kernel).
    int8_t acting_seat = d_acting_seats[env];
    int base120 = env * 120;
    int8_t pseat = d_piece_seat_arr[base120 + pid];
    if (pseat != acting_seat) return;
    if (!d_alive[base120 + pid]) return;
    int8_t ptype = d_piece_type_arr[base120 + pid];
    if (ptype == PT_JUNQI || ptype == PT_DILEI) return;
    int8_t sx = d_pos_x[base120 + pid];
    int8_t sy = d_pos_y[base120 + pid];
    if (sx < 0 || sy < 0) return;
    int16_t src_flat = (int16_t)(sy * 17 + sx);
    if (!ON_BOARD_FLAT[src_flat]) return;
    if (STRONGHOLD_FLAT[src_flat]) return;

    const int16_t* cpid = d_cell_piece_id + env * 289;
    int8_t acting_team = (int8_t)(acting_seat & 1);

    // Lambda: is this dst cell a legal landing for the acting piece?
    auto landable = [&](int16_t nb) -> bool {
        int16_t occ = cpid[nb];
        if (occ < 0) return true;  // empty
        int8_t occ_seat = d_piece_seat_arr[base120 + occ];
        int8_t occ_team = (int8_t)(occ_seat & 1);
        bool is_enemy = (occ_team != acting_team);
        return is_enemy && !CAMP_FLAT[nb];
    };

    // 0-3: orthogonal
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
        if (nb >= 0 && landable(nb)) out_row[k] = true;
    }
    // 4-7: diagonal-via-camp
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + 4 + k];
        if (nb >= 0 && landable(nb)) out_row[4 + k] = true;
    }

    // 8-79: rail long-range
    if (!RAIL_FLAT[src_flat]) return;
    bool is_engineer = (ptype == PT_GONGB);

    int16_t ortho0 = ADJACENT_CELLS[src_flat * 8 + 0];
    int16_t ortho1 = ADJACENT_CELLS[src_flat * 8 + 1];
    int16_t ortho2 = ADJACENT_CELLS[src_flat * 8 + 2];
    int16_t ortho3 = ADJACENT_CELLS[src_flat * 8 + 3];

    if (!is_engineer) {
        // Non-engineer: 4 directions × up to STRAIGHT_RAY_LEN cells each.
        // Slot 8..55 = 4 * 12; slot layout is 8 + dir*12 + k.
        // k=0 suppressed only when it duplicates an ortho neighbour.
        #pragma unroll
        for (int dir = 0; dir < 4; ++dir) {
            int ray_base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
            for (int k = 0; k < STRAIGHT_RAY_LEN; ++k) {
                int16_t nb = STRAIGHT_RAIL_RAYS[ray_base + k];
                if (nb < 0) break;
                bool is_ortho_of_src = (nb == ortho0) || (nb == ortho1)
                                    || (nb == ortho2) || (nb == ortho3);
                int16_t occ = cpid[nb];
                bool is_empty = (occ < 0);
                int slot_idx = 8 + dir * STRAIGHT_RAY_LEN + k;
                if (is_empty) {
                    if (!is_ortho_of_src) out_row[slot_idx] = true;
                } else {
                    if (!is_ortho_of_src) {
                        int8_t occ_seat = d_piece_seat_arr[base120 + occ];
                        int8_t occ_team = (int8_t)(occ_seat & 1);
                        bool is_enemy = (occ_team != acting_team);
                        if (is_enemy && !CAMP_FLAT[nb]) out_row[slot_idx] = true;
                    }
                    break;  // blocked
                }
            }
        }

        // Curve-rail BFS packed into slots 56..67 (up to 12 entries).
        int8_t src_curve = CURVE_RAIL_OF[src_flat];
        if (src_curve > 0) {
            int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
            if (src_ri >= 0) {
                uint64_t v0 = 0, v1 = 0;
                if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
                else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

                int8_t queue[ENG_NUM_RAIL];
                int qhead = 0, qtail = 0;
                queue[qtail++] = src_ri;

                int curve_slot = 56;
                const int curve_slot_end = 68;  // exclusive

                while (qhead < qtail) {
                    int8_t cur_ri = queue[qhead++];
                    for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                        int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                        if (nb_ri < 0) continue;
                        if (nb_ri < 64) {
                            uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                            if (v0 & b) continue;
                            v0 |= b;
                        } else {
                            uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                            if (v1 & b) continue;
                            v1 |= b;
                        }
                        int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                        if (CURVE_RAIL_OF[nb_flat] != src_curve) continue;

                        // Curve-BFS only emits diff-row-diff-col cells; same-axis
                        // overlaps with the straight-rail section above.
                        int nb_x = nb_flat % 17;
                        int nb_y = nb_flat / 17;
                        bool same_axis = (nb_x == sx) || (nb_y == sy);
                        int16_t occ = cpid[nb_flat];
                        bool is_empty = (occ < 0);
                        if (is_empty) {
                            if (!same_axis && curve_slot < curve_slot_end) {
                                out_row[curve_slot++] = true;
                            }
                            queue[qtail++] = nb_ri;
                        } else {
                            if (!same_axis && curve_slot < curve_slot_end) {
                                int8_t occ_seat = d_piece_seat_arr[base120 + occ];
                                int8_t occ_team = (int8_t)(occ_seat & 1);
                                bool is_enemy = (occ_team != acting_team);
                                if (is_enemy && !CAMP_FLAT[nb_flat]) {
                                    out_row[curve_slot++] = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        // Engineer BFS over the rail graph — packed into slots 8..79.
        int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
        if (src_ri < 0) return;
        uint64_t v0 = 0, v1 = 0;
        if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
        else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

        int8_t queue[ENG_NUM_RAIL];
        int qhead = 0, qtail = 0;
        queue[qtail++] = src_ri;

        int slot_cursor = 8;
        const int slot_end = SLOTS_PER_PIECE;  // 80

        while (qhead < qtail) {
            int8_t cur_ri = queue[qhead++];
            for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                if (nb_ri < 0) continue;
                if (nb_ri < 64) {
                    uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                    if (v0 & b) continue;
                    v0 |= b;
                } else {
                    uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                    if (v1 & b) continue;
                    v1 |= b;
                }
                int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                bool is_ortho_of_src = (nb_flat == ortho0) || (nb_flat == ortho1)
                                    || (nb_flat == ortho2) || (nb_flat == ortho3);
                int16_t occ = cpid[nb_flat];
                bool is_empty = (occ < 0);
                if (is_empty) {
                    if (!is_ortho_of_src && slot_cursor < slot_end) {
                        out_row[slot_cursor++] = true;
                    }
                    queue[qtail++] = nb_ri;
                } else {
                    if (!is_ortho_of_src && slot_cursor < slot_end) {
                        int8_t occ_seat = d_piece_seat_arr[base120 + occ];
                        int8_t occ_team = (int8_t)(occ_seat & 1);
                        bool is_enemy = (occ_team != acting_team);
                        if (is_enemy && !CAMP_FLAT[nb_flat]) {
                            out_row[slot_cursor++] = true;
                        }
                    }
                    // Blocked: do NOT enqueue.
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// legal_action_mask_batch launcher
// ---------------------------------------------------------------------------
LegalActionMaskResult legal_action_mask_batch(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_acting_seats,
    int /*stream_id*/)
{
    const int N = d_state.num_envs;
    GpuScratch& s = GpuScratch::instance();
    s.ensure_mask(N);

    // Grid: one thread per (env, piece) pair.
    int total_threads = N * 120;
    int block = 256;
    int grid  = (total_threads + block - 1) / block;

    legal_action_mask_kernel<<<grid, block>>>(
        N,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_cell_piece_id,
        d_acting_seats,
        s.d_piece_slot_mask
    );
    KERNEL_CHECK();

    LegalActionMaskResult r;
    r.d_mask = s.d_piece_slot_mask;
    r.N      = N;
    return r;
}

// ===========================================================================
// legal_mask_canonical_kernel — Dense (N, FLAT_ACTION_SPACE) bool mask in
// canonical frame.  Eliminates D2H/H2D roundtrip: the kernel writes directly
// into device memory that torch reads via __cuda_array_interface__.
//
// Thread mapping: one thread per (env, piece_id).
// Each thread enumerates the legal destinations of its piece (same logic as
// legal_action_mask_kernel) but instead of writing to per-piece slots, it
// computes the flat action id in CANONICAL coordinates:
//   canonical_src * 289 + canonical_dst
// and sets the corresponding bit in the output mask.
//
// Coordinate rotation (world → canonical, per ARCHITECTURE.md §3.3):
//   seat 0 (SOUTH): identity
//   seat 1 (WEST):  (x,y) → (y, 16-x)
//   seat 2 (NORTH): (x,y) → (16-x, 16-y)
//   seat 3 (EAST):  (x,y) → (16-y, x)
// ===========================================================================

__device__ __forceinline__ int16_t rotate_flat(int16_t world_flat, int8_t seat) {
    int x = world_flat % 17;
    int y = world_flat / 17;
    int cx, cy;
    switch (seat) {
        case 0: cx = x;      cy = y;      break;
        case 1: cx = y;      cy = 16 - x; break;
        case 2: cx = 16 - x; cy = 16 - y; break;
        case 3: cx = 16 - y; cy = x;      break;
        default: cx = x; cy = y; break;
    }
    return (int16_t)(cy * 17 + cx);
}

// Compact cell indexing: 289 → 129 on-board cells.
// FLAT_TO_COMPACT_MAP[flat289] → compact index [0,129) or -1 if off-board.
// COMPACT_TO_FLAT_MAP[compact129] → flat289 index.
static __constant__ int16_t FLAT_TO_COMPACT_MAP[289];
static __constant__ int16_t COMPACT_TO_FLAT_MAP[129];
static bool s_compact_maps_uploaded = false;

// Upload compact cell mapping (called once from Python via bindings).
void upload_compact_cell_maps(const int16_t* h_flat_to_compact,
                               const int16_t* h_compact_to_flat) {
    CUDA_CHECK(cudaMemcpyToSymbol(FLAT_TO_COMPACT_MAP, h_flat_to_compact,
                                   289 * sizeof(int16_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(COMPACT_TO_FLAT_MAP, h_compact_to_flat,
                                   129 * sizeof(int16_t)));
    s_compact_maps_uploaded = true;
}

// Device helper: rotate world_flat → canonical_flat, then map to compact index.
__device__ __forceinline__ int16_t rotate_flat_compact(int16_t world_flat, int8_t seat) {
    int16_t can_flat = rotate_flat(world_flat, seat);
    return FLAT_TO_COMPACT_MAP[can_flat];
}

__global__ void legal_mask_canonical_kernel(
    int            num_envs,
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const int16_t* d_cell_piece_id,
    const int8_t*  d_acting_seats,
    bool*          d_mask)   // (N, FLAT_ACTION_SPACE=16641)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int env = tid / 120;
    int pid = tid % 120;
    if (env >= num_envs) return;

    // Gate checks
    int8_t acting_seat = d_acting_seats[env];
    int base120 = env * 120;
    int8_t pseat = d_piece_seat_arr[base120 + pid];
    if (pseat != acting_seat) return;
    if (!d_alive[base120 + pid]) return;
    int8_t ptype = d_piece_type_arr[base120 + pid];
    if (ptype == PT_JUNQI || ptype == PT_DILEI) return;
    int8_t sx = d_pos_x[base120 + pid];
    int8_t sy = d_pos_y[base120 + pid];
    if (sx < 0 || sy < 0) return;
    int16_t src_flat = (int16_t)(sy * 17 + sx);
    if (!ON_BOARD_FLAT[src_flat]) return;
    if (STRONGHOLD_FLAT[src_flat]) return;

    const int16_t* cpid = d_cell_piece_id + env * 289;
    int8_t acting_team = (int8_t)(acting_seat & 1);

    // Rotate src to canonical frame, then map to compact index
    int16_t src_compact = rotate_flat_compact(src_flat, acting_seat);
    if (src_compact < 0) return;  // off-board in canonical (shouldn't happen)
    bool* mask_row = d_mask + (size_t)env * FLAT_ACTION_SPACE;

    // Lambda: check if dst cell is a legal landing
    auto landable = [&](int16_t nb) -> bool {
        int16_t occ = cpid[nb];
        if (occ < 0) return true;
        int8_t occ_seat = d_piece_seat_arr[base120 + occ];
        int8_t occ_team = (int8_t)(occ_seat & 1);
        return (occ_team != acting_team) && !CAMP_FLAT[nb];
    };

    // Lambda: emit one legal (src, dst) pair into the compact canonical mask
    auto emit = [&](int16_t dst_world) {
        int16_t dst_can_flat = rotate_flat(dst_world, acting_seat);
        int16_t dst_compact = FLAT_TO_COMPACT_MAP[dst_can_flat];
        if (dst_compact < 0) return;  // off-board in canonical
        int action_id = (int)src_compact * NUM_ON_BOARD + (int)dst_compact;
        mask_row[action_id] = true;
    };

    // 0-3: orthogonal
    for (int k = 0; k < 4; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + k];
        if (nb >= 0 && landable(nb)) emit(nb);
    }
    // 4-7: diagonal-via-camp
    for (int k = 0; k < 4; ++k) {
        int16_t nb = ADJACENT_CELLS[src_flat * 8 + 4 + k];
        if (nb >= 0 && landable(nb)) emit(nb);
    }

    // Rail long-range
    if (!RAIL_FLAT[src_flat]) return;
    bool is_engineer = (ptype == PT_GONGB);

    if (!is_engineer) {
        // Non-engineer: 4 straight-rail directions
        for (int dir = 0; dir < 4; ++dir) {
            int ray_base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
            for (int k = 0; k < STRAIGHT_RAY_LEN; ++k) {
                int16_t nb = STRAIGHT_RAIL_RAYS[ray_base + k];
                if (nb < 0) break;
                int16_t occ = cpid[nb];
                if (occ < 0) {
                    emit(nb);
                } else {
                    int8_t occ_seat = d_piece_seat_arr[base120 + occ];
                    int8_t occ_team = (int8_t)(occ_seat & 1);
                    if ((occ_team != acting_team) && !CAMP_FLAT[nb]) emit(nb);
                    break;  // blocked
                }
            }
        }

        // Curve-rail BFS
        int8_t src_curve = CURVE_RAIL_OF[src_flat];
        if (src_curve > 0) {
            int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
            if (src_ri >= 0) {
                uint64_t v0 = 0, v1 = 0;
                if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
                else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

                int8_t queue[ENG_NUM_RAIL];
                int qhead = 0, qtail = 0;
                queue[qtail++] = src_ri;

                while (qhead < qtail) {
                    int8_t cur_ri = queue[qhead++];
                    for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                        int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                        if (nb_ri < 0) continue;
                        if (nb_ri < 64) {
                            uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                            if (v0 & b) continue; v0 |= b;
                        } else {
                            uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                            if (v1 & b) continue; v1 |= b;
                        }
                        int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                        if (CURVE_RAIL_OF[nb_flat] != src_curve) continue;
                        int16_t occ = cpid[nb_flat];
                        if (occ < 0) {
                            emit(nb_flat);
                            queue[qtail++] = nb_ri;
                        } else {
                            int8_t occ_seat = d_piece_seat_arr[base120 + occ];
                            int8_t occ_team = (int8_t)(occ_seat & 1);
                            if ((occ_team != acting_team) && !CAMP_FLAT[nb_flat])
                                emit(nb_flat);
                        }
                    }
                }
            }
        }
    } else {
        // Engineer BFS over the full rail graph
        int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
        if (src_ri < 0) return;
        uint64_t v0 = 0, v1 = 0;
        if (src_ri < 64) v0 = (uint64_t)1 << (unsigned)src_ri;
        else             v1 = (uint64_t)1 << (unsigned)(src_ri - 64);

        int8_t queue[ENG_NUM_RAIL];
        int qhead = 0, qtail = 0;
        queue[qtail++] = src_ri;

        while (qhead < qtail) {
            int8_t cur_ri = queue[qhead++];
            for (int kk = 0; kk < ENG_ADJ_WIDTH; ++kk) {
                int8_t nb_ri = ENG_RAIL_ADJ[(int)cur_ri * ENG_ADJ_WIDTH + kk];
                if (nb_ri < 0) continue;
                if (nb_ri < 64) {
                    uint64_t b = (uint64_t)1 << (unsigned)nb_ri;
                    if (v0 & b) continue; v0 |= b;
                } else {
                    uint64_t b = (uint64_t)1 << (unsigned)(nb_ri - 64);
                    if (v1 & b) continue; v1 |= b;
                }
                int16_t nb_flat = ENG_RAIL_CELLS[nb_ri];
                int16_t occ = cpid[nb_flat];
                if (occ < 0) {
                    emit(nb_flat);
                    queue[qtail++] = nb_ri;
                } else {
                    int8_t occ_seat = d_piece_seat_arr[base120 + occ];
                    int8_t occ_team = (int8_t)(occ_seat & 1);
                    if ((occ_team != acting_team) && !CAMP_FLAT[nb_flat])
                        emit(nb_flat);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// legal_mask_canonical_batch launcher
// ---------------------------------------------------------------------------
static bool*   d_canonical_mask      = nullptr;
static int32_t canonical_mask_cap    = 0;

static void ensure_canonical_mask(int N) {
    if (N <= canonical_mask_cap) return;
    if (d_canonical_mask) cudaFree(d_canonical_mask);
    size_t bytes = (size_t)N * FLAT_ACTION_SPACE * sizeof(bool);
    CUDA_CHECK(cudaMalloc(&d_canonical_mask, bytes));
    canonical_mask_cap = N;
}

bool* legal_mask_canonical_batch(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_acting_seats,
    int /*stream_id*/)
{
    const int N = d_state.num_envs;
    ensure_canonical_mask(N);

    // Zero the mask first
    CUDA_CHECK(cudaMemset(d_canonical_mask, 0,
               (size_t)N * FLAT_ACTION_SPACE * sizeof(bool)));

    // Grid: one thread per (env, piece) pair
    int total_threads = N * 120;
    int block = 256;
    int grid  = (total_threads + block - 1) / block;

    legal_mask_canonical_kernel<<<grid, block>>>(
        N,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_cell_piece_id,
        d_acting_seats,
        d_canonical_mask
    );
    KERNEL_CHECK();

    return d_canonical_mask;
}
//
// Output buffers (d_valid / d_event / ...) must be pre-allocated by the caller
// (GpuScratch or Python-side).  Result struct is written entirely on-device;
// callers that want host-side MoveResultBatch must D2H the 6 output arrays.
//
// Kernel launch: one thread per env, blocks of 128 threads.
// ---------------------------------------------------------------------------
void step_batch(
    DeviceGameStateBatch& d_state,
    const int32_t* d_action_ids,
    bool*   d_valid_out,
    int8_t* d_event_out,
    bool*   d_terminated_out,
    int8_t* d_winner_out,
    bool*   d_draw_out,
    bool*   d_flag_captured_out,
    int     stream_id)
{
    const int N = d_state.num_envs;
    if (N <= 0) return;
    constexpr int BLOCK = 128;
    int grid = (N + BLOCK - 1) / BLOCK;
    cudaStream_t stream = 0;
    if (stream_id == 1) {
        auto& s = GpuScratch::instance();
        s.ensure_streams();
        stream = (cudaStream_t)s.stream_b;
    } else if (stream_id == 2) {
        auto& s = GpuScratch::instance();
        s.ensure_streams();
        stream = (cudaStream_t)s.stream_a;
    }

    step_batch_kernel<<<grid, BLOCK, 0, stream>>>(
        N, d_action_ids,
        d_state.d_cell_piece_id,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_move_count_arr,
        d_state.d_active_eat_arr,
        d_state.d_passive_surv_arr,
        d_state.d_death_reason_arr,
        d_state.d_death_step_arr,
        d_state.d_death_loc_flat_arr,
        d_state.d_seat_dead_arr,
        d_state.d_seat_flag_revealed_arr,
        d_state.d_turn,
        d_state.d_zobrist,
        d_state.d_move_counter,
        d_state.d_moves_since_last_combat,
        d_state.d_terminated,
        d_state.d_winner_team,
        d_state.d_draw,
        // CombatMemory v6 — all device-resident, no host traffic.
        d_state.d_cm_direct_lo,
        d_state.d_cm_direct_hi,
        d_state.d_cm_direct_type,
        d_state.d_cm_last_direct_step,
        d_state.d_cm_direct_other_count,
        d_state.d_cm_chain_lo,
        d_state.d_cm_chain_hi,
        d_state.d_cm_chain_type,
        d_state.d_cm_last_chain_step,
        d_state.d_cm_eaten_by_pid_lo,
        d_state.d_cm_eaten_by_pid_hi,
        d_state.d_cm_rank_floor,
        d_state.d_cm_rank_floor_step,
        d_state.d_cm_is_gongb,
        d_state.d_cm_not_gongb,
        d_state.d_cm_attacked_by_known_gongb,
        d_valid_out, d_event_out, d_terminated_out,
        d_winner_out, d_draw_out, d_flag_captured_out);
    KERNEL_CHECK();
}

// ===========================================================================
// Phase 3+4: Device-resident step pipeline (zero-CPU hot path)
//
// step_device() accepts CANONICAL-FRAME actions as a device pointer,
// rotates them to world-frame on GPU, steps, computes per-seat rewards,
// and leaves ALL outputs on device.  The caller (Python via torch tensors)
// never transfers actions, results, or rewards through the host.
// ===========================================================================

// Persistent output buffers (grow-only, never freed until process exit)
static struct StepDeviceBuffers {
    int32_t* d_world_actions = nullptr;  // (cap,) int32 — rotated actions
    bool*    d_valid         = nullptr;  // (cap,)
    int8_t*  d_event         = nullptr;  // (cap,)
    bool*    d_terminated_out= nullptr;  // (cap,) — step output (not d_state.d_terminated)
    int8_t*  d_winner_out    = nullptr;  // (cap,)
    bool*    d_draw_out      = nullptr;  // (cap,)
    bool*    d_flag_captured = nullptr;  // (cap,)
    float*   d_rewards       = nullptr;  // (cap,) float32
    int32_t  cap = 0;

    void ensure(int N) {
        if (N <= cap) return;
        if (d_world_actions) {
            cudaFree(d_world_actions);
            cudaFree(d_valid); cudaFree(d_event);
            cudaFree(d_terminated_out); cudaFree(d_winner_out);
            cudaFree(d_draw_out); cudaFree(d_flag_captured);
            cudaFree(d_rewards);
        }
        CUDA_CHECK(cudaMalloc(&d_world_actions,  (size_t)N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_valid,          (size_t)N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_event,          (size_t)N * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_terminated_out, (size_t)N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_winner_out,     (size_t)N * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_draw_out,       (size_t)N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_flag_captured,  (size_t)N * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_rewards,        (size_t)N * sizeof(float)));
        cap = N;
    }
} g_step_device_bufs;


// rotate_actions_kernel: compact canonical → world frame for each env's acting seat
// Input:  compact canonical action = compact_src * 129 + compact_dst
// Output: world-frame action = src_world_flat * 289 + dst_world_flat
//
// Steps:
//   1. Decompose compact action: compact_src, compact_dst
//   2. Map to canonical flat: COMPACT_TO_FLAT_MAP[compact_src/dst]
//   3. Rotate canonical → world coordinates
//   4. Encode as world flat action: src_world * 289 + dst_world
__global__ void rotate_actions_kernel(
    int N,
    const int32_t* d_canonical_actions,  // (N,) compact canonical-frame action ids
    const int8_t*  d_acting_seats,       // (N,) seat values
    int32_t*       d_world_actions)      // (N,) output world-frame full action ids
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int32_t can_act = d_canonical_actions[i];
    int8_t seat = d_acting_seats[i];

    // Decompose compact action
    int compact_src = can_act / NUM_ON_BOARD;  // [0, 129)
    int compact_dst = can_act % NUM_ON_BOARD;  // [0, 129)

    // Map to canonical flat (17×17 index)
    int src_can_flat = COMPACT_TO_FLAT_MAP[compact_src];
    int dst_can_flat = COMPACT_TO_FLAT_MAP[compact_dst];

    // canonical_to_world for src
    int scx = src_can_flat % 17, scy = src_can_flat / 17;
    int swx, swy;
    switch (seat) {
        case 0: swx = scx;      swy = scy;      break;
        case 1: swx = 16 - scy; swy = scx;      break;
        case 2: swx = 16 - scx; swy = 16 - scy; break;
        case 3: swx = scy;      swy = 16 - scx; break;
        default: swx = scx; swy = scy; break;
    }

    // canonical_to_world for dst
    int dcx = dst_can_flat % 17, dcy = dst_can_flat / 17;
    int dwx, dwy;
    switch (seat) {
        case 0: dwx = dcx;      dwy = dcy;      break;
        case 1: dwx = 16 - dcy; dwy = dcx;      break;
        case 2: dwx = 16 - dcx; dwy = 16 - dcy; break;
        case 3: dwx = dcy;      dwy = 16 - dcx; break;
        default: dwx = dcx; dwy = dcy; break;
    }

    // Output in world-frame FULL flat format (for step_batch_kernel)
    int src_world = swy * 17 + swx;
    int dst_world = dwy * 17 + dwx;
    d_world_actions[i] = src_world * 289 + dst_world;
}


// compute_rewards_kernel: per-seat terminal reward + reward shaping
// Called AFTER step_batch_kernel has written d_event, d_terminated_out, etc.
__global__ void compute_rewards_kernel(
    int N,
    const int8_t*  d_acting_seats,       // (N,) seat that just acted
    const bool*    d_was_active,          // (N,) true if step was applied (valid)
    const bool*    d_new_terminated,      // (N,) terminated AFTER this step
    const int8_t*  d_winner_team,        // (N,) -1/0/1
    const bool*    d_new_draw,           // (N,)
    const int8_t*  d_event,              // (N,) event code (0-4)
    float*         d_rewards)            // (N,) output rewards
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float reward = 0.0f;
    bool was_active = d_was_active[i];     // step was applied (not already terminated)
    bool now_term = d_new_terminated[i];
    bool fired = was_active && now_term;   // newly terminated THIS step

    if (fired) {
        bool is_draw = d_new_draw[i];
        if (!is_draw) {
            int8_t winner = d_winner_team[i];
            if (winner >= 0) {
                int8_t acting_seat = d_acting_seats[i];
                int8_t acting_team = acting_seat & 1;
                reward = (acting_team == winner) ? 1.0f : -1.0f;
            }
        }
    }

    // Pure terminal reward only — no intermediate shaping (mirrors Ataraxos exactly).
    // Terminal: +1.0 (win) / -1.0 (loss) / 0.0 (draw/ongoing).

    d_rewards[i] = reward;
}


// step_device: full device-resident step pipeline
// Input:  d_canonical_actions (device int32*, N) — canonical-frame actions
//         d_acting_seats      (device int8_t*, N) — per-env acting seat
// Output: persistent device buffers in g_step_device_bufs
StepDeviceResult step_device(
    DeviceGameStateBatch& d_state,
    const int32_t* d_canonical_actions,
    const int8_t*  d_acting_seats,
    int stream_id)
{
    const int N = d_state.num_envs;
    g_step_device_bufs.ensure(N);
    auto& b = g_step_device_bufs;

    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    // Step 1: rotate canonical → world actions
    rotate_actions_kernel<<<grid, BLOCK>>>(
        N, d_canonical_actions, d_acting_seats, b.d_world_actions);
    KERNEL_CHECK();

    // Step 2: save pre-step terminated state (for reward computation)
    // We need to know which envs were already terminated before the step.
    // Use d_rewards as temp storage for the bool → float cast (saves an alloc).
    // Actually simpler: pass d_state.d_terminated as "prev_terminated" to the
    // reward kernel.  But step_batch_kernel overwrites d_state.d_terminated.
    // So we need a copy.  Use a lightweight approach: the reward kernel reads
    // d_terminated_out which is the NEW state, and we compare with the
    // terminated field that step_batch_kernel writes into d_state.d_terminated.
    // Wait — step_batch_kernel writes to BOTH d_state.d_terminated AND
    // d_terminated_out.  We need the PREVIOUS d_state.d_terminated.
    // Solution: memcpy d_state.d_terminated to b.d_rewards temporarily.
    // Better: just add a prev_terminated buffer.
    // Simplest: the reward kernel can check `valid_out[i] == false` as proxy
    // for "was already terminated" (step_batch skips terminated envs, setting
    // valid=false).  So: fired = new_terminated AND valid.

    // Step 3: run step_batch_kernel
    step_batch(d_state, b.d_world_actions,
               b.d_valid, b.d_event, b.d_terminated_out,
               b.d_winner_out, b.d_draw_out, b.d_flag_captured,
               stream_id);

    // Step 4: compute rewards
    // d_valid[i] == true means the step was applied (env was active).
    // d_valid[i] == false means env was already terminated (step skipped).
    // "fired this step" = d_valid[i] && d_terminated_out[i]
    // For reward shaping, d_valid[i] && !d_terminated_out[i] means a live non-terminal step.
    compute_rewards_kernel<<<grid, BLOCK>>>(
        N, d_acting_seats,
        b.d_valid,           // used as "was active" flag
        b.d_terminated_out,
        b.d_winner_out,
        b.d_draw_out,
        b.d_event,
        b.d_rewards);
    KERNEL_CHECK();

    StepDeviceResult r;
    r.d_valid         = b.d_valid;
    r.d_event         = b.d_event;
    r.d_terminated    = b.d_terminated_out;
    r.d_winner_team   = b.d_winner_out;
    r.d_draw          = b.d_draw_out;
    r.d_flag_captured = b.d_flag_captured;
    r.d_rewards       = b.d_rewards;
    r.d_world_actions = b.d_world_actions;
    r.N               = N;
    return r;
}
int get_gpu_count() {
    int cnt = 0;
    cudaGetDeviceCount(&cnt);
    return cnt;
}

void set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

// ===========================================================================
// Phase 5: Device-side episode reset via pre-computed setup pool
//
// Ataraxos pattern: pre-generate a pool of valid initial game states on CPU,
// upload once to GPU.  When a terminated env needs reset, copy from pool.
// Zero CPU involvement in the hot path.
//
// Pool template per setup:
//   - piece_type_arr (120 int8) — which piece type at each slot
// World positions (pos_x, pos_y) are a FIXED function of (seat, slot),
// so they are stored in constant memory, not per-setup.
// Similarly piece_seat_arr is always the same (slot/30 = seat).
// cell_piece_id (289 int16) is rebuilt from (pos_x, pos_y) → cell mapping.
// ===========================================================================

// Constant tables for fresh game state reconstruction.
// Position lookup: POS_X_INIT[seat * 30 + slot], POS_Y_INIT[seat * 30 + slot]
// These are the same for every game (only piece_type varies with setup).
static __constant__ int8_t POS_X_INIT[120];  // 4 seats × 30 slots
static __constant__ int8_t POS_Y_INIT[120];
// Seat assignments for all 120 piece ids (piece_id / 30 = seat)
static __constant__ int8_t PIECE_SEAT_INIT[120];
// Camp slot mask: CAMP_SLOT[slot] = true if slot ∈ {6,8,12,16,18}
static __constant__ bool CAMP_SLOT[30];

// (s_reset_tables_uploaded removed — Python side tracks upload state)

// Pre-computed setup pool: device-resident, uploaded once from Python.
// Each pool entry has 120 int8 piece_type values for one 4-seat game.
struct SetupPool {
    int8_t* d_piece_types = nullptr;  // (pool_size, 120) int8
    int pool_size = 0;

    void upload(const int8_t* h_piece_types, int count) {
        pool_size = count;
        size_t sz = (size_t)count * 120 * sizeof(int8_t);
        if (d_piece_types) cudaFree(d_piece_types);
        CUDA_CHECK(cudaMalloc(&d_piece_types, sz));
        CUDA_CHECK(cudaMemcpy(d_piece_types, h_piece_types, sz,
                              cudaMemcpyHostToDevice));
    }

    ~SetupPool() {
        if (d_piece_types) cudaFree(d_piece_types);
    }
};

static SetupPool g_setup_pool;

// Upload constant tables (called once from Python).
void upload_reset_tables(
    const int8_t* h_pos_x,         // (120,) int8
    const int8_t* h_pos_y,         // (120,) int8
    const int8_t* h_piece_seat,    // (120,) int8
    const bool*   h_camp_slot)     // (30,)  bool
{
    CUDA_CHECK(cudaMemcpyToSymbol(POS_X_INIT, h_pos_x, 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(POS_Y_INIT, h_pos_y, 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(PIECE_SEAT_INIT, h_piece_seat, 120 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(CAMP_SLOT, h_camp_slot, 30 * sizeof(bool)));
}

// Upload setup pool (called once from Python).
void upload_setup_pool(const int8_t* h_piece_types, int count) {
    g_setup_pool.upload(h_piece_types, count);
}

// ---------------------------------------------------------------------------
// record_move_history_kernel
//
// One thread per env.  For each env where d_valid[env] is true (step was
// actually applied), decompose the world-frame action into (src_flat,
// dst_flat) and write it into the ring buffer.
// ---------------------------------------------------------------------------
__global__ void record_move_history_kernel(
    int num_envs,
    const int32_t* d_world_actions,  // (N,) src_flat*289+dst_flat
    const bool*    d_valid,          // (N,) from step result
    int16_t*       d_move_history,   // (N, MOVE_HISTORY_LEN, 2) int16
    int32_t*       d_history_write_idx,  // (N,)
    int32_t*       d_history_count)      // (N,)
{
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;
    if (!d_valid[env]) return;  // step was skipped (already terminated)

    int32_t action = d_world_actions[env];
    int16_t src_flat = (int16_t)(action / NUM_CELLS);    // action / 289
    int16_t dst_flat = (int16_t)(action % NUM_CELLS);    // action % 289

    int32_t widx = d_history_write_idx[env];
    int16_t* slot = d_move_history + (env * MOVE_HISTORY_LEN + widx) * 2;
    slot[0] = src_flat;
    slot[1] = dst_flat;

    d_history_write_idx[env] = (widx + 1) % MOVE_HISTORY_LEN;
    int32_t cnt = d_history_count[env];
    if (cnt < MOVE_HISTORY_LEN) d_history_count[env] = cnt + 1;
}

void record_move_history(
    DeviceGameStateBatch& d_state,
    const int32_t* d_world_actions,
    const bool*    d_valid,
    int /*stream_id*/)
{
    const int N = d_state.num_envs;
    if (N <= 0) return;
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    record_move_history_kernel<<<grid, BLOCK>>>(
        N, d_world_actions, d_valid,
        d_state.d_move_history,
        d_state.d_history_write_idx,
        d_state.d_history_count);
    KERNEL_CHECK();
}

// ---------------------------------------------------------------------------
// reset_terminated_envs_kernel
//
// One thread per env.  For each terminated env:
//   1. Pick a random pool entry via hash(env_id, seed)
//   2. Write fresh SoA arrays from the pool template + constant tables
//   3. Clear all game state counters
//
// Handles all 120 pieces per env + 289 cell_piece_id + 4 seat flags + scalars.
// ---------------------------------------------------------------------------
__global__ void reset_terminated_envs_kernel(
    int num_envs,
    // SoA arrays to overwrite
    int8_t*  d_piece_seat_arr,
    int8_t*  d_piece_type_arr,
    bool*    d_alive,
    int8_t*  d_pos_x,
    int8_t*  d_pos_y,
    int8_t*  d_zero_x,
    int8_t*  d_zero_y,
    int16_t* d_cell_piece_id_per_piece,
    int16_t* d_move_count_arr,
    int16_t* d_active_eat_arr,
    int16_t* d_passive_surv_arr,
    int8_t*  d_death_reason_arr,
    int16_t* d_death_step_arr,
    int16_t* d_death_loc_flat_arr,
    int16_t* d_cell_piece_id,
    bool*    d_seat_dead_arr,
    bool*    d_seat_flag_revealed_arr,
    int8_t*  d_turn,
    int64_t* d_zobrist,
    int32_t* d_move_counter,
    int32_t* d_moves_since_last_combat,
    bool*    d_terminated,
    int8_t*  d_winner_team,
    bool*    d_draw,
    // Move history ring buffer (clear on reset)
    int16_t* d_move_history,        // (N, 32, 2)
    int32_t* d_history_write_idx,   // (N,)
    int32_t* d_history_count,       // (N,)
    // CombatMemory v6 — must be wiped to initial state on episode reset, or
    // carry-over from the previous game pollutes the new game's observation.
    uint64_t* d_cm_direct_lo,
    uint64_t* d_cm_direct_hi,
    uint16_t* d_cm_direct_type,
    int16_t*  d_cm_last_direct_step,
    int16_t*  d_cm_direct_other_count,
    uint64_t* d_cm_chain_lo,
    uint64_t* d_cm_chain_hi,
    uint16_t* d_cm_chain_type,
    int16_t*  d_cm_last_chain_step,
    uint64_t* d_cm_eaten_by_pid_lo,
    uint64_t* d_cm_eaten_by_pid_hi,
    int8_t*   d_cm_rank_floor,
    int16_t*  d_cm_rank_floor_step,
    bool*     d_cm_is_gongb,
    bool*     d_cm_not_gongb,
    bool*     d_cm_attacked_by_known_gongb,
    // Pool + seed
    const int8_t* d_pool_piece_types,  // (pool_size, 120)
    int pool_size,
    int64_t seed)
{
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;
    if (!d_terminated[env]) return;  // only reset terminated envs

    // Hash-based pool index selection (cheap device RNG)
    // Use a simple hash combining env index and seed for deterministic-ish
    // but well-distributed selection.
    uint64_t h = (uint64_t)env * 2654435761ULL + (uint64_t)seed;
    h ^= (h >> 17);
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= (h >> 31);
    int pool_idx = (int)(h % (uint64_t)pool_size);

    const int8_t* pool_types = d_pool_piece_types + pool_idx * 120;

    // === Clear cell_piece_id (289 cells) ===
    int16_t* cpid = d_cell_piece_id + env * 289;
    for (int c = 0; c < 289; ++c) {
        cpid[c] = -1;
    }

    // === Write 120 piece slots ===
    int base120 = env * 120;
    for (int pid = 0; pid < 120; ++pid) {
        int slot = pid % 30;
        int8_t ptype = pool_types[pid];

        d_piece_seat_arr[base120 + pid] = PIECE_SEAT_INIT[pid];
        d_piece_type_arr[base120 + pid] = ptype;

        bool is_camp = CAMP_SLOT[slot];
        bool is_alive = !is_camp;  // camp slots have NONE type and are not alive
        d_alive[base120 + pid] = is_alive;

        int8_t px = is_alive ? POS_X_INIT[pid] : (int8_t)-1;
        int8_t py = is_alive ? POS_Y_INIT[pid] : (int8_t)-1;
        d_pos_x[base120 + pid] = px;
        d_pos_y[base120 + pid] = py;
        d_zero_x[base120 + pid] = px;
        d_zero_y[base120 + pid] = py;

        // cell_piece_id_per_piece: flat = py * 17 + px for alive pieces
        d_cell_piece_id_per_piece[base120 + pid] =
            is_alive ? (int16_t)(py * 17 + px) : (int16_t)-1;

        // Place alive piece on the board
        if (is_alive) {
            int flat = py * 17 + px;
            cpid[flat] = (int16_t)pid;
        }

        // Zero statistics
        d_move_count_arr[base120 + pid] = 0;
        d_active_eat_arr[base120 + pid] = 0;
        d_passive_surv_arr[base120 + pid] = 0;
        d_death_reason_arr[base120 + pid] = 0;
        d_death_step_arr[base120 + pid] = 0;
        d_death_loc_flat_arr[base120 + pid] = -1;
    }

    // === Seat flags ===
    int base4 = env * 4;
    for (int s = 0; s < 4; ++s) {
        d_seat_dead_arr[base4 + s] = false;
        d_seat_flag_revealed_arr[base4 + s] = false;
    }

    // === Scalars ===
    d_turn[env] = 0;  // SOUTH starts
    d_zobrist[env] = 0;  // TODO: compute proper initial zobrist if needed
    d_move_counter[env] = 0;
    d_moves_since_last_combat[env] = 0;
    d_terminated[env] = false;
    d_winner_team[env] = -1;
    d_draw[env] = false;

    // === Clear move history ===
    d_history_write_idx[env] = 0;
    d_history_count[env] = 0;
    int16_t* hist = d_move_history + env * MOVE_HISTORY_LEN * 2;
    for (int i = 0; i < MOVE_HISTORY_LEN * 2; ++i) {
        hist[i] = 0;
    }

    // === Clear CombatMemory v6 (4 observers × 120 piece_ids) ===
    // Without this wipe the new episode inherits the previous game's
    // chain bitmaps, is_gongb / not_gongb / dilei_candidate flags, etc. —
    // every CombatMemory channel becomes stale "ghost" data after the
    // first reset, which compounds across episodes and silently corrupts
    // the observation tensor.  Mirrors the constructor cudaMemset values:
    // 0 for bitmaps / counts / bool flags; -1 (0xff) for the three int16
    // *_step sentinels.
    int cm_base = env * 4 * 120;
    int cm_n = 4 * 120;
    for (int j = 0; j < cm_n; ++j) {
        d_cm_direct_lo               [cm_base + j] = (uint64_t)0;
        d_cm_direct_hi               [cm_base + j] = (uint64_t)0;
        d_cm_direct_type             [cm_base + j] = (uint16_t)0;
        d_cm_last_direct_step        [cm_base + j] = (int16_t)-1;
        d_cm_direct_other_count      [cm_base + j] = (int16_t)0;
        d_cm_chain_lo                [cm_base + j] = (uint64_t)0;
        d_cm_chain_hi                [cm_base + j] = (uint64_t)0;
        d_cm_chain_type              [cm_base + j] = (uint16_t)0;
        d_cm_last_chain_step         [cm_base + j] = (int16_t)-1;
        d_cm_eaten_by_pid_lo         [cm_base + j] = (uint64_t)0;
        d_cm_eaten_by_pid_hi         [cm_base + j] = (uint64_t)0;
        d_cm_rank_floor              [cm_base + j] = (int8_t)0;
        d_cm_rank_floor_step         [cm_base + j] = (int16_t)-1;
        d_cm_is_gongb                [cm_base + j] = false;
        d_cm_not_gongb               [cm_base + j] = false;
        d_cm_attacked_by_known_gongb [cm_base + j] = false;
    }
}

// ---------------------------------------------------------------------------
// reset_terminated_envs: launcher
// ---------------------------------------------------------------------------
void reset_terminated_envs(
    DeviceGameStateBatch& d_state,
    int64_t seed)
{
    if (g_setup_pool.pool_size <= 0) {
        // No pool uploaded — cannot reset
        return;
    }
    const int N = d_state.num_envs;
    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;

    reset_terminated_envs_kernel<<<grid, BLOCK>>>(
        N,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_zero_x,
        d_state.d_zero_y,
        d_state.d_cell_piece_id_per_piece,
        d_state.d_move_count_arr,
        d_state.d_active_eat_arr,
        d_state.d_passive_surv_arr,
        d_state.d_death_reason_arr,
        d_state.d_death_step_arr,
        d_state.d_death_loc_flat_arr,
        d_state.d_cell_piece_id,
        d_state.d_seat_dead_arr,
        d_state.d_seat_flag_revealed_arr,
        d_state.d_turn,
        d_state.d_zobrist,
        d_state.d_move_counter,
        d_state.d_moves_since_last_combat,
        d_state.d_terminated,
        d_state.d_winner_team,
        d_state.d_draw,
        d_state.d_move_history,
        d_state.d_history_write_idx,
        d_state.d_history_count,
        // CombatMemory v6 — must wipe on episode reset.
        d_state.d_cm_direct_lo,
        d_state.d_cm_direct_hi,
        d_state.d_cm_direct_type,
        d_state.d_cm_last_direct_step,
        d_state.d_cm_direct_other_count,
        d_state.d_cm_chain_lo,
        d_state.d_cm_chain_hi,
        d_state.d_cm_chain_type,
        d_state.d_cm_last_chain_step,
        d_state.d_cm_eaten_by_pid_lo,
        d_state.d_cm_eaten_by_pid_hi,
        d_state.d_cm_rank_floor,
        d_state.d_cm_rank_floor_step,
        d_state.d_cm_is_gongb,
        d_state.d_cm_not_gongb,
        d_state.d_cm_attacked_by_known_gongb,
        g_setup_pool.d_piece_types,
        g_setup_pool.pool_size,
        seed);
    KERNEL_CHECK();
}

}  // namespace junqi_cuda
