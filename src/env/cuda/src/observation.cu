/*
 * observation.cu
 * DeviceObservationBatch memory management + full 256-channel observation kernel.
 *
 * Observation layout per (env, observer_seat) pair:
 *   Spatial:  [NUM_OBS_CHANNELS=256, BOARD_SIZE=17, BOARD_SIZE=17]  float32
 *   Global:   [NUM_GLOBAL_DIMS=28]                                   float32
 *
 * Channel layout (mirrors junqi_core/observation.py CHANNEL_LAYOUT):
 *   [0..11]    piece_own           — own pieces by tracked type (12)
 *   [12..23]   prob_teammate       — teammate belief probs (12)
 *   [24]       dark_teammate       — teammate presence mask (1)
 *   [25]       piece_left_enemy    — left enemy presence (1)
 *   [26]       piece_right_enemy   — right enemy presence (1)
 *   [27..38]   belief_left_side    — left enemy belief probs (12)
 *   [39..50]   belief_right_side   — right enemy belief probs (12)
 *   [51..53]   dead_flags          — teammate/left/right dead (3, me removed)
 *   [54..57]   flag_revealed       — per-seat flag revealed, observer-sorted (4)
 *   [58..63]   board_static        — camp,stronghold,rail,ninegrid,curve1,curve2 (6)
 *   [64..65]   turn_history        — draw_progress, move_progress (2)
 *   [66..73]   move_bucket         — 4 ours exact + 4 theirs exact (8)
 *   [74..81]   active_eat_bucket   — 4 ours cumul + 4 theirs cumul (8)
 *   [82..89]   passive_surv_bucket — 4 ours cumul + 4 theirs cumul (8)
 *   [90..101]  death_reason        — 3 me + 3 teammate + 3 left + 3 right (12)
 *   [102..103] dead_at_zero        — 1 ours + 1 theirs (2)
 *   [104..223] piece_id            — 120-channel one-hot per piece (120)
 *   [224..255] move_history        — 32-channel src_dst_planes (32)
 *
 * Global layout:
 *   [0..11]  remaining_left_side  — 12 type counts, left enemy
 *   [12..23] remaining_right_side — 12 type counts, right enemy
 *   [24..27] flag_revealed        — per-seat in observer-sorted order
 *
 * Canonical-frame rotation (observer sits at SOUTH in canonical frame):
 *   SOUTH (k=0): (x,y) → (x,y)
 *   WEST  (k=1): (x,y) → (y, 16-x)
 *   NORTH (k=2): (x,y) → (16-x, 16-y)
 *   EAST  (k=3): (x,y) → (16-y, x)
 *
 * Seat team: SOUTH(0)/NORTH(2) = team 0; WEST(1)/EAST(3) = team 1.
 * Seat relationships from observer seat s:
 *   teammate       = (s+2)%4
 *   left_enemy     = (s+1)%4
 *   right_enemy    = (s+3)%4
 */

#include "observation.cuh"
#include "junqi_cuda.h"
#include "common.cuh"
#include "tables.cuh"
#include "combat_memory.cuh"
#include <cuda_runtime.h>

namespace junqi_cuda {

// ---------------------------------------------------------------------------
// Constants for bucket encoding (must match Python observation.py)
// ---------------------------------------------------------------------------
static constexpr int MAX_BUCKET = 3;

// ---------------------------------------------------------------------------
// Rotation helper: world (x,y) → canonical (cx, cy) for given observer seat
// ---------------------------------------------------------------------------
__device__ __forceinline__ void world_to_canonical(
    int x, int y, int obs_seat, int& cx, int& cy)
{
    switch (obs_seat) {
        case 0: cx = x;      cy = y;      break;  // SOUTH: identity
        case 1: cx = y;      cy = 16 - x; break;  // WEST: CCW 90°
        case 2: cx = 16 - x; cy = 16 - y; break;  // NORTH: 180°
        case 3: cx = 16 - y; cy = x;      break;  // EAST: CW 90°
        default: cx = x; cy = y; break;
    }
}

// ---------------------------------------------------------------------------
// Write a 1.0 at canonical (cx, cy) for channel ch in the spatial output.
// spatial is a pointer to the start of this (env, seat) spatial block.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void write_spatial(
    float* spatial, int ch, int cx, int cy)
{
    spatial[ch * (BOARD_SIZE * BOARD_SIZE) + cy * BOARD_SIZE + cx] = 1.0f;
}

// ---------------------------------------------------------------------------
// Fill all 17×17 cells of channel ch with value v.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fill_channel(float* spatial, int ch, float v)
{
    if (v == 0.0f) return;
    float* dst = spatial + ch * (BOARD_SIZE * BOARD_SIZE);
    for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i)
        dst[i] = v;
}

// ---------------------------------------------------------------------------
// DeviceObservationBatch — constructor
// ---------------------------------------------------------------------------
DeviceObservationBatch::DeviceObservationBatch(int n) : num_envs(n) {
    CUDA_CHECK(cudaMalloc(&d_spatial,
        (size_t)n * NUM_SEATS * NUM_OBS_CHANNELS * BOARD_SIZE * BOARD_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global,
        (size_t)n * NUM_SEATS * NUM_GLOBAL_DIMS * sizeof(float)));
}

// ---------------------------------------------------------------------------
// DeviceObservationBatch — destructor
// ---------------------------------------------------------------------------
DeviceObservationBatch::~DeviceObservationBatch() {
    if (d_spatial) { cudaFree(d_spatial); d_spatial = nullptr; }
    if (d_global)  { cudaFree(d_global);  d_global  = nullptr; }
}

// ---------------------------------------------------------------------------
// copy_to_host
// ---------------------------------------------------------------------------
void DeviceObservationBatch::copy_to_host(
    float* h_spatial, float* h_global, int /*stream_id*/) const
{
    CUDA_CHECK(cudaMemcpy(h_spatial, d_spatial,
        (size_t)num_envs * NUM_SEATS * NUM_OBS_CHANNELS * BOARD_SIZE * BOARD_SIZE * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_global, d_global,
        (size_t)num_envs * NUM_SEATS * NUM_GLOBAL_DIMS * sizeof(float),
        cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// observation_kernel — full 256-channel implementation
// ---------------------------------------------------------------------------
__global__ void observation_kernel(
    int num_envs, int num_seats,
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
    const bool*    d_seat_dead_arr,
    const bool*    d_seat_flag_revealed_arr,
    const int8_t*  d_turn,
    const int32_t* d_move_counter,
    const int32_t* d_moves_since_last_combat,
    const float*   d_belief_tensor,
    const int8_t*  d_observer_seats,
    // Move history ring buffer
    const int16_t* d_move_history,       // (N, MOVE_HISTORY_LEN, 2)
    const int32_t* d_history_write_idx,  // (N,)
    const int32_t* d_history_count,      // (N,)
    // CombatMemory v4 (N × 4 observers × 120 pids).  Read-only here.
    const uint64_t* d_cm_direct_lo,
    const uint64_t* d_cm_direct_hi,
    const uint16_t* d_cm_direct_type,
    const int16_t*  d_cm_direct_other_count,
    const uint64_t* d_cm_chain_lo,
    const uint64_t* d_cm_chain_hi,
    const uint16_t* d_cm_chain_type,
    const int8_t*   d_cm_rank_floor,
    const bool*     d_cm_is_gongb,
    const bool*     d_cm_not_gongb,
    const bool*     d_cm_attacked_by_known_gongb,
    // CombatMemory v5 layer-3 — additional state pointers used for recency.
    const int16_t*  d_cm_last_direct_step,
    const int16_t*  d_cm_last_chain_step,
    const int16_t*  d_cm_rank_floor_step,
    // Outputs
    float*         d_obs_spatial,
    float*         d_obs_global,
    int8_t         show_mode)   // 0=BRIGHT,1=DARK,2=HALF_DARK
{
    // ---------------------------------------------------------------------
    // Block-cooperative layout:
    //   * One CUDA block per (env, seat) pair.
    //   * Threads within a block cooperate on the zero-init, piece pass,
    //     board-static pass, and global scalars.
    //   * Block size is fixed to 128 threads.  Piece iteration uses 120 of
    //     them; static / zero-init passes stripe over all 128.
    //
    // This is ~10× faster than the earlier thread-per-(env,seat) kernel
    // because modern GPUs can't amortise the per-(env,seat) 29 k serial
    // stores over the 4 k-thread grid of the old design.
    // ---------------------------------------------------------------------
    const int env  = blockIdx.x;
    const int slot = blockIdx.y;
    if (env >= num_envs || slot >= num_seats) return;

    const int tid = threadIdx.x;
    const int BLK = blockDim.x;

    const int SPATIAL_SZ = NUM_OBS_CHANNELS * BOARD_SIZE * BOARD_SIZE;  // 256*17*17

    // Observer seat + relationships (kept in shared so every thread reads
    // the same values without redundant per-thread gmem loads).
    __shared__ int s_obs_seat, s_teammate, s_left, s_right, s_obs_team;
    __shared__ int s_order_vals[4];
    __shared__ int s_turn_seat, s_move_counter, s_since_combat;

    // Per-env per-piece arrays: we read individual lanes from threads, so
    // no need to stash the whole 120-entry array in shmem — L1/L2 is
    // adequate.  However, we DO stash the belief slice for obs_seat to
    // avoid 120 × 12 = 1440 gmem reads per thread.  bel_base[obs_seat]
    // is 12 × 289 = 3468 floats (13.5 KiB) — fits in shared.  We use
    // threads to cooperatively load it.
    __shared__ float s_bel_obs[NUM_TRACKED_TYPES * NUM_CELLS];  // 3468 floats

    // Per-seat boolean flags (4 entries each) for this env.
    __shared__ bool s_seat_dead[4];
    __shared__ bool s_seat_flagr[4];

    if (tid == 0) {
        s_obs_seat  = (int)d_observer_seats[env * num_seats + slot];
        s_teammate  = (s_obs_seat + 2) & 3;
        s_left      = (s_obs_seat + 1) & 3;
        s_right     = (s_obs_seat + 3) & 3;
        s_obs_team  = s_obs_seat & 1;
        s_order_vals[0] = s_obs_seat;
        s_order_vals[1] = s_teammate;
        s_order_vals[2] = s_left;
        s_order_vals[3] = s_right;
        s_turn_seat     = (int)d_turn[env];
        s_move_counter  = (int)d_move_counter[env];
        s_since_combat  = (int)d_moves_since_last_combat[env];
    }
    if (tid < 4) {
        s_seat_dead[tid]  = d_seat_dead_arr[env * 4 + tid];
        s_seat_flagr[tid] = d_seat_flag_revealed_arr[env * 4 + tid];
    }
    __syncthreads();

    const int obs_seat = s_obs_seat;
    const int teammate_seat = s_teammate;
    const int left_seat     = s_left;
    const int right_seat    = s_right;
    const int obs_team      = s_obs_team;

    // Co-op load of the observer's belief slice into shared.
    // Source base: d_belief_tensor + env * NUM_SEATS*12*289 + obs_seat*12*289
    const float* bel_env_base = d_belief_tensor
        + (size_t)env * NUM_SEATS * NUM_TRACKED_TYPES * NUM_CELLS;
    const float* bel_obs_src = bel_env_base
        + (size_t)obs_seat * NUM_TRACKED_TYPES * NUM_CELLS;
    for (int i = tid; i < NUM_TRACKED_TYPES * NUM_CELLS; i += BLK) {
        s_bel_obs[i] = bel_obs_src[i];
    }
    __syncthreads();

    // Per-env SoA pointers.
    const int8_t*  piece_seat   = d_piece_seat_arr  + env * 120;
    const int8_t*  piece_type   = d_piece_type_arr  + env * 120;
    const bool*    alive        = d_alive            + env * 120;
    const int8_t*  pos_x        = d_pos_x            + env * 120;
    const int8_t*  pos_y        = d_pos_y            + env * 120;
    const int8_t*  zero_x_arr   = d_zero_x           + env * 120;
    const int8_t*  zero_y_arr   = d_zero_y           + env * 120;
    const int16_t* move_cnt     = d_move_count_arr   + env * 120;
    const int16_t* eat_cnt      = d_active_eat_arr   + env * 120;
    const int16_t* surv_cnt     = d_passive_surv_arr + env * 120;
    const int8_t*  death_reason = d_death_reason_arr + env * 120;
    const int16_t* death_loc    = d_death_loc_flat_arr + env * 120;

    // Output pointers.
    float* spatial = d_obs_spatial
        + ((size_t)env * num_seats + slot) * SPATIAL_SZ;
    float* global_out = d_obs_global
        + ((size_t)env * num_seats + slot) * NUM_GLOBAL_DIMS;

    // --------- Co-op zero-init of the spatial + global blocks ----------
    for (int i = tid; i < SPATIAL_SZ; i += BLK) spatial[i] = 0.0f;
    if (tid < NUM_GLOBAL_DIMS) global_out[tid] = 0.0f;
    __syncthreads();

    // Channel base offsets (must match Python CHANNEL_LAYOUT exactly)
    constexpr int CH_OWN             =  0;
    constexpr int CH_TEAMMATE        = 12;
    constexpr int CH_DARK_TEAMMATE   = 24;
    constexpr int CH_LEFT_ENEMY      = 25;
    constexpr int CH_RIGHT_ENEMY     = 26;
    constexpr int CH_BELIEF_LEFT     = 27;
    constexpr int CH_BELIEF_RIGHT    = 39;
    constexpr int CH_DEAD_FLAGS      = 51;  // 3 ch: teammate, left, right (me removed)
    constexpr int CH_FLAG_REVEALED   = 54;
    constexpr int CH_BOARD_STATIC    = 58;
    constexpr int CH_TURN_HISTORY    = 64;  // 2 ch: draw_progress, move_progress
    constexpr int CH_MOVE_BUCKET     = 66;
    constexpr int CH_EAT_BUCKET      = 74;
    constexpr int CH_SURV_BUCKET     = 82;
    constexpr int CH_DEATH_REASON    = 90;  // 12 ch: 3 me + 3 teammate + 3 left + 3 right
    constexpr int CH_DEAD_AT_ZERO    = 102;
    constexpr int CH_PIECE_ID        = 104; // 120 ch: one-hot per piece_id (0..119)
    constexpr int CH_MOVE_HIST       = 224; // 32 ch: src_dst_planes history
    constexpr int PLANE              = BOARD_SIZE * BOARD_SIZE;

    // --------------------------------------------------------------------
    // Per-piece accumulators for global "remaining_left/right"
    // (thread-local, reduced at the end via atomicAdd into global_out).
    // --------------------------------------------------------------------
    float local_rem_left [NUM_TRACKED_TYPES] = {0};
    float local_rem_right[NUM_TRACKED_TYPES] = {0};

    // --------------------------------------------------------------------
    // Pass 1 — live pieces.  Thread p ∈ [0,120) processes piece p.
    // --------------------------------------------------------------------
    if (tid < 120) {
        int p = tid;
        if (alive[p]) {
            int px = (int)pos_x[p];
            int py = (int)pos_y[p];
            if (px >= 0 && py >= 0) {
                int ps = (int)piece_seat[p];
                int pt = (int)piece_type[p];
                int type_idx = (pt >= 2 && pt <= 13) ? (pt - 2) : -1;
                int cx, cy;
                world_to_canonical(px, py, obs_seat, cx, cy);
                int flat_w = py * BOARD_SIZE + px;
                int plane_off = cy * BOARD_SIZE + cx;

                // ch 0..11 : piece_own
                if (ps == obs_seat && type_idx >= 0) {
                    spatial[(CH_OWN + type_idx) * PLANE + plane_off] = 1.0f;
                }

                // ch 12..23 : prob_teammate (observer's belief at teammate cell)
                if (ps == teammate_seat) {
                    for (int ti = 0; ti < NUM_TRACKED_TYPES; ++ti) {
                        float prob = s_bel_obs[ti * NUM_CELLS + flat_w];
                        if (prob > 0.0f)
                            spatial[(CH_TEAMMATE + ti) * PLANE + plane_off] = prob;
                    }
                }

                // ch 24 : dark_teammate (only under DARK show_mode)
                if (ps == teammate_seat && show_mode == 1) {
                    spatial[CH_DARK_TEAMMATE * PLANE + plane_off] = 1.0f;
                }

                // ch 25 : piece_left_enemy
                if (ps == left_seat) {
                    spatial[CH_LEFT_ENEMY * PLANE + plane_off] = 1.0f;
                }

                // ch 26 : piece_right_enemy
                if (ps == right_seat) {
                    spatial[CH_RIGHT_ENEMY * PLANE + plane_off] = 1.0f;
                }

                // ch 27..38 : belief_left_side
                if (ps == left_seat) {
                    for (int ti = 0; ti < NUM_TRACKED_TYPES; ++ti) {
                        float prob = s_bel_obs[ti * NUM_CELLS + flat_w];
                        if (prob > 0.0f)
                            spatial[(CH_BELIEF_LEFT + ti) * PLANE + plane_off] = prob;
                    }
                }

                // ch 39..50 : belief_right_side
                if (ps == right_seat) {
                    for (int ti = 0; ti < NUM_TRACKED_TYPES; ++ti) {
                        float prob = s_bel_obs[ti * NUM_CELLS + flat_w];
                        if (prob > 0.0f)
                            spatial[(CH_BELIEF_RIGHT + ti) * PLANE + plane_off] = prob;
                    }
                }

                // Bucket visibility: always true.
                // Move/eat/survive counts are derived from public MoveResult
                // broadcasts — no hidden information involved.
                int ps_team = ps & 1;
                bool bucket_visible = true;

                // ch 69..76 : move_bucket (exact)
                if (bucket_visible) {
                    int c   = (int)move_cnt[p];
                    int bkt = (c <= 0) ? 0 : (c == 1 ? 1 : (c == 2 ? 2 : 3));
                    int base = (ps_team == obs_team) ? CH_MOVE_BUCKET : (CH_MOVE_BUCKET + 4);
                    spatial[(base + bkt) * PLANE + plane_off] = 1.0f;
                }

                // ch 77..84 : active_eat_bucket (cumulative)
                if (bucket_visible) {
                    int c   = (int)eat_cnt[p];
                    int top = (c <= 0) ? 0 : (c >= 3 ? 3 : c);
                    int base = (ps_team == obs_team) ? CH_EAT_BUCKET : (CH_EAT_BUCKET + 4);
                    for (int k = 0; k <= top; ++k)
                        spatial[(base + k) * PLANE + plane_off] = 1.0f;
                }

                // ch 85..92 : passive_surv_bucket (cumulative)
                if (bucket_visible) {
                    int c   = (int)surv_cnt[p];
                    int top = (c <= 0) ? 0 : (c >= 3 ? 3 : c);
                    int base = (ps_team == obs_team) ? CH_SURV_BUCKET : (CH_SURV_BUCKET + 4);
                    for (int k = 0; k <= top; ++k)
                        spatial[(base + k) * PLANE + plane_off] = 1.0f;
                }

                // ch 107..226 : piece_id one-hot (120 channels).
                // Each live piece p writes 1.0 at channel (CH_PIECE_ID + p).
                // piece_id IS public information: all players can observe which
                // cell a piece occupies and track its identity across moves,
                // even though the piece TYPE is hidden.
                spatial[(CH_PIECE_ID + p) * PLANE + plane_off] = 1.0f;

                // Accumulate remaining_left/right (for global reduce).
                if (ps == left_seat) {
                    for (int ti = 0; ti < NUM_TRACKED_TYPES; ++ti)
                        local_rem_left[ti] += s_bel_obs[ti * NUM_CELLS + flat_w];
                } else if (ps == right_seat) {
                    for (int ti = 0; ti < NUM_TRACKED_TYPES; ++ti)
                        local_rem_right[ti] += s_bel_obs[ti * NUM_CELLS + flat_w];
                }
            }
        }

        // ----------------------------------------------------------------
        // Pass 2 — dead-piece channels (same thread handles its piece).
        // ----------------------------------------------------------------
        if (!alive[p]) {
            int ps = (int)piece_seat[p];
            int ps_team = ps & 1;

            // ch 93..104 : death_reason (3 me + 3 teammate + 3 left + 3 right)
            int16_t dloc = death_loc[p];
            int8_t  dr   = death_reason[p];
            if (dloc >= 0 && dr >= 0 && dr <= 2) {
                int dxw = (int)(dloc % BOARD_SIZE);
                int dyw = (int)(dloc / BOARD_SIZE);
                int cx, cy;
                world_to_canonical(dxw, dyw, obs_seat, cx, cy);
                int base;
                if (ps == obs_seat)            base = CH_DEATH_REASON;       // me:       93..95
                else if (ps == teammate_seat)  base = CH_DEATH_REASON + 3;   // teammate:  96..98
                else if (ps == left_seat)      base = CH_DEATH_REASON + 6;   // left:      99..101
                else                           base = CH_DEATH_REASON + 9;   // right:    102..104
                spatial[(base + dr) * PLANE + cy * BOARD_SIZE + cx] = 1.0f;
            }

            // ch 105..106 : dead_at_zero
            int zx = (int)zero_x_arr[p];
            int zy = (int)zero_y_arr[p];
            if (zx >= 0 && zy >= 0) {
                int cx, cy;
                world_to_canonical(zx, zy, obs_seat, cx, cy);
                int ch = (ps_team == obs_team) ? CH_DEAD_AT_ZERO : (CH_DEAD_AT_ZERO + 1);
                spatial[ch * PLANE + cy * BOARD_SIZE + cx] = 1.0f;
            }
        }
    }

    // --------------------------------------------------------------------
    // Pass 3 — static board planes (co-op over 289 cells).
    // Each thread handles a stripe of cells.  Channel writes are
    // conditional but never race (different cells).
    // --------------------------------------------------------------------
    for (int f = tid; f < NUM_CELLS; f += BLK) {
        int wx = f % BOARD_SIZE;
        int wy = f / BOARD_SIZE;
        int cx, cy;
        world_to_canonical(wx, wy, obs_seat, cx, cy);
        int off = cy * BOARD_SIZE + cx;
        if (CAMP_FLAT[f])       spatial[(CH_BOARD_STATIC + 0) * PLANE + off] = 1.0f;
        if (STRONGHOLD_FLAT[f]) spatial[(CH_BOARD_STATIC + 1) * PLANE + off] = 1.0f;
        if (RAIL_FLAT[f])       spatial[(CH_BOARD_STATIC + 2) * PLANE + off] = 1.0f;
        if (NINE_GRID_FLAT[f])  spatial[(CH_BOARD_STATIC + 3) * PLANE + off] = 1.0f;
        // curve_rail planes (4,5) are always 0 in this kernel.
    }

    // --------------------------------------------------------------------
    // Pass 4 — dead_flags / flag_revealed constant planes (co-op).
    // dead_flags: 3 channels (teammate, left, right — me removed: always 0).
    // flag_revealed: 4 channels (me, teammate, left, right).
    // --------------------------------------------------------------------
    // dead_flags: skip me (order_vals[0]), use order_vals[1..3]
    for (int k = 0; k < 3; ++k) {
        int s = s_order_vals[k + 1];  // teammate=1, left=2, right=3
        if (s_seat_dead[s]) {
            for (int i = tid; i < PLANE; i += BLK)
                spatial[(CH_DEAD_FLAGS + k) * PLANE + i] = 1.0f;
        }
    }
    // flag_revealed: all 4 seats
    for (int k = 0; k < 4; ++k) {
        int s = s_order_vals[k];
        if (s_seat_flagr[s]) {
            for (int i = tid; i < PLANE; i += BLK)
                spatial[(CH_FLAG_REVEALED + k) * PLANE + i] = 1.0f;
        }
    }

    // --------------------------------------------------------------------
    // Pass 5 — turn_history (2 scalar channels: draw_progress, move_progress).
    // is_my_turn removed (always 1 in single-seat hot path).
    // game_phase removed (deterministic discretisation of move_progress).
    // --------------------------------------------------------------------
    __shared__ float s_draw_prog, s_move_prog;
    if (tid == 0) {
        s_draw_prog = (s_since_combat >= 200) ? 1.0f
                    : (float)s_since_combat / 200.0f;
        s_move_prog = (s_move_counter >= 4000) ? 1.0f
                    : (float)s_move_counter / 4000.0f;
    }
    __syncthreads();

    if (s_draw_prog != 0.0f) {
        for (int i = tid; i < PLANE; i += BLK)
            spatial[(CH_TURN_HISTORY + 0) * PLANE + i] = s_draw_prog;
    }
    if (s_move_prog != 0.0f) {
        for (int i = tid; i < PLANE; i += BLK)
            spatial[(CH_TURN_HISTORY + 1) * PLANE + i] = s_move_prog;
    }

    // --------------------------------------------------------------------
    // Pass 6 — reduce per-thread remaining_left/right into global_out
    // via atomicAdd.  Only threads 0..119 have non-zero partials.
    // --------------------------------------------------------------------
    if (tid < 120) {
        for (int ti = 0; ti < NUM_TRACKED_TYPES; ++ti) {
            if (local_rem_left[ti] != 0.0f)
                atomicAdd(&global_out[ti], local_rem_left[ti]);
            if (local_rem_right[ti] != 0.0f)
                atomicAdd(&global_out[12 + ti], local_rem_right[ti]);
        }
    }

    // --------------------------------------------------------------------
    // Pass 7 — global[24..27]: flag_revealed (observer-sorted).
    // --------------------------------------------------------------------
    if (tid < 4) {
        global_out[24 + tid] = s_seat_flagr[s_order_vals[tid]] ? 1.0f : 0.0f;
    }

    // --------------------------------------------------------------------
    // Pass 8 — move_history src_dst_planes (32 channels, co-op).
    //
    // Ring buffer layout: d_move_history[(env * MOVE_HISTORY_LEN + slot) * 2]
    //   slot 0/1 = src_flat / dst_flat (world frame, int16).
    // d_history_write_idx[env] = next write position (mod MOVE_HISTORY_LEN).
    // d_history_count[env] = valid entries (0..MOVE_HISTORY_LEN).
    //
    // Channel delta=0 is the MOST RECENT move; delta=31 is the oldest.
    // src cell gets -1, dst cell gets +1 (matching Ataraxos src_dst_planes).
    // Coordinates are rotated to canonical frame via world_to_canonical.
    // --------------------------------------------------------------------
    __shared__ int s_hist_widx, s_hist_count;
    if (tid == 0) {
        s_hist_widx  = d_history_write_idx[env];
        s_hist_count = d_history_count[env];
    }
    __syncthreads();

    // Use threads 0..31 for 32 history slots.
    if (tid < MOVE_HISTORY_LEN) {
        int delta = tid;  // 0 = most recent
        if (delta < s_hist_count) {
            // Read from ring buffer: most-recent is at (write_idx - 1 - delta + LEN) % LEN
            int ring_idx = (s_hist_widx - 1 - delta + MOVE_HISTORY_LEN) % MOVE_HISTORY_LEN;
            const int16_t* slot = d_move_history
                + ((size_t)env * MOVE_HISTORY_LEN + ring_idx) * 2;
            int src_flat = (int)slot[0];
            int dst_flat = (int)slot[1];

            int src_wx = src_flat % BOARD_SIZE;
            int src_wy = src_flat / BOARD_SIZE;
            int dst_wx = dst_flat % BOARD_SIZE;
            int dst_wy = dst_flat / BOARD_SIZE;

            int scx, scy, dcx, dcy;
            world_to_canonical(src_wx, src_wy, obs_seat, scx, scy);
            world_to_canonical(dst_wx, dst_wy, obs_seat, dcx, dcy);

            int ch = CH_MOVE_HIST + delta;
            spatial[ch * PLANE + scy * BOARD_SIZE + scx] = -1.0f;
            spatial[ch * PLANE + dcy * BOARD_SIZE + dcx] =  1.0f;
        }
    }
    __syncthreads();

    // Pass 9 — CombatMemory v4 (50 channels at indices [256, 306)) +
    // v5 layer-3 (46 channels at [306, 352)).
    // The writer is sequential over the 120 pids; we let thread 0 do it
    // to avoid atomic contention.  All inputs are device-resident; no
    // host traffic.  Pre-condition: spatial slice [256:] has been zeroed
    // by the co-op zero-init pass at the start of the kernel.
    if (tid == 0) {
        const size_t cm_env_off = (size_t)env * 4 * 120;
        cm_write_channels_device(
            env,
            obs_seat,
            /*spatial_base=*/0,
            spatial,
            d_cm_direct_lo                + cm_env_off,
            d_cm_direct_hi                + cm_env_off,
            d_cm_direct_type              + cm_env_off,
            d_cm_direct_other_count       + cm_env_off,
            d_cm_chain_lo                 + cm_env_off,
            d_cm_chain_hi                 + cm_env_off,
            d_cm_chain_type               + cm_env_off,
            d_cm_rank_floor               + cm_env_off,
            d_cm_is_gongb                 + cm_env_off,
            d_cm_not_gongb                + cm_env_off,
            d_cm_attacked_by_known_gongb  + cm_env_off,
            // Layer-3 step trackers (4 × 120 stride per env, same indexing).
            d_cm_last_direct_step         + cm_env_off,
            d_cm_last_chain_step          + cm_env_off,
            d_cm_rank_floor_step          + cm_env_off,
            (int)d_move_counter[env],
            piece_seat,
            piece_type,
            alive,
            pos_x,
            pos_y,
            zero_x_arr,
            zero_y_arr,
            move_cnt
        );
    }
}

// ---------------------------------------------------------------------------
// build_observation_batch launcher
//
// Grid: (num_envs, NUM_SEATS).  Block: 128 threads.
// Each block builds one (env, seat) observation cooperatively.
// ---------------------------------------------------------------------------
void build_observation_batch(
    const DeviceGameStateBatch& d_state,
    const float* d_beliefs,
    const int8_t* d_observer_seats,
    DeviceObservationBatch& d_obs_out,
    int8_t show_mode,
    int /*stream_id*/)
{
    const int n = d_state.num_envs;
    if (n <= 0) return;
    dim3 grid(n, NUM_SEATS);
    dim3 block(128);
    observation_kernel<<<grid, block>>>(
        n, NUM_SEATS,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_zero_x,
        d_state.d_zero_y,
        d_state.d_move_count_arr,
        d_state.d_active_eat_arr,
        d_state.d_passive_surv_arr,
        d_state.d_death_reason_arr,
        d_state.d_death_loc_flat_arr,
        d_state.d_seat_dead_arr,
        d_state.d_seat_flag_revealed_arr,
        d_state.d_turn,
        d_state.d_move_counter,
        d_state.d_moves_since_last_combat,
        d_beliefs,
        d_observer_seats,
        d_state.d_move_history,
        d_state.d_history_write_idx,
        d_state.d_history_count,
        // CombatMemory v4 — device pointers, never copied through host.
        d_state.d_cm_direct_lo,
        d_state.d_cm_direct_hi,
        d_state.d_cm_direct_type,
        d_state.d_cm_direct_other_count,
        d_state.d_cm_chain_lo,
        d_state.d_cm_chain_hi,
        d_state.d_cm_chain_type,
        d_state.d_cm_rank_floor,
        d_state.d_cm_is_gongb,
        d_state.d_cm_not_gongb,
        d_state.d_cm_attacked_by_known_gongb,
        // CombatMemory v5 layer-3 step trackers.
        d_state.d_cm_last_direct_step,
        d_state.d_cm_last_chain_step,
        d_state.d_cm_rank_floor_step,
        d_obs_out.d_spatial,
        d_obs_out.d_global,
        show_mode
    );
    KERNEL_CHECK();
}

// ===========================================================================
// DeviceObservationSingleBatch — single-seat variant (N, 256, 17, 17)
// ===========================================================================
DeviceObservationSingleBatch::DeviceObservationSingleBatch(int n) : num_envs(n) {
    CUDA_CHECK(cudaMalloc(&d_spatial,
        (size_t)n * NUM_OBS_CHANNELS * BOARD_SIZE * BOARD_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_global,
        (size_t)n * NUM_GLOBAL_DIMS * sizeof(float)));
}

DeviceObservationSingleBatch::~DeviceObservationSingleBatch() {
    if (d_spatial) { cudaFree(d_spatial); d_spatial = nullptr; }
    if (d_global)  { cudaFree(d_global);  d_global  = nullptr; }
}

// ---------------------------------------------------------------------------
// build_observation_single_seat — builds obs for ONE seat per env.
//
// The existing observation_kernel already parameterises on `num_seats`:
//   blockIdx.y = slot index (0..num_seats-1)
//   d_observer_seats[env * num_seats + slot] → which seat to observe
//   output offset = (env * num_seats + slot) * SPATIAL_SZ
//
// With num_seats=1 and slot=0, each block builds exactly the acting seat's
// observation and writes to a contiguous (N, 256, 17, 17) buffer.
//
// d_acting_seats is a device pointer (N,) int8 — the per-env seat to observe.
// We copy it into a scratch buffer formatted as d_observer_seats[env * 1 + 0].
// ---------------------------------------------------------------------------
void build_observation_single_seat(
    const DeviceGameStateBatch& d_state,
    const float* d_beliefs,
    const int8_t* d_acting_seats,      // device pointer (N,) int8
    DeviceObservationSingleBatch& d_obs_out,
    int8_t show_mode,
    int /*stream_id*/)
{
    const int n = d_state.num_envs;
    if (n <= 0) return;

    // d_acting_seats is already (N,) int8 — same layout as
    // d_observer_seats[env * 1 + 0] for num_seats=1.
    // We can pass it directly as the observer_seats pointer.

    dim3 grid(n, 1);     // 1 seat per env
    dim3 block(128);
    observation_kernel<<<grid, block>>>(
        n, /*num_seats=*/1,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_zero_x,
        d_state.d_zero_y,
        d_state.d_move_count_arr,
        d_state.d_active_eat_arr,
        d_state.d_passive_surv_arr,
        d_state.d_death_reason_arr,
        d_state.d_death_loc_flat_arr,
        d_state.d_seat_dead_arr,
        d_state.d_seat_flag_revealed_arr,
        d_state.d_turn,
        d_state.d_move_counter,
        d_state.d_moves_since_last_combat,
        d_beliefs,
        d_acting_seats,         // observer_seats: env*1+0 = acting_seats[env]
        d_state.d_move_history,
        d_state.d_history_write_idx,
        d_state.d_history_count,
        // CombatMemory v4 — device pointers, never copied through host.
        d_state.d_cm_direct_lo,
        d_state.d_cm_direct_hi,
        d_state.d_cm_direct_type,
        d_state.d_cm_direct_other_count,
        d_state.d_cm_chain_lo,
        d_state.d_cm_chain_hi,
        d_state.d_cm_chain_type,
        d_state.d_cm_rank_floor,
        d_state.d_cm_is_gongb,
        d_state.d_cm_not_gongb,
        d_state.d_cm_attacked_by_known_gongb,
        // CombatMemory v5 layer-3 step trackers.
        d_state.d_cm_last_direct_step,
        d_state.d_cm_last_chain_step,
        d_state.d_cm_rank_floor_step,
        d_obs_out.d_spatial,
        d_obs_out.d_global,
        show_mode
    );
    KERNEL_CHECK();
}

}  // namespace junqi_cuda

