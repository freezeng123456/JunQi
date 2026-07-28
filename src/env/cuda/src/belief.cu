/*
 * belief.cu
 * GPU-side belief tensor init + update kernels for imperfect-information JunQi.
 *
 * Phase 1 scope:
 *   - belief_init_kernel: HALF_DARK prior seeding on env reset
 *   - belief_update_kernel: deductive rules R1, R4, R5/R7, R6, R9, I5
 *   - Pre-step flag snapshot for detecting new seat_flag_revealed / seat_dead
 *
 * Remaining-inventory maintenance is deferred to Phase 2.
 */

#include "belief.cuh"
#include "junqi_cuda.h"
#include "common.cuh"
#include "tables.cuh"
#include <cuda_runtime.h>

namespace junqi_cuda {

// ---------------------------------------------------------------------------
// Constant-memory tables (definitions)
// ---------------------------------------------------------------------------
__constant__ float   BELIEF_PRIOR_TABLE[30 * 12];
__constant__ int16_t SEAT_STRONGHOLDS[4 * 2];

// ---------------------------------------------------------------------------
// Piece-type constants (must match rules.py / game_state.cu)
// ---------------------------------------------------------------------------
static constexpr int8_t  PT_JUNQI  = 2;
static constexpr int8_t  PT_DILEI  = 3;
// static constexpr int8_t  PT_ZHADAN = 4;  // Phase 2: used for ZHADAN-related rules
// static constexpr int8_t  PT_SILING = 5;  // Phase 2: used for SILING reveal rules
static constexpr int8_t  PT_GONGB  = 13;

// Event constants (must match rules.py)
static constexpr int8_t  EV_MOVE   = 1;
static constexpr int8_t  EV_EAT    = 2;
static constexpr int8_t  EV_BOMB   = 3;
static constexpr int8_t  EV_KILLED = 4;

// ---------------------------------------------------------------------------
// Pre-step flag snapshot buffer (persistent, grow-only)
// Also stores the pre-reset terminated mask for belief re-init.
// ---------------------------------------------------------------------------
static struct PreStepSnapshot {
    bool* d_prev_seat_flag_revealed = nullptr;  // (cap, 4)
    bool* d_prev_seat_dead = nullptr;           // (cap, 4)
    bool* d_pre_reset_terminated = nullptr;     // (cap,) for belief init after reset
    int   cap = 0;

    void ensure(int N) {
        if (N <= cap) return;
        if (d_prev_seat_flag_revealed) {
            cudaFree(d_prev_seat_flag_revealed);
            cudaFree(d_prev_seat_dead);
            cudaFree(d_pre_reset_terminated);
        }
        CUDA_CHECK(cudaMalloc(&d_prev_seat_flag_revealed, (size_t)N * 4 * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_prev_seat_dead,          (size_t)N * 4 * sizeof(bool)));
        CUDA_CHECK(cudaMalloc(&d_pre_reset_terminated,    (size_t)N * sizeof(bool)));
        cap = N;
    }
} g_pre_step;

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Check if a 12-element belief vector is one-hot (max prob > 1 - eps).
__device__ __forceinline__ bool is_one_hot_12(const float* vec) {
    float mx = 0.0f;
    for (int i = 0; i < NUM_TRACKED_TYPES; ++i)
        mx = fmaxf(mx, vec[i]);
    return mx > (1.0f - 1e-6f);
}

// Return the argmax of a 12-element belief vector.
__device__ __forceinline__ int argmax_12(const float* vec) {
    int best = 0;
    float best_val = vec[0];
    for (int i = 1; i < NUM_TRACKED_TYPES; ++i) {
        if (vec[i] > best_val) {
            best_val = vec[i];
            best = i;
        }
    }
    return best;
}

// Write a one-hot vector at belief[type_idx] = 1.0, rest = 0.
__device__ __forceinline__ void write_one_hot(float* vec, int type_idx) {
    for (int i = 0; i < NUM_TRACKED_TYPES; ++i)
        vec[i] = (i == type_idx) ? 1.0f : 0.0f;
}

// Zero out a 12-element belief vector.
__device__ __forceinline__ void write_zero(float* vec) {
    for (int i = 0; i < NUM_TRACKED_TYPES; ++i)
        vec[i] = 0.0f;
}

// Copy a 12-element belief vector: dst = src.
__device__ __forceinline__ void copy_12(float* dst, const float* src) {
    for (int i = 0; i < NUM_TRACKED_TYPES; ++i)
        dst[i] = src[i];
}

// Check if a 12-element vector is all-zero (no piece at this cell).
__device__ __forceinline__ bool is_zero_12(const float* vec) {
    float s = 0.0f;
    for (int i = 0; i < NUM_TRACKED_TYPES; ++i)
        s += vec[i];
    return s < 1e-8f;
}

// Pointer to belief[env, obs_seat, :, cell_flat].
// Layout: [env * 4*12*289 + seat * 12*289 + type * 289 + cell].
// Returns pointer to the 12 type slots at stride 289.
// Usage: ptr[type_idx * 289] to read/write probability for type_idx.
//
// NOTE: belief layout is [env, seat, type, cell], so the 12 type values
// for one cell are NOT contiguous — they are strided by 289.
// For efficiency, we use a small local buffer of 12 floats.
__device__ __forceinline__ float* bel_ptr(float* d_belief, int env, int seat, int cell) {
    return d_belief + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS)
                    + (size_t)seat * (NUM_TRACKED_TYPES * NUM_CELLS)
                    + cell;
    // Access type t: bel_ptr[t * NUM_CELLS]
}

// Read belief vector at (env, seat, cell) into local buf[12].
__device__ __forceinline__ void read_belief(
    const float* d_belief, int env, int seat, int cell, float* buf)
{
    const float* base = d_belief
        + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS)
        + (size_t)seat * (NUM_TRACKED_TYPES * NUM_CELLS)
        + cell;
    for (int t = 0; t < NUM_TRACKED_TYPES; ++t)
        buf[t] = base[t * NUM_CELLS];
}

// Write belief vector from local buf[12] to (env, seat, cell).
__device__ __forceinline__ void write_belief(
    float* d_belief, int env, int seat, int cell, const float* buf)
{
    float* base = d_belief
        + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS)
        + (size_t)seat * (NUM_TRACKED_TYPES * NUM_CELLS)
        + cell;
    for (int t = 0; t < NUM_TRACKED_TYPES; ++t)
        base[t * NUM_CELLS] = buf[t];
}

// Zero out belief at (env, seat, cell).
__device__ __forceinline__ void zero_belief(
    float* d_belief, int env, int seat, int cell)
{
    float* base = d_belief
        + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS)
        + (size_t)seat * (NUM_TRACKED_TYPES * NUM_CELLS)
        + cell;
    for (int t = 0; t < NUM_TRACKED_TYPES; ++t)
        base[t * NUM_CELLS] = 0.0f;
}

// ===========================================================================
// belief_init_kernel
// ===========================================================================

__global__ void belief_init_kernel(
    int num_envs,
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const bool*    d_just_reset,      // (N,) true for envs that need init
    float*         d_belief)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;

    // Only initialise envs that were just reset.
    // For the very first call (all envs), pass an all-true mask.
    if (!d_just_reset[env]) return;

    int tid = threadIdx.x;

    // --- Zero the entire belief block for this env ---
    // Size = 4 * 12 * 289 = 13,872 floats
    float* bel_env = d_belief + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS);
    int total = 4 * NUM_TRACKED_TYPES * NUM_CELLS;
    for (int i = tid; i < total; i += blockDim.x)
        bel_env[i] = 0.0f;
    __syncthreads();

    // --- Populate beliefs from piece arrays ---
    // Each thread handles a subset of the 120 pieces × 4 observers = 480 items.
    // We loop over the 120 pieces; for each, we update all 4 observer seats.
    const int8_t* seat_arr = d_piece_seat_arr + env * 120;
    const int8_t* type_arr = d_piece_type_arr + env * 120;
    const bool*   alive_arr = d_alive + env * 120;
    const int8_t* px_arr = d_pos_x + env * 120;
    const int8_t* py_arr = d_pos_y + env * 120;

    for (int pid = tid; pid < 120; pid += blockDim.x) {
        if (!alive_arr[pid]) continue;

        int8_t piece_seat = seat_arr[pid];
        int8_t piece_type = type_arr[pid];
        int px = (int)px_arr[pid];
        int py = (int)py_arr[pid];
        if (px < 0 || py < 0) continue;
        int cell_flat = py * BOARD_SIZE + px;

        // type_idx in [0,11] for TRACKED_TYPES (JUNQI=0 .. GONGB=11)
        int type_idx = (piece_type >= 2 && piece_type <= 13)
                       ? (piece_type - 2) : -1;
        if (type_idx < 0) continue;

        // Seat-local slot index: pid % 30
        int slot = pid % 30;

        // For each observer seat
        for (int obs = 0; obs < 4; ++obs) {
            // DARK mode: only observer's OWN pieces get one-hot.
            // Teammate + enemy pieces all get per-slot prior.
            if (obs == piece_seat) {
                // Own piece: one-hot (known type)
                float* base = d_belief
                    + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS)
                    + (size_t)obs * (NUM_TRACKED_TYPES * NUM_CELLS)
                    + cell_flat;
                base[type_idx * NUM_CELLS] = 1.0f;
            } else {
                // Teammate or enemy: use per-slot prior from constant table
                const float* prior = BELIEF_PRIOR_TABLE + slot * NUM_TRACKED_TYPES;
                float* base = d_belief
                    + (size_t)env * (4 * NUM_TRACKED_TYPES * NUM_CELLS)
                    + (size_t)obs * (NUM_TRACKED_TYPES * NUM_CELLS)
                    + cell_flat;
                for (int t = 0; t < NUM_TRACKED_TYPES; ++t)
                    base[t * NUM_CELLS] = prior[t];
            }
        }
    }
}

// ===========================================================================
// belief_update_kernel
// ===========================================================================

__global__ void belief_update_kernel(
    int num_envs,
    const int8_t*  d_piece_seat_arr,
    const int8_t*  d_piece_type_arr,
    const bool*    d_alive,
    const int8_t*  d_pos_x,
    const int8_t*  d_pos_y,
    const int16_t* d_cell_piece_id,
    const bool*    d_seat_dead_arr,
    const bool*    d_seat_flag_revealed,
    const bool*    d_terminated,
    const int8_t*  d_event,
    const bool*    d_flag_captured,
    const int32_t* d_world_actions,
    const bool*    d_prev_seat_flag_revealed,
    const bool*    d_prev_seat_dead,
    float*         d_belief)
{
    int env = blockIdx.x;
    if (env >= num_envs) return;
    if (d_terminated[env]) return;

    int tid = threadIdx.x;

    // --- Parse step result ---
    int8_t  event_val = d_event[env];
    if (event_val == 0) return;  // invalid / no-op step

    bool    flag_cap  = d_flag_captured[env];
    int32_t action    = d_world_actions[env];
    int     src_flat  = action / 289;
    int     dst_flat  = action % 289;

    // --- Post-step state pointers ---
    const int8_t*  seat_arr = d_piece_seat_arr + env * 120;
    const int8_t*  type_arr = d_piece_type_arr + env * 120;
    const bool*    alive_e  = d_alive + env * 120;
    const int16_t* cpid_e   = d_cell_piece_id + env * 289;
    const bool*    seat_dead = d_seat_dead_arr + env * 4;
    const bool*    seat_flagr = d_seat_flag_revealed + env * 4;
    const bool*    prev_flagr = d_prev_seat_flag_revealed + env * 4;
    const bool*    prev_dead  = d_prev_seat_dead + env * 4;

    // Identify attacker (src) and defender (dst) seats.
    // After step:
    //   MOVE: attacker moved to dst, src is empty.
    //   EAT: attacker moved to dst (replacing defender who died).
    //   KILLED: attacker died, src is empty, dst unchanged.
    //   BOMB: both died, both cells empty.
    //
    // For MOVE/EAT: attacker pid is now at dst.
    // For KILLED: defender pid is still at dst.
    // For BOMB: both cells are empty.

    int16_t dst_pid_post = cpid_e[dst_flat];

    // Determine defender seat (before step, the piece at dst).
    // For EAT/KILLED/BOMB, the defender existed.  For MOVE, no defender.
    // We can infer defender_seat from the event:
    //   EAT: dst_pid_post is the attacker; defender is dead.
    //        We need defender_seat to handle R4/R6/R9.
    //        After EAT, the dead defender's seat is in death records.
    //   KILLED: dst_pid_post is still the defender (survived).
    //   BOMB: both dead.
    //
    // For R4/R6 we need defender_seat.  We can get it from:
    //   - flag_captured: only if defender is JUNQI -> we know defender_seat
    //     from the seat that just became dead or had flag revealed.
    //   - R6: dst is a stronghold -> we need to know whose stronghold it is.
    //     This is inferable from position geometry (which seat's rectangle
    //     contains dst_flat).

    // Helper: determine which seat "owns" a flat cell (by initial geometry).
    // This is needed for stronghold ownership in R6.
    // Seat ownership is fixed by the board topology:
    //   SOUTH: y ∈ [11,16], x ∈ [6,10]  (rows 11-16, cols 6-10)
    //   WEST:  y ∈ [6,10],  x ∈ [0,5]   (rows 6-10, cols 0-5)
    //   NORTH: y ∈ [0,5],   x ∈ [6,10]  (rows 0-5, cols 6-10)
    //   EAST:  y ∈ [6,10],  x ∈ [11,16] (rows 6-10, cols 11-16)
    // But for belief update we mostly need this for strongholds,
    // which we precomputed in SEAT_STRONGHOLDS.

    // --- Thread 0 computes shared data, broadcasts via shared mem ---
    __shared__ int8_t  s_event;
    __shared__ bool    s_flag_cap;
    __shared__ int     s_src_flat, s_dst_flat;
    __shared__ bool    s_new_dead[4];   // newly dead this step

    if (tid == 0) {
        s_event = event_val;
        s_flag_cap = flag_cap;
        s_src_flat = src_flat;
        s_dst_flat = dst_flat;

        // Detect newly dead seats
        for (int s = 0; s < 4; ++s) {
            s_new_dead[s]  = seat_dead[s] && !prev_dead[s];
        }

        // NOTE: defender seat (s_dst_seat) resolution is deferred to Phase 2.
        // For Phase 1, the cases that need it use alternative approaches:
        //   R4: flag_cap → use s_new_dead to find which seat died
        //   R6: stronghold → use precomputed SEAT_STRONGHOLDS
        //   R5/R7: read belief at dst before clearing, type check
    }
    __syncthreads();

    // For R4 (flag_captured), identify the surrendered seat from newly-dead.
    // In most cases exactly one seat dies when its flag is captured.
    __shared__ int8_t s_surrendered_seat;
    if (tid == 0) {
        s_surrendered_seat = -1;
        if (s_flag_cap) {
            for (int s = 0; s < 4; ++s) {
                if (s_new_dead[s]) {
                    s_surrendered_seat = s;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // For R6 (stronghold deduction), determine if dst is a stronghold
    // and if so, which seat owns it and where the other stronghold is.
    __shared__ bool   s_dst_is_stronghold;
    __shared__ int8_t s_stronghold_owner;   // seat that owns the stronghold at dst
    __shared__ int16_t s_other_stronghold;  // flat cell of the other stronghold
    if (tid == 0) {
        s_dst_is_stronghold = STRONGHOLD_FLAT[s_dst_flat];
        s_stronghold_owner = -1;
        s_other_stronghold = -1;
        if (s_dst_is_stronghold) {
            // Find which seat owns dst as a stronghold
            for (int s = 0; s < 4; ++s) {
                if (SEAT_STRONGHOLDS[s * 2 + 0] == s_dst_flat) {
                    s_stronghold_owner = s;
                    s_other_stronghold = SEAT_STRONGHOLDS[s * 2 + 1];
                    break;
                }
                if (SEAT_STRONGHOLDS[s * 2 + 1] == s_dst_flat) {
                    s_stronghold_owner = s;
                    s_other_stronghold = SEAT_STRONGHOLDS[s * 2 + 0];
                    break;
                }
            }
        }
    }
    __syncthreads();

    // =====================================================================
    // Per-observer belief update (4 observers, cooperatively with threads)
    // Each thread handles one observer seat.  With 128 threads:
    //   tid 0-3:   observer seats 0-3
    //   tid 4-127: idle for this phase (could be used for vectorization)
    //
    // For simplicity in Phase 1, we use 4 threads (one per observer).
    // This is not bandwidth-bound — the kernel is compute-light.
    // =====================================================================
    if (tid < 4) {
        int obs = tid;  // observer seat
        int obs_team = obs & 1;

        // Temporary buffers for belief at src and dst before modification
        float bel_src[12], bel_dst[12];

        // Read pre-update beliefs at src and dst cells
        read_belief(d_belief, env, obs, s_src_flat, bel_src);
        read_belief(d_belief, env, obs, s_dst_flat, bel_dst);

        // =================================================================
        // R5/R7: Engineer signature — EAT + defender belief is one-hot DILEI
        // → attacker belief becomes one-hot GONGB.
        //
        // Must run BEFORE R1 migration (which overwrites beliefs at src/dst).
        // In HALF_DARK: observer may know defender is DILEI if it's own/teammate.
        // =================================================================
        if (s_event == EV_EAT) {
            if (is_one_hot_12(bel_dst)) {
                int def_type = argmax_12(bel_dst);
                if (def_type == (PT_DILEI - 2)) {  // DILEI is type_idx 1
                    // Attacker must be GONGB (only engineer eats mines)
                    if (!is_one_hot_12(bel_src)) {
                        write_one_hot(bel_src, PT_GONGB - 2);  // GONGB is type_idx 11
                    }
                }
            }
        }

        // =================================================================
        // R1: Piece migration — move belief vectors based on event.
        // =================================================================
        switch (s_event) {
        case EV_MOVE:
            // src → dst; src becomes empty
            write_belief(d_belief, env, obs, s_dst_flat, bel_src);
            zero_belief(d_belief, env, obs, s_src_flat);
            break;

        case EV_EAT:
            // Attacker wins: attacker belief migrates from src to dst.
            // Defender belief at dst is destroyed (piece is dead).
            write_belief(d_belief, env, obs, s_dst_flat, bel_src);
            zero_belief(d_belief, env, obs, s_src_flat);
            break;

        case EV_KILLED:
            // Attacker dies: src belief destroyed.  Dst unchanged.
            zero_belief(d_belief, env, obs, s_src_flat);
            break;

        case EV_BOMB:
            // Both die: both cells cleared.
            zero_belief(d_belief, env, obs, s_src_flat);
            zero_belief(d_belief, env, obs, s_dst_flat);
            break;
        }

        // =================================================================
        // R6: Stronghold deduction — EAT at a stronghold that is NOT flag capture
        // → the OTHER stronghold of the same seat must be the JUNQI (flag).
        // =================================================================
        if (s_event == EV_EAT && s_dst_is_stronghold && !s_flag_cap
            && s_stronghold_owner >= 0 && s_other_stronghold >= 0)
        {
            // Only update if the other stronghold has a piece with unknown belief.
            float bel_other[12];
            read_belief(d_belief, env, obs, s_other_stronghold, bel_other);
            if (!is_zero_12(bel_other) && !is_one_hot_12(bel_other)) {
                // The other stronghold must hold the flag
                write_one_hot(bel_other, PT_JUNQI - 2);  // JUNQI is type_idx 0
                write_belief(d_belief, env, obs, s_other_stronghold, bel_other);
            }
        }

        // =================================================================
        // R4: Flag captured → the surrendered seat's pieces are all removed.
        // Clear all beliefs for pieces belonging to that seat.
        // =================================================================
        if (s_flag_cap && s_surrendered_seat >= 0) {
            // Scan all cells and zero beliefs for pieces of the surrendered seat.
            // After surrender, those pieces are already dead (alive=false).
            // The I5 sweep below will also catch this, but we do it explicitly
            // for clarity and to handle edge cases.
            for (int c = 0; c < NUM_CELLS; ++c) {
                int16_t pid_c = cpid_e[c];
                if (pid_c < 0) {
                    // Cell empty — zero any stale belief
                    zero_belief(d_belief, env, obs, c);
                }
            }
        }

        // =================================================================
        // R9: Seat death — clear beliefs for all pieces of newly-dead seats.
        // =================================================================
        for (int s = 0; s < 4; ++s) {
            if (s_new_dead[s]) {
                // All pieces of seat s are dead — zero their belief cells.
                // Already handled by I5 below, but this is explicit.
            }
        }

        // =================================================================
        // I5: Consistency sweep — ensure belief matches live board state.
        // For every cell:
        //   - If cell is empty (cpid < 0) but belief is non-zero → clear.
        //   - If cell has a live piece but belief is zero → seed fallback.
        // =================================================================
        for (int c = 0; c < NUM_CELLS; ++c) {
            int16_t pid_c = cpid_e[c];
            if (pid_c < 0 || !alive_e[pid_c]) {
                // Cell is empty — zero any stale belief.
                // (Except we need to avoid zeroing cells that were just written
                // by R1 above.  Since R1 already wrote the correct values,
                // and empty cells should have zero belief, this is safe.)
                float buf[12];
                read_belief(d_belief, env, obs, c, buf);
                if (!is_zero_12(buf)) {
                    zero_belief(d_belief, env, obs, c);
                }
            } else {
                // Cell has a live piece — check if belief is missing.
                float buf[12];
                read_belief(d_belief, env, obs, c, buf);
                if (is_zero_12(buf)) {
                    // Missing belief entry — provide a fallback.
                    int8_t ps = seat_arr[pid_c];
                    int8_t pt = type_arr[pid_c];
                    int p_team = ps & 1;
                    int type_idx = (pt >= 2 && pt <= 13) ? (pt - 2) : -1;

                    if (obs_team == p_team && type_idx >= 0) {
                        // Own/teammate: one-hot
                        float oh[12];
                        write_one_hot(oh, type_idx);
                        write_belief(d_belief, env, obs, c, oh);
                    } else {
                        // Enemy: uniform fallback (conservative)
                        float uf[12];
                        float val = 1.0f / (float)NUM_TRACKED_TYPES;
                        for (int t = 0; t < NUM_TRACKED_TYPES; ++t)
                            uf[t] = val;
                        write_belief(d_belief, env, obs, c, uf);
                    }
                }
            }
        }
    }
    // tid >= 4: no work in Phase 1 (could be used for vectorized cell sweeps later)
}

// ===========================================================================
// snapshot_pre_step_flags_kernel
// ===========================================================================
__global__ void snapshot_pre_step_flags_kernel(
    int num_envs,
    const bool* d_seat_flag_revealed,  // (N, 4)
    const bool* d_seat_dead_arr,       // (N, 4)
    bool* d_prev_flag_revealed,        // (N, 4) output
    bool* d_prev_dead)                 // (N, 4) output
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_envs * 4;
    if (idx < total) {
        d_prev_flag_revealed[idx] = d_seat_flag_revealed[idx];
        d_prev_dead[idx] = d_seat_dead_arr[idx];
    }
}

// ===========================================================================
// Host-callable launchers
// ===========================================================================

void upload_belief_prior_table(const float* h_table) {
    CUDA_CHECK(cudaMemcpyToSymbol(
        BELIEF_PRIOR_TABLE, h_table, 30 * 12 * sizeof(float)));
}

void upload_seat_strongholds(const int16_t* h_strongholds) {
    CUDA_CHECK(cudaMemcpyToSymbol(
        SEAT_STRONGHOLDS, h_strongholds, 4 * 2 * sizeof(int16_t)));
}

void init_beliefs_for_reset_envs(
    const DeviceGameStateBatch& d_state,
    float* d_belief,
    int /*stream_id*/)
{
    int N = d_state.num_envs;
    if (N <= 0) return;

    g_pre_step.ensure(N);

    // Use the pre-reset terminated snapshot as the "just reset" mask.
    const bool* mask = g_pre_step.d_pre_reset_terminated;

    dim3 grid(N);
    dim3 block(128);
    belief_init_kernel<<<grid, block>>>(
        N, d_state.d_piece_seat_arr, d_state.d_piece_type_arr,
        d_state.d_alive, d_state.d_pos_x, d_state.d_pos_y,
        mask, d_belief);
    KERNEL_CHECK();
}

void init_all_beliefs(
    const DeviceGameStateBatch& d_state,
    float* d_belief,
    int /*stream_id*/)
{
    int N = d_state.num_envs;
    if (N <= 0) return;

    // Allocate a temporary all-true mask so every env is initialised.
    bool* d_mask = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mask, (size_t)N * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_mask, 1, (size_t)N * sizeof(bool)));

    dim3 grid(N);
    dim3 block(128);
    belief_init_kernel<<<grid, block>>>(
        N, d_state.d_piece_seat_arr, d_state.d_piece_type_arr,
        d_state.d_alive, d_state.d_pos_x, d_state.d_pos_y,
        d_mask, d_belief);
    KERNEL_CHECK();

    cudaFree(d_mask);
}

void snapshot_pre_step_flags(
    const DeviceGameStateBatch& d_state,
    int /*stream_id*/)
{
    int N = d_state.num_envs;
    if (N <= 0) return;

    g_pre_step.ensure(N);

    int total = N * 4;
    int block = 256;
    int grid = (total + block - 1) / block;
    snapshot_pre_step_flags_kernel<<<grid, block>>>(
        N,
        d_state.d_seat_flag_revealed_arr,
        d_state.d_seat_dead_arr,
        g_pre_step.d_prev_seat_flag_revealed,
        g_pre_step.d_prev_seat_dead);
    KERNEL_CHECK();

    // Also snapshot d_terminated for belief re-init after reset.
    CUDA_CHECK(cudaMemcpy(
        g_pre_step.d_pre_reset_terminated,
        d_state.d_terminated,
        (size_t)N * sizeof(bool),
        cudaMemcpyDeviceToDevice));
}

void update_beliefs_after_step(
    const DeviceGameStateBatch& d_state,
    const int8_t* d_event,
    const bool* d_flag_captured,
    const int32_t* d_world_actions,
    float* d_belief,
    int /*stream_id*/)
{
    int N = d_state.num_envs;
    if (N <= 0) return;

    g_pre_step.ensure(N);

    dim3 grid(N);
    dim3 block(128);
    belief_update_kernel<<<grid, block>>>(
        N,
        d_state.d_piece_seat_arr,
        d_state.d_piece_type_arr,
        d_state.d_alive,
        d_state.d_pos_x,
        d_state.d_pos_y,
        d_state.d_cell_piece_id,
        d_state.d_seat_dead_arr,
        d_state.d_seat_flag_revealed_arr,
        d_state.d_terminated,
        d_event,
        d_flag_captured,
        d_world_actions,
        g_pre_step.d_prev_seat_flag_revealed,
        g_pre_step.d_prev_seat_dead,
        d_belief);
    KERNEL_CHECK();
}

}  // namespace junqi_cuda
