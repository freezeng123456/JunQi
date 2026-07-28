/*
 * combat_memory.cu — GPU-side CombatMemory v4.
 *
 * Strict no-host-interaction rule:
 *   * cm_apply_event_dev runs INSIDE step_batch_kernel (no host calls).
 *   * cm_move_requires_gongb_dev does its rail BFS purely in registers /
 *     constant memory (rail tables already device-resident).
 *   * cm_write_channels_device runs INSIDE observation_kernel.
 *
 * The only host↔device traffic is the one-time DeviceGameStateBatch ctor
 * (cudaMalloc + cudaMemset) and the parity-test copy_from_host /
 * copy_to_host plumbing (NOT used during training).
 */

#include "combat_memory.cuh"
#include "tables.cuh"
#include "common.cuh"

namespace junqi_cuda {

// ---------------------------------------------------------------------------
// PieceType.value tags (mirror common.cuh / game_state.cu PT_* if any).
// ---------------------------------------------------------------------------
static constexpr int8_t CM_PT_DILEI = 3;
static constexpr int8_t CM_PT_GONGB = 13;

// Indexing helper: state[(obs, pid)] → flat (4×120).
__device__ inline int cm_idx(int obs, int pid) {
    return obs * CM_NUM_PIDS_DEV + pid;
}

// ===========================================================================
// cm_move_requires_gongb_dev
//
// Returns true iff src→dst is a path that ONLY a GONGB can take given
// the current board occupancy (cpid_env).  Mirrors
// junqi_core/move_gen.py::move_requires_gongb.
//
// Logic:
//   1) 1-step orthogonal: any piece → return false.
//   2) 1-step diagonal with one endpoint a camp: any piece → false.
//   3) Otherwise must be a rail move; off-rail endpoints → false.
//   4) Same row/col on rail and straight-rail-clear → false (non-engineer can do it).
//   5) Different row+col on rail and curve_rail_of(src)==curve_rail_of(dst)
//      AND curve-rail clear → false (non-engineer can do it via curve).
//   6) Otherwise the only way is engineer BFS — return true if BFS reaches.
// ===========================================================================

__device__ static bool straight_rail_clear_dev(
    const int16_t* cpid_env,
    int16_t src_flat, int16_t dst_flat)
{
    // Walk the precomputed STRAIGHT_RAIL_RAYS from src; require src/dst on
    // same row OR same column; at each ray cell, the cell must be empty
    // until we hit dst_flat (which itself is allowed to be occupied —
    // that's the destination cell).
    int sx = src_flat % 17, sy = src_flat / 17;
    int dx = dst_flat % 17, dy = dst_flat / 17;
    if (sx != dx && sy != dy) return false;

    // Scan the 4 rays from src; the one containing dst is the legal route.
    for (int dir = 0; dir < 4; ++dir) {
        int base = src_flat * 4 * STRAIGHT_RAY_LEN + dir * STRAIGHT_RAY_LEN;
        for (int k = 0; k < STRAIGHT_RAY_LEN; ++k) {
            int16_t cell = STRAIGHT_RAIL_RAYS[base + k];
            if (cell < 0) break;
            if (cell == dst_flat) {
                // All cells before this k must be empty (cells emitted
                // by the ray earlier).  We re-scan to verify.
                for (int j = 0; j < k; ++j) {
                    int16_t c = STRAIGHT_RAIL_RAYS[base + j];
                    if (cpid_env[c] >= 0) return false;
                }
                return true;
            }
            // Continue ray; if cell is occupied AND it's not dst, ray
            // is blocked.  We don't break here because dst may still
            // appear later — but the legacy semantic in move_gen.py is
            // "intermediate cells must be empty".  So if any cell
            // before dst is occupied, this ray is blocked: no need to
            // continue, but dst could be reached via a DIFFERENT ray
            // (different dir).  So just break inner loop.
            if (cpid_env[cell] >= 0) break;
        }
    }
    return false;
}

__device__ static bool curve_rail_clear_dev(
    const int16_t* cpid_env,
    int16_t src_flat, int16_t dst_flat)
{
    // Same curve id and connected via curve-rail (rail-graph BFS staying
    // on the same curve).  Mirrors move_gen.py::_curve_rail_clear.
    int8_t src_curve = CURVE_RAIL_OF[src_flat];
    int8_t dst_curve = CURVE_RAIL_OF[dst_flat];
    if (src_curve <= 0 || src_curve != dst_curve) return false;

    int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
    int8_t dst_ri = ENG_RAIL_TO_IDX[dst_flat];
    if (src_ri < 0 || dst_ri < 0) return false;

    // Two-word visited bitmask covers ENG_NUM_RAIL=73 entries.
    uint64_t v0 = 0, v1 = 0;
    auto mark = [&](int8_t ri) {
        if (ri < 64) { v0 |= (uint64_t)1 << ri; }
        else         { v1 |= (uint64_t)1 << (ri - 64); }
    };
    auto seen = [&](int8_t ri) -> bool {
        return (ri < 64) ? ((v0 >> ri) & 1ULL) : ((v1 >> (ri - 64)) & 1ULL);
    };

    int8_t queue[ENG_NUM_RAIL];
    int qhead = 0, qtail = 0;
    queue[qtail++] = src_ri;
    mark(src_ri);

    while (qhead < qtail) {
        int8_t cur = queue[qhead++];
        int16_t cur_flat = ENG_RAIL_CELLS[cur];
        bool is_src = (cur_flat == src_flat);
        // Block if intermediate cell is occupied (but src/dst themselves OK).
        if (!is_src && cur_flat != dst_flat && cpid_env[cur_flat] >= 0) continue;
        if (cur_flat == dst_flat) return true;
        for (int k = 0; k < ENG_ADJ_WIDTH; ++k) {
            int8_t nb = ENG_RAIL_ADJ[cur * ENG_ADJ_WIDTH + k];
            if (nb < 0) continue;
            if (CURVE_RAIL_OF[ENG_RAIL_CELLS[nb]] != src_curve) continue;
            if (seen(nb)) continue;
            mark(nb);
            queue[qtail++] = nb;
        }
    }
    return false;
}

__device__ static bool engineer_can_reach_dev(
    const int16_t* cpid_env,
    int16_t src_flat, int16_t dst_flat)
{
    int8_t src_ri = ENG_RAIL_TO_IDX[src_flat];
    int8_t dst_ri = ENG_RAIL_TO_IDX[dst_flat];
    if (src_ri < 0 || dst_ri < 0) return false;

    uint64_t v0 = 0, v1 = 0;
    auto mark = [&](int8_t ri) {
        if (ri < 64) v0 |= (uint64_t)1 << ri;
        else         v1 |= (uint64_t)1 << (ri - 64);
    };
    auto seen = [&](int8_t ri) -> bool {
        return (ri < 64) ? ((v0 >> ri) & 1ULL) : ((v1 >> (ri - 64)) & 1ULL);
    };

    int8_t queue[ENG_NUM_RAIL];
    int qhead = 0, qtail = 0;
    queue[qtail++] = src_ri;
    mark(src_ri);

    while (qhead < qtail) {
        int8_t cur = queue[qhead++];
        int16_t cur_flat = ENG_RAIL_CELLS[cur];
        bool is_src = (cur_flat == src_flat);
        if (!is_src && cur_flat != dst_flat && cpid_env[cur_flat] >= 0) continue;
        if (cur_flat == dst_flat) return true;
        for (int k = 0; k < ENG_ADJ_WIDTH; ++k) {
            int8_t nb = ENG_RAIL_ADJ[cur * ENG_ADJ_WIDTH + k];
            if (nb < 0) continue;
            if (seen(nb)) continue;
            mark(nb);
            queue[qtail++] = nb;
        }
    }
    return false;
}

__device__ bool cm_move_requires_gongb_dev(
    const int16_t* cpid_env,
    int16_t src_flat,
    int16_t dst_flat)
{
    if (src_flat == dst_flat) return false;
    int sx = src_flat % 17, sy = src_flat / 17;
    int dx = dst_flat % 17, dy = dst_flat / 17;
    int adx = sx - dx; if (adx < 0) adx = -adx;
    int ady = sy - dy; if (ady < 0) ady = -ady;

    // 1-step orthogonal: any piece can do this.
    if (adx + ady == 1) return false;
    // 1-step diagonal via camp: any piece.
    if (adx == 1 && ady == 1 && (CAMP_FLAT[src_flat] || CAMP_FLAT[dst_flat]))
        return false;

    // Beyond here must be a rail move.
    if (!RAIL_FLAT[src_flat] || !RAIL_FLAT[dst_flat]) return false;

    // Same row/col straight rail: non-engineer can do iff straight clear.
    if (sx == dx || sy == dy) {
        if (straight_rail_clear_dev(cpid_env, src_flat, dst_flat)) return false;
        return engineer_can_reach_dev(cpid_env, src_flat, dst_flat);
    }
    // Different row+col on rails: curve or BFS.
    if (curve_rail_clear_dev(cpid_env, src_flat, dst_flat)) return false;
    return engineer_can_reach_dev(cpid_env, src_flat, dst_flat);
}

// ===========================================================================
// cm_apply_event_dev
// ===========================================================================

__device__ void cm_apply_event_dev(
    CMEnvPtrs cm,
    bool   is_eat,
    int    attacker_pid,
    int    defender_pid,
    int    attacker_seat,
    int    defender_seat,
    int8_t attacker_type_val,
    int8_t defender_type_val,
    int16_t /*defender_pos_flat*/,    // present for parity with CPU API; unused on GPU
    int    death_step)
{
    // K = survivor, V = corpse.
    int K, V, K_seat, V_seat;
    int8_t V_type;
    if (is_eat) {
        K = attacker_pid; V = defender_pid;
        K_seat = attacker_seat; V_seat = defender_seat;
        V_type = defender_type_val;
    } else {
        K = defender_pid; V = attacker_pid;
        K_seat = defender_seat; V_seat = attacker_seat;
        V_type = attacker_type_val;
    }
    if (K < 0 || V < 0) return;

    int8_t v_idx_t = cm_type_to_idx(V_type);   // -1 if not tracked

    // ---------- KILLED preflight: was the dead attacker (V) a known GONGB? ----------
    // For each observer:
    //   v_known_gongb[obs] = is_gongb[obs][V] OR (V_seat == obs AND attacker_type == GONGB)
    // (Only KILLED needs this; EAT does not require attacked_by_known_gongb update.)
    bool v_known[CM_NUM_OBSERVERS_DEV];
    if (!is_eat) {
        for (int obs = 0; obs < CM_NUM_OBSERVERS_DEV; ++obs) {
            bool flag = cm.is_gongb[cm_idx(obs, V)];
            if (V_seat == obs && attacker_type_val == CM_PT_GONGB) flag = true;
            v_known[obs] = flag;
            if (flag) {
                cm.attacked_by_known_gongb[cm_idx(obs, K)] = true;
            }
        }
    }

    // ---------- Chain propagation (all observers) ----------
    // chain[K] |= direct_my[V] | chain[V] | bit(V)
    // chain_type[K] |= direct_type[V] | chain_type[V]
    // last_chain_step[K] = death_step
    uint64_t v_bit_lo = 0, v_bit_hi = 0;
    if (V < 64) v_bit_lo = (uint64_t)1 << V;
    else        v_bit_hi = (uint64_t)1 << (V - 64);

    for (int obs = 0; obs < CM_NUM_OBSERVERS_DEV; ++obs) {
        int kIdx = cm_idx(obs, K);
        int vIdx = cm_idx(obs, V);
        cm.chain_lo[kIdx]   |= cm.direct_lo[vIdx];
        cm.chain_lo[kIdx]   |= cm.chain_lo[vIdx];
        cm.chain_lo[kIdx]   |= v_bit_lo;
        cm.chain_hi[kIdx]   |= cm.direct_hi[vIdx];
        cm.chain_hi[kIdx]   |= cm.chain_hi[vIdx];
        cm.chain_hi[kIdx]   |= v_bit_hi;
        cm.chain_type[kIdx] |= cm.direct_type[vIdx];
        cm.chain_type[kIdx] |= cm.chain_type[vIdx];
        cm.last_chain_step[kIdx] = (int16_t)death_step;

        // §8 chain rank-floor: K's floor ≥ V.floor + 1 (capped at 9).
        int8_t v_floor = cm.rank_floor[vIdx];
        if (v_floor > 0) {
            int8_t prop = (int8_t)(v_floor + 1);
            if (prop > 9) prop = 9;
            int8_t cur = cm.rank_floor[kIdx];
            if (prop > cur) {
                cm.rank_floor[kIdx]      = prop;
                cm.rank_floor_step[kIdx] = (int16_t)death_step;
            }
            // K can't be GONGB (chain-killed an ordinary).
            cm.not_gongb[kIdx] = true;
        }
    }

    // ---------- Per-observer dispatch on V visibility ----------
    for (int obs = 0; obs < CM_NUM_OBSERVERS_DEV; ++obs) {
        int kIdx = cm_idx(obs, K);
        bool v_visible = (V_seat == obs);

        if (!v_visible) {
            // DARK rule: type unknown to obs, count only.
            int16_t cur = cm.direct_other_count[kIdx];
            if (cur < 32767) cm.direct_other_count[kIdx] = (int16_t)(cur + 1);
            cm.last_direct_step[kIdx] = (int16_t)death_step;
            continue;
        }

        // Direct bitmap + type mask.
        cm.direct_lo[kIdx] |= v_bit_lo;
        cm.direct_hi[kIdx] |= v_bit_hi;
        if (v_idx_t >= 0) {
            uint16_t bit = (uint16_t)1 << v_idx_t;
            cm.direct_type[kIdx] |= bit;
            cm.chain_type[kIdx]  |= bit;
        }
        cm.last_direct_step[kIdx] = (int16_t)death_step;

        // GONGB / not-GONGB flags.
        if (is_eat) {
            if (V_type == CM_PT_DILEI) {
                cm.is_gongb[kIdx] = true;       // A1
            } else {
                cm.not_gongb[kIdx] = true;      // B1
            }
            // EAT direct floor lift on K.
            if (cm_rank_of_type(V_type) > 0) {
                int8_t prop = cm_next_floor_after_eat(V_type);
                if (prop > cm.rank_floor[kIdx]) {
                    cm.rank_floor[kIdx]      = prop;
                    cm.rank_floor_step[kIdx] = (int16_t)death_step;
                }
            }
        } else {
            // KILLED: V is the dead attacker; K (defender) survived.
            if (cm_rank_of_type(V_type) > 0) {
                int8_t prop = cm_next_floor_after_eat(V_type);
                if (prop > cm.rank_floor[kIdx]) {
                    cm.rank_floor[kIdx]      = prop;
                    cm.rank_floor_step[kIdx] = (int16_t)death_step;
                }
            }
            // Note: we do NOT flag not_gongb here — defender could be DILEI.
        }
    }
}

// ===========================================================================
// Channel writer (called from observation_kernel; canonical-frame rotation
// applied per observer; binary 0/1 only).
//
// Pre-condition: caller has already memset the v4 channel slice to 0.
// ===========================================================================

// Local rotation copy (observation.cu has its own; we redeclare to keep
// this TU self-contained without including the .cu).
__device__ static __forceinline__ void cm_rotate(
    int x, int y, int obs_seat, int& cx, int& cy)
{
    switch (obs_seat) {
        case 0: cx = x;      cy = y;      break;
        case 1: cx = y;      cy = 16 - x; break;
        case 2: cx = 16 - x; cy = 16 - y; break;
        case 3: cx = 16 - y; cy = x;      break;
        default: cx = x; cy = y; break;
    }
}

// CombatMemory v4 channel offsets relative to spatial_base.  Must match
// junqi_core/observation.py CHANNEL_LAYOUT exactly.
//   [256..267]  cm_kill_mine_type      (12)
//   [268..270]  cm_kill_mine_ge        (3)
//   [271..273]  cm_kill_other_ge       (3)
//   [274..285]  cm_chain_type          (12)
//   [286..288]  cm_chain_ge            (3)
//   [289..297]  cm_floor_ge            (9)
//   [298]       cm_is_gongb
//   [299]       cm_not_gongb
//   [300]       cm_dilei_candidate
//   [301..303]  cm_my_kill_count_ge    (3)
//   [304]       cm_my_is_gongb
//   [305]       cm_my_dilei_candidate
static constexpr int CH_CM_BASE                  = 256;
static constexpr int CH_CM_KILL_MINE_TYPE        = CH_CM_BASE +   0;
static constexpr int CH_CM_KILL_MINE_GE          = CH_CM_BASE +  12;
static constexpr int CH_CM_KILL_OTHER_GE         = CH_CM_BASE +  15;
static constexpr int CH_CM_CHAIN_TYPE            = CH_CM_BASE +  18;
static constexpr int CH_CM_CHAIN_GE              = CH_CM_BASE +  30;
static constexpr int CH_CM_FLOOR_GE              = CH_CM_BASE +  33;
static constexpr int CH_CM_IS_GONGB              = CH_CM_BASE +  42;
static constexpr int CH_CM_NOT_GONGB             = CH_CM_BASE +  43;
static constexpr int CH_CM_DILEI_CANDIDATE       = CH_CM_BASE +  44;
static constexpr int CH_CM_MY_KILL_COUNT_GE      = CH_CM_BASE +  45;
static constexpr int CH_CM_MY_IS_GONGB           = CH_CM_BASE +  48;
static constexpr int CH_CM_MY_DILEI_CANDIDATE    = CH_CM_BASE +  49;
// Layer 3 (ADR-129 v5) — per-pid identity tail.
static constexpr int CH_CM_KILL_MINE_COUNT       = CH_CM_BASE +  50;  // 12 channels
static constexpr int CH_CM_KILL_MINE_SLOT        = CH_CM_BASE +  62;  // 30 channels
static constexpr int CH_CM_RECENCY               = CH_CM_BASE +  92;  //  4 channels
// Total v5 tail: 96 channels (0..95).

static constexpr int CM_PLANE = 17 * 17;   // 289

__device__ inline void cm_set_plane(float* spatial, int ch, int cx, int cy) {
    spatial[ch * CM_PLANE + cy * 17 + cx] = 1.0f;
}

__device__ inline int cm_popcount64(uint64_t x) {
    return __popcll(x);
}

__device__ void cm_write_channels_device(
    int   /*env_id*/,
    int   observer_seat,
    int   /*spatial_base*/,    // not used directly; spatial pointer already env+slot offset
    float* spatial,
    const uint64_t* cm_direct_lo_env,
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
    const int16_t*  cm_last_direct_step_env,
    const int16_t*  cm_last_chain_step_env,
    const int16_t*  cm_rank_floor_step_env,
    int             move_counter,
    const int8_t*   piece_seat_env,
    const int8_t*   piece_type_env,
    const bool*     alive_env,
    const int8_t*   pos_x_env,
    const int8_t*   pos_y_env,
    const int8_t*   zero_x_env,
    const int8_t*   zero_y_env,
    const int16_t*  move_count_env)
{
    const int obs_team = observer_seat & 1;
    // Two opponents (left = +1 mod 4, right = +3 mod 4).
    const int opp_left  = (observer_seat + 1) & 3;
    const int opp_right = (observer_seat + 3) & 3;

    // Pre-compute per-tracked-type pid masks for observer's 30 pids.  These
    // are needed by Layer 3.1 (kill_mine_count) and only depend on the
    // observer's piece_type_arr restricted to its 30-pid range.  Built
    // once before the per-enemy loop so we don't redo the type scan.
    uint64_t obs_type_mask_lo[CM_NUM_TRACKED_TYPES] = {0};
    uint64_t obs_type_mask_hi[CM_NUM_TRACKED_TYPES] = {0};
    {
        int obs_pid_lo = observer_seat * 30;
        for (int s = 0; s < 30; ++s) {
            int gp = obs_pid_lo + s;
            int8_t pt = piece_type_env[gp];
            int8_t ti = cm_type_to_idx(pt);
            if (ti < 0 || ti >= CM_NUM_TRACKED_TYPES) continue;
            if (gp < 64) obs_type_mask_lo[ti] |= (uint64_t)1 << gp;
            else         obs_type_mask_hi[ti] |= (uint64_t)1 << (gp - 64);
        }
    }

    for (int pid = 0; pid < CM_NUM_PIDS_DEV; ++pid) {
        if (!alive_env[pid]) continue;
        int8_t pseat = piece_seat_env[pid];
        if (pseat < 0) continue;
        int8_t pteam = pseat & 1;
        int wx = pos_x_env[pid], wy = pos_y_env[pid];
        if (wx < 0 || wy < 0) continue;
        int cx, cy;
        cm_rotate(wx, wy, observer_seat, cx, cy);

        // ============== Layer 1 (only for enemies of observer) ==============
        if (pteam != obs_team) {
            int idx = cm_idx(observer_seat, pid);
            uint64_t dlo = cm_direct_lo_env[idx];
            uint64_t dhi = cm_direct_hi_env[idx];
            uint16_t dtm = cm_direct_type_env[idx];
            int16_t  oth = cm_direct_other_count_env[idx];
            uint64_t clo = cm_chain_lo_env[idx];
            uint64_t chi = cm_chain_hi_env[idx];
            uint16_t ctm = cm_chain_type_env[idx];
            int8_t   rf  = cm_rank_floor_env[idx];
            bool     isg = cm_is_gongb_env[idx];
            bool     nog = cm_not_gongb_env[idx];
            bool     atk = cm_attacked_by_known_gongb_env[idx];

            int mine_count = cm_popcount64(dlo) + cm_popcount64(dhi);
            // kill_mine_ge[k]  ⇔  mine_count ≥ k+1
            if (mine_count >= 1) cm_set_plane(spatial, CH_CM_KILL_MINE_GE + 0, cx, cy);
            if (mine_count >= 2) cm_set_plane(spatial, CH_CM_KILL_MINE_GE + 1, cx, cy);
            if (mine_count >= 3) cm_set_plane(spatial, CH_CM_KILL_MINE_GE + 2, cx, cy);
            // kill_mine_type multi-hot.
            #pragma unroll
            for (int t = 0; t < CM_NUM_TRACKED_TYPES; ++t) {
                if ((dtm >> t) & 1U) {
                    cm_set_plane(spatial, CH_CM_KILL_MINE_TYPE + t, cx, cy);
                }
            }
            if (oth >= 1) cm_set_plane(spatial, CH_CM_KILL_OTHER_GE + 0, cx, cy);
            if (oth >= 2) cm_set_plane(spatial, CH_CM_KILL_OTHER_GE + 1, cx, cy);
            if (oth >= 3) cm_set_plane(spatial, CH_CM_KILL_OTHER_GE + 2, cx, cy);

            int chain_count = cm_popcount64(clo) + cm_popcount64(chi);
            if (chain_count >= 1) cm_set_plane(spatial, CH_CM_CHAIN_GE + 0, cx, cy);
            if (chain_count >= 2) cm_set_plane(spatial, CH_CM_CHAIN_GE + 1, cx, cy);
            if (chain_count >= 3) cm_set_plane(spatial, CH_CM_CHAIN_GE + 2, cx, cy);
            #pragma unroll
            for (int t = 0; t < CM_NUM_TRACKED_TYPES; ++t) {
                if ((ctm >> t) & 1U) {
                    cm_set_plane(spatial, CH_CM_CHAIN_TYPE + t, cx, cy);
                }
            }
            // floor_ge cumulative (≥1 ... ≥9).
            #pragma unroll
            for (int g = 0; g < CM_NUM_RANK_FLOORS_DEV; ++g) {
                if (rf >= (g + 1)) cm_set_plane(spatial, CH_CM_FLOOR_GE + g, cx, cy);
            }
            if (isg) cm_set_plane(spatial, CH_CM_IS_GONGB,  cx, cy);
            if (nog) cm_set_plane(spatial, CH_CM_NOT_GONGB, cx, cy);

            // dilei_candidate runtime check.
            int16_t zf = (int16_t)zero_y_env[pid] * 17 + (int16_t)zero_x_env[pid];
            if (zero_x_env[pid] >= 0 && zero_y_env[pid] >= 0
                && cm_in_back_two_rows(zf, pseat)
                && move_count_env[pid] == 0
                && !atk)
            {
                cm_set_plane(spatial, CH_CM_DILEI_CANDIDATE, cx, cy);
            }

            // ============== Layer 3 (v5) — per-pid identity tail ==============
            // Layer 3.1: cm_kill_mine_count[12] — per-type popcount of
            // direct_lo/hi restricted to victims of that tracked type.
            // Normalized to [0, 1] by /3.
            for (int t = 0; t < CM_NUM_TRACKED_TYPES; ++t) {
                int cnt = cm_popcount64(dlo & obs_type_mask_lo[t])
                        + cm_popcount64(dhi & obs_type_mask_hi[t]);
                if (cnt > 0) {
                    float v = (cnt >= 3) ? 1.0f : (float)cnt / 3.0f;
                    spatial[(CH_CM_KILL_MINE_COUNT + t) * CM_PLANE
                            + cy * 17 + cx] = v;
                }
            }

            // Layer 3.2: cm_kill_mine_slot[30] — slot-i bit lit if
            // direct-pid bitmap covers observer's pid (obs_pid_lo + i).
            {
                int obs_pid_lo = observer_seat * 30;
                for (int s = 0; s < 30; ++s) {
                    int gp = obs_pid_lo + s;
                    bool hit = (gp < 64)
                        ? ((dlo >> gp)        & 1ULL) != 0ULL
                        : ((dhi >> (gp - 64)) & 1ULL) != 0ULL;
                    if (hit) {
                        cm_set_plane(spatial, CH_CM_KILL_MINE_SLOT + s, cx, cy);
                    }
                }
            }

            // Layer 3.3: cm_recency[4] — exponential fresh signals.
            // Plane 0: direct, tau=32   ; plane 1: direct, tau=256
            // Plane 2: chain,  tau=64   ; plane 3: rank-floor, tau=128
            int idx_obs = cm_idx(observer_seat, pid);
            int16_t ld = cm_last_direct_step_env[idx_obs];
            int16_t lc = cm_last_chain_step_env[idx_obs];
            int16_t lf = cm_rank_floor_step_env[idx_obs];
            const float taus[4] = {32.0f, 256.0f, 64.0f, 128.0f};
            const int16_t lasts[4] = {ld, ld, lc, lf};
            for (int k = 0; k < 4; ++k) {
                int16_t last_step = lasts[k];
                if (last_step < 0) continue;
                int elapsed = move_counter - (int)last_step;
                if (elapsed < 0) elapsed = 0;
                float v = expf(-(float)elapsed / taus[k]);
                if (v > 0.0f) {
                    spatial[(CH_CM_RECENCY + k) * CM_PLANE
                            + cy * 17 + cx] = v;
                }
            }
        }

        // ============== Layer 2 (only for own pieces) ==============
        if (pseat == observer_seat) {
            int li = cm_idx(opp_left,  pid);
            int ri = cm_idx(opp_right, pid);

            int l_mine = cm_popcount64(cm_direct_lo_env[li]) + cm_popcount64(cm_direct_hi_env[li]);
            int r_mine = cm_popcount64(cm_direct_lo_env[ri]) + cm_popcount64(cm_direct_hi_env[ri]);
            int l_total = l_mine + (int)cm_direct_other_count_env[li];
            int r_total = r_mine + (int)cm_direct_other_count_env[ri];
            int public_count = (l_total < r_total) ? l_total : r_total;
            if (public_count >= 1) cm_set_plane(spatial, CH_CM_MY_KILL_COUNT_GE + 0, cx, cy);
            if (public_count >= 2) cm_set_plane(spatial, CH_CM_MY_KILL_COUNT_GE + 1, cx, cy);
            if (public_count >= 3) cm_set_plane(spatial, CH_CM_MY_KILL_COUNT_GE + 2, cx, cy);

            // is_gongb: AND of two opponents (path-revealed only).
            if (cm_is_gongb_env[li] && cm_is_gongb_env[ri]) {
                cm_set_plane(spatial, CH_CM_MY_IS_GONGB, cx, cy);
            }

            // dilei_candidate: alive ∧ at_zero ∧ in_back ∧ ¬(any opponent saw a known-GONGB attack).
            int16_t zf = (int16_t)zero_y_env[pid] * 17 + (int16_t)zero_x_env[pid];
            bool public_atk = cm_attacked_by_known_gongb_env[li]
                           || cm_attacked_by_known_gongb_env[ri];
            if (zero_x_env[pid] >= 0 && zero_y_env[pid] >= 0
                && cm_in_back_two_rows(zf, pseat)
                && move_count_env[pid] == 0
                && !public_atk)
            {
                cm_set_plane(spatial, CH_CM_MY_DILEI_CANDIDATE, cx, cy);
            }
        }
    }
}

}  // namespace junqi_cuda
