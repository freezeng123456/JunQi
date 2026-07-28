"""Detailed profile of generate_legal_action_ids_n internals."""
import sys, time
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.batched_state import BatchedGameState
import junqi_core._movegen_tables as _T
from junqi_core.move_gen import (
    _compute_occupancy_masks_n, _split_by_env, _batch_engineer_bfs_n,
    _ADJ_STRAIGHT_PAD, _ADJ_DIAG_PAD, _SRAYS_PAD, _SRAYS_L,
    _EMPTY_INT32, NUM_CELLS, BOARD_SIZE,
)

def _new_game():
    setups = generate_random_setup()
    return GameState.new_game(setups)

N = 256
states = [_new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
for _ in range(10):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

REPS = 10

# Full manual execution of generate_legal_action_ids_n with timing
seat_per_env = b.turn
n_idx = np.arange(b.num_envs, dtype=np.intp)
sv_idx = seat_per_env.astype(np.intp)
dead_acting = b.seat_dead_arr[n_idx, sv_idx]
terminated = b.terminated | dead_acting

timings = {}

for trial in range(REPS):
    # 1. Occupancy
    t0 = time.perf_counter()
    empty, landable_pad, empty_pad = _compute_occupancy_masks_n(
        b.cell_piece_id, seat_per_env, b.cell_team_arr
    )
    enemy_att = landable_pad[:, :NUM_CELLS] & ~empty
    timings.setdefault('1_occupancy', []).append(time.perf_counter() - t0)

    # 2. Candidates
    t0 = time.perf_counter()
    ai = np.nonzero(~terminated)[0]
    A = len(ai)
    cpid = b.cell_piece_id[ai]
    psa = b.piece_seat_arr[ai]
    pta = b.piece_type_arr[ai]
    alv = b.alive[ai]
    px = b.pos_x[ai]
    py = b.pos_y[ai]
    spe = seat_per_env[ai]
    cta = b.cell_team_arr[ai]

    seat_match = alv & (psa == spe[:, np.newaxis])
    pt_vals_full = pta.astype(np.intp, copy=False)
    mobile_full = ~_T.IS_IMMOBILE_TYPE[pt_vals_full]
    on_board_full = (px >= 0) & (py >= 0)
    sx_full = px.astype(np.int32, copy=False)
    sy_full = py.astype(np.int32, copy=False)
    sf_full = sy_full * BOARD_SIZE + sx_full
    sf_clip = np.clip(sf_full, 0, NUM_CELLS - 1).astype(np.intp)
    not_sh_full = ~_T.IS_STRONGHOLD_FLAT[sf_clip]
    cand = seat_match & mobile_full & on_board_full & not_sh_full
    c_a, c_p = np.nonzero(cand)
    C = c_a.shape[0]
    c_env = ai[c_a].astype(np.int32)
    c_sf = sf_clip[c_a, c_p]
    c_pt = pt_vals_full[c_a, c_p]
    is_rail_c = _T.IS_RAIL_FLAT[c_sf]
    is_eng_c = _T.IS_ENGINEER_TYPE[c_pt]
    src_act = c_sf.astype(np.int32) * NUM_CELLS
    lp_flat = landable_pad.ravel()
    ep_flat = empty_pad.ravel()
    row_stride = NUM_CELLS + 1
    timings.setdefault('2_candidates', []).append(time.perf_counter() - t0)

    env_parts = []
    act_parts = []

    # 3a. Ortho
    t0 = time.perf_counter()
    adj = _ADJ_STRAIGHT_PAD[c_sf]
    fi_a = c_a[:, np.newaxis] * row_stride + adj
    aok = lp_flat[fi_a.ravel()].reshape(C, 4)
    if aok.any():
        ci, dk = np.nonzero(aok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + adj[ci, dk])
    timings.setdefault('3a_ortho', []).append(time.perf_counter() - t0)

    # 3b. Diagonal
    t0 = time.perf_counter()
    diag = _ADJ_DIAG_PAD[c_sf]
    fi_d = c_a[:, np.newaxis] * row_stride + diag
    dok = lp_flat[fi_d.ravel()].reshape(C, 4)
    if dok.any():
        ci, dk = np.nonzero(dok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + diag[ci, dk])
    timings.setdefault('3b_diag', []).append(time.perf_counter() - t0)

    # 3c. Straight rail
    t0 = time.perf_counter()
    rail_ne = is_rail_c & ~is_eng_c
    if rail_ne.any():
        r_sf = c_sf[rail_ne]
        r_a = c_a[rail_ne]
        r_env = c_env[rail_ne]
        r_sc = src_act[rail_ne]
        rays = _SRAYS_PAD[r_sf]
        Cr = r_sf.shape[0]
        L = _SRAYS_L

        fi_r = (r_a[:, np.newaxis, np.newaxis] * row_stride + rays).ravel()
        re = ep_flat[fi_r].reshape(Cr, 4, L)
        rl = lp_flat[fi_r].reshape(Cr, 4, L)

        if L == 4:
            pref = np.empty((Cr, 4, 4), dtype=bool)
            pref[..., 0] = True
            pref[..., 1] = re[..., 0]
            pref[..., 2] = re[..., 0] & re[..., 1]
            pref[..., 3] = re[..., 0] & re[..., 1] & re[..., 2]
        else:
            sh = np.empty_like(re)
            sh[..., 0] = True
            if L > 1:
                sh[..., 1:] = re[..., :-1]
            pref = np.cumprod(sh, axis=-1).astype(bool, copy=False)

        ok = pref & rl
        if ok.any():
            ci, di, li = np.nonzero(ok)
            env_parts.append(r_env[ci])
            act_parts.append(r_sc[ci] + rays[ci, di, li])
    timings.setdefault('3c_straight_rail', []).append(time.perf_counter() - t0)

    # 3d. Engineer BFS
    t0 = time.perf_counter()
    eng_on_rail = is_rail_c & is_eng_c
    if eng_on_rail.any():
        eng_sf = c_sf[eng_on_rail].astype(np.int32)
        eng_a = c_a[eng_on_rail].astype(np.int32)
        eng_env = c_env[eng_on_rail]
        ev, av = _batch_engineer_bfs_n(eng_sf, eng_a, eng_env, empty, enemy_att)
        if ev is not None:
            env_parts.append(ev)
            act_parts.append(av)
    timings.setdefault('3d_engineer_bfs', []).append(time.perf_counter() - t0)

    # 3e. Curve-rail
    t0 = time.perf_counter()
    curve_cand = is_rail_c & ~is_eng_c
    if curve_cand.any() and _T.CURVE_CELLS:
        cids = _T.CURVE_ID_OF[c_sf]
        curve_mask = curve_cand & (cids > 0)
        if curve_mask.any():
            from junqi_core.move_gen import _curve_dests_soa
            for kidx in np.nonzero(curve_mask)[0].tolist():
                sf = int(c_sf[kidx])
                cid = int(cids[kidx])
                a_idx = int(c_a[kidx])
                dests = _curve_dests_soa(sf, cid, empty[a_idx], enemy_att[a_idx])
                if dests:
                    d_arr = np.asarray(dests, dtype=np.int32)
                    env_parts.append(
                        np.full(len(dests), int(c_env[kidx]), dtype=np.int32)
                    )
                    act_parts.append(sf * NUM_CELLS + d_arr)
    timings.setdefault('3e_curve_rail', []).append(time.perf_counter() - t0)

    # 4. Assembly
    t0 = time.perf_counter()
    if env_parts:
        env_col_all = np.concatenate(env_parts)
        act_col_all = np.concatenate(act_parts)
        result = _split_by_env(env_col_all, act_col_all, N)
    timings.setdefault('4_assembly', []).append(time.perf_counter() - t0)

print(f"\n=== Detailed profiling (N={N}, REPS={REPS}) ===")
print(f"C={C} candidates, {is_rail_c.sum()} rail, {is_eng_c.sum()} eng, {(is_rail_c & ~is_eng_c).sum()} rail-noeng")
total_ms = 0
for key in sorted(timings.keys()):
    avg_ms = np.mean(timings[key]) * 1000
    total_ms += avg_ms
    print(f"  {key:25s}: {avg_ms:8.3f} ms  ({avg_ms/12.8*100:5.1f}% of total 12.8ms)")
print(f"  {'TOTAL':25s}: {total_ms:8.3f} ms")
