"""Fine-grained profiling inside generate_legal_action_ids_n to find the ~7ms gap."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np

from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.board import NUM_CELLS, BOARD_SIZE
from junqi_core import move_gen as mg
from junqi_core.move_gen import (
    _compute_occupancy_masks_n, _split_by_env,
    _ADJ_STRAIGHT_PAD, _ADJ_DIAG_PAD, _SRAYS_PAD, _SRAYS_L,
    _EMPTY_INT32,
)
from junqi_core.move_gen import _T

def new_game():
    return GameState.new_game(generate_random_setup())

N = 1024
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)

# Warm up
for _ in range(30):
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)

STEPS = 50
timings = {k: 0.0 for k in [
    't01_active', 't02_subviews', 't03_occupancy', 't04_cand_before_nonzero',
    't05_nonzero', 't06_cenv_csf_cpt', 't07_rail_check', 't08_src_act_flat',
    't09_ortho', 't10_diag', 't11_rail', 't12_engineer', 't13_concat', 't14_split',
]}

for step in range(STEPS):
    seat_per_env = b.turn
    n_idx = np.arange(N, dtype=np.intp)
    sv_idx = seat_per_env.astype(np.intp)
    dead_acting = b.seat_dead_arr[n_idx, sv_idx]
    terminated = b.terminated | dead_acting
    cell_piece_id = b.cell_piece_id
    piece_seat_arr = b.piece_seat_arr
    piece_type_arr = b.piece_type_arr
    alive = b.alive
    pos_x = b.pos_x
    pos_y = b.pos_y
    cell_team_arr = b.cell_team_arr

    # t01: active mask
    t0 = time.perf_counter()
    active_mask = ~terminated
    active_idx = np.nonzero(active_mask)[0]
    A = len(active_idx)
    ai = active_idx
    timings['t01_active'] += time.perf_counter() - t0

    # t02: sub-batch views
    t0 = time.perf_counter()
    cpid = cell_piece_id[ai]
    psa  = piece_seat_arr[ai]
    pta  = piece_type_arr[ai]
    alv  = alive[ai]
    px   = pos_x[ai]
    py   = pos_y[ai]
    spe  = seat_per_env[ai]
    cta  = cell_team_arr[ai]
    timings['t02_subviews'] += time.perf_counter() - t0

    # t03: occupancy
    t0 = time.perf_counter()
    empty, landable_pad, empty_pad = _compute_occupancy_masks_n(cpid, spe, cta)
    timings['t03_occupancy'] += time.perf_counter() - t0

    # t04: candidate computation (everything before nonzero)
    t0 = time.perf_counter()
    seat_match    = alv & (psa == spe[:, np.newaxis])
    pt_vals_full  = pta.astype(np.intp, copy=False)
    mobile_full   = ~_T.IS_IMMOBILE_TYPE[pt_vals_full]
    on_board_full = (px >= 0) & (py >= 0)
    sx_full = px.astype(np.int32, copy=False)
    sy_full = py.astype(np.int32, copy=False)
    sf_full = sy_full * BOARD_SIZE + sx_full
    sf_clip = np.clip(sf_full, 0, NUM_CELLS - 1).astype(np.intp)
    not_sh_full = ~_T.IS_STRONGHOLD_FLAT[sf_clip]
    cand = seat_match & mobile_full & on_board_full & not_sh_full
    timings['t04_cand_before_nonzero'] += time.perf_counter() - t0

    # t05: nonzero
    t0 = time.perf_counter()
    c_a, c_p = np.nonzero(cand)
    C = c_a.shape[0]
    timings['t05_nonzero'] += time.perf_counter() - t0

    # t06: c_env, c_sf, c_pt extraction
    t0 = time.perf_counter()
    c_env = ai[c_a].astype(np.int32)
    c_sf  = sf_clip[c_a, c_p]
    c_pt  = pt_vals_full[c_a, c_p]
    is_rail_c = _T.IS_RAIL_FLAT[c_sf]
    is_eng_c  = _T.IS_ENGINEER_TYPE[c_pt]
    timings['t06_cenv_csf_cpt'] += time.perf_counter() - t0

    # t07: src_act + flat setup
    t0 = time.perf_counter()
    src_act = c_sf.astype(np.int32) * NUM_CELLS
    lp_flat = landable_pad.ravel()
    ep_flat = empty_pad.ravel()
    row_stride = NUM_CELLS + 1  # 290
    timings['t07_rail_check'] += time.perf_counter() - t0

    env_parts = []
    act_parts = []

    # t09: ortho
    t0 = time.perf_counter()
    adj = _ADJ_STRAIGHT_PAD[c_sf]
    fi_a = c_a[:, np.newaxis] * row_stride + adj
    aok  = lp_flat[fi_a.ravel()].reshape(C, 4)
    if aok.any():
        ci, dk = np.nonzero(aok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + adj[ci, dk])
    timings['t09_ortho'] += time.perf_counter() - t0

    # t10: diag
    t0 = time.perf_counter()
    diag = _ADJ_DIAG_PAD[c_sf]
    fi_d = c_a[:, np.newaxis] * row_stride + diag
    dok  = lp_flat[fi_d.ravel()].reshape(C, 4)
    if dok.any():
        ci, dk = np.nonzero(dok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + diag[ci, dk])
    timings['t10_diag'] += time.perf_counter() - t0

    # t11: rail
    t0 = time.perf_counter()
    rail_ne = is_rail_c & ~is_eng_c
    if rail_ne.any():
        r_sf  = c_sf[rail_ne]
        r_a   = c_a[rail_ne]
        r_env = c_env[rail_ne]
        r_sc  = src_act[rail_ne]
        rays  = _SRAYS_PAD[r_sf]
        Cr    = r_sf.shape[0]
        L     = _SRAYS_L
        fi_r  = (r_a[:, np.newaxis, np.newaxis] * row_stride + rays).ravel()
        re    = ep_flat[fi_r].reshape(Cr, 4, L)
        rl    = lp_flat[fi_r].reshape(Cr, 4, L)
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
    timings['t11_rail'] += time.perf_counter() - t0

    # t12: engineer BFS
    t0 = time.perf_counter()
    eng_on_rail = is_rail_c & is_eng_c
    # (skip actual engineer BFS for timing; it's rare)
    timings['t12_engineer'] += time.perf_counter() - t0

    # t13: concatenate
    t0 = time.perf_counter()
    if env_parts:
        env_col_all = np.concatenate(env_parts)
        act_col_all = np.concatenate(act_parts)
    timings['t13_concat'] += time.perf_counter() - t0

    # t14: split
    t0 = time.perf_counter()
    if env_parts:
        result = _split_by_env(env_col_all, act_col_all, N)
    timings['t14_split'] += time.perf_counter() - t0

    # Actually step (to keep state realistic)
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)

print(f"\n=== Fine-grained profiling over {STEPS} steps, N={N} ===")
total_accounted = 0.0
for k, v in timings.items():
    ms = v / STEPS * 1000
    total_accounted += ms
    print(f"  {k}: {ms:.3f} ms")
print(f"\n  Total accounted: {total_accounted:.3f} ms")

# Also measure the actual function call for comparison
b2 = BatchedGameState.from_game_states([new_game() for _ in range(N)])
for _ in range(20):
    ids_list = b2.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b2.step_batch(act)

MEASURE = 50
t0 = time.perf_counter()
for _ in range(MEASURE):
    ids_list = b2.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b2.step_batch(act)
elapsed = time.perf_counter() - t0
tput = N * MEASURE / elapsed
print(f"\nActual legal_action_ids_batch call: {elapsed/MEASURE*1000:.3f} ms/step")
print(f"Throughput: {tput:,.0f} env-steps/sec")
