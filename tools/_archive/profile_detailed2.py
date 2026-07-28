"""Cleaner profiling: measure phases WITHOUT running the actual function in the same loop."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np

from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.board import NUM_CELLS, BOARD_SIZE
from junqi_core.move_gen import (
    _compute_occupancy_masks_n, _split_by_env,
    _ADJ_STRAIGHT_PAD, _ADJ_DIAG_PAD, _SRAYS_PAD, _SRAYS_L,
    _EMPTY_INT32,
)
from junqi_core.move_gen import _T

def new_game():
    return GameState.new_game(generate_random_setup())

N = 1024

def make_batch():
    states = [new_game() for _ in range(N)]
    b = BatchedGameState.from_game_states(states)
    # Advance the game state ~50 steps
    for _ in range(50):
        ids_list = b.legal_action_ids_batch()
        act = np.array([
            int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
            for ids in ids_list
        ], dtype=np.int32)
        b.step_batch(act)
    return b

b = make_batch()

# ======================================================
# Part 1: Measure actual legal_action_ids_batch alone
# ======================================================
WARMUP = 20
MEASURE = 100

for _ in range(WARMUP):
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)

times_legal = []
times_build = []
times_step = []
for _ in range(MEASURE):
    t0 = time.perf_counter()
    ids_list = b.legal_action_ids_batch()
    t1 = time.perf_counter()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    t2 = time.perf_counter()
    b.step_batch(act)
    t3 = time.perf_counter()
    times_legal.append(t1 - t0)
    times_build.append(t2 - t1)
    times_step.append(t3 - t2)

print(f"=== Phase timing (N={N}, {MEASURE} steps) ===")
print(f"  legal_action_ids_batch: median={np.median(times_legal)*1000:.2f}ms, mean={np.mean(times_legal)*1000:.2f}ms")
print(f"  action_build (Python):  median={np.median(times_build)*1000:.2f}ms, mean={np.mean(times_build)*1000:.2f}ms")
print(f"  step_batch:             median={np.median(times_step)*1000:.2f}ms, mean={np.mean(times_step)*1000:.2f}ms")
total = np.mean(times_legal) + np.mean(times_build) + np.mean(times_step)
print(f"  total (mean):           {total*1000:.2f}ms => {N/total:,.0f} env-steps/sec")

# ======================================================
# Part 2: Profile individual operations within legal_action_ids_batch
# (each step run WITHOUT the full function call in same loop)
# ======================================================
b2 = make_batch()
for _ in range(WARMUP):
    ids_list = b2.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b2.step_batch(act)

PROFILE_STEPS = 50

timings = {
    'subviews': 0.0, 'occupancy': 0.0, 'cand_mask': 0.0,
    'nonzero': 0.0, 'c_extract': 0.0, 'ortho': 0.0, 'diag': 0.0,
    'rail': 0.0, 'concat': 0.0, 'split': 0.0,
}

saved_states = []  # capture states to step after profiling

for step in range(PROFILE_STEPS):
    # Capture current state without modifying
    seat_per_env = b2.turn.copy()
    n_idx = np.arange(N, dtype=np.intp)
    sv_idx = seat_per_env.astype(np.intp)
    dead_acting = b2.seat_dead_arr[n_idx, sv_idx]
    skip = b2.terminated | dead_acting
    cell_piece_id = b2.cell_piece_id
    piece_seat_arr = b2.piece_seat_arr
    piece_type_arr = b2.piece_type_arr
    alive = b2.alive
    pos_x = b2.pos_x
    pos_y = b2.pos_y
    cell_team_arr = b2.cell_team_arr
    terminated = skip

    # Subviews
    t0 = time.perf_counter()
    active_mask = ~terminated
    active_idx = np.nonzero(active_mask)[0]
    ai = active_idx
    cpid = cell_piece_id[ai]
    psa  = piece_seat_arr[ai]
    pta  = piece_type_arr[ai]
    alv  = alive[ai]
    px   = pos_x[ai]
    py   = pos_y[ai]
    spe  = seat_per_env[ai]
    cta  = cell_team_arr[ai]
    timings['subviews'] += time.perf_counter() - t0

    # Occupancy
    t0 = time.perf_counter()
    empty, landable_pad, empty_pad = _compute_occupancy_masks_n(cpid, spe, cta)
    timings['occupancy'] += time.perf_counter() - t0

    # Candidate mask (before nonzero)
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
    timings['cand_mask'] += time.perf_counter() - t0

    # Nonzero
    t0 = time.perf_counter()
    c_a, c_p = np.nonzero(cand)
    C = c_a.shape[0]
    timings['nonzero'] += time.perf_counter() - t0

    # Candidate extraction
    t0 = time.perf_counter()
    c_env = ai[c_a].astype(np.int32)
    c_sf  = sf_clip[c_a, c_p]
    c_pt  = pt_vals_full[c_a, c_p]
    is_rail_c = _T.IS_RAIL_FLAT[c_sf]
    is_eng_c  = _T.IS_ENGINEER_TYPE[c_pt]
    src_act = c_sf.astype(np.int32) * NUM_CELLS
    lp_flat = landable_pad.ravel()
    ep_flat = empty_pad.ravel()
    row_stride = NUM_CELLS + 1
    timings['c_extract'] += time.perf_counter() - t0

    env_parts = []
    act_parts = []

    # Ortho
    t0 = time.perf_counter()
    adj = _ADJ_STRAIGHT_PAD[c_sf]
    fi_a = c_a[:, np.newaxis] * row_stride + adj
    aok  = lp_flat[fi_a.ravel()].reshape(C, 4)
    if aok.any():
        ci, dk = np.nonzero(aok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + adj[ci, dk])
    timings['ortho'] += time.perf_counter() - t0

    # Diag
    t0 = time.perf_counter()
    diag = _ADJ_DIAG_PAD[c_sf]
    fi_d = c_a[:, np.newaxis] * row_stride + diag
    dok  = lp_flat[fi_d.ravel()].reshape(C, 4)
    if dok.any():
        ci, dk = np.nonzero(dok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + diag[ci, dk])
    timings['diag'] += time.perf_counter() - t0

    # Rail
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
    timings['rail'] += time.perf_counter() - t0

    # Concat
    t0 = time.perf_counter()
    if env_parts:
        env_col_all = np.concatenate(env_parts)
        act_col_all = np.concatenate(act_parts)
    timings['concat'] += time.perf_counter() - t0

    # Split
    t0 = time.perf_counter()
    if env_parts:
        result = _split_by_env(env_col_all, act_col_all, N)
    timings['split'] += time.perf_counter() - t0

    # Now actually step the game (after profiling)
    ids_list = b2.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b2.step_batch(act)

print(f"\n=== Individual operation timing ({PROFILE_STEPS} steps) ===")
total_acc = 0.0
for k, v in timings.items():
    ms = v / PROFILE_STEPS * 1000
    total_acc += ms
    print(f"  {k}: {ms:.3f} ms")
print(f"\n  total accounted: {total_acc:.3f} ms")

# Check M sizes
if env_parts and env_col_all is not None:
    print(f"\n  M (total moves in last step): {len(env_col_all)}")
    print(f"  C (total candidates): {C}")
    print(f"  Cr (rail candidates): {c_sf[is_rail_c & ~is_eng_c].shape[0] if (is_rail_c & ~is_eng_c).any() else 0}")
