"""Quick profile of BatchedGameState throughput bottleneck."""
import sys, time
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.rules import Seat
from junqi_core.batched_state import BatchedGameState

def _new_game():
    setups = generate_random_setup()
    return GameState.new_game(setups)

N = 256
states = [_new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)

# Warmup
for _ in range(10):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Profile individual components
STEPS = 50

# 1. Profile legal_action_ids_batch only
t0 = time.perf_counter()
for _ in range(STEPS):
    active_ids = b.legal_action_ids_batch()
t1 = time.perf_counter()
legal_time = (t1 - t0) / STEPS
print(f"legal_action_ids_batch: {legal_time*1000:.1f} ms/step = {N/legal_time:,.0f} env·legal/s")

# 2. Profile action selection
t0 = time.perf_counter()
for _ in range(STEPS):
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
t1 = time.perf_counter()
select_time = (t1 - t0) / STEPS
print(f"action selection: {select_time*1000:.1f} ms/step")

# 3. Profile step_batch only
t0 = time.perf_counter()
for _ in range(STEPS):
    bc = b.clone()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    bc.step_batch(action_ids)
t1 = time.perf_counter()
step_time = (t1 - t0) / STEPS - select_time
print(f"step_batch: {step_time*1000:.1f} ms/step = {N/step_time:,.0f} env·step/s")

# 4. Combined throughput
t0 = time.perf_counter()
for _ in range(STEPS):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)
t1 = time.perf_counter()
total_time = (t1 - t0) / STEPS
combined_throughput = N / total_time
print(f"\nCombined: {total_time*1000:.1f} ms/step = {combined_throughput:,.0f} env·steps/s")
print(f"Breakdown: legal={legal_time/total_time*100:.0f}% select={select_time/total_time*100:.0f}% step={step_time/total_time*100:.0f}%")

# 5. Profile inside generate_legal_action_ids_n
from junqi_core.move_gen import (
    _compute_occupancy_masks_n, _split_by_env, _batch_engineer_bfs_n,
    _ADJ_STRAIGHT_PAD, _ADJ_DIAG_PAD, _SRAYS_PAD, _SRAYS_L,
    _EMPTY_INT32, NUM_CELLS, _ENG_RAIL_TO_IDX, _ENG_RAIL_CELLS, _NUM_RAIL_CELLS,
)
import junqi_core._movegen_tables as _T

print("\n--- Internal breakdown of legal_action_ids_n ---")

# Reset for fresh state
states = [_new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
for _ in range(10):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Now time individual phases
seat_per_env = b.turn
n_idx = np.arange(b.num_envs, dtype=np.intp)
sv_idx = seat_per_env.astype(np.intp)
dead_acting = b.seat_dead_arr[n_idx, sv_idx]
skip = b.terminated | dead_acting

REPS = 20

# Phase 1: occupancy
t0 = time.perf_counter()
for _ in range(REPS):
    empty, landable_pad, empty_pad = _compute_occupancy_masks_n(
        b.cell_piece_id, seat_per_env, b.cell_team_arr
    )
t1 = time.perf_counter()
print(f"  occupancy: {(t1-t0)/REPS*1000:.2f} ms")

# Phase 2: candidate filtering
t0 = time.perf_counter()
for _ in range(REPS):
    ai = np.nonzero(~skip)[0]
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
    sf_full = sy_full * 17 + sx_full
    sf_clip = np.clip(sf_full, 0, 288).astype(np.intp)
    not_sh_full = ~_T.IS_STRONGHOLD_FLAT[sf_clip]
    cand = seat_match & mobile_full & on_board_full & not_sh_full
    c_a, c_p = np.nonzero(cand)
    C = c_a.shape[0]
t1 = time.perf_counter()
print(f"  candidates: {(t1-t0)/REPS*1000:.2f} ms (C={C})")

# Count rail/engineer/curve pieces
c_sf = sf_clip[c_a, c_p]
c_pt = pt_vals_full[c_a, c_p]
is_rail_c = _T.IS_RAIL_FLAT[c_sf]
is_eng_c = _T.IS_ENGINEER_TYPE[c_pt]
print(f"  -> rail pieces: {is_rail_c.sum()}, engineers: {is_eng_c.sum()}")
curve_cand = is_rail_c & ~is_eng_c
cids = _T.CURVE_ID_OF[c_sf]
curve_on = curve_cand & (cids > 0)
print(f"  -> curve-rail non-eng: {curve_on.sum()}")
print(f"  -> straight-rail non-eng: {(is_rail_c & ~is_eng_c).sum()}")
print(f"  -> engineers on rail: {(is_rail_c & is_eng_c).sum()}")
