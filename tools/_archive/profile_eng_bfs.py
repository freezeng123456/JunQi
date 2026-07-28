"""Check engineer BFS iteration count."""
import sys, time
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.batched_state import BatchedGameState
from junqi_core.move_gen import (
    _ENG_RAIL_CELLS, _ENG_RAIL_TO_IDX, _NUM_RAIL_CELLS,
    _ENG_RAIL_ADJ_CLAMP, _ENG_RAIL_ADJ_VALID,
    _compute_occupancy_masks_n, NUM_CELLS, BOARD_SIZE,
)
import junqi_core._movegen_tables as _T

def _new_game():
    return GameState.new_game(generate_random_setup())

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

# Extract engineer data
seat_per_env = b.turn
skip = b.terminated | b.seat_dead_arr[np.arange(N), seat_per_env.astype(np.intp)]
ai = np.nonzero(~skip)[0]
A = len(ai)
alv = b.alive[ai]
psa = b.piece_seat_arr[ai]
pta = b.piece_type_arr[ai]
px = b.pos_x[ai]
py = b.pos_y[ai]
spe = seat_per_env[ai]

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
c_sf = sf_clip[c_a, c_p]
c_pt = pt_vals_full[c_a, c_p]
is_rail_c = _T.IS_RAIL_FLAT[c_sf]
is_eng_c = _T.IS_ENGINEER_TYPE[c_pt]
eng_on_rail = is_rail_c & is_eng_c

print(f"Total candidates: {len(c_a)}")
print(f"Engineers on rail: {eng_on_rail.sum()}")

# Run BFS manually and count iterations
empty, landable_pad, empty_pad = _compute_occupancy_masks_n(
    b.cell_piece_id, spe, b.cell_team_arr[ai]
)
enemy_att = landable_pad[:, :NUM_CELLS] & ~empty

eng_sf = c_sf[eng_on_rail].astype(np.int32)
eng_a = c_a[eng_on_rail].astype(np.int32)
E = int(eng_sf.shape[0])

R = _NUM_RAIL_CELLS
rail_flats = _ENG_RAIL_CELLS
empty_e = empty[eng_a[:, np.newaxis], rail_flats[np.newaxis, :]]
attack_e = enemy_att[eng_a[:, np.newaxis], rail_flats[np.newaxis, :]]

src_ri = _ENG_RAIL_TO_IDX[eng_sf]
reached = np.zeros((E, R), dtype=bool)
reached[np.arange(E), src_ri] = True
passable = empty_e.copy()
passable[np.arange(E), src_ri] = True

adj = _ENG_RAIL_ADJ_CLAMP
adj_valid = _ENG_RAIL_ADJ_VALID

iters = 0
for _ in range(R):
    reached_nb = reached[:, adj]
    reached_nb &= adj_valid[np.newaxis, :, :]
    new_reached = reached.copy()
    any_nb_reached = reached_nb.any(axis=2)
    new_reached |= any_nb_reached & passable
    iters += 1
    if np.array_equal(new_reached, reached):
        break
    reached = new_reached

print(f"BFS converged after {iters} iterations (max possible: {R})")
print(f"Average cells reached per engineer: {reached.sum() / E:.1f}")

# Profile the individual BFS steps
t0 = time.perf_counter()
REPS = 20
total_iters = 0
for _ in range(REPS):
    reached2 = np.zeros((E, R), dtype=bool)
    reached2[np.arange(E), src_ri] = True
    passable2 = empty_e.copy()
    passable2[np.arange(E), src_ri] = True
    for it in range(R):
        reached_nb = reached2[:, adj]
        reached_nb &= adj_valid[np.newaxis, :, :]
        new_reached = reached2.copy()
        any_nb_reached = reached_nb.any(axis=2)
        new_reached |= any_nb_reached & passable2
        total_iters += 1
        if np.array_equal(new_reached, reached2):
            break
        reached2 = new_reached
t1 = time.perf_counter()
print(f"\nBFS time: {(t1-t0)/REPS*1000:.2f} ms ({total_iters/REPS:.0f} iters avg)")
print(f"  Per-iteration: {(t1-t0)/total_iters*1000:.3f} ms")
print(f"  E={E}, R={R}, adj shape={adj.shape}")
