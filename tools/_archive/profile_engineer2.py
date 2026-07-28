"""Test vectorized batch BFS for engineer moves."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np

from junqi_core._movegen_tables import (
    ENGINEER_RAIL_NEIGHBORS, ENGINEER_RAIL_NEIGHBORS_PAD,
    IS_RAIL_FLAT, NUM_CELLS,
)
from junqi_core.move_gen import _engineer_dests_soa

# -----------------------------------------------------------------------
# Build precomputed tables
# -----------------------------------------------------------------------
RAIL_CELLS = np.nonzero(IS_RAIL_FLAT)[0].astype(np.int32)   # (64,) flat indices
R = len(RAIL_CELLS)  # 64

RAIL_TO_IDX = np.full(NUM_CELLS, -1, dtype=np.int32)
RAIL_TO_IDX[RAIL_CELLS] = np.arange(R, dtype=np.int32)

# Build (R, R) bool adjacency matrix for the rail graph
RAIL_ADJ = np.zeros((R, R), dtype=np.uint8)   # uint8 for matmul
for i, f in enumerate(RAIL_CELLS.tolist()):
    for nb in ENGINEER_RAIL_NEIGHBORS[f]:
        j = int(RAIL_TO_IDX[nb])
        if j >= 0:
            RAIL_ADJ[i, j] = 1

print(f"RAIL_CELLS: {R} cells")
print(f"RAIL_ADJ density: {RAIL_ADJ.sum()} / {R*R}")
print(f"Max BFS diameter (all empty): ", end="")
# Compute diameter: BFS from cell 0
visited = np.zeros(R, dtype=bool)
visited[0] = True
frontier = visited.astype(np.uint8).reshape(1, R)
depth = 0
while True:
    new = (frontier @ RAIL_ADJ).astype(bool) & ~visited
    if not new.any():
        break
    visited = visited | new.ravel()
    frontier = new.ravel().astype(np.uint8).reshape(1, R)
    depth += 1
print(depth)

# -----------------------------------------------------------------------
# Batch vectorized BFS
# -----------------------------------------------------------------------
def batch_engineer_bfs_vec(
    eng_sf: np.ndarray,         # (E,) int32 — src flat cell indices
    eng_env_in_batch: np.ndarray,  # (E,) int32 — which sub-batch env
    empty_batch: np.ndarray,    # (A, 289) bool — empty cells per sub-batch env
    enemy_att_batch: np.ndarray, # (A, 289) bool — enemy attackable per sub-batch env
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized engineer BFS for all E engineers simultaneously.

    Returns (env_col, act_col) pairs.
    """
    E = len(eng_sf)
    if E == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # Map src flat cells to rail indices
    src_rail_idx = RAIL_TO_IDX[eng_sf]  # (E,) int32, values in [0, 64)

    # Build per-engineer empty_rail and attackable_rail masks (E, 64)
    # empty_batch[eng_env_in_batch[e], RAIL_CELLS[j]] = empty at that rail cell for eng e
    empty_rail   = empty_batch[eng_env_in_batch][:, RAIL_CELLS]  # (E, 64) bool
    attack_rail  = enemy_att_batch[eng_env_in_batch][:, RAIL_CELLS]  # (E, 64) bool

    # Initialize frontier as one-hot at src_rail_idx
    frontier = np.zeros((E, R), dtype=np.uint8)
    frontier[np.arange(E), src_rail_idx] = 1   # (E, 64) uint8

    # BFS state
    visited     = frontier.astype(bool)  # (E, 64) bool
    all_dests   = np.zeros((E, R), dtype=bool)  # accumulate destinations

    # BFS loop (max depth = diameter of rail graph ≈ 30, but terminates early)
    for _ in range(35):  # max iterations = rail diameter
        # Expand: find all unvisited rail neighbors of current frontier
        # neighbors shape: (E, R) — each entry = number of frontier neighbors
        neighbors_reach = (frontier @ RAIL_ADJ).astype(bool)  # (E, R) bool

        # New cells: unvisited AND (empty_rail OR attack_rail)
        # But: can only PASS THROUGH empty cells (not enemy), so:
        # - destinations: new cells that are empty OR attackable (not yet visited as dest)
        # - can continue through: only empty cells
        new_empty_reach    = neighbors_reach & empty_rail & ~visited
        new_attack_reach   = neighbors_reach & attack_rail & ~visited

        # Accumulate destinations (both empty and attackable landing spots)
        new_dests = new_empty_reach | new_attack_reach
        all_dests |= new_dests

        # Continue frontier only through empty cells
        frontier = new_empty_reach.astype(np.uint8)
        visited |= new_empty_reach  # only visited for passthrough purposes
        # Note: visited_for_dest = all_dests (prevents re-adding already found destinations)
        visited |= new_attack_reach  # don't revisit attackable cells either

        if not new_dests.any():
            break

    # Convert all_dests back to (env, flat_cell) pairs
    if not all_dests.any():
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    ei, ri = np.nonzero(all_dests)   # (M,) each
    # ei = engineer index, ri = rail index
    # Map back to original env and flat cell
    env_col = eng_env_in_batch[ei].astype(np.int32)  # sub-batch env index
    flat_col = RAIL_CELLS[ri].astype(np.int32)       # flat cell id
    return env_col, flat_col


# -----------------------------------------------------------------------
# Benchmark: vectorized vs Python BFS
# -----------------------------------------------------------------------
from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core._movegen_tables import IS_ENGINEER_TYPE, IS_RAIL_FLAT as _IRL
from junqi_core.board import BOARD_SIZE

def new_game():
    return GameState.new_game(generate_random_setup())

N = 1024
b = BatchedGameState.from_game_states([new_game() for _ in range(N)])
for _ in range(50):
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)

# Extract a snapshot for benchmarking
from junqi_core.move_gen import _compute_occupancy_masks_n, _T
seat_per_env = b.turn.copy()
n_idx = np.arange(N, dtype=np.intp)
sv_idx = seat_per_env.astype(np.intp)
dead_acting = b.seat_dead_arr[n_idx, sv_idx]
skip = b.terminated | dead_acting
terminated = skip

active_mask = ~terminated
active_idx  = np.nonzero(active_mask)[0]
ai = active_idx
A  = len(ai)

cpid  = b.cell_piece_id[ai]
psa   = b.piece_seat_arr[ai]
pta   = b.piece_type_arr[ai]
alv   = b.alive[ai]
px    = b.pos_x[ai]
py    = b.pos_y[ai]
spe   = seat_per_env[ai]
cta   = b.cell_team_arr[ai]

empty, landable_pad, empty_pad = _compute_occupancy_masks_n(cpid, spe, cta)

# Compute candidates (same as generate_legal_action_ids_n step 2)
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

c_a, c_p = np.nonzero(cand)
C = c_a.shape[0]
c_sf  = sf_clip[c_a, c_p]
c_pt  = pt_vals_full[c_a, c_p]
is_rail_c = _T.IS_RAIL_FLAT[c_sf]
is_eng_c  = _T.IS_ENGINEER_TYPE[c_pt]

eng_on_rail = is_rail_c & is_eng_c
eng_sf  = c_sf[eng_on_rail]
eng_a   = c_a[eng_on_rail].astype(np.int32)
c_env   = ai[c_a].astype(np.int32)
eng_env = c_env[eng_on_rail]

print(f"\nA (active envs): {A}")
print(f"C (candidates): {C}")
print(f"E (engineers on rail): {len(eng_sf)}")

# Benchmark Python BFS (current approach)
TRIALS = 200

t0 = time.perf_counter()
for _ in range(TRIALS):
    env_parts_py = []
    act_parts_py = []
    ea_cache: dict[int, np.ndarray] = {}
    for k in range(len(eng_sf)):
        j  = int(eng_a[k])
        oi = int(eng_env[k])
        if j not in ea_cache:
            ea_cache[j] = landable_pad[j, :NUM_CELLS] & ~empty[j]
        ea_1d = ea_cache[j]
        sf_k  = int(eng_sf[k])
        dests = _engineer_dests_soa(sf_k, empty[j], ea_1d)
        if dests:
            d_arr = np.asarray(dests, dtype=np.int32)
            env_parts_py.append(np.full(len(d_arr), oi, dtype=np.int32))
            act_parts_py.append(sf_k * NUM_CELLS + d_arr)
t_py = (time.perf_counter() - t0) / TRIALS * 1000

print(f"\nPython BFS: {t_py:.3f}ms per step  ({len(eng_sf)} engineers)")

# Benchmark vectorized BFS
t0 = time.perf_counter()
for _ in range(TRIALS):
    env_col_vec, flat_col_vec = batch_engineer_bfs_vec(
        eng_sf.astype(np.int32), eng_a, empty, empty & _T.IS_RAIL_FLAT[np.newaxis, :]
        # Wait, enemy attackable is wrong here - let me fix this
    )
    # Fix: use correct enemy_att (enemy_att = landable & ~empty)
    enemy_att_batch = landable_pad[:, :NUM_CELLS] & ~empty
    env_col_vec, flat_col_vec = batch_engineer_bfs_vec(
        eng_sf.astype(np.int32), eng_a, empty, enemy_att_batch
    )
t_vec = (time.perf_counter() - t0) / TRIALS * 1000

print(f"Vectorized BFS: {t_vec:.3f}ms per step")

# Verify correctness: compare results
# Build action_ids from Python version
if env_parts_py:
    py_envs = np.concatenate([e for e in env_parts_py])
    py_acts = np.concatenate([a for a in act_parts_py])
else:
    py_envs = np.empty(0, dtype=np.int32)
    py_acts = np.empty(0, dtype=np.int32)

# Build from vec version
# env_col_vec is sub-batch index, flat_col_vec is flat cell
src_acts_eng = eng_sf.astype(np.int32)[
    np.isin(np.arange(len(eng_sf)),
            np.where(eng_on_rail)[0][:len(eng_on_rail)])
]

# Actually the fix is: eng_env[ei] × NUM_CELLS + flat_col_vec needs to match
# Let me re-run the vectorized BFS with correct output format
def batch_engineer_bfs_correct(eng_sf, eng_a, eng_env_orig, empty_batch, enemy_att_batch):
    """Returns (env_parts, act_parts) in same format as Python version."""
    E = len(eng_sf)
    if E == 0:
        return [], []

    src_rail_idx = RAIL_TO_IDX[eng_sf]
    empty_rail  = empty_batch[eng_a][:, RAIL_CELLS]   # (E, 64) bool
    attack_rail = enemy_att_batch[eng_a][:, RAIL_CELLS]  # (E, 64) bool
    # Only rail enemy cells count
    attack_rail &= True  # already rail-scoped by indexing RAIL_CELLS

    frontier = np.zeros((E, R), dtype=np.uint8)
    frontier[np.arange(E), src_rail_idx] = 1
    visited  = frontier.astype(bool)
    all_dests = np.zeros((E, R), dtype=bool)

    for _ in range(35):
        neighbors = (frontier @ RAIL_ADJ).astype(bool)
        new_emp  = neighbors & empty_rail  & ~visited
        new_atk  = neighbors & attack_rail & ~visited
        new_dests = new_emp | new_atk
        all_dests |= new_dests
        frontier = new_emp.astype(np.uint8)
        visited |= new_emp | new_atk
        if not new_dests.any():
            break

    if not all_dests.any():
        return [], []

    ei, ri = np.nonzero(all_dests)
    env_parts_out = []
    act_parts_out = []

    for k in range(len(ei)):
        e = int(ei[k])
        r = int(ri[k])
        oi = int(eng_env_orig[e])
        sf = int(eng_sf[e])
        dst = int(RAIL_CELLS[r])
        env_parts_out.append(oi)
        act_parts_out.append(sf * NUM_CELLS + dst)

    if not env_parts_out:
        return [], []
    return (
        [np.array(env_parts_out, dtype=np.int32)],
        [np.array(act_parts_out, dtype=np.int32)],
    )

enemy_att = landable_pad[:, :NUM_CELLS] & ~empty
env_parts_v, act_parts_v = batch_engineer_bfs_correct(
    eng_sf.astype(np.int32), eng_a, eng_env, empty, enemy_att
)

# Compare
def sort_pairs(env_arr, act_arr):
    if len(env_arr) == 0 or not env_arr:
        return set()
    if isinstance(env_arr, list):
        if not env_arr:
            return set()
        e = np.concatenate(env_arr)
        a = np.concatenate(act_arr)
    else:
        e, a = env_arr, act_arr
    return set(zip(e.tolist(), a.tolist()))

py_pairs = sort_pairs(env_parts_py, act_parts_py)
vec_pairs = sort_pairs(env_parts_v, act_parts_v)

print(f"\nPython BFS found: {len(py_pairs)} (env,act) pairs")
print(f"Vectorized BFS found: {len(vec_pairs)} (env,act) pairs")
if py_pairs == vec_pairs:
    print("RESULTS MATCH!")
else:
    missing = py_pairs - vec_pairs
    extra   = vec_pairs - py_pairs
    print(f"MISMATCH! missing: {list(missing)[:5]}, extra: {list(extra)[:5]}")

# Final timing
t0 = time.perf_counter()
for _ in range(TRIALS):
    env_col_v, act_col_v = batch_engineer_bfs_correct(
        eng_sf.astype(np.int32), eng_a, eng_env, empty, enemy_att
    )
t_vec2 = (time.perf_counter() - t0) / TRIALS * 1000
print(f"\nVectorized BFS (correct version): {t_vec2:.3f}ms per step")
print(f"Python BFS: {t_py:.3f}ms per step")
print(f"Speedup: {t_py/t_vec2:.1f}x")
