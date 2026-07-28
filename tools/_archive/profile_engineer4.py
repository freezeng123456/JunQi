"""Design precomputed engineer reach tables for ultra-fast vectorized BFS.

The rail graph has 4 independent components of 16 cells each (simple cycles).
From any source, BFS expands in 2 directions around the ring.
We precompute per-source ordered reach sequences, then use cumsum-style masking.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import time
import numpy as np
from collections import deque
from junqi_core._movegen_tables import (
    ENGINEER_RAIL_NEIGHBORS, IS_RAIL_FLAT, NUM_CELLS
)
from junqi_core.move_gen import _engineer_dests_soa

RAIL_CELLS = np.nonzero(IS_RAIL_FLAT)[0].astype(np.int32)
R = len(RAIL_CELLS)
RAIL_TO_IDX = np.full(NUM_CELLS, -1, dtype=np.int32)
RAIL_TO_IDX[RAIL_CELLS] = np.arange(R, dtype=np.int32)

# Build per-component adjacency list
adj_list = [[] for _ in range(R)]
for i, f in enumerate(RAIL_CELLS.tolist()):
    for nb in ENGINEER_RAIL_NEIGHBORS[f]:
        j = int(RAIL_TO_IDX[nb])
        if j >= 0:
            adj_list[i].append(j)

# For each src rail_idx, compute the BFS traversal ORDER.
# Since each node has degree 2, BFS from src explores in 2 directions.
# The reachable sequence starting from each of the 2 neighbors:
#   branch_A: src -> A0 -> A1 -> A2 ... (one direction)
#   branch_B: src -> B0 -> B1 -> B2 ... (other direction)
# Both branches have at most 15 cells (ring of 16 total - src itself)

# Precompute: for each src (R), 2 directions each of length up to 15
# Shape: ENG_REACH[src_rail_idx] = array of (up to 15) rail_idxs in BFS order
#   but split into 2 branches

BRANCH_LEN = 15  # max cells in one direction (ring of 16)
# ENG_BRANCH[src, dir, k] = rail_idx of kth cell in direction dir from src
# dir=0 or dir=1; -1 = end of chain

ENG_BRANCHES = np.full((R, 2, BRANCH_LEN), -1, dtype=np.int32)

for src in range(R):
    nbrs = adj_list[src]
    assert len(nbrs) == 2, f"Expected degree-2 at rail_idx {src}"
    for d, start in enumerate(nbrs):
        prev = src
        cur = start
        for k in range(BRANCH_LEN):
            ENG_BRANCHES[src, d, k] = cur
            # Find next: the neighbor that isn't prev
            next_nbrs = adj_list[cur]
            nxt = [x for x in next_nbrs if x != prev]
            if not nxt:
                break  # end of chain
            prev = cur
            cur = nxt[0]

# Verify: all branches of length exactly 15 (ring of 16)
print(f"ENG_BRANCHES shape: {ENG_BRANCHES.shape}")
branch_lens = [(ENG_BRANCHES[i, d, :] >= 0).sum() for i in range(R) for d in range(2)]
print(f"Branch lengths: min={min(branch_lens)}, max={max(branch_lens)}, mean={sum(branch_lens)/len(branch_lens):.1f}")

# -----------------------------------------------------------------------
# Fast branch-based engineer move generation
# -----------------------------------------------------------------------
def batch_engineer_bfs_branch(
    eng_sf_flat: np.ndarray,  # (E,) int32 flat cell ids
    eng_a: np.ndarray,        # (E,) int32 sub-batch env idx
    eng_env_orig: np.ndarray, # (E,) int32 original env ids
    empty_batch: np.ndarray,  # (A, 289) bool
    enemy_att_batch: np.ndarray,  # (A, 289) bool
) -> tuple[list, list]:
    E = len(eng_sf_flat)
    if E == 0:
        return [], []

    src_ri = RAIL_TO_IDX[eng_sf_flat]  # (E,) rail indices

    # For each engineer, for each direction (2), for each step (15):
    # cell_ri = ENG_BRANCHES[src_ri, d, k]
    # Shape: (E, 2, 15)
    branch_ri = ENG_BRANCHES[src_ri]  # (E, 2, 15) int32, -1=padding

    # Map to flat cell ids (with -1 staying as NUM_CELLS-1 = 288 as sentinel,
    # or we handle padding separately)
    valid_mask = branch_ri >= 0  # (E, 2, 15)

    # Map rail_idx -> flat cell id, clamp -1 to 0 (will be masked out anyway)
    branch_ri_clamp = np.where(valid_mask, branch_ri, 0)  # (E, 2, 15)
    branch_flat = RAIL_CELLS[branch_ri_clamp]  # (E, 2, 15) int32 flat cells

    # Look up empty[eng_a[e], flat] and enemy_att[eng_a[e], flat]
    # Shape: empty_batch[eng_a, :] → (E, 289) but we need (E, 2, 15)
    # Use advanced indexing: eng_a[:, None, None] broadcasts with branch_flat
    e_idx = eng_a[:, np.newaxis, np.newaxis]  # (E, 1, 1)
    is_empty_branch  = empty_batch[e_idx, branch_flat]     # (E, 2, 15)
    is_attack_branch = enemy_att_batch[e_idx, branch_flat] # (E, 2, 15)

    # Mask out padding
    is_empty_branch  &= valid_mask
    is_attack_branch &= valid_mask

    # For each branch: compute prefix empty (can pass through up to this cell)
    # pref_empty[e, d, k] = all(is_empty_branch[e, d, 0:k])
    # We need prefix to allow LANDING: can land if all PRIOR cells are empty
    # (the cell itself can be empty or attackable)
    # pref[e, d, 0] = True (no prior cells needed)
    # pref[e, d, k] = is_empty_branch[e, d, k-1] & pref[e, d, k-1]
    pref = np.ones((E, 2, BRANCH_LEN), dtype=bool)
    for k in range(1, BRANCH_LEN):
        pref[:, :, k] = pref[:, :, k-1] & is_empty_branch[:, :, k-1]

    # Can land at position k if: pref[k] AND (is_empty OR is_attack) AND valid
    can_land = pref & (is_empty_branch | is_attack_branch) & valid_mask

    if not can_land.any():
        return [], []

    # Gather results
    ei, di, ki = np.nonzero(can_land)  # (M,) each
    dest_flat = branch_flat[ei, di, ki]  # (M,) flat cell ids
    dest_src_flat = eng_sf_flat[ei]     # (M,) src flat cell ids
    orig_env = eng_env_orig[ei]          # (M,) original env ids

    # Build action ids
    act_ids = dest_src_flat.astype(np.int32) * NUM_CELLS + dest_flat.astype(np.int32)
    return [orig_env.astype(np.int32)], [act_ids]


# -----------------------------------------------------------------------
# Verify correctness and benchmark
# -----------------------------------------------------------------------
from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.move_gen import _compute_occupancy_masks_n, _T
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

seat_per_env = b.turn.copy()
n_idx = np.arange(N, dtype=np.intp)
dead_acting = b.seat_dead_arr[n_idx, seat_per_env.astype(np.intp)]
skip = b.terminated | dead_acting
active_idx = np.nonzero(~skip)[0]
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
enemy_att = landable_pad[:, :NUM_CELLS] & ~empty

# Candidates
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
c_sf  = sf_clip[c_a, c_p].astype(np.int32)
c_pt  = pt_vals_full[c_a, c_p]
is_rail_c = _T.IS_RAIL_FLAT[c_sf]
is_eng_c  = _T.IS_ENGINEER_TYPE[c_pt]
c_env = ai[c_a].astype(np.int32)

eng_on_rail = is_rail_c & is_eng_c
eng_sf   = c_sf[eng_on_rail]
eng_a    = c_a[eng_on_rail].astype(np.int32)
eng_env  = c_env[eng_on_rail]

print(f"E (engineers on rail): {len(eng_sf)}")

TRIALS = 500

# Python BFS (current)
t0 = time.perf_counter()
for _ in range(TRIALS):
    env_parts_py = []
    act_parts_py = []
    ea_cache = {}
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

# New branch BFS
t0 = time.perf_counter()
for _ in range(TRIALS):
    env_parts_v, act_parts_v = batch_engineer_bfs_branch(
        eng_sf, eng_a, eng_env, empty, enemy_att
    )
t_vec = (time.perf_counter() - t0) / TRIALS * 1000

print(f"\nPython BFS: {t_py:.3f}ms per step")
print(f"Branch BFS: {t_vec:.3f}ms per step")
print(f"Speedup: {t_py/t_vec:.1f}x")

# Correctness check
def to_pairs(env_p, act_p):
    if not env_p:
        return set()
    return set(zip(np.concatenate(env_p).tolist(), np.concatenate(act_p).tolist()))

py_pairs  = to_pairs(env_parts_py, act_parts_py)
vec_pairs = to_pairs(env_parts_v, act_parts_v)

print(f"\nPython BFS: {len(py_pairs)} (env,act) pairs")
print(f"Branch BFS: {len(vec_pairs)} (env,act) pairs")
if py_pairs == vec_pairs:
    print("CORRECT!")
else:
    missing = list(py_pairs - vec_pairs)[:5]
    extra   = list(vec_pairs - py_pairs)[:5]
    print(f"MISMATCH! missing={missing}, extra={extra}")
