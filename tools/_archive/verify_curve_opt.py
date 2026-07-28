"""Verify curve chain ray tables and test correctness."""
import sys
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
import junqi_core._movegen_tables as _T

print('CURVE_CHAIN_RAYS_PAD shape:', _T.CURVE_CHAIN_RAYS_PAD.shape)
print('IS_CURVE_ACTIVE count:', int(_T.IS_CURVE_ACTIVE.sum()))

# Verify for curve 1 cell (5,10) flat=175
flat = 175
print(f'\nCell flat={flat}: curve_id={_T.CURVE_ID_OF[flat]}, degree={len(_T.CURVE_NEIGHBORS[flat])}')
print(f'  Forward ray: {_T.CURVE_CHAIN_RAYS_PAD[flat, 0].tolist()}')
print(f'  Backward ray: {_T.CURVE_CHAIN_RAYS_PAD[flat, 1].tolist()}')

# Now test correctness: compare vectorized vs original BFS
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.move_gen import _curve_dests_soa, generate_legal_action_ids_n, _compute_occupancy_masks_soa, generate_legal_action_ids_batch
from junqi_core.batched_state import BatchedGameState

N = 64
states = [GameState.new_game(generate_random_setup()) for _ in range(N)]
b = BatchedGameState.from_game_states(states)

# Step a few times to get some diversity
for _ in range(30):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Now get the vectorized output
result_vec = b.legal_action_ids_batch()

# Compare with per-env output using the original single-env function
mismatches = 0
for i in range(N):
    if b.terminated[i]:
        continue
    sv = int(b.turn[i])
    ref = generate_legal_action_ids_batch(
        b.cell_piece_id[i], b.piece_seat_arr[i], b.piece_type_arr[i],
        b.alive[i], b.pos_x[i], b.pos_y[i], sv,
    )
    vec_set = set(result_vec[i].tolist())
    ref_set = set(ref.tolist())
    if vec_set != ref_set:
        only_vec = vec_set - ref_set
        only_ref = ref_set - vec_set
        mismatches += 1
        if mismatches <= 3:
            print(f"Env {i}: vec has {len(only_vec)} extra, ref has {len(only_ref)} extra")
            if only_vec:
                for a in list(only_vec)[:5]:
                    sf, df = a // 289, a % 289
                    print(f"  extra in vec: {sf} -> {df}")
            if only_ref:
                for a in list(only_ref)[:5]:
                    sf, df = a // 289, a % 289
                    print(f"  missing in vec: {sf} -> {df}")

if mismatches == 0:
    print(f"\nCORRECTNESS CHECK PASSED: {N} envs, all match!")
else:
    print(f"\nCORRECTNESS CHECK FAILED: {mismatches}/{N} envs mismatched")
