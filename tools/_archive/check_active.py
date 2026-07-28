"""Check active env count during benchmark."""
import sys, time
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.batched_state import BatchedGameState

def _new_game():
    return GameState.new_game(generate_random_setup())

N = 1024
states = [_new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)

for step in range(220):
    active_ids = b.legal_action_ids_batch()
    n_active = int((~b.terminated).sum())
    n_combat = 0
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    if step % 20 == 0:
        print(f"Step {step:4d}: active={n_active}/{N} term={int(b.terminated.sum())}")
    b.step_batch(action_ids)

print(f"\nFinal: active={int((~b.terminated).sum())}/{N}")
