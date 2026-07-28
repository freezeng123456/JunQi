"""Detailed profile v3 - calls the actual function."""
import sys, time
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.batched_state import BatchedGameState

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

# Profile with cProfile
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()

STEPS = 30
for _ in range(STEPS):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(30)
print(s.getvalue())

# Also time-sorted
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(30)
print(s2.getvalue())
