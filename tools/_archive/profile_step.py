"""Profile step_batch at N=1024."""
import sys, time, cProfile, pstats, io
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.batched_state import BatchedGameState

N = 1024
states = [GameState.new_game(generate_random_setup()) for _ in range(N)]
b = BatchedGameState.from_game_states(states)
for _ in range(20):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Pre-compute actions
active_ids = b.legal_action_ids_batch()
action_ids = np.array([
    int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
    for ids in active_ids
], dtype=np.int32)

# Profile step_batch
pr = cProfile.Profile()
pr.enable()
STEPS = 20
for _ in range(STEPS):
    bc = b.clone()
    bc.step_batch(action_ids)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats(20)
print(s.getvalue())
