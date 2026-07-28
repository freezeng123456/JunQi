"""Profile INSIDE the actual function call to find the gap."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
import cProfile
import pstats
import io

from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup

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

# Profile
pr = cProfile.Profile()
pr.enable()
for _ in range(50):
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())

# Also by tottime
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(30)
print("=== By tottime ===")
print(s2.getvalue())
