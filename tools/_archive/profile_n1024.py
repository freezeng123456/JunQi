"""Profile at N=1024 to match the failing test."""
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

WARMUP = 10
MEASURE = 30

for _ in range(WARMUP):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Separate timing
# 1. Legal only
t0 = time.perf_counter()
for _ in range(MEASURE):
    active_ids = b.legal_action_ids_batch()
t1 = time.perf_counter()
legal_ms = (t1-t0) / MEASURE * 1000
print(f"legal_action_ids_batch: {legal_ms:.1f} ms/step = {N/(legal_ms/1000):,.0f} env/s")

# 2. Step only
t0 = time.perf_counter()
for _ in range(MEASURE):
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)
t1 = time.perf_counter()
step_ms = (t1-t0) / MEASURE * 1000
print(f"step_batch+select: {step_ms:.1f} ms/step")

# 3. Combined
t0 = time.perf_counter()
for _ in range(MEASURE):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)
t1 = time.perf_counter()
combined_ms = (t1-t0) / MEASURE * 1000
throughput = N / (combined_ms / 1000)
print(f"\nCombined: {combined_ms:.1f} ms/step = {throughput:,.0f} env·steps/s")
print(f"Target: 50,000 env·steps/s")
print(f"Need: {1024/(50000)*1000:.1f} ms/step")
