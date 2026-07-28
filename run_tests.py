import sys, numpy as np, os
os.chdir('/data/home/freezeng/data/workspace/JunQi')
sys.path.insert(0, '.')
from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup

def new_game():
    return GameState.new_game(generate_random_setup())

print("=" * 60)
print("TEST 1: Legal action parity N=16")
print("=" * 60)
N = 16
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
batch_ids = b.legal_action_ids_batch()
ok = True
for i, gs in enumerate(states):
    expected = set(gs.legal_action_ids().tolist())
    actual = set(batch_ids[i].tolist())
    if actual != expected:
        print(f'FAIL env {i}: expected {len(expected)} got {len(actual)}')
        ok = False
if ok:
    print('PASS: legal action parity N=16')

print()
print("=" * 60)
print("TEST 2: step_matches_single_env")
print("=" * 60)
from junqi_core.board import NUM_CELLS
from junqi_core.state import Action

N = 8
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
action_ids = np.array([int(gs.legal_action_ids()[0]) for gs in states], dtype=np.int32)

for i, gs in enumerate(states):
    aid = int(action_ids[i])
    src_flat = aid // NUM_CELLS
    dst_flat = aid % NUM_CELLS
    src = (src_flat % 17, src_flat // 17)
    dst = (dst_flat % 17, dst_flat // 17)
    gs.step_inplace(Action(seat=gs.turn, src=src, dst=dst))

b.step_batch(action_ids)

ok = True
for i, gs in enumerate(states):
    a1 = b.alive[i]; a2 = gs.alive
    if not (a1 == a2).all():
        print(f'FAIL env {i}: alive mismatch'); ok = False
    p1 = b.pos_x[i]; p2 = gs.pos_x
    if not (p1 == p2).all():
        print(f'FAIL env {i}: pos_x mismatch'); ok = False
if ok:
    print('PASS: step_matches_single_env')

print()
print("=" * 60)
print("TEST 3: Benchmark N=1024")
print("=" * 60)
import time

N = 1024
TARGET = 50000
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)

# Warmup 20 steps
for _ in range(20):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)

# Measure 200 steps
t0 = time.perf_counter()
for _ in range(200):
    active_ids = b.legal_action_ids_batch()
    action_ids = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in active_ids
    ], dtype=np.int32)
    b.step_batch(action_ids)
elapsed = time.perf_counter() - t0

total_env_steps = N * 200
throughput = total_env_steps / elapsed
print(f'N={N}: {200} steps in {elapsed:.2f}s => {throughput:,.0f} env-steps/sec')
print(f'Target: {TARGET:,}')
print(f'PASS' if throughput >= TARGET else f'FAIL (need {TARGET - throughput:,.0f} more)')
