"""Quick benchmark and correctness check for BatchedGameState."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np
from junqi_core.batched_state import BatchedGameState
from junqi_core.state import GameState, Action
from junqi_core.setup import generate_random_setup
from junqi_core.board import NUM_CELLS

def new_game():
    return GameState.new_game(generate_random_setup())

# -----------------------------------------------------------------------
# Test 1: legal action parity
# -----------------------------------------------------------------------
print("=== Test 1: Legal action parity N=16 ===")
N = 16
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
batch_ids = b.legal_action_ids_batch()
ok = True
for i, gs in enumerate(states):
    expected = set(gs.legal_action_ids().tolist())
    actual = set(batch_ids[i].tolist())
    if actual != expected:
        print(f"  FAIL env {i}: expected {len(expected)} got {len(actual)}")
        missing = expected - actual
        extra = actual - expected
        if missing: print(f"    missing: {list(missing)[:5]}")
        if extra:   print(f"    extra:   {list(extra)[:5]}")
        ok = False
print(f"  {'PASS' if ok else 'FAIL'}")

# -----------------------------------------------------------------------
# Test 2: step matches single env
# -----------------------------------------------------------------------
print("\n=== Test 2: step_batch matches single-env ===")
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
    if not (b.alive[i] == gs.alive).all():
        print(f"  FAIL env {i}: alive mismatch"); ok = False
    if not (b.pos_x[i] == gs.pos_x).all():
        print(f"  FAIL env {i}: pos_x mismatch"); ok = False
    if not (b.pos_y[i] == gs.pos_y).all():
        print(f"  FAIL env {i}: pos_y mismatch"); ok = False
    if not (b.cell_piece_id[i] == gs.cell_piece_id).all():
        print(f"  FAIL env {i}: cell_piece_id mismatch"); ok = False
print(f"  {'PASS' if ok else 'FAIL'}")

# -----------------------------------------------------------------------
# Test 3: Zobrist consistency over 5 steps
# -----------------------------------------------------------------------
print("\n=== Test 3: Zobrist consistency 5 steps N=4 ===")
N = 4
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
ok = True
for step in range(5):
    action_ids = np.array([int(gs.legal_action_ids()[0]) for gs in states], dtype=np.int32)
    for i, gs in enumerate(states):
        aid = int(action_ids[i])
        sf = aid // NUM_CELLS; df = aid % NUM_CELLS
        gs.step_inplace(Action(seat=gs.turn, src=(sf%17, sf//17), dst=(df%17, df//17)))
    b.step_batch(action_ids)
    for i, gs in enumerate(states):
        if int(b.zobrist[i]) != gs.zobrist:
            print(f"  FAIL step {step} env {i}: batch={int(b.zobrist[i]):#x} single={gs.zobrist:#x}")
            ok = False
print(f"  {'PASS' if ok else 'FAIL'}")

# -----------------------------------------------------------------------
# Test 4: Multi-step rollout N=32 (100 steps)
# -----------------------------------------------------------------------
print("\n=== Test 4: Multi-step rollout N=32 x 100 ===")
N = 32
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)
ok = True
try:
    for step in range(100):
        active_ids = b.legal_action_ids_batch()
        action_ids = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if b.terminated[i] or len(active_ids[i]) == 0:
                action_ids[i] = 0
            else:
                action_ids[i] = active_ids[i][np.random.randint(len(active_ids[i]))]
        b.step_batch(action_ids)
    # Check no NaN/inf in zobrist
    if not np.isfinite(b.zobrist.astype(float)).all():
        print("  FAIL: NaN/inf in zobrist")
        ok = False
except Exception as e:
    print(f"  FAIL: exception: {e}")
    import traceback; traceback.print_exc()
    ok = False
print(f"  {'PASS' if ok else 'FAIL'}")

# -----------------------------------------------------------------------
# Benchmark N=1024
# -----------------------------------------------------------------------
print("\n=== Benchmark N=1024 ===")
N = 1024
TARGET = 50_000
states = [new_game() for _ in range(N)]
b = BatchedGameState.from_game_states(states)

# Warmup
for _ in range(20):
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)

# Measure
MEASURE = 200
t0 = time.perf_counter()
for _ in range(MEASURE):
    ids_list = b.legal_action_ids_batch()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    b.step_batch(act)
elapsed = time.perf_counter() - t0

total = N * MEASURE
tput = total / elapsed
print(f"  {MEASURE} steps × N={N} = {total:,} env-steps in {elapsed:.2f}s")
print(f"  Throughput: {tput:,.0f} env-steps/sec  (target: {TARGET:,})")
print(f"  {'PASS' if tput >= TARGET else 'FAIL'}")

# Also time individual phases
print("\n=== Phase timing breakdown (20 steps N=1024) ===")
N = 1024
states = [new_game() for _ in range(N)]
b2 = BatchedGameState.from_game_states(states)

t_legal = 0.0
t_action_build = 0.0
t_step = 0.0
PROFILE_STEPS = 20

for _ in range(PROFILE_STEPS):
    t0 = time.perf_counter()
    ids_list = b2.legal_action_ids_batch()
    t_legal += time.perf_counter() - t0

    t0 = time.perf_counter()
    act = np.array([
        int(ids[np.random.randint(len(ids))]) if len(ids) > 0 else 0
        for ids in ids_list
    ], dtype=np.int32)
    t_action_build += time.perf_counter() - t0

    t0 = time.perf_counter()
    b2.step_batch(act)
    t_step += time.perf_counter() - t0

print(f"  legal_action_ids_batch: {1000*t_legal/PROFILE_STEPS:.2f} ms/step")
print(f"  action build (Python):  {1000*t_action_build/PROFILE_STEPS:.2f} ms/step")
print(f"  step_batch:             {1000*t_step/PROFILE_STEPS:.2f} ms/step")
total_ms = 1000*(t_legal + t_action_build + t_step)/PROFILE_STEPS
print(f"  total per step:         {total_ms:.2f} ms  => {1000/total_ms*N:,.0f} env-steps/sec")
