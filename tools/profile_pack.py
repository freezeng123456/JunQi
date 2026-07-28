"""tools/profile_pack.py — profile where _pack_state_arrays spends its time."""

from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import random

from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays


def make_envs(n, seed_base=0xABCDEF):
    envs = []
    for i in range(n):
        rng = random.Random(seed_base + i)
        env = JunqiEnv()
        env.reset(seed=seed_base + i)
        for _ in range(i % 50):
            if env.state.terminated:
                break
            aids = env.legal_action_ids()
            if aids.size == 0:
                break
            env._step_game_only(int(rng.choice(aids)))
        envs.append(env)
    return envs


def main():
    N = 1024
    envs = make_envs(N)
    # warm
    for _ in range(5):
        _pack_state_arrays(envs)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        _pack_state_arrays(envs)
    pr.disable()

    stats = pstats.Stats(pr).sort_stats("cumulative")
    stats.print_stats(30)

    # Fine-grained timing of each stage
    t = {}
    NIT = 50
    # ---- stage: np.stack on bool ----
    states = [e.state for e in envs]
    t0 = time.perf_counter()
    for _ in range(NIT):
        _ = np.stack([s.alive for s in states])
    t["stack_bool_120"] = (time.perf_counter() - t0)/NIT
    t0 = time.perf_counter()
    for _ in range(NIT):
        _ = np.stack([s.pos_x for s in states])
    t["stack_int8_120"] = (time.perf_counter() - t0)/NIT
    t0 = time.perf_counter()
    for _ in range(NIT):
        _ = np.stack([s.cell_piece_id for s in states])
    t["stack_int16_289"] = (time.perf_counter() - t0)/NIT
    t0 = time.perf_counter()
    for _ in range(NIT):
        _ = np.stack([s.move_count_arr for s in states])
    t["stack_int16_120"] = (time.perf_counter() - t0)/NIT
    t0 = time.perf_counter()
    for _ in range(NIT):
        _ = np.stack([s.seat_dead_arr for s in states])
    t["stack_bool_4"] = (time.perf_counter() - t0)/NIT
    t0 = time.perf_counter()
    for _ in range(NIT):
        _ = np.fromiter((s.turn.value for s in states), dtype=np.int8, count=N)
    t["fromiter_turn"] = (time.perf_counter() - t0)/NIT

    for k, v in t.items():
        print(f"  {k:25s}: {v*1e3:6.3f} ms")


if __name__ == "__main__":
    main()
