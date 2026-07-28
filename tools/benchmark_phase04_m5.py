"""Phase 0.4 M5 / ADR-121 — batch observation builder benchmark.

Compares three paths:
  1. per-state loop calling `ObservationBuilder.build()` + snapshot
  2. per-state loop calling `ObservationBuilder.build_into(...)` into a
     preallocated batch slab
  3. one call to `ObservationBuilder.build_observations_batch(...)`

The target is 64-way obs build in < 15 ms (PHASE_0.4_PERF_TODO §6.3).
"""
from __future__ import annotations
import random
import sys
from time import perf_counter

import numpy as np

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import (
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationBuilder,
)
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState

BOARD_SIZE = 17
N = 64
ITERS = 50


def make_fixture(
    n: int,
) -> tuple[list[GameState], list[BeliefTensor], list[Seat]]:
    rng = random.Random(0xB107)
    states: list[GameState] = []
    beliefs: list[BeliefTensor] = []
    observers: list[Seat] = []
    for i in range(n):
        r = random.Random(rng.random())
        state = GameState.new_game(
            generate_random_setup(r), show_mode=ShowMode.HALF_DARK
        )
        # Advance a handful of steps so per-state data is realistically varied.
        for _ in range(i % 20):
            if state.terminated:
                break
            la = state.legal_actions()
            if not la:
                break
            state, _ = state.step(r.choice(la))
        observer = ALL_SEATS[i % 4]
        if state.info[observer].dead:
            observer = state.turn
        belief = BeliefTensor.initial(state, observer)
        states.append(state)
        beliefs.append(belief)
        observers.append(observer)
    return states, beliefs, observers


def main() -> int:
    states, beliefs, observers = make_fixture(N)
    builder = ObservationBuilder()

    # Warmup.
    for i in range(N):
        builder.build(states[i], beliefs[i], observers[i]).snapshot()

    # 1) per-state .build() + snapshot (detached copies).
    t0 = perf_counter()
    for _ in range(ITERS):
        for i in range(N):
            builder.build(states[i], beliefs[i], observers[i]).snapshot()
    dt_build_snap = (perf_counter() - t0) / ITERS * 1e3  # ms per batch

    # 2) per-state build_into into slab.
    slab_s = np.zeros(
        (N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
    )
    slab_g = np.zeros((N, OBS_GLOBAL_DIMS), dtype=np.float32)
    t0 = perf_counter()
    for _ in range(ITERS):
        for i in range(N):
            builder.build_into(
                states[i], beliefs[i], observers[i], slab_s[i], slab_g[i],
            )
    dt_build_into = (perf_counter() - t0) / ITERS * 1e3

    # 3) build_observations_batch (one call).
    slab_s.fill(0.0); slab_g.fill(0.0)
    t0 = perf_counter()
    for _ in range(ITERS):
        builder.build_observations_batch(
            states, beliefs, observers, slab_s, slab_g,
        )
    dt_batch = (perf_counter() - t0) / ITERS * 1e3

    print(f"batch N={N} iters={ITERS} show_mode=HALF_DARK")
    print(f"  per-state .build()+snapshot : {dt_build_snap:.2f} ms/batch "
          f"({N / dt_build_snap * 1e3:.0f} obs/s)")
    print(f"  per-state build_into        : {dt_build_into:.2f} ms/batch "
          f"({N / dt_build_into * 1e3:.0f} obs/s)")
    print(f"  build_observations_batch    : {dt_batch:.2f} ms/batch "
          f"({N / dt_batch * 1e3:.0f} obs/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
