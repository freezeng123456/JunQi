"""End-to-end rollout benchmark (Phase 0.4 M4 acceptance).

Drives two rollout modes so we can see the effect of the ADR-119 flat
action-id API independently from ``step()`` overhead.

    legacy  — ``state.legal_actions()`` returning List[Action]
    ids     — ``state.legal_action_ids()`` returning ndarray[int32]

Writes a compact report to ``/tmp/m4_rollout.log`` and prints headline
numbers to stdout.
"""
from __future__ import annotations
import random
import sys
from pathlib import Path
from time import perf_counter

from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState

BOARD_SIZE = 17
NUM_CELLS = 289


def rollout_legacy(seed: int, max_plays: int = 300) -> int:
    rng = random.Random(seed)
    state = GameState.new_game(generate_random_setup(rng))
    plays = 0
    while plays < max_plays and not state.terminated:
        la = state.legal_actions()
        if not la:
            break
        a = rng.choice(la)
        state, _ = state.step(a)
        plays += 1
    return plays


def rollout_ids(seed: int, max_plays: int = 300) -> int:
    rng = random.Random(seed)
    state = GameState.new_game(generate_random_setup(rng))
    plays = 0
    while plays < max_plays and not state.terminated:
        ids = state.legal_action_ids()
        if ids.size == 0:
            break
        flat_id = int(ids[rng.randrange(ids.size)])
        src_flat = flat_id // NUM_CELLS
        dst_flat = flat_id %  NUM_CELLS
        src = (src_flat %  BOARD_SIZE, src_flat // BOARD_SIZE)
        dst = (dst_flat %  BOARD_SIZE, dst_flat // BOARD_SIZE)
        a = Action(seat=state.turn, src=src, dst=dst)
        state, _ = state.step(a)
        plays += 1
    return plays


def bench_legal_actions() -> tuple[float, float]:
    """Return (opening_us_per_call_batch, opening_us_per_call_legacy)."""
    rng = random.Random(0)
    state = GameState.new_game(generate_random_setup(rng))
    seat_val = state.turn.value

    from junqi_core import move_gen

    for _ in range(200):
        state.legal_action_ids()  # warmup batch
        state.legal_actions()

    N = 1000
    t0 = perf_counter()
    for _ in range(N):
        state.legal_action_ids()
    batch_us = (perf_counter() - t0) / N * 1e6

    # Legacy dict version still available via raw move_gen API.
    t0 = perf_counter()
    for _ in range(N):
        move_gen.generate_legal_actions(state.pieces, state.turn)
    legacy_us = (perf_counter() - t0) / N * 1e6

    return batch_us, legacy_us


def main() -> int:
    # Warmup.
    rollout_legacy(0, 50)
    rollout_ids(0, 50)

    N = 20
    results: list[str] = []
    for label, fn in (("legacy", rollout_legacy), ("ids", rollout_ids)):
        total_plays = 0
        t0 = perf_counter()
        for i in range(N):
            total_plays += fn(i, 300)
        dt = perf_counter() - t0
        line = (f"{label:7s} {N} rollouts, {total_plays:5d} plays in "
                f"{dt:.2f}s -> {total_plays / dt:.0f} plays/s")
        print(line)
        results.append(line)

    b_us, l_us = bench_legal_actions()
    line = (f"legal_action_ids (batch) : {b_us:.1f} us/call "
            f"= {1e6/b_us:.0f}/s")
    print(line); results.append(line)
    line = (f"generate_legal_actions   : {l_us:.1f} us/call "
            f"= {1e6/l_us:.0f}/s  (legacy PieceMap path)")
    print(line); results.append(line)

    Path("/tmp/m4_rollout.log").write_text("\n".join(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
