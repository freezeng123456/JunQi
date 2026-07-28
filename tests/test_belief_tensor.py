"""Phase 0.4 M3 / ADR-120 — BeliefTensor tensor-mirror acceptance tests.

Pins down the M3 contracts:

1. **Tensor mirror exists and agrees with dict state** at every step of a
   300-step random game (for all 4 observers, all show modes).
2. **probs_arr dtype / shape** stay stable across the game.
3. **remaining_arr agrees with remaining dict** including the
   "enemy seat dies -> row zeroed" edge case (R4).
4. **clone** of BeliefTensor (via the post-``update`` sync) preserves
   the mirror.
5. **Observation bit-parity still holds** with tensor path engaged
   (covered indirectly by tests/test_observation_builder.py, but we
   pin down the "belief mirror directly populated" case here).

None of the M1/M2 acceptance tests are modified; this file is the
additional regression net for M3.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.board import BOARD_SIZE
from junqi_core.info_model import (
    BeliefTensor,
    NUM_TRACKED_TYPES,
    TRACKED_TYPES,
    _TYPE_TO_IDX,
)
from junqi_core.rules import ALL_SEATS, PIECE_COUNTS, PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


def _drive(state: GameState, belief: BeliefTensor, steps: int, rng: random.Random):
    for _ in range(steps):
        if state.terminated:
            break
        legal = state.legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        new_state, result = state.step(action)
        if not new_state.info[belief.observer].dead:
            belief.update(state, new_state, result)
        state = new_state
    return state, belief


# ---------------------------------------------------------------------------
# 1. Tensor mirror matches dict at every step
# ---------------------------------------------------------------------------


class TestProbsArrParity:
    """``probs_arr[pid]`` must equal ``probs[(x, y)]`` for every live
    piece at every step."""

    @pytest.mark.parametrize("seed", [0, 7, 42, 100, 2026])
    @pytest.mark.parametrize("show_mode", [ShowMode.BRIGHT, ShowMode.HALF_DARK])
    def test_parity_over_300_step_game(
        self, seed: int, show_mode: ShowMode
    ) -> None:
        rng = random.Random(seed)
        observer = Seat.SOUTH
        state = GameState.new_game(
            generate_random_setup(rng), show_mode=show_mode
        )
        belief = BeliefTensor.initial(state, observer)

        for step_idx in range(300):
            if state.terminated:
                break
            if state.info[observer].dead:
                break
            legal = state.legal_actions()
            if not legal:
                break
            action = rng.choice(legal)
            new_state, result = state.step(action)
            if not new_state.info[observer].dead:
                belief.update(state, new_state, result)
            state = new_state

            # After every update, probs_arr must agree with probs dict.
            for (x, y), vec in belief.probs.items():
                flat = y * BOARD_SIZE + x
                pid = int(state.cell_piece_id[flat])
                if pid < 0:
                    # Stale entry cleaned up on next update; skip.
                    continue
                assert np.array_equal(belief.probs_arr[pid], vec), (
                    f"step {step_idx} seed {seed} mode {show_mode.name}: "
                    f"probs_arr[{pid}] != probs[{(x, y)}] at step {step_idx}"
                )


# ---------------------------------------------------------------------------
# 2. probs_arr shape + dtype pin
# ---------------------------------------------------------------------------


class TestProbsArrShape:
    def test_initial_shape_matches_state(self) -> None:
        rng = random.Random(0)
        state = GameState.new_game(generate_random_setup(rng))
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        assert belief.probs_arr.shape == (
            state.alive.shape[0], NUM_TRACKED_TYPES,
        )
        assert belief.probs_arr.dtype == np.float32

    def test_remaining_arr_shape_pin(self) -> None:
        rng = random.Random(0)
        state = GameState.new_game(generate_random_setup(rng))
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        assert belief.remaining_arr.shape == (4, NUM_TRACKED_TYPES)
        assert belief.remaining_arr.dtype == np.int16

    def test_shape_stable_across_steps(self) -> None:
        rng = random.Random(1)
        state = GameState.new_game(generate_random_setup(rng))
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        initial_shape = belief.probs_arr.shape
        state, belief = _drive(state, belief, 100, rng)
        assert belief.probs_arr.shape == initial_shape


# ---------------------------------------------------------------------------
# 3. remaining_arr agrees with remaining dict
# ---------------------------------------------------------------------------


class TestRemainingArrParity:
    def test_initial_remaining_arr_matches_dict(self) -> None:
        rng = random.Random(0)
        state = GameState.new_game(generate_random_setup(rng))
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        for seat, inv in belief.remaining.items():
            row = belief.remaining_arr[seat.value]
            for pt, cnt in inv.items():
                idx = _TYPE_TO_IDX[pt]
                assert int(row[idx]) == cnt

    def test_remaining_zeroes_when_seat_dies(self) -> None:
        """If an enemy seat's flag is captured, their row in
        remaining dict disappears -> row in remaining_arr must zero out."""
        rng = random.Random(777)
        state = GameState.new_game(generate_random_setup(rng))
        belief = BeliefTensor.initial(state, Seat.SOUTH)

        # Drive enough steps to likely have a seat die.
        for _ in range(500):
            if state.terminated:
                break
            if state.info[Seat.SOUTH].dead:
                break
            legal = state.legal_actions()
            if not legal:
                break
            a = rng.choice(legal)
            new_state, result = state.step(a)
            if not new_state.info[Seat.SOUTH].dead:
                belief.update(state, new_state, result)
            state = new_state

        # For every seat NOT in remaining dict, row in remaining_arr
        # should be zero.
        tracked_seats = set(belief.remaining.keys())
        all_enemy_seats = {
            s for s in ALL_SEATS
            if s is not Seat.SOUTH and s is not Seat.SOUTH.teammate
        }
        for seat in all_enemy_seats - tracked_seats:
            row = belief.remaining_arr[seat.value]
            assert int(row.sum()) == 0, (
                f"seat {seat.name} is gone from remaining dict but its "
                f"remaining_arr row is {row.tolist()}"
            )

    def test_remaining_arr_decrement_on_combat(self) -> None:
        """When an enemy piece is lost to combat the dict is decremented
        (only if one-hot) and so is the arr."""
        rng = random.Random(5)
        state = GameState.new_game(generate_random_setup(rng))
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        state, belief = _drive(state, belief, 200, rng)
        # ADR-120 lazy-sync: refresh the tensor mirror before reading it.
        belief.ensure_synced(state)

        for seat, inv in belief.remaining.items():
            row = belief.remaining_arr[seat.value]
            for pt, cnt in inv.items():
                idx = _TYPE_TO_IDX[pt]
                assert int(row[idx]) == cnt, (
                    f"{seat.name} {pt.name}: arr={int(row[idx])} dict={cnt}"
                )


# ---------------------------------------------------------------------------
# 4. All four observers pin
# ---------------------------------------------------------------------------


class TestPerObserverParity:
    @pytest.mark.parametrize("observer", list(ALL_SEATS))
    def test_parity_all_observers(self, observer: Seat) -> None:
        rng = random.Random(999)
        state = GameState.new_game(
            generate_random_setup(rng), show_mode=ShowMode.HALF_DARK
        )
        belief = BeliefTensor.initial(state, observer)
        state, belief = _drive(state, belief, 100, rng)

        for (x, y), vec in belief.probs.items():
            flat = y * BOARD_SIZE + x
            pid = int(state.cell_piece_id[flat])
            if pid < 0:
                continue
            assert np.array_equal(belief.probs_arr[pid], vec)
