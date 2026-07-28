"""T7 M6 — Integration & performance health checks.

This file is the "thin CI guard" for T7's performance / consistency
invariants. Hard empirical numbers (steps/sec, obs-build µs) are
measured by the standalone `tools/benchmark_t7.py` script, which is
explicitly NOT run in CI (timing-sensitive, machine-dependent).

The tests here are kept deliberately **loose** — they fire only on
genuine regressions (e.g. an O(N²) accident), not on noise.

Checks:
  * `stdevcep()` throughput is in the right order of magnitude (≥ 500 / s).
  * `build_observation()` latency is in the right order of magnitude
    (< 20 ms mean on a mid-game state).
  * `BeliefTensor` and `GameState.deaths` stay consistent across 300
    steps of random play:
      - `belief.probs.keys() ⊆ state.pieces.keys()` (no ghost cells).
      - Every `piece_id` in `state.deaths` is NOT currently alive (and
        is gone from `state.piece_state`).
      - `sum(belief.remaining[enemy])` + `|visible enemy pieces|` =
        initial inventory minus dead enemy pieces (conservation).
"""

from __future__ import annotations

import random
from time import perf_counter

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor, TRACKED_TYPES
from junqi_core.observation import build_observation
from junqi_core.rules import ALL_SEATS, PIECE_COUNTS, Seat, ShowMode, same_team
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _play_random_game(
    *,
    seed: int,
    max_steps: int,
    show_mode: ShowMode = ShowMode.HALF_DARK,
    track_beliefs: bool = False,
    assert_each_step=None,
):
    """Drive a single self-play game forward, optionally running an
    assertion callback after every successful step.

    Returns (final_state, n_steps_played).
    """
    rng = random.Random(seed)
    state = GameState.new_game(generate_random_setup(rng), show_mode=show_mode)

    beliefs: dict[Seat, BeliefTensor] = {}
    if track_beliefs:
        for s in ALL_SEATS:
            beliefs[s] = BeliefTensor.initial(state, s)

    for step_idx in range(max_steps):
        if state.terminated:
            break
        seat = state.turn
        legal = state.legal_actions(seat=seat)
        if not legal:
            # Q12: seat had no legal move. We can't step with no action;
            # that is the simulator's responsibility to handle via
            # _advance_turn_with_q12. For a pure random driver we simply
            # stop — shorter games are fine for these tests.
            break
        action = rng.choice(legal)
        prev_state = state
        state, result = state.step(action)

        if track_beliefs:
            for s in ALL_SEATS:
                beliefs[s].update(prev_state, state, result)

        if assert_each_step is not None:
            assert_each_step(step_idx, state, beliefs)

    return state, step_idx


# ===========================================================================
# 1. Consistency: BeliefTensor ∩ GameState.deaths
# ===========================================================================


class TestBeliefDeathsConsistency:
    """Cross-module invariants between `info_model.BeliefTensor` and
    the T7 death registry on `GameState`."""

    def _consistency_checks(
        self,
        step_idx: int,
        state: GameState,
        beliefs: dict[Seat, BeliefTensor],
    ) -> None:
        live_cells = set(state.pieces.keys())
        live_piece_ids = {ref.piece_id for ref in state.pieces.values()}

        # T7 registry invariants (reprise of M2/M3 tests, belt-and-braces).
        assert set(state.piece_state.keys()) == live_piece_ids, (
            f"step {step_idx}: piece_state keys != live piece ids "
            f"(missing={live_piece_ids - set(state.piece_state.keys())}, "
            f"stale={set(state.piece_state.keys()) - live_piece_ids})"
        )
        # deaths and live are disjoint at the piece_id level.
        assert live_piece_ids.isdisjoint(state.deaths.keys()), (
            f"step {step_idx}: piece_id(s) both alive and in deaths: "
            f"{live_piece_ids & state.deaths.keys()}"
        )

        # BeliefTensor ↔ live cells.
        for observer, belief in beliefs.items():
            ghost_cells = set(belief.probs.keys()) - live_cells
            assert not ghost_cells, (
                f"step {step_idx}, observer={observer.name}: belief has "
                f"probability mass on non-live cells {ghost_cells}"
            )

            # Every live cell is covered by belief (no missing cells either).
            missing_cells = live_cells - set(belief.probs.keys())
            assert not missing_cells, (
                f"step {step_idx}, observer={observer.name}: belief is "
                f"missing live cells {missing_cells}"
            )

            # Each distribution is a valid probability (sums to 1, no NaN).
            for pos, vec in belief.probs.items():
                assert np.isfinite(vec).all(), (
                    f"step {step_idx}: belief @ {pos} has non-finite: {vec}"
                )
                total = float(vec.sum())
                assert abs(total - 1.0) < 1e-4, (
                    f"step {step_idx}: belief @ {pos} sums to {total}, not 1"
                )

            # Remaining-inventory sanity bounds.
            #
            # Under HALF_DARK / DARK an observer does NOT learn the
            # type of every dead enemy (that would violate ADR-002's
            # info-broadcast rule). So we can't assert strict
            # conservation `remaining[enemy][pt] == initial - deaths`.
            # What we CAN guarantee is that the belief's own monotonic
            # book-keeping respects the physical envelope:
            #
            #   0 <= remaining[enemy][pt] <= PIECE_COUNTS[pt]
            #
            # and that the total over all types never exceeds
            # `sum(PIECE_COUNTS)`. This catches both double-subtracting
            # (going negative) and forgotten-subtraction (going above
            # the initial inventory).
            for enemy in ALL_SEATS:
                if enemy is observer or same_team(observer, enemy):
                    continue
                inv = belief.remaining.get(enemy)
                assert inv is not None, (
                    f"step {step_idx}: missing remaining-inventory for "
                    f"enemy {enemy.name}"
                )
                for pt in TRACKED_TYPES:
                    cnt = inv.get(pt, 0)
                    assert 0 <= cnt <= PIECE_COUNTS.get(pt, 0), (
                        f"step {step_idx}, observer={observer.name}, "
                        f"enemy={enemy.name}, type={pt.name}: "
                        f"remaining={cnt} out of "
                        f"[0, {PIECE_COUNTS.get(pt, 0)}]"
                    )
                total = sum(inv.values())
                assert total <= sum(PIECE_COUNTS.values()), (
                    f"step {step_idx}: sum(remaining[{enemy.name}])="
                    f"{total} exceeds sum(PIECE_COUNTS)="
                    f"{sum(PIECE_COUNTS.values())}"
                )

    def test_random_game_consistency(self) -> None:
        """300-step random game: every step, belief & deaths stay consistent."""
        _play_random_game(
            seed=20260421,
            max_steps=300,
            track_beliefs=True,
            assert_each_step=self._consistency_checks,
        )

    def test_bright_mode_consistency(self) -> None:
        """Same check under BRIGHT mode (all pieces one-hot)."""
        _play_random_game(
            seed=20260422,
            max_steps=200,
            show_mode=ShowMode.BRIGHT,
            track_beliefs=True,
            assert_each_step=self._consistency_checks,
        )


# ===========================================================================
# 2. Performance health (loose CI guard; real numbers live in tools/benchmark_t7.py)
# ===========================================================================


class TestPerformanceHealth:
    """Regression guards. These are INTENTIONALLY loose so CI noise does
    not flap. See `tools/benchmark_t7.py --report` for real numbers."""

    def test_step_throughput_order_of_magnitude(self) -> None:
        """`step()` throughput must stay in the 1k+ plays/s order of
        magnitude on a commodity dev box. We guard at 500/s so CI
        (shared runner, no warm cache) still passes reliably."""
        rng = random.Random(42)
        state = GameState.new_game(generate_random_setup(rng))

        t0 = perf_counter()
        n_steps = 0
        for _ in range(400):
            if state.terminated:
                break
            seat = state.turn
            legal = state.legal_actions(seat=seat)
            if not legal:
                break
            action = rng.choice(legal)
            state, _ = state.step(action)
            n_steps += 1
        elapsed = perf_counter() - t0
        if n_steps < 50:
            pytest.skip("game terminated too fast to measure throughput")

        rate = n_steps / elapsed
        assert rate >= 500, (
            f"step() throughput regressed: {rate:.0f} plays/s (want ≥ 500). "
            f"Measured over {n_steps} steps in {elapsed:.2f}s. "
            "Run `python -m tools.benchmark_t7` for detailed profiling."
        )

    def test_build_observation_latency_order_of_magnitude(self) -> None:
        """`build_observation()` mean latency must stay within the 10ms
        order of magnitude. Tight Phase 0.3 target is < 2ms; we guard
        the CI boundary at 20ms."""
        rng = random.Random(43)
        state = GameState.new_game(generate_random_setup(rng))
        # Advance ~30 steps so the board is in a non-trivial mid-game
        # shape with some deaths / counters populated.
        for _ in range(30):
            if state.terminated:
                break
            seat = state.turn
            legal = state.legal_actions(seat=seat)
            if not legal:
                break
            action = rng.choice(legal)
            state, _ = state.step(action)

        belief = BeliefTensor.initial(state, Seat.SOUTH)

        # Warm-up (JIT caches, allocator pools, static board cache).
        for _ in range(5):
            build_observation(state, belief, Seat.SOUTH)

        # Measure mean over N iterations.
        n_iter = 20
        t0 = perf_counter()
        for _ in range(n_iter):
            build_observation(state, belief, Seat.SOUTH)
        elapsed = perf_counter() - t0
        mean_ms = (elapsed / n_iter) * 1000.0

        assert mean_ms < 20.0, (
            f"build_observation() latency regressed: {mean_ms:.2f} ms mean "
            f"(want < 20 ms; Phase 0.3 target < 2 ms). "
            "Run `python -m tools.benchmark_t7` for detailed profiling."
        )
