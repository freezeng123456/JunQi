"""Phase 0.4 M6 — BeliefTensor lazy-sync contract (ADR-120 revision)."""
from __future__ import annotations
import random

import numpy as np

from junqi_core.info_model import BeliefTensor
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


def _open_game(seed: int = 0) -> tuple[GameState, BeliefTensor]:
    rng = random.Random(seed)
    state = GameState.new_game(
        generate_random_setup(rng), show_mode=ShowMode.HALF_DARK
    )
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    return state, belief


class TestLazySync:
    def test_initial_is_synced(self) -> None:
        """After BeliefTensor.initial the mirror is fresh — _dirty_state is None."""
        _state, belief = _open_game(0)
        assert belief._dirty_state is None
        # probs_arr should already have non-zero entries for tracked own pieces.
        assert belief.probs_arr.sum() > 0

    def test_update_marks_dirty(self) -> None:
        state, belief = _open_game(0)
        arr_before = belief.probs_arr.copy()
        # Advance one ply; belief.update should be dirty-marked.
        la = state.legal_actions()
        new_state, result = state.step(la[0])
        belief.update(state, new_state, result)
        assert belief._dirty_state is new_state
        # The mirror itself has not been touched yet.
        np.testing.assert_array_equal(belief.probs_arr, arr_before)

    def test_ensure_synced_refreshes(self) -> None:
        state, belief = _open_game(0)
        la = state.legal_actions()
        new_state, result = state.step(la[0])
        belief.update(state, new_state, result)
        assert belief._dirty_state is new_state
        belief.ensure_synced()
        assert belief._dirty_state is None
        # A second call is a no-op.
        belief.ensure_synced()
        assert belief._dirty_state is None

    def test_obs_build_triggers_sync(self) -> None:
        """ObservationBuilder._fill_into must sync belief before reading."""
        from junqi_core.observation import ObservationBuilder

        state, belief = _open_game(0)
        la = state.legal_actions()
        new_state, result = state.step(la[0])
        belief.update(state, new_state, result)
        # Dirty before build.
        assert belief._dirty_state is new_state

        builder = ObservationBuilder()
        builder.build(new_state, belief, Seat.SOUTH)
        # After build, mirror is fresh.
        assert belief._dirty_state is None
