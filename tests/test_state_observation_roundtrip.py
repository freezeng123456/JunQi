"""A midgame snapshot must preserve the inputs consumed by the policy."""
from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import ObservationBuilder
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


def _midgame(steps: int = 5):
    seed = 913101
    state = GameState.new_game(
        generate_random_setup(random.Random(seed)), show_mode=ShowMode.DARK
    )
    beliefs = {seat: BeliefTensor.initial(state, seat) for seat in Seat}
    actions_rng = random.Random(seed)
    for _ in range(steps):
        assert not state.terminated
        after, result = state.step(actions_rng.choice(state.legal_actions()))
        for belief in beliefs.values():
            belief.update(state, after, result)
        state = after
    assert state.deaths, "the reproduction must exercise a real combat death"
    return state, beliefs


@pytest.mark.parametrize("steps", [5, 40])
def test_recent_move_history_survives_json_round_trip(steps):
    state, _ = _midgame(steps)
    restored = GameState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert len(state.move_history) == min(steps, 32)
    assert restored.move_history == state.move_history
    assert restored.move_history is not state.move_history
    assert all(isinstance(move, tuple) for move in restored.move_history)


def test_dead_piece_types_survive_json_round_trip():
    state, _ = _midgame()
    restored = GameState.from_dict(json.loads(json.dumps(state.to_dict())))

    dead_pids = list(state.deaths)
    assert np.all(state.piece_type_arr[dead_pids] >= 0)
    np.testing.assert_array_equal(restored.piece_type_arr, state.piece_type_arr)


@pytest.mark.parametrize("steps", [5, 40])
def test_policy_inputs_survive_midgame_snapshot_and_next_action(steps):
    state, beliefs = _midgame(steps)
    restored = GameState.from_dict(json.loads(json.dumps(state.to_dict())))
    restored_beliefs = copy.deepcopy(beliefs)

    for continuation in range(2):
        for observer in Seat:
            if state.seat_dead_arr[observer.value]:
                continue
            expected = ObservationBuilder().build(state, beliefs[observer], observer)
            actual = ObservationBuilder().build(
                restored, restored_beliefs[observer], observer
            )
            np.testing.assert_array_equal(actual.spatial, expected.spatial)
            np.testing.assert_array_equal(actual.global_, expected.global_)
            np.testing.assert_array_equal(
                restored.legal_action_ids(observer), state.legal_action_ids(observer)
            )
        if continuation == 0:
            action = state.legal_actions()[0]
            after, result = state.step(action)
            restored_after, restored_result = restored.step(action)
            assert restored_result == result
            for observer in Seat:
                beliefs[observer].update(state, after, result)
                restored_beliefs[observer].update(
                    restored, restored_after, restored_result
                )
            state, restored = after, restored_after


def test_older_snapshot_without_move_history_still_loads():
    state, _ = _midgame()
    payload = state.to_dict()
    payload.pop("move_history", None)

    restored = GameState.from_dict(json.loads(json.dumps(payload)))

    assert restored.move_history == []
    assert restored.move_counter == state.move_counter
    assert restored.deaths == state.deaths
