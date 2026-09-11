"""Observations must depend on information available to their observer."""
import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor
from junqi_core.move_gen import PieceRef
from junqi_core.observation import build_observation, CHANNEL_LAYOUT, GLOBAL_LAYOUT
from junqi_core.rules import Event, PieceType, Seat, ShowMode
from junqi_core.state import Action, GameState, SeatInfo


def position(attacker, defender):
    pieces = {
        (2, 7): PieceRef(Seat.SOUTH, attacker),
        (1, 7): PieceRef(Seat.WEST, defender),
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    return GameState(pieces=pieces, turn=Seat.SOUTH, move_counter=0,
                     moves_since_last_combat=0, info={s: SeatInfo() for s in Seat},
                     terminated=False, winner_team=None, draw=False, show_mode=ShowMode.DARK)


def transition(attacker, defender, observer):
    state = position(attacker, defender)
    belief = BeliefTensor.initial(state, observer)
    before = build_observation(state, belief, observer)
    after, result = state.step(Action(seat=Seat.SOUTH, src=(2, 7), dst=(1, 7)))
    belief.update(state, after, result)
    return before, build_observation(after, belief, observer), belief, result


def test_hidden_mine_capture_does_not_reveal_engineer():
    mine = transition(PieceType.GONGB, PieceType.DILEI, Seat.NORTH)
    ordinary = transition(PieceType.LIANZH, PieceType.PAIZH, Seat.NORTH)
    assert mine[3].event is ordinary[3].event is Event.EAT
    np.testing.assert_array_equal(mine[0].spatial, ordinary[0].spatial)
    np.testing.assert_allclose(mine[2].probs[(1, 7)], ordinary[2].probs[(1, 7)])
    np.testing.assert_array_equal(mine[1].spatial, ordinary[1].spatial)
    np.testing.assert_array_equal(mine[1].global_, ordinary[1].global_)


def test_hidden_defender_does_not_disclose_private_death_cause():
    mine = transition(PieceType.PAIZH, PieceType.DILEI, Seat.SOUTH)
    ordinary = transition(PieceType.PAIZH, PieceType.LIANZH, Seat.SOUTH)
    assert mine[3].event is ordinary[3].event is Event.KILLED
    np.testing.assert_array_equal(mine[0].spatial, ordinary[0].spatial)
    np.testing.assert_array_equal(mine[1].spatial, ordinary[1].spatial)
    np.testing.assert_array_equal(mine[1].global_, ordinary[1].global_)


def test_global_inventory_counts_live_belief_mass():
    state = position(PieceType.PAIZH, PieceType.LIANZH)
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    obs = build_observation(state, belief, Seat.SOUTH)
    for key in ('remaining_left_side', 'remaining_right_side'):
        # Each opposing seat has two (WEST) or one (EAST) surviving pieces.
        seat = (Seat.SOUTH.left_side_enemy if key == 'remaining_left_side'
                else Seat.SOUTH.right_side_enemy)
        expected = sum((belief.probs[pos] for pos, piece in state.pieces.items()
                        if piece.seat is seat), np.zeros(12))
        np.testing.assert_allclose(obs.global_[GLOBAL_LAYOUT[key]], expected)


def test_own_mine_candidate_does_not_read_opponent_private_engineer():
    observations = []
    for attacker in (PieceType.GONGB, PieceType.PAIZH):
        state = position(PieceType.LIANZH, PieceType.PAIZH)
        pieces = dict(state.pieces)
        del pieces[(2, 7)]
        del pieces[(1, 7)]
        pieces[(6, 14)] = PieceRef(Seat.WEST, attacker)
        pieces[(6, 15)] = PieceRef(Seat.SOUTH, PieceType.LIANZH)
        state = GameState(pieces=pieces, turn=Seat.WEST, move_counter=0,
                          moves_since_last_combat=0, info={s: SeatInfo() for s in Seat},
                          terminated=False, winner_team=None, draw=False, show_mode=ShowMode.DARK)
        # A stationary defender started here; hand-built positions have no zero_board.
        pid = state.pieces[(6, 15)].piece_id
        state.zero_x[pid], state.zero_y[pid] = 6, 15
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        before = build_observation(state, belief, Seat.SOUTH)
        assert before.spatial[CHANNEL_LAYOUT["cm_my_dilei_candidate"], 15, 6].item() == 1.0
        after, result = state.step(Action(seat=Seat.WEST, src=(6, 14), dst=(6, 15)))
        assert result.event is Event.KILLED
        belief.update(state, after, result)
        observations.append((before, build_observation(after, belief, Seat.SOUTH)))
    np.testing.assert_array_equal(observations[0][0].spatial, observations[1][0].spatial)
    np.testing.assert_array_equal(observations[0][1].spatial, observations[1][1].spatial)
    np.testing.assert_array_equal(observations[0][1].global_, observations[1][1].global_)


def test_native_rollout_rejects_unsupported_visibility_before_allocation():
    from junqi_rl.gpu_rollout import GpuRollout
    with pytest.raises(ValueError, match='DARK only'):
        GpuRollout(1, show_mode=ShowMode.HALF_DARK)


def test_public_engineer_attack_removes_own_mine_candidate():
    state = position(PieceType.PAIZH, PieceType.LIANZH)
    pid = state.pieces[(2, 7)].piece_id
    # Project a stationary back-row identity with public attacked-by-engineer memory.
    state.zero_x[pid], state.zero_y[pid] = 6, 15
    for opponent in (Seat.SOUTH.left_side_enemy, Seat.SOUTH.right_side_enemy):
        state.combat_memory.attacked_by_known_gongb[opponent.value, pid] = True
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    obs = build_observation(state, belief, Seat.SOUTH)
    assert obs.spatial[CHANNEL_LAYOUT['cm_my_dilei_candidate'], 7, 2].item() == 0.0
