"""Public board events must reach each player's actual policy observation."""

import json
import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor, TRACKED_TYPES
from junqi_core.move_gen import PieceRef
from junqi_core.observation import CHANNEL_LAYOUT, ObservationBuilder
from junqi_core.rotation import canonical_to_world, world_to_canonical
from junqi_core.rules import PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup, validate_setup
from junqi_core.state import Action, GameState, SeatInfo


def commander_position(owner=Seat.SOUTH, mode=ShowMode.DARK):
    """Legal commander-on-mine action, with real flags and surviving movers."""
    base = {
        (2, 7): (Seat.SOUTH, PieceType.SILING),
        (1, 7): (Seat.WEST, PieceType.DILEI),
        (10, 11): (Seat.SOUTH, PieceType.PAIZH),
        (5, 6): (Seat.WEST, PieceType.PAIZH),
        (10, 5): (Seat.NORTH, PieceType.PAIZH),
        (11, 8): (Seat.EAST, PieceType.PAIZH),
        (7, 16): (Seat.SOUTH, PieceType.JUNQI),
        (9, 16): (Seat.SOUTH, PieceType.LIANZH),
        (0, 7): (Seat.WEST, PieceType.JUNQI),
        (9, 0): (Seat.NORTH, PieceType.JUNQI),
        (16, 9): (Seat.EAST, PieceType.JUNQI),
    }
    pieces = {
        canonical_to_world(*pos, owner): PieceRef(Seat((seat.value + owner.value) % 4), typ)
        for pos, (seat, typ) in base.items()
    }
    state = GameState(
        pieces=pieces, turn=owner, move_counter=0, moves_since_last_combat=0,
        info={seat: SeatInfo() for seat in Seat}, terminated=False,
        winner_team=None, draw=False, show_mode=mode,
    )
    action = Action(owner, canonical_to_world(2, 7, owner), canonical_to_world(1, 7, owner))
    return state, action, canonical_to_world(7, 16, owner)


@pytest.mark.parametrize("mode", [ShowMode.DARK, ShowMode.HALF_DARK])
@pytest.mark.parametrize("owner", list(Seat))
@pytest.mark.parametrize("observer", list(Seat))
def test_public_flag_reaches_policy_and_restored_observation(owner, observer, mode):
    state, action, flag_pos = commander_position(owner, mode)
    belief = BeliefTensor.initial(state, observer)
    after, result = state.step(action)
    assert result.flag_reveal_src
    assert after.info[owner].flag_revealed
    assert after.pieces[flag_pos].piece_type is PieceType.JUNQI
    belief.update(state, after, result)

    # Use the real observation builder, including the observer's rotation.
    builder = ObservationBuilder()
    for current, current_belief in (
        (after, belief),
        (GameState.from_dict(json.loads(json.dumps(after.to_dict()))), None),
    ):
        current_belief = current_belief or BeliefTensor.initial(current, observer)
        obs = builder.build(current, current_belief, observer)
        x, y = world_to_canonical(*flag_pos, observer)
        key = (
            "piece_own" if owner is observer else
            "prob_teammate" if owner is observer.teammate else
            "belief_left_side" if owner is observer.left_side_enemy else
            "belief_right_side"
        )
        assert obs.spatial[CHANNEL_LAYOUT[key].start, y, x] == 1.0
        for pos, piece in current.pieces.items():
            if piece.seat is owner and pos != flag_pos:
                assert current_belief.probs[pos][TRACKED_TYPES.index(PieceType.JUNQI)] == 0.0


@pytest.mark.parametrize("observer", list(Seat))
@pytest.mark.parametrize("mode", [ShowMode.DARK, ShowMode.HALF_DARK])
def test_hidden_lineup_permutations_preserve_all_policy_inputs(observer, mode):
    """Two legal worlds with identical visible openings, not just low entropy."""
    builder = ObservationBuilder()
    for seed in range(6):
        setups = generate_random_setup(random.Random(seed))
        altered = [list(lineup) for lineup in setups]
        for seat in Seat:
            if seat is observer or (mode is ShowMode.HALF_DARK and seat is observer.teammate):
                continue
            # Ordinary ranks share placement and movement rules. Preserve counts.
            indices = [i for i, t in enumerate(altered[seat.value]) if PieceType.SILING <= t <= PieceType.PAIZH]
            vals = [altered[seat.value][i] for i in indices]
            for i, val in zip(indices, vals[1:] + vals[:1], strict=True):
                altered[seat.value][i] = val
        assert validate_setup(altered).ok
        a = GameState.new_game(setups, show_mode=mode, first_seat=observer)
        b = GameState.new_game(altered, show_mode=mode, first_seat=observer)
        ba, bb = [BeliefTensor.initial(s, observer) for s in (a, b)]
        # Drive actual legal transitions in both worlds. Keep only actions with
        # the same public result; hidden types are allowed to differ throughout.
        accepted = 0
        for _ in range(32):
            oa, ob = builder.build(a, ba, observer), builder.build(b, bb, observer)
            np.testing.assert_array_equal(oa.spatial, ob.spatial)
            np.testing.assert_array_equal(oa.global_, ob.global_)
            np.testing.assert_array_equal(a.legal_action_ids(observer), b.legal_action_ids(observer))
            if a.terminated or b.terminated:
                break
            legal_b = set(b.legal_actions())
            choices = [act for act in a.legal_actions() if act in legal_b]
            random.Random(seed * 100 + accepted).shuffle(choices)
            for act in choices:
                na, ra = a.step(act)
                nb, rb = b.step(act)
                if ra != rb or na.terminated != nb.terminated:
                    continue
                # Include public flag locations, which are meaningful once revealed.
                public_a = {(p.seat, pos) for pos, p in na.pieces.items() if p.piece_type is PieceType.JUNQI and na.info[p.seat].flag_revealed}
                public_b = {(p.seat, pos) for pos, p in nb.pieces.items() if p.piece_type is PieceType.JUNQI and nb.info[p.seat].flag_revealed}
                if public_a != public_b:
                    continue
                ba.update(a, na, ra)
                bb.update(b, nb, rb)
                a, b = na, nb
                accepted += 1
                break
            else:
                break
        assert accepted >= 16, "fixture must exercise a substantial shared public history"
