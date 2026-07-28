
"""Tests for junqi_core.info_model — BeliefTensor invariants & inference rules.

Covers:
  - I1/I2: probability vectors sum to 1 and are non-negative
  - I3: own + teammate (Q11) pieces are one-hot in HALF_DARK
  - I3 extended: BRIGHT mode makes everything one-hot
  - I4: dead seats have no belief entries
  - I5: belief keys == live piece positions
  - R1 migration: MOVE / EAT / KILLED / BOMB updates
  - R3 (Q7): SILING-flag-reveal causes remaining-inventory decrement
  - R4 (Q1): flag capture clears the defender's inventory
  - R6: eating a non-flag piece at a stronghold reveals the JUNQI at the other
  - R5/R7: GONGB eats DILEI → attacker revealed as GONGB
  - R9: Q12 seat death purges stale belief
  - to_world_tensor() output shape + content
"""

from __future__ import annotations

import random
from typing import cast

import numpy as np
import pytest

from junqi_core.info_model import (
    BeliefTensor,
    NUM_TRACKED_TYPES,
    TRACKED_TYPES,
    _is_one_hot,
    _observer_sees_truth,
    one_hot,
)
from junqi_core.move_gen import PieceRef
from junqi_core.rules import (
    BACK_TWO_ROWS_INDICES,
    CAMP_INDICES,
    FRONT_ROW_INDICES,
    PIECE_COUNTS,
    STRONGHOLD_INDICES,
    Event,
    PieceType,
    Seat,
    ShowMode,
)
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState, MoveResult, SeatInfo


# ===========================================================================
# Helpers
# ===========================================================================


def _build_state(
    pieces: dict[tuple[int, int], PieceRef],
    turn: Seat = Seat.SOUTH,
    *,
    info: dict[Seat, SeatInfo] | None = None,
    show_mode: ShowMode = ShowMode.HALF_DARK,
) -> GameState:
    if info is None:
        info = {s: SeatInfo() for s in Seat}
    return GameState(
        pieces=dict(pieces),
        turn=turn,
        move_counter=0,
        moves_since_last_combat=0,
        info=info,
        terminated=False,
        winner_team=None,
        draw=False,
        show_mode=show_mode,
    )


def _random_opening(seed: int = 0) -> GameState:
    rng = random.Random(seed)
    return GameState.new_game(generate_random_setup(rng))


# ===========================================================================
# 1. Initial prior invariants (I1–I5)
# ===========================================================================


def test_initial_sums_to_one() -> None:
    st = _random_opening(seed=42)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    for pos, vec in b.probs.items():
        s = float(vec.sum())
        assert abs(s - 1.0) < 1e-5, f"belief at {pos} sums to {s}"
        assert (vec >= 0).all(), f"negative prob at {pos}"


def test_initial_own_pieces_one_hot() -> None:
    """Observer sees own pieces as one-hot."""
    st = _random_opening(seed=1)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    for pos, piece in st.pieces.items():
        if piece.seat is Seat.SOUTH:
            assert _is_one_hot(b.probs[pos]), f"own piece at {pos} not one-hot"
            # And that one-hot points to the true type
            true_idx = TRACKED_TYPES.index(piece.piece_type)
            assert b.probs[pos][true_idx] == pytest.approx(1.0)


def test_initial_teammate_one_hot_in_half_dark() -> None:
    """Q11: HOME sees OPPS (teammate) pieces as one-hot in HALF_DARK."""
    st = _random_opening(seed=2)
    assert st.show_mode is ShowMode.HALF_DARK
    b = BeliefTensor.initial(st, Seat.SOUTH)
    opps_count = 0
    for pos, piece in st.pieces.items():
        if piece.seat is Seat.NORTH:
            assert _is_one_hot(b.probs[pos]), f"teammate at {pos} not one-hot"
            opps_count += 1
    assert opps_count == 25  # every OPPS piece is visible


def test_initial_enemy_pieces_have_distribution() -> None:
    """Enemies' pieces have non-trivial distributions."""
    st = _random_opening(seed=3)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    non_trivial = 0
    for pos, piece in st.pieces.items():
        if piece.seat in (Seat.WEST, Seat.EAST):
            vec = b.probs[pos]
            # True type must have nonzero probability.
            true_idx = TRACKED_TYPES.index(piece.piece_type)
            assert vec[true_idx] > 0, (
                f"enemy at {pos} has zero prob on TRUE type {piece.piece_type.name}"
            )
            if not _is_one_hot(vec):
                non_trivial += 1
    # The vast majority of enemy cells should be truly uncertain
    assert non_trivial >= 20


def test_initial_bright_mode_all_one_hot() -> None:
    """In BRIGHT mode, every piece is known — belief is one-hot everywhere."""
    rng = random.Random(4)
    setups = generate_random_setup(rng)
    st = GameState.new_game(setups, show_mode=ShowMode.BRIGHT)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    for pos, piece in st.pieces.items():
        assert _is_one_hot(b.probs[pos]), (
            f"BRIGHT mode but piece at {pos} not one-hot"
        )


def test_initial_remaining_inventory_only_for_enemies() -> None:
    st = _random_opening(seed=5)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    # Own + teammate → no inventory (fully known)
    assert Seat.SOUTH not in b.remaining
    assert Seat.NORTH not in b.remaining
    # Enemies → full PIECE_COUNTS inventory
    for s in (Seat.WEST, Seat.EAST):
        assert b.remaining[s] == dict(PIECE_COUNTS)


def test_initial_stronghold_has_junqi_probability() -> None:
    """At game start, both strongholds of each enemy seat should assign some
    probability mass to JUNQI (since the flag is in exactly one stronghold
    but we don't know which)."""
    st = _random_opening(seed=6)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    junqi_idx = TRACKED_TYPES.index(PieceType.JUNQI)
    for enemy in (Seat.WEST, Seat.EAST):
        from junqi_core.board import index_to_pos
        for local in STRONGHOLD_INDICES:
            pos = index_to_pos(enemy, local)
            if pos in b.probs:
                assert b.probs[pos][junqi_idx] > 0, (
                    f"stronghold {pos} has 0 JUNQI prob"
                )


def test_initial_dilei_zero_prob_on_front_row() -> None:
    """DILEI is confined to back two rows → front-row cells have zero DILEI prob."""
    st = _random_opening(seed=7)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    dilei_idx = TRACKED_TYPES.index(PieceType.DILEI)
    from junqi_core.board import index_to_pos
    for enemy in (Seat.WEST, Seat.EAST):
        for local in FRONT_ROW_INDICES:
            pos = index_to_pos(enemy, local)
            if pos in b.probs:
                assert b.probs[pos][dilei_idx] == 0, (
                    f"front-row {pos} has nonzero DILEI prob"
                )


def test_initial_zhadan_zero_prob_on_front_row() -> None:
    """ZHADAN never on front row."""
    st = _random_opening(seed=8)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    zhadan_idx = TRACKED_TYPES.index(PieceType.ZHADAN)
    from junqi_core.board import index_to_pos
    for enemy in (Seat.WEST, Seat.EAST):
        for local in FRONT_ROW_INDICES:
            pos = index_to_pos(enemy, local)
            if pos in b.probs:
                assert b.probs[pos][zhadan_idx] == 0


def test_initial_junqi_only_on_strongholds() -> None:
    """JUNQI must be exactly on a stronghold → zero JUNQI prob elsewhere."""
    st = _random_opening(seed=9)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    junqi_idx = TRACKED_TYPES.index(PieceType.JUNQI)
    from junqi_core.board import index_to_pos
    for enemy in (Seat.WEST, Seat.EAST):
        for local in range(30):
            if local in CAMP_INDICES or local in STRONGHOLD_INDICES:
                continue
            pos = index_to_pos(enemy, local)
            if pos in b.probs:
                assert b.probs[pos][junqi_idx] == 0, (
                    f"non-stronghold {pos} has nonzero JUNQI prob"
                )


# ===========================================================================
# 2. R1 Piece migration
# ===========================================================================


def test_r1_move_migrates_belief() -> None:
    """MOVE event: belief follows the piece from src to dst."""
    pieces = {(8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH)}
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    assert (8, 11) in b.probs
    assert (8, 12) not in b.probs

    st2, result = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 12)))
    b.update(st, st2, result)

    assert (8, 11) not in b.probs
    assert (8, 12) in b.probs
    assert _is_one_hot(b.probs[(8, 12)])  # own piece


def test_r1_eat_migrates_and_deletes_defender() -> None:
    """EAT: attacker's belief moves to dst; old defender belief removed."""
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (6, 10): PieceRef(Seat.WEST, PieceType.PAIZH),
        # RIGHT needs a second mobile piece or it dies via Phase-4 dead-sweep
        # and the dead-sweep wipes its remaining-inventory.
        (5, 8): PieceRef(Seat.WEST, PieceType.LIANZH),
    }
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    assert (6, 11) in b.probs and (6, 10) in b.probs

    st2, result = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert result.event is Event.EAT
    b.update(st, st2, result)

    # src deleted, dst now has attacker's belief (which was known: SILING)
    assert (6, 11) not in b.probs
    assert (6, 10) in b.probs
    assert _is_one_hot(b.probs[(6, 10)])
    # The defender belief at (6,10) was NOT one-hot (enemy uncertain), so we
    # conservatively do NOT decrement any specific type. Inventory unchanged.
    assert b.remaining[Seat.WEST][PieceType.PAIZH] == PIECE_COUNTS[PieceType.PAIZH]


def test_r1_bomb_deletes_both() -> None:
    """BOMB: src and dst both deleted."""
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (6, 10): PieceRef(Seat.WEST, PieceType.SILING),
        # Keep seats alive
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    st2, result = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert result.event is Event.BOMB
    b.update(st, st2, result)
    assert (6, 11) not in b.probs
    assert (6, 10) not in b.probs


def test_r1_killed_deletes_src_keeps_dst() -> None:
    """KILLED: attacker dies, defender stays."""
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),     # weaker attacker
        (6, 10): PieceRef(Seat.WEST, PieceType.SILING),   # stronger defender
        (10, 11): PieceRef(Seat.SOUTH, PieceType.LIANZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    st2, result = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert result.event is Event.KILLED
    b.update(st, st2, result)
    assert (6, 11) not in b.probs
    assert (6, 10) in b.probs  # defender unchanged


# ===========================================================================
# 3. R3 Q7 SILING flag reveal
# ===========================================================================


def test_r3_siling_reveal_decrements_inventory() -> None:
    """RIGHT's SILING dies via same-rank BOMB → remaining[RIGHT][SILING] -= 1."""
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (6, 10): PieceRef(Seat.WEST, PieceType.SILING),
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    assert b.remaining[Seat.WEST][PieceType.SILING] == 1
    st2, result = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert result.flag_reveal_dst is True
    b.update(st, st2, result)
    # SILING is dead on RIGHT's side → inventory goes to 0
    assert b.remaining[Seat.WEST][PieceType.SILING] == 0


# ===========================================================================
# 4. R4 Q1 Flag capture clears defender's inventory
# ===========================================================================


def test_r4_flag_capture_clears_inventory() -> None:
    """Capturing RIGHT's flag → remaining[RIGHT] is removed entirely."""
    pieces = {
        (1, 9): PieceRef(Seat.SOUTH, PieceType.SILING),
        (0, 9): PieceRef(Seat.WEST, PieceType.JUNQI),
        (2, 9): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 16): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (6, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    assert Seat.WEST in b.remaining
    st2, result = st.step(Action(seat=Seat.SOUTH, src=(1, 9), dst=(0, 9)))
    assert result.flag_captured is True
    b.update(st, st2, result)
    # RIGHT's inventory is gone
    assert Seat.WEST not in b.remaining
    # No RIGHT pieces in belief
    assert not any(pos == (2, 9) for pos in b.probs)


# ===========================================================================
# 5. R6 Stronghold non-flag eat → other stronghold is flag
# ===========================================================================


def test_r6_stronghold_eat_reveals_other_stronghold_as_flag() -> None:
    """HOME eats RIGHT's non-flag piece at stronghold (0,9).  By R6, the flag
    must be at the other stronghold (0,7).

    NOTE: RIGHT needs at least one MOBILE piece in addition to the flag + mine,
    otherwise after the attack the Q12 check on RIGHT's turn sees no legal
    moves, kills RIGHT, and purges (0,7) from belief before we can assert on it.
    """
    pieces = {
        (1, 9): PieceRef(Seat.SOUTH, PieceType.SILING),   # attacker adjacent
        (0, 9): PieceRef(Seat.WEST, PieceType.PAIZH),   # NON-flag at stronghold #1
        (0, 7): PieceRef(Seat.WEST, PieceType.JUNQI),   # real flag at stronghold #2
        (1, 7): PieceRef(Seat.WEST, PieceType.DILEI),
        # Extra mobile RIGHT piece so RIGHT survives Q12 after HOME acts
        (3, 8): PieceRef(Seat.WEST, PieceType.LIANZH),
        # Keep others alive
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    junqi_idx = TRACKED_TYPES.index(PieceType.JUNQI)

    # Before: (0,7) has some (but not full) JUNQI probability
    before = float(b.probs[(0, 7)][junqi_idx])
    assert 0 < before < 1, f"expected uncertain JUNQI at (0,7), got {before}"

    st2, result = st.step(Action(seat=Seat.SOUTH, src=(1, 9), dst=(0, 9)))
    assert result.event is Event.EAT
    assert not result.flag_captured
    assert Seat.WEST not in result.seats_died_this_step, (
        "test requires RIGHT to stay alive; check setup"
    )
    b.update(st, st2, result)

    # After: (0,7) MUST be JUNQI one-hot
    after = b.probs[(0, 7)]
    assert _is_one_hot(after)
    assert after[junqi_idx] == pytest.approx(1.0)


# ===========================================================================
# 6. R5 / R7  GONGB eats DILEI → attacker is provably GONGB
# ===========================================================================


def test_r5_r7_gongb_eats_dilei_reveals_attacker() -> None:
    """HOME's attacker (unknown to OPPOSING observer) eats RIGHT's DILEI.
    Observer deduces attacker must be GONGB.

    We set up observer = LEFT (so HOME pieces are unknown to LEFT), and HOME
    attacks a known DILEI placed where LEFT can watch.
    """
    pieces = {
        # HOME's GONGB attacks RIGHT's DILEI
        (6, 11): PieceRef(Seat.SOUTH, PieceType.GONGB),
        (6, 10): PieceRef(Seat.WEST, PieceType.DILEI),
        # Keep all 4 seats alive
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces)
    # Observer is LEFT — HOME pieces are enemies (different team).
    b = BeliefTensor.initial(st, Seat.EAST)

    # But LEFT does NOT know DILEI is DILEI (RIGHT is LEFT's teammate so the
    # mine IS visible). Wait — LEFT + RIGHT are teammates, so LEFT SEES
    # RIGHT's DILEI as one-hot. Good. LEFT does NOT see HOME's GONGB (enemy).

    # HOME's GONGB at (6,11) should NOT be one-hot in LEFT's belief
    before = b.probs[(6, 11)]
    assert not _is_one_hot(before)
    gongb_idx = TRACKED_TYPES.index(PieceType.GONGB)
    assert before[gongb_idx] > 0

    # Execute the attack
    st2, result = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert result.event is Event.EAT
    b.update(st, st2, result)

    # After: the attacker (now at (6,10)) is provably GONGB
    after = b.probs[(6, 10)]
    assert _is_one_hot(after), "GONGB should be revealed as one-hot"
    assert after[gongb_idx] == pytest.approx(1.0)
    # HOME's GONGB inventory decremented (LEFT tracks HOME since they're enemies)
    assert b.remaining[Seat.SOUTH][PieceType.GONGB] == PIECE_COUNTS[PieceType.GONGB] - 1


# ===========================================================================
# 7. to_world_tensor rendering
# ===========================================================================


def test_to_world_tensor_shape_and_content() -> None:
    st = _random_opening(seed=11)
    b = BeliefTensor.initial(st, Seat.SOUTH)
    t = b.to_world_tensor()
    from junqi_core.board import BOARD_SIZE
    assert t.shape == (BOARD_SIZE, BOARD_SIZE, NUM_TRACKED_TYPES)
    assert t.dtype == np.float32
    # Every cell with a belief → rows sum to 1
    for (x, y), vec in b.probs.items():
        assert np.allclose(t[x, y], vec)
    # Empty cells → zero vector
    assert t[8, 8].sum() == 0  # (8,8) is nine-grid center, empty


# ===========================================================================
# 8. I5 stale-key cleanup
# ===========================================================================


def test_i5_no_orphan_keys_after_sequence_of_moves() -> None:
    """After several moves, belief keys == live piece positions."""
    st = _random_opening(seed=12)
    beliefs = {s: BeliefTensor.initial(st, s) for s in Seat}

    # Play 20 random moves (or until terminated)
    rng = random.Random(100)
    for _ in range(20):
        if st.terminated:
            break
        acts = st.legal_actions()
        if not acts:
            break
        a = rng.choice(acts)
        st_new, result = st.step(a)
        for s in Seat:
            if beliefs[s].observer in result.seats_died_this_step:
                continue
            beliefs[s].update(st, st_new, result)
        st = st_new

    # Invariant: belief keys ⊆ live piece positions (I5)
    for s, b in beliefs.items():
        for pos in b.probs:
            assert pos in st.pieces, f"{s.name}: orphan belief at {pos}"
        for pos, piece in st.pieces.items():
            assert pos in b.probs, f"{s.name}: missing belief at {pos}"
