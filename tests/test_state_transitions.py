
"""Tests for junqi_core.state — GameState.step() and termination logic.

Covers all 14 canonical rule decisions end-to-end:
  - Q1 flag capture → team surrender
  - Q7 SILING flag reveals (via combat rules test but re-checked here)
  - Q10 draw thresholds (4000 / 200)
  - Q12 no-legal-moves → seat dies
  - Q14 mutual destruction → attacker's team wins (not draw)
  - General step() invariants (immutability, hash stability, legal mask)

All test scenarios use ONLY valid on-board coordinates and legal moves
(verified against is_legal_move + board topology).

Key coordinate facts used below:
  HOME zone: x ∈ [6,10], y ∈ [11,16]; stronghold indices 26,28 → (9,16),(7,16)
  RIGHT:    x ∈ [0,5],  y ∈ [6,10];  stronghold → (0,9),(0,7)
  OPPS:     x ∈ [6,10], y ∈ [0,5];   stronghold → (7,0),(9,0)  (index 26,28)
  LEFT:     x ∈ [11,16], y ∈ [6,10]; stronghold → (16,7),(16,9) (index 26,28)
  NineGrid center cells: (6,6) (8,6) (10,6) (6,8) (8,8) (10,8) (6,10) (8,10) (10,10)

Reachability notes:
  - HOME front (y=11) and OPPS front (y=5) are rails, but the column
    connecting them passes through nine-grid which is NOT rail, so rail
    long-range cannot hop between HOME/OPPS fronts. Use adjacent moves via
    nine-grid cells instead.
  - HOME (8,11) <-> (8,10) is legal (adjacent orthogonal, dst is nine-grid).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from junqi_core.move_gen import PieceRef
from junqi_core.rules import (
    MAX_NUM_MOVES,
    MAX_NUM_MOVES_BETWEEN_ATTACKS,
    Event,
    PieceType,
    Seat,
    ShowMode,
)
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState, MoveResult, SeatInfo


# ===========================================================================
# Helper: build a surgical GameState without full setup validation
# ===========================================================================


def _build_state(
    pieces: dict[tuple[int, int], PieceRef],
    turn: Seat = Seat.SOUTH,
    *,
    info: dict[Seat, SeatInfo] | None = None,
    move_counter: int = 0,
    moves_since_last_combat: int = 0,
    show_mode: ShowMode = ShowMode.HALF_DARK,
) -> GameState:
    """Construct a GameState bypassing validate_setup (for surgical tests)."""
    if info is None:
        info = {s: SeatInfo() for s in Seat}
    return GameState(
        pieces=dict(pieces),
        turn=turn,
        move_counter=move_counter,
        moves_since_last_combat=moves_since_last_combat,
        info=info,
        terminated=False,
        winner_team=None,
        draw=False,
        show_mode=show_mode,
    )


# ===========================================================================
# 1. Basic new_game & step() invariants
# ===========================================================================


def test_new_game_rejects_invalid_setup() -> None:
    """Invalid setups raise ValueError."""
    import random as _random
    setups = generate_random_setup(_random.Random(0))
    bad = list(setups[0])
    bad[6] = PieceType.PAIZH  # camp index 6 — C1 violation
    with pytest.raises(ValueError):
        GameState.new_game((tuple(bad), setups[1], setups[2], setups[3]))


def test_new_game_from_random() -> None:
    import random as _random
    setups = generate_random_setup(_random.Random(1))
    st = GameState.new_game(setups)
    assert not st.terminated
    assert st.turn is Seat.SOUTH
    assert len(st.pieces) == 4 * 25
    for s in Seat:
        assert not st.info[s].dead


def test_step_immutability() -> None:
    """step() returns a new state; original untouched."""
    pieces = {(8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH)}
    st = _build_state(pieces)
    acts = st.legal_actions()
    assert len(acts) > 0
    st2, _ = st.step(acts[0])
    assert st2 is not st
    assert st2.move_counter == 1
    assert st.move_counter == 0


def test_step_rejects_wrong_seat() -> None:
    """Cannot act out of turn."""
    pieces = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 8): PieceRef(Seat.WEST, PieceType.PAIZH),
    }
    st = _build_state(pieces, turn=Seat.SOUTH)
    with pytest.raises(ValueError, match="action.seat"):
        # RIGHT's (5,8) -> (5,7) is a legal adj move for RIGHT if it were RIGHT's turn
        st.step(Action(seat=Seat.WEST, src=(5, 8), dst=(5, 7)))


def test_step_rejects_illegal_move() -> None:
    """Cannot execute an illegal move (on-board but not reachable)."""
    pieces = {(8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH)}
    st = _build_state(pieces, turn=Seat.SOUTH)
    # (8,11) -> (8,9) is NOT on-board (y=9 is off-board in HOME zone), so
    # Action() itself rejects it. Use an ON-BOARD but unreachable target:
    # (8,11) -> (6,10) is a nine-grid diagonal over distance 2 — neither
    # adjacent nor rail-reachable in one step, so is_legal_move returns False.
    with pytest.raises(ValueError, match="illegal action"):
        st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(6, 10)))


# ===========================================================================
# 2. Combat semantics (Q7 + event types)
# ===========================================================================


def test_plain_move_emits_move_event() -> None:
    pieces = {(8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH)}
    st = _build_state(pieces)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 12)))
    assert r.event is Event.MOVE
    assert not r.flag_reveal_src
    assert not r.flag_reveal_dst
    assert not r.flag_captured
    assert st2.pieces[(8, 12)].piece_type is PieceType.PAIZH
    assert (8, 11) not in st2.pieces
    # moves_since_last_combat increments for MOVE
    assert st2.moves_since_last_combat == 1


def test_combat_resets_moves_since_combat_counter() -> None:
    """Any EAT/KILLED/BOMB resets moves_since_last_combat to 0.

    Setup: HOME SILING at (6, 11) attacks RIGHT PAIZH at (6, 10).
    (6, 11) is HOME i=4 rail; (6, 10) is nine-grid. Adj orthogonal 1-step → legal.
    """
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (6, 10): PieceRef(Seat.WEST, PieceType.PAIZH),
    }
    st = _build_state(pieces, turn=Seat.SOUTH, moves_since_last_combat=77)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert r.event is Event.EAT
    assert st2.moves_since_last_combat == 0


def test_siling_vs_siling_both_reveal_and_die() -> None:
    """Q7 canonical: two SILINGs collide → BOMB + both flags reveal + both die.

    Setup: HOME SILING at (6, 11) front rail; RIGHT SILING at nine-grid (6, 10).
    NOTE: OPPS is HOME's TEAMMATE (not enemy). HOME's enemies are RIGHT + LEFT.
    (6,11) -> (6,10) is orthogonal 1-step (RIGHT is on nine-grid (6,10)).

    But wait — does HOME's SILING flag-reveal rule apply when a RIGHT enemy's
    SILING dies? Yes, Q7 is per-seat independent; either side’s SILING death
    reveals that side’s flag.
    """
    pieces = {
        # HOME attacker on rail front
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        # RIGHT enemy SILING at nine-grid (6, 10)
        (6, 10): PieceRef(Seat.WEST, PieceType.SILING),
        # Keep teams alive with dummy pieces
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces, turn=Seat.SOUTH)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

    assert r.event is Event.BOMB
    assert r.flag_reveal_src is True
    assert r.flag_reveal_dst is True
    assert not r.flag_captured
    assert st2.info[Seat.SOUTH].flag_revealed
    assert st2.info[Seat.WEST].flag_revealed
    # Both pieces gone
    assert (6, 11) not in st2.pieces
    assert (6, 10) not in st2.pieces
    # Neither seat should be dead (they still have other pieces)
    assert not st2.info[Seat.SOUTH].dead
    assert not st2.info[Seat.WEST].dead


def test_siling_bomb_src_only_reveal() -> None:
    """Q7 key deduction: SILING attacks ZHADAN → BOMB + only src reveals.

    HOME SILING at (6,11); RIGHT ZHADAN (enemy) at nine-grid (6,10).
    """
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (6, 10): PieceRef(Seat.WEST, PieceType.ZHADAN),
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces, turn=Seat.SOUTH)
    _, r = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert r.event is Event.BOMB
    assert r.flag_reveal_src is True
    assert r.flag_reveal_dst is False


# ===========================================================================
# 3. Q1: Flag capture → team surrender
# ===========================================================================


def test_flag_capture_triggers_surrender() -> None:
    """Q1 + §3.5: eating JUNQI removes ALL the defender's pieces.

    HOME's enemies are RIGHT + LEFT. So the attack target is a BLUE flag.
    Setup: RIGHT stronghold at (0, 9) = RIGHT i=26 holds its flag.
    Attacker: HOME SILING placed at (1, 9) (RIGHT i=21 rail) adjacent to (0,9).
    """
    pieces = {
        (1, 9): PieceRef(Seat.SOUTH, PieceType.SILING),    # adjacent attacker
        (0, 9): PieceRef(Seat.WEST, PieceType.JUNQI),    # RIGHT flag
        # RIGHT has more pieces to verify they all get removed
        (2, 9): PieceRef(Seat.WEST, PieceType.PAIZH),
        (0, 8): PieceRef(Seat.WEST, PieceType.DILEI),
        # Keep teammates / other seats alive
        (10, 16): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (6, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces, turn=Seat.SOUTH)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(1, 9), dst=(0, 9)))

    assert r.event is Event.EAT
    assert r.flag_captured is True
    assert st2.info[Seat.WEST].dead
    assert Seat.WEST in r.seats_died_this_step
    # All RIGHT pieces gone
    for pos, p in st2.pieces.items():
        assert p.seat is not Seat.WEST, f"leftover RIGHT piece at {pos}"
    # Attacker is now on the captured flag cell
    assert st2.pieces[(0, 9)].seat is Seat.SOUTH


def test_flag_capture_with_teammate_alone_defeats_team() -> None:
    """When flag-capture leaves the defending team fully dead → victory.

    Setup:
      - RIGHT already dead (blue teammate gone before this step)
      - LEFT only has its flag at (16, 7) (LEFT stronghold i=26)
      - HOME attacker adjacent at (15, 7)? No, (15, 7) is LEFT i=21 - occupied?
        Free actually. Make HOME SILING stand at (15,7) for adj attack on (16,7).
      Wait: LEFT zone is x ∈ [11,16]. (15,7) is inside LEFT. But it's a rail
      cell (LEFT i=21). That's fine; HOME can sit on any on-board cell as
      long as it's not occupied by another piece.
    """
    pieces = {
        (15, 7): PieceRef(Seat.SOUTH, PieceType.SILING),   # attacker 1 step away
        (16, 7): PieceRef(Seat.EAST, PieceType.JUNQI),    # LEFT flag (stronghold)
        (10, 16): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (6, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
    }
    info = {s: SeatInfo() for s in Seat}
    info[Seat.WEST].dead = True
    st = _build_state(pieces, turn=Seat.SOUTH, info=info)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(15, 7), dst=(16, 7)))

    assert r.flag_captured is True
    assert st2.info[Seat.EAST].dead
    assert st2.terminated is True
    assert st2.winner_team == 0  # red team wins


# ===========================================================================
# 4. Q12: No legal moves → seat dies
# ===========================================================================


def test_q12_no_moves_seat_dies_when_its_turn() -> None:
    """Q12: if the incoming seat has no legal moves, it dies immediately.

    Setup:
      - HOME has a mobile PAIZH
      - RIGHT has only immobile/stronghold-trapped pieces:
          flag on stronghold (0,9) i=26
          mine on stronghold (0,7) i=28
          one mobile PAIZH — wait that defeats the test. Instead, RIGHT has
          only pieces that are either immobile OR surrounded by own pieces.
      Simpler: give RIGHT only JUNQI + DILEI + DILEI, all immobile.
    """
    pieces = {
        # HOME mobile
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        # RIGHT — flag + 2 mines, all immobile
        (0, 9): PieceRef(Seat.WEST, PieceType.JUNQI),    # stronghold i=26
        (0, 7): PieceRef(Seat.WEST, PieceType.DILEI),    # stronghold i=28
        (1, 10): PieceRef(Seat.WEST, PieceType.DILEI),   # mine in back
        # OPPS mobile
        (8, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        # LEFT mobile
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(pieces, turn=Seat.SOUTH)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 10)))

    # RIGHT should have been killed by Q12 when it became its turn
    assert st2.info[Seat.WEST].dead, "RIGHT should be dead via Q12"
    assert Seat.WEST in r.seats_died_this_step
    # Turn should have advanced past RIGHT to OPPS
    assert st2.turn is Seat.NORTH
    # All RIGHT's pieces should be removed
    for pos, p in st2.pieces.items():
        assert p.seat is not Seat.WEST


def test_q12_chain_kills_both_blue_seats() -> None:
    """Cascading Q12 kills both blue seats → red team wins.

    HOME moves → turn goes to RIGHT (no moves → dies) → turn goes to OPPS (has
    moves → stop, no cascade to LEFT). So this test path stops at RIGHT.

    To test CASCADE to both blue, we must structure so OPPS also has no moves
    OR structure so that after RIGHT dies, turn goes to OPPS which DOES move,
    but we want to verify LEFT dies too. LEFT only dies when it's LEFT's turn.
    So we can't kill both blues in one HOME action via cascade (the cascade
    chain passes through OPPS which is red-team and mobile).

    Revised test: simulate two separate steps — HOME moves, then OPPS moves,
    at which point LEFT (also immobile) should die via Q12.
    """
    pieces = {
        # HOME mobile
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        # RIGHT immobile
        (0, 9): PieceRef(Seat.WEST, PieceType.JUNQI),
        (0, 7): PieceRef(Seat.WEST, PieceType.DILEI),
        (1, 10): PieceRef(Seat.WEST, PieceType.DILEI),
        # OPPS mobile
        (8, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        # LEFT immobile (flag + 2 mines)
        (16, 7): PieceRef(Seat.EAST, PieceType.JUNQI),
        (16, 9): PieceRef(Seat.EAST, PieceType.DILEI),
        (15, 8): PieceRef(Seat.EAST, PieceType.DILEI),
    }
    st = _build_state(pieces, turn=Seat.SOUTH)
    # Step 1: HOME plays — RIGHT dies by Q12 → turn=OPPS
    st1, r1 = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 10)))
    assert st1.info[Seat.WEST].dead
    assert not st1.info[Seat.EAST].dead   # LEFT is still alive (its turn hasn't come)
    assert st1.turn is Seat.NORTH
    assert not st1.terminated

    # Step 2: OPPS plays — LEFT dies by Q12 → blue team fully defeated → red wins
    opps_acts = st1.legal_actions()
    st2, r2 = st1.step(opps_acts[0])
    assert st2.info[Seat.EAST].dead
    assert Seat.EAST in r2.seats_died_this_step
    assert st2.terminated
    assert st2.winner_team == 0  # red wins


# ===========================================================================
# 5. Q14: Mutual destruction → attacker wins
# ===========================================================================


def test_q14_mutual_destruction_attacker_wins() -> None:
    """Canonical Q14: both teams reduced to one piece each; same-rank BOMB
    simultaneously kills both → attacker's team wins.

    Pre-condition: OPPS dead, LEFT dead; HOME has lone SHIZH, RIGHT has lone
    SHIZH. HOME attacks.

    Coordinates: HOME SHIZH at (6, 10) (nine-grid cell), RIGHT SHIZH at (5, 10)
    (RIGHT i=0 rail). (6,10) -> (5,10) is orthogonal 1-step adjacent (both
    on-board; non-diagonal so no camp needed).
    """
    pieces = {
        (6, 10): PieceRef(Seat.SOUTH, PieceType.SHIZH),    # HOME mobile (nine-grid)
        (5, 10): PieceRef(Seat.WEST, PieceType.SHIZH),   # RIGHT mobile
    }
    info = {s: SeatInfo() for s in Seat}
    info[Seat.NORTH].dead = True
    info[Seat.EAST].dead = True
    st = _build_state(pieces, turn=Seat.SOUTH, info=info)
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(6, 10), dst=(5, 10)))

    assert r.event is Event.BOMB
    assert st2.info[Seat.SOUTH].dead, "HOME's last piece died"
    assert st2.info[Seat.WEST].dead, "RIGHT's last piece died"
    assert st2.terminated is True
    # Q14: attacker (HOME, team=0) wins despite both teams defeated
    assert st2.winner_team == 0, (
        f"Q14 violation: expected attacker team=0, got {st2.winner_team}"
    )
    assert not st2.draw
    rewards = st2.team_rewards()
    assert rewards == (1, -1, 1, -1), f"team rewards: {rewards}"


def test_q14_mutual_destruction_blue_attacker_wins() -> None:
    """Mirror of Q14: RIGHT (blue) attacks HOME; blue team wins."""
    pieces = {
        (6, 10): PieceRef(Seat.SOUTH, PieceType.SHIZH),
        (5, 10): PieceRef(Seat.WEST, PieceType.SHIZH),
    }
    info = {s: SeatInfo() for s in Seat}
    info[Seat.NORTH].dead = True
    info[Seat.EAST].dead = True
    st = _build_state(pieces, turn=Seat.WEST, info=info)
    st2, r = st.step(Action(seat=Seat.WEST, src=(5, 10), dst=(6, 10)))

    assert r.event is Event.BOMB
    assert st2.terminated is True
    assert st2.winner_team == 1  # blue team wins


# ===========================================================================
# 6. Q10: Draw thresholds
# ===========================================================================


def test_q10_draw_by_move_counter() -> None:
    """Hitting MAX_NUM_MOVES triggers a draw."""
    pieces = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 8): PieceRef(Seat.WEST, PieceType.PAIZH),
        (8, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(
        pieces,
        turn=Seat.SOUTH,
        move_counter=MAX_NUM_MOVES - 1,
        moves_since_last_combat=0,
    )
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 12)))
    assert st2.terminated
    assert st2.draw
    assert st2.winner_team is None
    assert r.draw_after is True
    rewards = st2.team_rewards()
    assert rewards == (0, 0, 0, 0)


def test_q10_draw_by_no_combat_counter() -> None:
    """Hitting MAX_NUM_MOVES_BETWEEN_ATTACKS (200) triggers a draw."""
    pieces = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 8): PieceRef(Seat.WEST, PieceType.PAIZH),
        (8, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(
        pieces,
        turn=Seat.SOUTH,
        move_counter=500,
        moves_since_last_combat=MAX_NUM_MOVES_BETWEEN_ATTACKS - 1,
    )
    st2, _ = st.step(Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 12)))
    assert st2.terminated
    assert st2.draw


def test_q10_draw_not_triggered_by_combat() -> None:
    """Combat resets the no-combat counter, so we should NOT hit threshold."""
    pieces = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (6, 10): PieceRef(Seat.WEST, PieceType.PAIZH),
        (8, 0): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = _build_state(
        pieces,
        turn=Seat.SOUTH,
        move_counter=500,
        moves_since_last_combat=MAX_NUM_MOVES_BETWEEN_ATTACKS - 1,
    )
    st2, r = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))
    assert r.event is Event.EAT
    assert st2.moves_since_last_combat == 0
    assert not st2.draw
    assert not st2.terminated


# ===========================================================================
# 7. Queries: legal_actions, mask, clone, round-trip
# ===========================================================================


def test_legal_actions_from_empty_seat() -> None:
    """A dead seat has no legal actions."""
    pieces = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 8): PieceRef(Seat.WEST, PieceType.PAIZH),
    }
    info = {s: SeatInfo() for s in Seat}
    info[Seat.NORTH].dead = True
    st = _build_state(pieces, turn=Seat.SOUTH, info=info)
    assert st.legal_actions(seat=Seat.NORTH) == []
    assert len(st.legal_actions(seat=Seat.SOUTH)) > 0


def test_legal_action_mask_shape_and_content() -> None:
    pieces = {(8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH)}
    st = _build_state(pieces, turn=Seat.SOUTH)
    mask = st.legal_action_mask()
    assert mask.shape == (17, 17, 17, 17)
    assert mask.dtype == np.bool_
    assert mask[8, 11, 8, 12]
    assert not mask[8, 11, 9, 10]
    assert int(mask.sum()) == len(st.legal_actions())


def test_clone_independence() -> None:
    pieces = {(8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH)}
    st = _build_state(pieces)
    st2 = st.clone()
    assert st2.state_hash() == st.state_hash()
    st2.info[Seat.SOUTH].dead = True
    assert not st.info[Seat.SOUTH].dead


def test_to_from_dict_round_trip() -> None:
    pieces = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 8): PieceRef(Seat.WEST, PieceType.SILING),
    }
    st = _build_state(pieces, turn=Seat.SOUTH, move_counter=42)
    d = st.to_dict()
    st2 = GameState.from_dict(d)
    assert st2.state_hash() == st.state_hash()


# ===========================================================================
# 8. Integration: golden full_game scenarios
# ===========================================================================


GOLDEN_FULL_GAME_DIR = Path(__file__).parent / "golden" / "full_game"


def _piece_map_from_json(pieces_json: list[dict]) -> dict[tuple[int, int], PieceRef]:
    result: dict[tuple[int, int], PieceRef] = {}
    for p in pieces_json:
        if p.get("dead"):
            continue
        pos = (p["pos"][0], p["pos"][1])
        result[pos] = PieceRef(
            seat=Seat(p["seat"]),
            piece_type=PieceType[p["type"]],
            alive=True,
        )
    return result


def _info_from_json(info_json: list[dict]) -> dict[Seat, SeatInfo]:
    return {
        Seat(item["seat"]): SeatInfo(
            dead=item["dead"],
            flag_revealed=item["flag_revealed"],
        )
        for item in info_json
    }


def test_golden_q14_mutual_destruction_json() -> None:
    """Directly exercise the golden Q14 full_game scenario."""
    path = GOLDEN_FULL_GAME_DIR / "mutual_destruction_attacker_wins.json"
    doc = json.loads(path.read_text())

    pre = doc["pre_state"]
    pieces = _piece_map_from_json(pre["pieces"])
    info = _info_from_json(pre["info"])
    st = _build_state(
        pieces,
        turn=Seat(pre["turn"]),
        info=info,
        move_counter=pre["move_counter"],
        moves_since_last_combat=pre["moves_since_last_combat"],
    )

    action_json = doc["actions"][0]
    action = Action(
        seat=Seat(action_json["seat"]),
        src=tuple(action_json["src"]),  # type: ignore[arg-type]
        dst=tuple(action_json["dst"]),  # type: ignore[arg-type]
    )
    st2, r = st.step(action)

    expected_result = doc["expected"]["results"][0]
    assert r.event.name == expected_result["event"]
    assert r.flag_reveal_src == expected_result["flag_reveal_src"]
    assert r.flag_reveal_dst == expected_result["flag_reveal_dst"]
    assert r.flag_captured == expected_result["flag_captured"]
    expected_died = {Seat(s) for s in expected_result["seats_died_this_step"]}
    assert set(r.seats_died_this_step) == expected_died
    assert st2.terminated == doc["expected"]["terminated"]
    assert st2.winner_team == doc["expected"]["winner_team"]
    assert st2.draw == doc["expected"]["draw"]
