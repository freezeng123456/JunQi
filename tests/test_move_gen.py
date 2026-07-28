"""Tests for junqi_core.move_gen against golden move_gen scenarios.

For each case in tests/golden/move_gen/*.json we build a piece map from
pre_state.pieces and assert:
  1. The declared action is legal per is_legal_move.
  2. generate_legal_actions includes this action.
  3. (Optional) Hand-coded invariants per scenario tag.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from junqi_core.move_gen import (
    PieceMap,
    PieceRef,
    generate_legal_actions,
    has_any_legal_move,
    is_legal_move,
    legal_moves_from,
)
from junqi_core.rules import PieceType, Seat

GOLDEN_MOVE_GEN_DIR = Path(__file__).parent / "golden" / "move_gen"


def _build_piece_map(pieces_json: list[dict]) -> PieceMap:
    result: PieceMap = {}
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


def _load_move_gen_cases() -> list[tuple[str, dict]]:
    return [
        (p.name, json.loads(p.read_text()))
        for p in sorted(GOLDEN_MOVE_GEN_DIR.glob("*.json"))
    ]


@pytest.mark.parametrize(
    "case_name,doc",
    _load_move_gen_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.golden
def test_move_gen_golden_legal(case_name: str, doc: dict) -> None:
    """Every move in a golden move_gen scenario must be legal per junqi_core."""
    pre = doc["pre_state"]
    actions = doc["actions"]
    pieces = _build_piece_map(pre["pieces"])

    for idx, action in enumerate(actions):
        seat = Seat(action["seat"])
        src = (action["src"][0], action["src"][1])
        dst = (action["dst"][0], action["dst"][1])
        assert is_legal_move(pieces, src, dst, seat), (
            f"{case_name} action[{idx}]: move {src}->{dst} seat={seat.name} "
            f"should be legal but is_legal_move returned False"
        )
        # Also present in generate_legal_actions
        legal = generate_legal_actions(pieces, seat)
        assert (src, dst) in legal, (
            f"{case_name} action[{idx}]: {src}->{dst} not in legal actions "
            f"({len(legal)} actions total)"
        )


# ---- Hand-written invariant tests for representative move_gen scenarios ----


def test_home_paizh_front_row_moves() -> None:
    """SOUTH PAIZH at front-row (8, 11) should have: 4 orthogonal neighbors +
    straight-rail moves along y=11 to the left and right ends."""
    pieces: PieceMap = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
    }
    moves = legal_moves_from(pieces, (8, 11), Seat.SOUTH)
    # Neighbors
    assert (7, 11) in moves    # orthogonal rail neighbor
    assert (9, 11) in moves
    assert (8, 10) in moves    # nine-grid cell
    assert (8, 12) in moves    # adj SOUTH cell
    # Rail long-range on y=11
    assert (10, 11) in moves
    assert (6, 11) in moves
    # Diagonal without camp: NOT allowed
    assert (7, 10) not in moves
    assert (9, 10) not in moves


def test_stronghold_piece_cannot_move() -> None:
    """A piece on a stronghold cannot move (even non-flag)."""
    pieces: PieceMap = {
        (9, 16): PieceRef(Seat.SOUTH, PieceType.LIANZH),
    }
    moves = legal_moves_from(pieces, (9, 16), Seat.SOUTH)
    assert moves == []


def test_dilei_cannot_move() -> None:
    """Landmines are immobile."""
    pieces: PieceMap = {
        (10, 15): PieceRef(Seat.SOUTH, PieceType.DILEI),
    }
    moves = legal_moves_from(pieces, (10, 15), Seat.SOUTH)
    assert moves == []


def test_junqi_cannot_move() -> None:
    """Flag is immobile."""
    pieces: PieceMap = {
        (9, 16): PieceRef(Seat.SOUTH, PieceType.JUNQI),
    }
    moves = legal_moves_from(pieces, (9, 16), Seat.SOUTH)
    assert moves == []


def test_rail_blocked_by_own_piece() -> None:
    """Non-engineer cannot jump over own piece on rail."""
    pieces: PieceMap = {
        (10, 11): PieceRef(Seat.SOUTH, PieceType.LIANZH),
        (8, 11):  PieceRef(Seat.SOUTH, PieceType.PAIZH),
    }
    # (10,11) -> (6,11) is blocked by own (8,11)
    assert not is_legal_move(pieces, (10, 11), (6, 11), Seat.SOUTH)
    # But (10,11) -> (9,11) adjacent is fine
    assert is_legal_move(pieces, (10, 11), (9, 11), Seat.SOUTH)


def test_engineer_bfs_can_turn() -> None:
    """Engineer at rail corner (6, 11) can reach (10, 15) via BFS (requires turn)."""
    pieces: PieceMap = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.GONGB),
    }
    assert is_legal_move(pieces, (6, 11), (10, 15), Seat.SOUTH)


def test_non_engineer_cannot_turn_on_rail() -> None:
    """Non-engineer at (6, 11) cannot reach (10, 15) because it requires a turn."""
    pieces: PieceMap = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.LIANZH),
    }
    assert not is_legal_move(pieces, (6, 11), (10, 15), Seat.SOUTH)


def test_diagonal_into_camp_allowed() -> None:
    """Diagonal move is legal when destination is a camp."""
    pieces: PieceMap = {
        (10, 13): PieceRef(Seat.SOUTH, PieceType.LIANZH),
    }
    # (10,13) -> (9,12) is camp
    assert is_legal_move(pieces, (10, 13), (9, 12), Seat.SOUTH)


def test_diagonal_out_of_camp_allowed() -> None:
    """Piece sitting in a camp can move diagonally to another cell."""
    pieces: PieceMap = {
        (8, 13): PieceRef(Seat.SOUTH, PieceType.PAIZH),  # center camp
    }
    # (8,13) -> (7,12) is another camp
    assert is_legal_move(pieces, (8, 13), (7, 12), Seat.SOUTH)


def test_enemy_in_camp_unattackable() -> None:
    """Enemy piece sitting in a camp cannot be attacked."""
    pieces: PieceMap = {
        (10, 13): PieceRef(Seat.SOUTH, PieceType.SILING),
        (9, 12):  PieceRef(Seat.NORTH, PieceType.PAIZH),  # enemy in camp
    }
    assert not is_legal_move(pieces, (10, 13), (9, 12), Seat.SOUTH)


def test_same_team_blocks_dst() -> None:
    """Cannot move onto teammate's piece."""
    pieces: PieceMap = {
        (10, 13): PieceRef(Seat.SOUTH, PieceType.SILING),
        (10, 12): PieceRef(Seat.NORTH, PieceType.PAIZH),  # teammate
    }
    assert not is_legal_move(pieces, (10, 13), (10, 12), Seat.SOUTH)


def test_has_any_legal_move() -> None:
    """has_any_legal_move reflects reality."""
    pieces: PieceMap = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
    }
    assert has_any_legal_move(pieces, Seat.SOUTH)
    assert not has_any_legal_move(pieces, Seat.NORTH)


def test_seat_with_only_immobile_pieces_has_no_moves() -> None:
    """A seat holding only flag + mines + stronghold-piece has no legal moves."""
    pieces: PieceMap = {
        (9, 16): PieceRef(Seat.SOUTH, PieceType.JUNQI),   # flag in stronghold
        (7, 16): PieceRef(Seat.SOUTH, PieceType.DILEI),   # mine in other stronghold? no, (7,16) is stronghold too
        (10, 15): PieceRef(Seat.SOUTH, PieceType.DILEI),  # mine
        (6, 15): PieceRef(Seat.SOUTH, PieceType.DILEI),   # mine
    }
    assert not has_any_legal_move(pieces, Seat.SOUTH)


def test_gongb_diagonal_via_two_step_rail_is_legal() -> None:
    """Regression: GONGB at (12,10) can reach (11,9) via 2-step rail BFS
    ((12,10) → (11,10) → (11,9)), even though 1-step diagonal is illegal.

    This was a bug caught by T5 stress tester: is_legal_move's adjacent
    branch used to early-return False on a diagonal non-camp step, NEVER
    falling through to the rail BFS. Fixed by letting the adjacent branch
    fall through when it cannot prove legality. See docs/LEGACY_PARITY.md §3.
    """
    # Place only the GONGB; rails (11,10) and (11,9) must be empty so BFS
    # can walk through them.
    pieces = {(12, 10): PieceRef(Seat.EAST, PieceType.GONGB)}
    assert is_legal_move(pieces, (12, 10), (11, 9), Seat.EAST), (
        "regression: GONGB 2-step rail BFS should make (12,10)→(11,9) legal"
    )
    # legal_moves_from agrees
    dests = legal_moves_from(pieces, (12, 10), Seat.EAST)
    assert (11, 9) in dests


def test_non_gongb_diagonal_not_camp_still_illegal() -> None:
    """Non-engineer may NOT use 2-step rail to go around a diagonal: a LIANZH
    at (12,10) cannot land on (11,9) because rail travel requires same-x or
    same-y OR being on the same curve rail — not an arbitrary BFS."""
    pieces = {(12, 10): PieceRef(Seat.EAST, PieceType.LIANZH)}
    assert not is_legal_move(pieces, (12, 10), (11, 9), Seat.EAST)
    dests = legal_moves_from(pieces, (12, 10), Seat.EAST)
    assert (11, 9) not in dests
