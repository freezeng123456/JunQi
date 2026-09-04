
"""Pytest-based regression: validate every golden battle/siling_flag/inference
case against junqi_core.rules.resolve_combat + siling_reveals_*.

Each JSON under tests/golden/{battle,siling_flag,inference,stronghold}/ with a
single-action scenario must be reproducible by the pure rule engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from junqi_core.rules import (
    Event,
    PieceType,
    resolve_combat,
    siling_reveals_dst,
    siling_reveals_src,
)

GOLDEN_ROOT = Path(__file__).parent / "golden"
COMBAT_CATEGORIES = ("battle", "siling_flag", "inference", "stronghold")


def _load_combat_cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    for cat in COMBAT_CATEGORIES:
        cat_dir = GOLDEN_ROOT / cat
        for path in sorted(cat_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            # Only process single-action scenarios with a real combat.
            actions = doc.get("actions", [])
            if len(actions) != 1:
                continue
            results = doc.get("expected", {}).get("results", [])
            if len(results) != 1:
                continue
            cases.append((f"{cat}/{path.name}", doc))
    return cases


def _find_piece_at(pieces: list[dict], pos: tuple[int, int]) -> dict | None:
    for p in pieces:
        if p.get("dead"):
            continue
        if tuple(p["pos"]) == pos:
            return p
    return None


@pytest.mark.parametrize(
    "case_name,doc",
    _load_combat_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.golden
def test_combat_golden(case_name: str, doc: dict) -> None:
    """Every single-action combat/move case is reproducible by the rule engine."""
    pre = doc["pre_state"]
    action = doc["actions"][0]
    expected = doc["expected"]["results"][0]

    src_pos = tuple(action["src"])
    dst_pos = tuple(action["dst"])

    src_piece = _find_piece_at(pre["pieces"], src_pos)
    dst_piece = _find_piece_at(pre["pieces"], dst_pos)

    assert src_piece is not None, f"{case_name}: no piece at src={src_pos}"

    if dst_piece is None:
        # Empty destination → MOVE, no flag reveals, no capture.
        assert expected["event"] == "MOVE", f"{case_name}: expected MOVE"
        assert expected["flag_reveal_src"] is False
        assert expected["flag_reveal_dst"] is False
        assert expected["flag_captured"] is False
        return

    src_type = PieceType[src_piece["type"]]
    dst_type = PieceType[dst_piece["type"]]

    got_event = resolve_combat(src_type, dst_type)
    got_reveal_src = siling_reveals_src(src_type, dst_type, got_event)
    got_reveal_dst = siling_reveals_dst(src_type, dst_type, got_event)
    got_captured = dst_type is PieceType.JUNQI

    assert got_event.name == expected["event"], (
        f"{case_name}: event mismatch — rule says {got_event.name}, "
        f"golden says {expected['event']}"
    )
    assert got_reveal_src == expected["flag_reveal_src"], (
        f"{case_name}: flag_reveal_src mismatch — rule={got_reveal_src}, "
        f"golden={expected['flag_reveal_src']}"
    )
    assert got_reveal_dst == expected["flag_reveal_dst"], (
        f"{case_name}: flag_reveal_dst mismatch — rule={got_reveal_dst}, "
        f"golden={expected['flag_reveal_dst']}"
    )
    assert got_captured == expected["flag_captured"], (
        f"{case_name}: flag_captured mismatch — rule={got_captured}, "
        f"golden={expected['flag_captured']}"
    )


# ===========================================================================
# Exhaustive table check against the legacy authority
# ===========================================================================
#
# The golden files above are a hand-picked sample: ten battle cases out of the
# 120 valid (attacker, defender) pairs, with no bomb-vs-mine and no
# bomb-vs-flag among them. That gap let resolve_combat disagree with the rules
# on both of those pairs for a long time. This test closes it by comparing
# every pair against a transcription of the authoritative legacy resolver.

_NONE, _DARK, _JUNQI, _DILEI, _ZHADAN, _SILING = 0, 1, 2, 3, 4, 5
_GONGB = 13


def _legacy_compare_chess(a: int, d: int) -> str:
    """Transcription of legacy_gui/src/rule.c::CompareChess.

    Kept as a literal port, branch for branch and in the same order, so a
    reviewer can diff it against the C by eye. The ordering carries the rule:
    ZHADAN is tested first because a bomb dies together with anything it
    touches, the flag and the mine included.
    """
    if d == _NONE:
        return "MOVE"
    if d == _ZHADAN or a == _ZHADAN:
        return "BOMB"
    if d == _JUNQI:
        return "EAT"
    if d == _DILEI:
        return "EAT" if a == _GONGB else "KILLED"
    assert _SILING <= d <= _GONGB and _SILING <= a <= _GONGB
    if a == d:
        return "BOMB"
    return "EAT" if a < d else "KILLED"


_ATTACKERS = [p for p in PieceType if p.value >= _ZHADAN]   # mobile pieces
_DEFENDERS = [p for p in PieceType if p.value >= _JUNQI]    # anything on board


@pytest.mark.parametrize(
    "attacker,defender",
    [(a, d) for a in _ATTACKERS for d in _DEFENDERS],
    ids=lambda p: p.name if isinstance(p, PieceType) else "",
)
def test_combat_matches_legacy_table(
    attacker: PieceType, defender: PieceType
) -> None:
    got = resolve_combat(attacker, defender).name
    want = _legacy_compare_chess(attacker.value, defender.value)
    assert got == want, (
        f"{attacker.name} attacking {defender.name}: "
        f"resolve_combat says {got}, legacy CompareChess says {want}"
    )


def test_bomb_dies_with_everything() -> None:
    """炸弹和敌方任何棋子相遇则同归于尽，包括军旗和地雷.

    Spelled out separately from the table sweep because these two pairs are
    the ones that were wrong, and because the flag pair also decides games:
    a surviving bomb would take the flag cell and win outright.
    """
    for other in _DEFENDERS:
        assert resolve_combat(PieceType.ZHADAN, other) is Event.BOMB, (
            f"ZHADAN attacking {other.name} must be mutual death"
        )
    for attacker in _ATTACKERS:
        assert resolve_combat(attacker, PieceType.ZHADAN) is Event.BOMB, (
            f"{attacker.name} attacking ZHADAN must be mutual death"
        )
