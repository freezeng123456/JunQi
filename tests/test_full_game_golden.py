
"""Golden-JSON replay tests for full_game/ scenarios.

Each JSON file in tests/golden/full_game/ describes a pre-state, a
sequence of actions, and expected per-action results plus terminal
state. We load these declaratively and replay through
`GameState.step()`, asserting every expected field matches.

This module is the **bulk integration tester** for the engine; new
full_game JSONs added to the folder are picked up automatically.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from junqi_core.move_gen import PieceRef
from junqi_core.rules import Event, PieceType, Seat
from junqi_core.state import Action, GameState, SeatInfo


GOLDEN_DIR = Path(__file__).parent / "golden" / "full_game"


def _list_cases() -> list[tuple[str, Path]]:
    return [(p.name, p) for p in sorted(GOLDEN_DIR.glob("*.json"))]


def _pieces_from(pieces_json: list[dict]) -> dict[tuple[int, int], PieceRef]:
    out: dict[tuple[int, int], PieceRef] = {}
    for p in pieces_json:
        if p.get("dead"):
            continue
        pos = (p["pos"][0], p["pos"][1])
        out[pos] = PieceRef(
            seat=Seat(p["seat"]),
            piece_type=PieceType[p["type"]],
            alive=True,
        )
    return out


def _info_from(info_json: list[dict]) -> dict[Seat, SeatInfo]:
    return {
        Seat(item["seat"]): SeatInfo(
            dead=item["dead"],
            flag_revealed=item["flag_revealed"],
        )
        for item in info_json
    }


def _build_state(pre: dict) -> GameState:
    return GameState(
        pieces=_pieces_from(pre["pieces"]),
        turn=Seat(pre["turn"]),
        move_counter=pre["move_counter"],
        moves_since_last_combat=pre["moves_since_last_combat"],
        info=_info_from(pre["info"]),
        terminated=False,
        winner_team=None,
        draw=False,
    )


@pytest.mark.parametrize(
    "name,path", _list_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.golden
def test_replay_golden_full_game(name: str, path: Path) -> None:
    """Replay a golden full_game scenario and assert every expected field."""
    doc = json.loads(path.read_text())
    state = _build_state(doc["pre_state"])

    actions_json = doc["actions"]
    expected_results = doc["expected"]["results"]
    assert len(actions_json) == len(expected_results), (
        f"{name}: actions ({len(actions_json)}) and expected results "
        f"({len(expected_results)}) count mismatch"
    )

    for i, (a_json, exp) in enumerate(zip(actions_json, expected_results)):
        action = Action(
            seat=Seat(a_json["seat"]),
            src=(a_json["src"][0], a_json["src"][1]),
            dst=(a_json["dst"][0], a_json["dst"][1]),
        )
        state, result = state.step(action)

        # Check public broadcast fields
        assert result.event.name == exp["event"], (
            f"{name} step {i}: event {result.event.name} vs expected {exp['event']}"
        )
        assert result.flag_reveal_src == exp["flag_reveal_src"], (
            f"{name} step {i}: flag_reveal_src mismatch"
        )
        assert result.flag_reveal_dst == exp["flag_reveal_dst"], (
            f"{name} step {i}: flag_reveal_dst mismatch"
        )
        assert result.flag_captured == exp["flag_captured"], (
            f"{name} step {i}: flag_captured mismatch"
        )
        # seats_died_this_step is order-insensitive and optional
        if "seats_died_this_step" in exp:
            expected_died = {Seat(s) for s in exp["seats_died_this_step"]}
            actual_died = set(result.seats_died_this_step)
            assert actual_died == expected_died, (
                f"{name} step {i}: seats_died actual={actual_died} expected={expected_died}"
            )

    # Final terminal state assertions
    assert state.terminated == doc["expected"]["terminated"], (
        f"{name}: terminated {state.terminated} vs {doc['expected']['terminated']}"
    )
    assert state.winner_team == doc["expected"]["winner_team"], (
        f"{name}: winner_team {state.winner_team} vs {doc['expected']['winner_team']}"
    )
    assert state.draw == doc["expected"]["draw"], (
        f"{name}: draw {state.draw} vs {doc['expected']['draw']}"
    )
