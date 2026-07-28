
"""Universal replay tests for all golden JSON scenarios.

Covers: battle / siling_flag / stronghold / inference
(move_gen replay is already in test_move_gen.py;
 full_game replay is already in test_full_game_golden.py)

For each category we:
  1. Load pre_state into a GameState (bypassing validate_setup).
  2. Execute each action via state.step().
  3. Assert the MoveResult matches the declared 'expected' fields.
  4. Assert terminal state matches.

This was added in Phase 0.2 T6 after discovering that battle/, siling_flag/,
stronghold/, inference/ golden files were authored in Phase 0.1 but never
actually replayed end-to-end, leaving latent DSL bugs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from junqi_core.move_gen import PieceRef
from junqi_core.rules import PieceType, Seat, ShowMode
from junqi_core.state import Action, GameState, SeatInfo


GOLDEN_ROOT = Path(__file__).parent / "golden"
CATEGORIES = ("battle", "siling_flag", "stronghold", "inference")


def _build_piece_map(pieces_json: list[dict]) -> dict[tuple[int, int], PieceRef]:
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


def _build_info(info_json: list[dict]) -> dict[Seat, SeatInfo]:
    return {
        Seat(item["seat"]): SeatInfo(
            dead=item["dead"],
            flag_revealed=item["flag_revealed"],
        )
        for item in info_json
    }


def _build_state(pre: dict) -> GameState:
    return GameState(
        pieces=_build_piece_map(pre["pieces"]),
        turn=Seat(pre["turn"]),
        move_counter=pre["move_counter"],
        moves_since_last_combat=pre["moves_since_last_combat"],
        info=_build_info(pre["info"]),
        terminated=False,
        winner_team=None,
        draw=False,
        show_mode=ShowMode.HALF_DARK,
    )


def _collect_cases() -> list[tuple[str, str, Path]]:
    cases: list[tuple[str, str, Path]] = []
    for category in CATEGORIES:
        cat_dir = GOLDEN_ROOT / category
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.glob("*.json")):
            cases.append((category, p.name, p))
    return cases


_CASE_PARAMS = _collect_cases()


@pytest.mark.parametrize(
    "category,name,path",
    _CASE_PARAMS,
    ids=[f"{c}/{n}" for c, n, _ in _CASE_PARAMS],
)
@pytest.mark.golden
def test_golden_replays_through_step(
    category: str, name: str, path: Path,
) -> None:
    """Load and replay a golden scenario; assert every expected field."""
    doc = json.loads(path.read_text())
    state = _build_state(doc["pre_state"])

    actions = doc["actions"]
    expected_results = doc["expected"]["results"]
    assert len(actions) == len(expected_results), (
        f"{category}/{name}: actions ({len(actions)}) and expected "
        f"results ({len(expected_results)}) count mismatch"
    )

    for i, (a_json, exp) in enumerate(zip(actions, expected_results)):
        action = Action(
            seat=Seat(a_json["seat"]),
            src=(a_json["src"][0], a_json["src"][1]),
            dst=(a_json["dst"][0], a_json["dst"][1]),
        )
        state, result = state.step(action)

        assert result.event.name == exp["event"], (
            f"{category}/{name} step {i}: "
            f"event {result.event.name} != expected {exp['event']}"
        )
        assert result.flag_reveal_src == exp["flag_reveal_src"], (
            f"{category}/{name} step {i}: flag_reveal_src mismatch "
            f"(actual={result.flag_reveal_src} vs {exp['flag_reveal_src']})"
        )
        assert result.flag_reveal_dst == exp["flag_reveal_dst"], (
            f"{category}/{name} step {i}: flag_reveal_dst mismatch "
            f"(actual={result.flag_reveal_dst} vs {exp['flag_reveal_dst']})"
        )
        assert result.flag_captured == exp["flag_captured"], (
            f"{category}/{name} step {i}: flag_captured mismatch "
            f"(actual={result.flag_captured} vs {exp['flag_captured']})"
        )

        # Optional: seats_died_this_step (if the JSON declares it)
        if "seats_died_this_step" in exp:
            expected_died = {Seat(s) for s in exp["seats_died_this_step"]}
            actual_died = set(result.seats_died_this_step)
            assert actual_died == expected_died, (
                f"{category}/{name} step {i}: seats_died "
                f"actual={actual_died} vs expected={expected_died}"
            )

    # Terminal assertions
    assert state.terminated == doc["expected"]["terminated"], (
        f"{category}/{name}: terminated "
        f"(actual={state.terminated} vs {doc['expected']['terminated']})"
    )
    assert state.winner_team == doc["expected"]["winner_team"], (
        f"{category}/{name}: winner_team "
        f"(actual={state.winner_team} vs {doc['expected']['winner_team']})"
    )
    assert state.draw == doc["expected"]["draw"], (
        f"{category}/{name}: draw "
        f"(actual={state.draw} vs {doc['expected']['draw']})"
    )
