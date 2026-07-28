
"""Pytest-based regression: validate every golden setup_validation case.

Each JSON under tests/golden/setup_validation/ asserts a specific outcome from
`junqi_core.setup.validate_lineup` / `validate_setup`. This test loops over all
such files and checks that the validator agrees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from junqi_core.setup import lineup_from_names, validate_lineup, validate_setup

GOLDEN_SETUP_DIR = Path(__file__).parent / "golden" / "setup_validation"


def _load_golden_cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    for path in sorted(GOLDEN_SETUP_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            cases.append((path.name, json.load(f)))
    return cases


@pytest.mark.parametrize(
    "case_name,doc",
    _load_golden_cases(),
    ids=lambda x: x if isinstance(x, str) else "",
)
@pytest.mark.golden
def test_setup_validation_golden(case_name: str, doc: dict) -> None:
    """Every setup_validation golden case must match junqi_core.setup output."""
    su = doc.get("setup_only")
    assert su is not None, f"{case_name}: missing 'setup_only'"
    lineups = su["lineups"]
    expect_valid: bool = su["expect_valid"]
    expect_viol_prefixes: set[str] = set(su.get("expect_violations", []))

    # The convention in our scenarios is: seat 0's lineup is the one being
    # tested (the others are always the canonical-valid template). For
    # LENGTH cases we have < 30 entries.
    lu = lineups[0]

    # Try to parse; LENGTH violation comes first if the wrong length.
    try:
        parsed = lineup_from_names(lu)
        # Run full validate (length == SLOTS_PER_SEAT branch)
        result = validate_lineup(parsed)
    except Exception:  # noqa: BLE001 — LENGTH path bypasses parse
        result = validate_lineup(lu)

    got_prefixes = {v.split(":")[0] for v in result.violations}
    assert result.ok == expect_valid, (
        f"{case_name}: expected ok={expect_valid}, got ok={result.ok}, "
        f"violations={result.violations}"
    )
    if expect_viol_prefixes:
        assert expect_viol_prefixes.issubset(got_prefixes), (
            f"{case_name}: expected prefixes {expect_viol_prefixes} not all "
            f"found in {got_prefixes}; full violations: {result.violations}"
        )


@pytest.mark.golden
def test_canonical_setup_is_valid() -> None:
    """The canonical 4-seat setup passes validate_setup."""
    from tests.golden.scenarios import _canonical_setup

    raw = _canonical_setup()
    parsed_seats = [lineup_from_names(lu) for lu in raw]
    r = validate_setup(parsed_seats)
    assert r.ok, f"canonical setup should be valid; violations: {r.violations}"
