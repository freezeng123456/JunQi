
"""Setup (布阵) validation and generation.

Implements the placement hard constraints C1-C5 from RULES.md §1.3:
  C1: Camps must be empty (indices {6, 8, 12, 16, 18}).
  C2: Flag must be in a stronghold (exactly one of indices {26, 28}).
  C3: Landmines only in the back two rows (index ≥ 20).
  C4: Bombs not in the front row (index ≥ 5).
  C5: Piece counts must match rules.PIECE_COUNTS exactly.

Also provides a uniform-random setup generator for test data / RL rollouts
(respects all C1-C5 by construction).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .rules import (
    ALL_SEATS,
    CAMP_INDICES,
    PIECE_COUNTS,
    SLOTS_PER_SEAT,
    STRONGHOLD_INDICES,
    PieceType,
    Seat,
    is_back_two_rows_index,
    is_front_row_index,
    is_stronghold_index,
)

# ===========================================================================
# Types
# ===========================================================================

# A Lineup is a 30-element sequence of PieceType (or its int value), one per
# seat-local index. Camp indices must carry PieceType.NONE.
Lineup = tuple[PieceType, ...]
SetupArray = tuple[Lineup, Lineup, Lineup, Lineup]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of `validate_lineup` / `validate_setup`."""

    ok: bool
    violations: tuple[str, ...]   # one or more codes: C1, C2, C3, C4, C5

    def __bool__(self) -> bool:
        return self.ok


# ===========================================================================
# Lineup validation (single seat)
# ===========================================================================


def validate_lineup(lineup: Sequence[PieceType | int]) -> ValidationResult:
    """Validate a single seat's 30-element lineup against C1-C5.

    Accepts either PieceType values or raw ints (for JSON-load convenience).
    Returns a ValidationResult with ALL violation codes found (not just the first).
    """
    violations: list[str] = []

    # ---- Pre-check: correct length ----
    if len(lineup) != SLOTS_PER_SEAT:
        return ValidationResult(
            ok=False,
            violations=(f"LENGTH:expected={SLOTS_PER_SEAT},got={len(lineup)}",),
        )

    # Coerce to PieceType
    try:
        pieces = [PieceType(int(p)) for p in lineup]
    except (ValueError, TypeError) as e:
        return ValidationResult(ok=False, violations=(f"TYPE:{e}",))

    # ---- C1: Camps must be empty ----
    for i in sorted(CAMP_INDICES):
        if pieces[i] is not PieceType.NONE:
            violations.append(f"C1:camp_index_{i}_has_{pieces[i].name}")

    # ---- C2: Flag in exactly one stronghold; not elsewhere ----
    flag_positions = [i for i, p in enumerate(pieces) if p is PieceType.JUNQI]
    if len(flag_positions) != 1:
        violations.append(f"C2:flag_count={len(flag_positions)}(expected=1)")
    else:
        (flag_i,) = flag_positions
        if not is_stronghold_index(flag_i):
            violations.append(f"C2:flag_at_non_stronghold_{flag_i}")

    # ---- C3: Landmines only in back two rows ----
    for i, p in enumerate(pieces):
        if p is PieceType.DILEI and not is_back_two_rows_index(i):
            violations.append(f"C3:dilei_at_non_back_index_{i}")

    # ---- C4: Bombs not in front row ----
    for i, p in enumerate(pieces):
        if p is PieceType.ZHADAN and is_front_row_index(i):
            violations.append(f"C4:zhadan_at_front_index_{i}")

    # ---- C5: Piece counts ----
    counts: dict[PieceType, int] = {}
    for p in pieces:
        counts[p] = counts.get(p, 0) + 1

    # NONE count must equal number of camps (5) — since ALL non-camp slots
    # must carry a piece (25 pieces + 5 camps = 30).
    if counts.get(PieceType.NONE, 0) != len(CAMP_INDICES):
        violations.append(
            f"C5:none_count={counts.get(PieceType.NONE, 0)}"
            f"(expected={len(CAMP_INDICES)})"
        )

    # Each placeable piece type must have the expected count
    for piece_type, expected_count in PIECE_COUNTS.items():
        got = counts.get(piece_type, 0)
        if got != expected_count:
            violations.append(
                f"C5:{piece_type.name}_count={got}(expected={expected_count})"
            )

    # DARK should never appear in a setup
    if counts.get(PieceType.DARK, 0) != 0:
        violations.append(f"C5:dark_count={counts[PieceType.DARK]}(expected=0)")

    return ValidationResult(ok=(len(violations) == 0), violations=tuple(violations))


def validate_setup(setups: Sequence[Sequence[PieceType | int]]) -> ValidationResult:
    """Validate all 4 seats' lineups at once.

    Returns a combined ValidationResult. Violation codes are prefixed with the
    seat name, e.g. "HOME:C1:camp_index_6_has_PAIZH".
    """
    if len(setups) != 4:
        return ValidationResult(
            ok=False,
            violations=(f"LENGTH:expected=4_seats,got={len(setups)}",),
        )

    all_violations: list[str] = []
    for seat, lineup in zip(ALL_SEATS, setups, strict=True):
        result = validate_lineup(lineup)
        for v in result.violations:
            all_violations.append(f"{seat.name}:{v}")

    return ValidationResult(
        ok=(len(all_violations) == 0),
        violations=tuple(all_violations),
    )


# ===========================================================================
# Random setup generation
# ===========================================================================


def generate_random_lineup(rng: random.Random | None = None) -> Lineup:
    """Generate a single valid random lineup for one seat.

    Uses rejection sampling on a few constrained slots, then fills the rest
    uniformly. Always returns a lineup that passes `validate_lineup`.
    """
    rng = rng or random.Random()

    # 1. Place the flag in one of the two strongholds.
    flag_slot = rng.choice(sorted(STRONGHOLD_INDICES))

    # 2. Determine remaining piece inventory (exclude 1 flag).
    inventory: dict[PieceType, int] = dict(PIECE_COUNTS)
    inventory[PieceType.JUNQI] -= 1

    # 3. Collect valid cells for each piece type.
    # All non-camp, non-flag-slot indices.
    free_slots = [
        i for i in range(SLOTS_PER_SEAT)
        if i not in CAMP_INDICES and i != flag_slot
    ]
    rng.shuffle(free_slots)

    # Constrained placements:
    # - DILEI: only in indices {20..29} (excluding flag slot if in {26,28})
    # - ZHADAN: only in indices {5..29}
    dilei_valid = [i for i in free_slots if is_back_two_rows_index(i)]
    zhadan_valid = [i for i in free_slots if not is_front_row_index(i)]

    placements: dict[int, PieceType] = {flag_slot: PieceType.JUNQI}

    # Place DILEI first (most constrained)
    dilei_count = inventory[PieceType.DILEI]
    if len(dilei_valid) < dilei_count:
        raise RuntimeError("insufficient DILEI slots (board misconfigured)")
    rng.shuffle(dilei_valid)
    chosen_dilei = dilei_valid[:dilei_count]
    for i in chosen_dilei:
        placements[i] = PieceType.DILEI
    inventory[PieceType.DILEI] = 0

    # Remove used slots from all subsequent pools
    used = set(chosen_dilei)
    zhadan_valid = [i for i in zhadan_valid if i not in used]

    # Place ZHADAN next
    zhadan_count = inventory[PieceType.ZHADAN]
    if len(zhadan_valid) < zhadan_count:
        raise RuntimeError("insufficient ZHADAN slots (board misconfigured)")
    rng.shuffle(zhadan_valid)
    chosen_zhadan = zhadan_valid[:zhadan_count]
    for i in chosen_zhadan:
        placements[i] = PieceType.ZHADAN
    inventory[PieceType.ZHADAN] = 0
    used.update(chosen_zhadan)

    # Remaining ranked combatants fill the rest uniformly
    remaining_slots = [i for i in free_slots if i not in used]
    rng.shuffle(remaining_slots)

    remaining_pieces: list[PieceType] = []
    for p, count in inventory.items():
        if p in (PieceType.NONE, PieceType.DARK):
            continue
        remaining_pieces.extend([p] * count)
    rng.shuffle(remaining_pieces)

    if len(remaining_pieces) != len(remaining_slots):
        raise RuntimeError(
            f"piece/slot count mismatch: {len(remaining_pieces)} pieces "
            f"vs {len(remaining_slots)} slots"
        )
    for slot, piece in zip(remaining_slots, remaining_pieces, strict=True):
        placements[slot] = piece

    # Camps get NONE
    for i in CAMP_INDICES:
        placements[i] = PieceType.NONE

    lineup = tuple(placements[i] for i in range(SLOTS_PER_SEAT))

    # Self-check: freshly generated lineups must always be valid.
    result = validate_lineup(lineup)
    if not result.ok:
        raise AssertionError(
            f"generator produced invalid lineup: {result.violations}"
        )

    return lineup


def generate_random_setup(
    rng: random.Random | None = None,
) -> SetupArray:
    """Generate valid random lineups for all 4 seats."""
    rng = rng or random.Random()
    return tuple(generate_random_lineup(rng) for _ in range(4))  # type: ignore[return-value]


# ===========================================================================
# Serialization helpers (for JSON golden tests)
# ===========================================================================


def lineup_to_names(lineup: Sequence[PieceType | int]) -> list[str]:
    """Convert a lineup to a list of enum NAMES for JSON dumping."""
    return [PieceType(int(p)).name for p in lineup]


def lineup_from_names(names: Sequence[str]) -> Lineup:
    """Convert JSON string list back to a Lineup tuple."""
    return tuple(PieceType[name] for name in names)


def setup_to_names(setups: Sequence[Sequence[PieceType | int]]) -> list[list[str]]:
    return [lineup_to_names(lu) for lu in setups]


def setup_from_names(setups: Sequence[Sequence[str]]) -> SetupArray:
    return tuple(lineup_from_names(lu) for lu in setups)  # type: ignore[return-value]


# ===========================================================================
# piece_id assignment (T7 / ADR-114)
# ===========================================================================
# See docs/PHASE_0.3_T7_TODO.md §1.1.
# Rule:
#   piece_id = seat.value * SLOTS_PER_SEAT + setup_slot
#   for every slot whose lineup entry is NOT PieceType.NONE
#
# Camp slots (PieceType.NONE) get INVALID_PIECE_ID (-1). Since piece_id is
# only used as a dict key in GameState.piece_state / deaths / death_info,
# this sentinel is never actually stored anywhere — callers only assign ids
# to real pieces. All 30 piece_ids per seat are nonetheless pre-reserved
# (i.e. we do NOT pack them). This preserves the 1-to-1 mapping between
# setup_slot and piece_id, which makes `zero_board` reconstruction from
# piece_id trivial (piece_id % SLOTS_PER_SEAT = slot index).
#
# Result: 4 seats × 30 slots = 120 piece_id values in [0, 119]; 25 per seat
# are actually used (5 camps per seat remain "unassigned" — callers simply
# skip them when building PieceRef objects).

INVALID_PIECE_ID: int = -1


def assign_piece_ids(
    setups: Sequence[Sequence[PieceType | int]],
) -> dict[tuple[Seat, int], int]:
    """Compute piece_id for every non-NONE slot in `setups`.

    Returns a dict keyed by (seat, setup_slot) → piece_id.

    Preconditions
    -------------
    `setups` must pass `validate_setup` (caller is expected to have validated
    already; we do not re-validate here to avoid duplicate work in `new_game`).

    Invariants
    ----------
    - Each returned piece_id is unique across the whole dict.
    - piece_id is deterministic given (seat, setup_slot): this is the sole
      contract that `zero_board` writers downstream rely on.
    """
    if len(setups) != len(ALL_SEATS):
        raise ValueError(
            f"setups must have {len(ALL_SEATS)} lineups, got {len(setups)}"
        )

    mapping: dict[tuple[Seat, int], int] = {}
    for seat, lineup in zip(ALL_SEATS, setups, strict=True):
        if len(lineup) != SLOTS_PER_SEAT:
            raise ValueError(
                f"lineup for {seat.name} has wrong length: "
                f"expected {SLOTS_PER_SEAT}, got {len(lineup)}"
            )
        for slot_idx, piece_entry in enumerate(lineup):
            try:
                piece_type = PieceType(int(piece_entry))
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"lineup[{seat.name}][{slot_idx}]={piece_entry!r} "
                    f"is not a valid PieceType: {exc}"
                ) from exc
            if piece_type is PieceType.NONE:
                continue  # camps (and only camps) carry NONE under C1.
            mapping[(seat, slot_idx)] = seat.value * SLOTS_PER_SEAT + slot_idx

    return mapping


# ===========================================================================
# Self-test
# ===========================================================================


def _self_test() -> None:  # pragma: no cover
    rng = random.Random(42)

    # 1. Generate 100 random setups, all should validate
    for trial in range(100):
        setup = generate_random_setup(rng)
        result = validate_setup(setup)
        assert result.ok, f"trial {trial}: {result.violations}"

    # 2. Intentionally break C1 (camp has a piece)
    bad = list(generate_random_lineup(rng))
    bad[6] = PieceType.PAIZH   # camp slot
    bad[0] = PieceType.NONE    # compensate counts
    result = validate_lineup(bad)
    assert not result.ok
    assert any(v.startswith("C1:") for v in result.violations)

    # 3. Break C2 (flag at non-stronghold)
    bad = list(generate_random_lineup(rng))
    # find current flag, move to an ordinary slot
    flag_i = next(i for i, p in enumerate(bad) if p is PieceType.JUNQI)
    # pick an ordinary non-camp, non-stronghold slot
    other = 25
    bad[flag_i], bad[other] = bad[other], bad[flag_i]
    result = validate_lineup(bad)
    assert not result.ok
    assert any(v.startswith("C2:") for v in result.violations)

    # 4. Break C3 (landmine in row 0)
    bad = list(generate_random_lineup(rng))
    # find a DILEI, move to front row
    dilei_i = next(i for i, p in enumerate(bad) if p is PieceType.DILEI)
    bad[dilei_i], bad[0] = bad[0], bad[dilei_i]
    result = validate_lineup(bad)
    assert not result.ok
    assert any(v.startswith("C3:") for v in result.violations)

    # 5. Break C4 (bomb in front row)
    bad = list(generate_random_lineup(rng))
    zhadan_i = next(i for i, p in enumerate(bad) if p is PieceType.ZHADAN)
    bad[zhadan_i], bad[0] = bad[0], bad[zhadan_i]
    result = validate_lineup(bad)
    assert not result.ok
    assert any(v.startswith("C4:") for v in result.violations)

    # 6. Break C5 (extra SILING)
    bad = list(generate_random_lineup(rng))
    # find a GONGB, turn into SILING (count becomes 2 vs 1 expected)
    gongb_i = next(i for i, p in enumerate(bad) if p is PieceType.GONGB)
    bad[gongb_i] = PieceType.SILING
    result = validate_lineup(bad)
    assert not result.ok
    assert any(v.startswith("C5:") for v in result.violations)

    # 7. Round-trip serialization
    setup = generate_random_setup(rng)
    names = setup_to_names(setup)
    round_trip = setup_from_names(names)
    assert round_trip == setup

    # 8. piece_id assignment (T7 / ADR-114)
    ids = assign_piece_ids(setup)
    # 25 real pieces per seat × 4 seats = 100 ids total.
    assert len(ids) == 4 * 25, f"expected 100 piece_ids, got {len(ids)}"
    # Uniqueness
    all_ids = list(ids.values())
    assert len(set(all_ids)) == len(all_ids), "piece_ids must be unique"
    # Encoding rule: id = seat.value * 30 + slot
    for (seat, slot), pid in ids.items():
        assert pid == seat.value * SLOTS_PER_SEAT + slot
        assert 0 <= pid < 4 * SLOTS_PER_SEAT
    # Camp slots never appear in the mapping
    for (_seat, slot) in ids:
        assert slot not in CAMP_INDICES

    print("junqi_core.setup self-test: OK")


if __name__ == "__main__":
    _self_test()
