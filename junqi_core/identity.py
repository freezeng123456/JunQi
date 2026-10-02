"""Piece vocabulary and exact opening restrictions, without identity probabilities."""
from __future__ import annotations

from typing import Final

from .rules import (
    ALL_PLACEABLE_PIECES,
    BACK_TWO_ROWS_INDICES,
    CAMP_INDICES,
    FRONT_ROW_INDICES,
    STRONGHOLD_INDICES,
    PieceType,
)

TRACKED_TYPES: Final[tuple[PieceType, ...]] = tuple(
    t for t in ALL_PLACEABLE_PIECES if t not in (PieceType.NONE, PieceType.DARK)
)
NUM_TRACKED_TYPES: Final[int] = len(TRACKED_TYPES)
TYPE_TO_INDEX: Final[dict[PieceType, int]] = {t: i for i, t in enumerate(TRACKED_TYPES)}


def _allowed_types(slot: int) -> tuple[PieceType, ...]:
    if slot in CAMP_INDICES:
        return ()
    return tuple(
        t for t in TRACKED_TYPES
        if (not t.is_flag or slot in STRONGHOLD_INDICES)
        and (not t.is_mine or slot in BACK_TWO_ROWS_INDICES)
        and (not t.is_bomb or slot not in FRONT_ROW_INDICES)
    )


ADMISSIBLE_BY_INDEX: Final[tuple[tuple[PieceType, ...], ...]] = tuple(
    _allowed_types(slot) for slot in range(30)
)
