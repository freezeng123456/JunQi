"""Board-facing identities, separate from hidden engine truth and beliefs.

An inference is not a face-up piece: clients may show it as a deduction, but
must not silently render a probability estimate as the true piece label.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .combat_memory import is_publicly_revealed_mine
from .rules import PieceType, Seat, ShowMode

if TYPE_CHECKING:
    from .move_gen import PieceRef
    from .state import GameState


def sees_army_types(observer: Seat, owner: Seat, mode: ShowMode) -> bool:
    return (
        mode is ShowMode.BRIGHT
        or observer is owner
        or (mode is ShowMode.HALF_DARK and observer.team == owner.team)
    )


def visible_piece_type(state: GameState, observer: Seat, piece: PieceRef) -> PieceType:
    """Literal label visible on the board to this seat at this moment."""
    if sees_army_types(observer, piece.seat, state.show_mode):
        return piece.piece_type
    if state.info[piece.seat].flag_revealed and piece.piece_type is PieceType.JUNQI:
        return PieceType.JUNQI
    return PieceType.DARK


def public_deduced_type(state: GameState, piece: PieceRef) -> PieceType | None:
    """Exact public deductions supported by the recorded event/path history."""
    pid = piece.piece_id
    if is_publicly_revealed_mine(state.combat_memory, pid):
        return PieceType.DILEI
    if state.combat_memory.is_gongb[:, pid].all():
        return PieceType.GONGB
    return None
