
"""17×17 board geometry for 4-player Junqi.

Pure geometric / topological data structures. NO game state here.
See `rules.py` for piece/rule data; see `state.py` for GameState.

This module computes, at import time:
  - The world-coordinate `(x, y)` of every seat-local index i (4×30 positions).
  - Per-cell flags: is_camp, is_stronghold, is_railway, is_nine_grid,
    owning_seat, seat_local_index, curve_rail_id.
  - The railway adjacency graph (used by move-gen BFS).

Everything is precomputed and exposed as immutable tables so move-gen and
observation-tensor construction can hot-loop without recomputation.

All coordinates follow the WORLD frame (legacy engine compatible):
  x ∈ [0,16], y ∈ [0,16], origin top-left.
  HOME:  y ∈ [11,16] bottom
  RIGHT: x ∈ [0,5]   left side of board
  OPPS:  y ∈ [0,5]   top
  LEFT:  x ∈ [11,16] right side of board

Coordinate formulas replicated from legacy_engine/src/junqi.c::SetChess:
  HOME  : x = 10 - i%5   , y = 11 + i//5
  RIGHT : x = 5 - i//5   , y = 10 - i%5
  OPPS  : x = 6 + i%5    , y = 5 - i//5
  LEFT  : x = 11 + i//5  , y = 6 + i%5

NineGrid (center 3×3 on a 2-step grid):
  (x, y) = (10 - (i%3)*2, 6 + (i//3)*2)  for i ∈ [0,8]

Curve rails (4 special segments): per legacy observation, the 4 curved
railway pieces connect the corners of the central rail ring. We enumerate
them below (see `CURVE_RAIL_CELLS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .rules import (
    ALL_SEATS,
    CAMP_INDICES,
    SLOTS_PER_SEAT,
    STRONGHOLD_INDICES,
    Seat,
)

# ===========================================================================
# Board size
# ===========================================================================

BOARD_SIZE: Final[int] = 17
NUM_CELLS: Final[int] = BOARD_SIZE * BOARD_SIZE  # 289

# Flat index helpers
def xy_to_flat(x: int, y: int) -> int:
    """Convert (x, y) world coord to flat 0..288 index."""
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(f"({x},{y}) out of board range [0,{BOARD_SIZE})")
    return y * BOARD_SIZE + x


def flat_to_xy(flat: int) -> tuple[int, int]:
    """Convert flat index 0..288 to (x, y)."""
    if not 0 <= flat < NUM_CELLS:
        raise ValueError(f"flat {flat} out of range [0,{NUM_CELLS})")
    return flat % BOARD_SIZE, flat // BOARD_SIZE


# ===========================================================================
# Seat-local index → world (x, y)
# ===========================================================================
# Direct port from legacy_engine/src/junqi.c::SetChess. Verified against
# legacy behavior by tests/test_rules.py::test_coord_parity.


def index_to_pos(seat: Seat, i: int) -> tuple[int, int]:
    """Return (x, y) world coordinate for seat `seat`'s local piece index `i`."""
    if not 0 <= i < SLOTS_PER_SEAT:
        raise ValueError(f"index {i} out of [0,{SLOTS_PER_SEAT})")
    col = i % 5
    row = i // 5
    if seat is Seat.SOUTH:
        return (10 - col, 11 + row)
    if seat is Seat.WEST:
        return (5 - row, 10 - col)
    if seat is Seat.NORTH:
        return (6 + col, 5 - row)
    if seat is Seat.EAST:
        return (11 + row, 6 + col)
    raise AssertionError(f"unreachable seat {seat!r}")


def nine_grid_pos(i: int) -> tuple[int, int]:
    """Return (x, y) for the i-th NineGrid cell, i ∈ [0,8]."""
    if not 0 <= i < 9:
        raise ValueError(f"nine-grid index {i} out of [0,9)")
    return (10 - (i % 3) * 2, 6 + (i // 3) * 2)


# ===========================================================================
# Cell role (per world-coordinate cell)
# ===========================================================================


@dataclass(frozen=True, slots=True)
class CellInfo:
    """Static topology info for one 17×17 cell. Immutable; computed at import."""

    x: int
    y: int
    # Which seat owns this cell (if any); None for nine-grid / empty transit rails.
    owner: Seat | None
    # Seat-local index within the owner's 30-slot layout; None if owner is None.
    seat_index: int | None
    # Topological role flags (mutually exclusive where applicable)
    is_camp: bool
    is_stronghold: bool
    is_railway: bool
    is_nine_grid: bool
    # Curve-rail id: 0 if not on a curve rail; 1..4 otherwise.
    curve_rail: int
    # True iff this cell is a valid "board" cell at all. Transit corners
    # between seat zones are still board cells (rail) but not owned.
    is_on_board: bool


# ===========================================================================
# Build the full cell table at import time
# ===========================================================================


def _build_cell_table() -> tuple[CellInfo, ...]:
    """Compute per-cell topology. Called once at module import."""
    # Start with all cells marked "off-board" (not part of any logical zone).
    # We'll flip on cells as we discover them below.
    table: list[CellInfo | None] = [None] * NUM_CELLS

    # ------------------------------------------------------------------
    # (1) Seat zones (30 cells per seat)
    # ------------------------------------------------------------------
    # Curve-rail assignment mirrors legacy InitCurveRail (junqi.c:283-305):
    #   for cid in 1..4:
    #       seat_a = cid - 1, seat_b = cid % 4
    #       slots j in {4,9,14,19,24,29} of seat_a  → curve_rail = cid
    #       slots (j-4) ∈ {0,5,10,15,20,25} of seat_b → curve_rail = cid
    _curve_lookup: dict[int, int] = {}
    for _cid in range(1, 5):
        _seat_a = ALL_SEATS[_cid - 1]
        _seat_b = ALL_SEATS[_cid % 4]
        for _j in range(SLOTS_PER_SEAT):
            if _j % 5 == 4:
                _xa, _ya = index_to_pos(_seat_a, _j)
                _xb, _yb = index_to_pos(_seat_b, _j - 4)
                _curve_lookup[xy_to_flat(_xa, _ya)] = _cid
                _curve_lookup[xy_to_flat(_xb, _yb)] = _cid

    for seat in ALL_SEATS:
        for i in range(SLOTS_PER_SEAT):
            x, y = index_to_pos(seat, i)
            flat = xy_to_flat(x, y)
            is_camp = i in CAMP_INDICES
            is_stronghold = i in STRONGHOLD_INDICES
            # A cell is railway (per legacy SetBoardRailway):
            #   i < 25 AND (row==0 OR row==4 OR col==0 OR col==4)
            row, col = i // 5, i % 5
            is_rail = (i < 25) and (row in (0, 4) or col in (0, 4))
            curve_rail_id = _curve_lookup.get(flat, 0)
            if table[flat] is not None:
                raise AssertionError(
                    f"duplicate cell assignment at ({x},{y}) for seat {seat!r} i={i}"
                )
            table[flat] = CellInfo(
                x=x, y=y,
                owner=seat,
                seat_index=i,
                is_camp=is_camp,
                is_stronghold=is_stronghold,
                is_railway=is_rail,
                is_nine_grid=False,
                curve_rail=curve_rail_id,
                is_on_board=True,
            )

    # ------------------------------------------------------------------
    # (2) NineGrid (9 center cells) — 3×3 on a 2-step lattice
    #
    # CRITICAL: NineGrid cells ARE railways (legacy InitNineGrid in
    # junqi.c:152-166 sets isRailway=1).  The engineer BFS traverses
    # through the central 9-grid to connect all four seats' rail rings.
    # ------------------------------------------------------------------
    for i in range(9):
        x, y = nine_grid_pos(i)
        flat = xy_to_flat(x, y)
        if table[flat] is not None:
            # Should not overlap with seat cells under correct layout
            raise AssertionError(f"nine-grid cell {i} at ({x},{y}) conflicts")
        table[flat] = CellInfo(
            x=x, y=y,
            owner=None,
            seat_index=None,
            is_camp=False,
            is_stronghold=False,
            is_railway=True,    # legacy: NineGrid cells are railways.
            is_nine_grid=True,
            curve_rail=0,
            is_on_board=True,
        )

    # ------------------------------------------------------------------
    # (3) Fill remaining cells as off-board. Any later code treating
    #     them as board cells is a bug.
    # ------------------------------------------------------------------
    for flat in range(NUM_CELLS):
        if table[flat] is None:
            x, y = flat_to_xy(flat)
            table[flat] = CellInfo(
                x=x, y=y,
                owner=None,
                seat_index=None,
                is_camp=False,
                is_stronghold=False,
                is_railway=False,
                is_nine_grid=False,
                curve_rail=0,
                is_on_board=False,
            )

    return tuple(ci for ci in table if ci is not None)


CELL_TABLE: Final[tuple[CellInfo, ...]] = _build_cell_table()
assert len(CELL_TABLE) == NUM_CELLS


# ===========================================================================
# Compact cell indexing: 289 (17×17) → 129 on-board cells
# ===========================================================================
# Only 129 of the 289 cells are actually on the board (4×30 seat cells + 9
# nine-grid).  For Transformer / action-head efficiency we define a compact
# indexing [0, 129) that excludes the 160 off-board cells.
#
# ON_BOARD_INDICES[compact_idx]  = flat_idx   (129 entries, sorted ascending)
# FLAT_TO_COMPACT[flat_idx]      = compact_idx or -1 if off-board
# COMPACT_TO_FLAT[compact_idx]   = flat_idx   (same as ON_BOARD_INDICES)

_on_board_flat_list: list[int] = [
    i for i, ci in enumerate(CELL_TABLE) if ci.is_on_board
]
_on_board_flat_list.sort()

NUM_ON_BOARD_CELLS: Final[int] = len(_on_board_flat_list)
assert NUM_ON_BOARD_CELLS == 129, f"expected 129, got {NUM_ON_BOARD_CELLS}"

ON_BOARD_INDICES: Final[tuple[int, ...]] = tuple(_on_board_flat_list)

COMPACT_TO_FLAT: Final[np.ndarray] = np.array(
    _on_board_flat_list, dtype=np.int16
)

FLAT_TO_COMPACT: Final[np.ndarray] = np.full(
    NUM_CELLS, -1, dtype=np.int16
)
for _compact_i, _flat_i in enumerate(_on_board_flat_list):
    FLAT_TO_COMPACT[_flat_i] = _compact_i

# Compact action space: 129 × 129 = 16,641 (vs 289 × 289 = 83,521)
COMPACT_ACTION_DIM: Final[int] = NUM_ON_BOARD_CELLS * NUM_ON_BOARD_CELLS
assert COMPACT_ACTION_DIM == 16641


def cell_info(x: int, y: int) -> CellInfo:
    """Look up static info for cell (x, y)."""
    return CELL_TABLE[xy_to_flat(x, y)]


def cell_info_flat(flat: int) -> CellInfo:
    return CELL_TABLE[flat]


# ===========================================================================
# Convenience predicates (on world coordinates)
# ===========================================================================


def is_camp(x: int, y: int) -> bool:
    return cell_info(x, y).is_camp


def is_stronghold(x: int, y: int) -> bool:
    return cell_info(x, y).is_stronghold


def is_railway(x: int, y: int) -> bool:
    return cell_info(x, y).is_railway


def is_nine_grid(x: int, y: int) -> bool:
    return cell_info(x, y).is_nine_grid


def is_on_board(x: int, y: int) -> bool:
    """True iff (x, y) is a logical board cell (inside seat zones OR nine-grid)."""
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        return False
    return cell_info(x, y).is_on_board


def owner_of(x: int, y: int) -> Seat | None:
    return cell_info(x, y).owner


def seat_index_of(x: int, y: int) -> int | None:
    return cell_info(x, y).seat_index


# ===========================================================================
# Seat-region helpers
# ===========================================================================


def all_cells_of_seat(seat: Seat) -> tuple[tuple[int, int], ...]:
    """Return the 30 world-coord cells of `seat` in local-index order."""
    return tuple(index_to_pos(seat, i) for i in range(SLOTS_PER_SEAT))


def all_camps() -> tuple[tuple[int, int], ...]:
    """Return the 20 camp cells on the board (5 per seat × 4 seats)."""
    return tuple(
        index_to_pos(seat, i)
        for seat in ALL_SEATS
        for i in sorted(CAMP_INDICES)
    )


def all_strongholds() -> tuple[tuple[int, int], ...]:
    """Return the 8 stronghold cells (2 per seat × 4 seats)."""
    return tuple(
        index_to_pos(seat, i)
        for seat in ALL_SEATS
        for i in sorted(STRONGHOLD_INDICES)
    )


def all_rails() -> tuple[tuple[int, int], ...]:
    """Return every railway cell on the board (order unspecified)."""
    return tuple((ci.x, ci.y) for ci in CELL_TABLE if ci.is_railway)


def all_nine_grid() -> tuple[tuple[int, int], ...]:
    return tuple(nine_grid_pos(i) for i in range(9))


# ===========================================================================
# Adjacency (used by move-gen)
# ===========================================================================


def orthogonal_neighbors(x: int, y: int) -> tuple[tuple[int, int], ...]:
    """4 orthogonal neighbors of (x, y), filtered to on-board cells."""
    result: list[tuple[int, int]] = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if is_on_board(nx, ny):
            result.append((nx, ny))
    return tuple(result)


def eight_neighbors(x: int, y: int) -> tuple[tuple[int, int], ...]:
    """All 8 (incl. diagonal) neighbors, filtered to on-board cells."""
    result: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if is_on_board(nx, ny):
                result.append((nx, ny))
    return tuple(result)


def nine_grid_rail_neighbors(x: int, y: int) -> tuple[tuple[int, int], ...]:
    """Extra 'jump' neighbors for NineGrid cells (step of 2, not 1).

    Used by the engineer pathing algorithm in legacy `CanMovetoJunqi` for the
    nine-grid's diagonal rail shortcuts. For non-nine-grid cells this returns ().
    """
    if not is_nine_grid(x, y):
        return ()
    result: list[tuple[int, int]] = []
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if is_on_board(nx, ny):
                result.append((nx, ny))
    return tuple(result)


# ===========================================================================
# Self-check at import time (fail loudly if topology is broken)
# ===========================================================================


def _self_check() -> None:
    # Seat count invariant: 4 × 30 = 120 seat cells + 9 nine-grid = 129 on-board cells.
    on_board = sum(1 for ci in CELL_TABLE if ci.is_on_board)
    assert on_board == 4 * SLOTS_PER_SEAT + 9, (
        f"expected {4*SLOTS_PER_SEAT + 9} on-board cells, got {on_board}"
    )
    # 20 camps (5 per seat × 4 seats).
    assert sum(1 for ci in CELL_TABLE if ci.is_camp) == 20
    # 8 strongholds (2 × 4 seats).
    assert sum(1 for ci in CELL_TABLE if ci.is_stronghold) == 8
    # 9 nine-grid cells.
    assert sum(1 for ci in CELL_TABLE if ci.is_nine_grid) == 9

    # Spot-check HOME seat coords per RULES.md §1.4 table.
    assert index_to_pos(Seat.SOUTH, 0) == (10, 11)
    assert index_to_pos(Seat.SOUTH, 4) == (6, 11)
    assert index_to_pos(Seat.SOUTH, 26) == (9, 16)
    assert index_to_pos(Seat.SOUTH, 28) == (7, 16)
    # HOME camps at {6,8,12,16,18}
    assert index_to_pos(Seat.SOUTH, 6) == (9, 12)
    assert index_to_pos(Seat.SOUTH, 8) == (7, 12)
    assert index_to_pos(Seat.SOUTH, 12) == (8, 13)
    assert index_to_pos(Seat.SOUTH, 16) == (9, 14)
    assert index_to_pos(Seat.SOUTH, 18) == (7, 14)

    # Spot-check other seats' centers (home-of-each)
    assert index_to_pos(Seat.WEST, 0) == (5, 10)
    assert index_to_pos(Seat.NORTH, 0) == (6, 5)
    assert index_to_pos(Seat.EAST, 0) == (11, 6)

    # NineGrid center (i=4) at (8,8)
    assert nine_grid_pos(4) == (8, 8)

    # Camps should be correctly classified
    assert is_camp(9, 12)
    assert is_camp(8, 13)
    assert not is_camp(10, 13)  # an ordinary HOME cell
    # Strongholds
    assert is_stronghold(9, 16)
    assert is_stronghold(7, 16)
    # Rails
    assert is_railway(*index_to_pos(Seat.SOUTH, 0))  # front row
    assert not is_railway(*index_to_pos(Seat.SOUTH, 12))  # center camp
    assert not is_railway(*index_to_pos(Seat.SOUTH, 28))  # stronghold


_self_check()


# ===========================================================================
# Manual run mode (python3 -m junqi_core.board)
# ===========================================================================


def _print_summary() -> None:  # pragma: no cover
    print(f"Board: {BOARD_SIZE}×{BOARD_SIZE} = {NUM_CELLS} cells")
    print(f"On-board cells: {sum(1 for ci in CELL_TABLE if ci.is_on_board)}")
    print(f"Camps: {sum(1 for ci in CELL_TABLE if ci.is_camp)}")
    print(f"Strongholds: {sum(1 for ci in CELL_TABLE if ci.is_stronghold)}")
    print(f"Rails: {sum(1 for ci in CELL_TABLE if ci.is_railway)}")
    print(f"NineGrid: {sum(1 for ci in CELL_TABLE if ci.is_nine_grid)}")
    for seat in ALL_SEATS:
        corners = [index_to_pos(seat, i) for i in (0, 4, 25, 29)]
        print(f"  {seat.name:5s} corners (i=0,4,25,29): {corners}")
    print("junqi_core.board self-check: OK")


if __name__ == "__main__":
    _print_summary()
