"""junqi_core/rail_topology.py — authoritative rail graph, derived from legacy.

This module is the **single source of truth** for the Junqi railway topology.
Everything is derived directly from the legacy C code under
``legacy_engine/src/junqi.c`` — those files must not be modified.

Legacy source correspondence
----------------------------
* ``RAIL_CELLS`` + ``IS_RAILWAY`` / ``IS_NINEGRID`` — mirror
  ``SetBoardRailway`` (per-seat ``i<25 && (row 0/4 || col 0/4)``) plus
  ``InitNineGrid`` (9 central cells, also marked ``isRailway=1``).
* ``RAIL_ADJ`` — mirrors ``InitBoardGraph``:
    (a) every rail cell links to ortho rail neighbours,
    (b) NineGrid cells additionally link to 2-step ortho NineGrid neighbours,
    (c) ``AddSpcNode`` adds 4 inner-corner diagonal edges.
* ``CURVE_RAIL_OF`` — mirrors ``InitCurveRail``: every rail cell belongs to
  at most one curve rail in ``{1,2,3,4}`` (0 = "not on any curve").

All constants are plain NumPy arrays / Python tuples. No runtime dependency
on the CUDA backend.  This module is imported by ``board.py`` and
``_movegen_tables.py`` to provide the corrected rail data for the entire
engine (CPU and GPU alike).
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE: Final[int] = 17
SEAT_SLOTS: Final[int] = 30

# Seat ids: 0=SOUTH/HOME, 1=WEST/RIGHT, 2=NORTH/OPPS, 3=EAST/LEFT
# (same ordering as junqi_core.rules.Seat)

# Camp slot indices within a seat's 30-cell zone (legacy SetBoardCamp).
_CAMP_IDX: Final[frozenset[int]] = frozenset({6, 8, 12, 16, 18})
# Stronghold slot indices within a seat's 30-cell zone.
_STRONGHOLD_IDX: Final[frozenset[int]] = frozenset({26, 28})

# SpcRail values (RAIL1..RAIL4), legacy curve-rail identifiers.
CURVE_RAIL_NONE: Final[int] = 0
CURVE_RAIL_RAIL1: Final[int] = 1
CURVE_RAIL_RAIL2: Final[int] = 2
CURVE_RAIL_RAIL3: Final[int] = 3
CURVE_RAIL_RAIL4: Final[int] = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seat_pos(dir_: int, i: int) -> tuple[int, int]:
    """Legacy ``SetChess`` mapping: seat + slot → (x, y)."""
    if dir_ == 0:   # HOME / SOUTH
        return (10 - i % 5, 11 + i // 5)
    if dir_ == 1:   # RIGHT / WEST
        return (5 - i // 5, 10 - i % 5)
    if dir_ == 2:   # OPPS / NORTH
        return (6 + i % 5, 5 - i // 5)
    if dir_ == 3:   # LEFT / EAST
        return (11 + i // 5, 6 + i % 5)
    raise ValueError(f"Invalid seat dir={dir_}")


def _xy_to_flat(x: int, y: int) -> int:
    return y * BOARD_SIZE + x


def _flat_to_xy(flat: int) -> tuple[int, int]:
    return flat % BOARD_SIZE, flat // BOARD_SIZE


# ---------------------------------------------------------------------------
# Per-cell tables — seat zones + NineGrid
# ---------------------------------------------------------------------------

NC: Final[int] = BOARD_SIZE * BOARD_SIZE  # 289

_is_on_board: list[int] = [0] * NC
_is_railway: list[int] = [0] * NC
_is_ninegrid: list[int] = [0] * NC
_is_camp: list[int] = [0] * NC
_is_stronghold: list[int] = [0] * NC
_owner: list[int] = [-1] * NC   # -1 for NineGrid / off-board
_seat_index: list[int] = [-1] * NC

# Populate seat zones (legacy SetChess + SetBoardCamp + SetBoardRailway).
for _dir in range(4):
    for _i in range(SEAT_SLOTS):
        _x, _y = _seat_pos(_dir, _i)
        _f = _xy_to_flat(_x, _y)
        _is_on_board[_f] = 1
        _owner[_f] = _dir
        _seat_index[_f] = _i
        if _i in _CAMP_IDX:
            _is_camp[_f] = 1
        if _i in _STRONGHOLD_IDX:
            _is_stronghold[_f] = 1
        # SetBoardRailway: i<25 && (row 0/4 || col 0/4)
        if _i < 25 and (_i // 5 in (0, 4) or _i % 5 in (0, 4)):
            _is_railway[_f] = 1

# InitNineGrid: 9 central cells at (10-(i%3)*2, 6+(i/3)*2), all rail + ninegrid.
for _i in range(9):
    _x = 10 - (_i % 3) * 2
    _y = 6 + (_i // 3) * 2
    _f = _xy_to_flat(_x, _y)
    _is_on_board[_f] = 1
    _is_railway[_f] = 1
    _is_ninegrid[_f] = 1


# Public per-cell NumPy arrays.
IS_ON_BOARD:   Final[np.ndarray] = np.array(_is_on_board,   dtype=bool)
IS_RAILWAY:    Final[np.ndarray] = np.array(_is_railway,    dtype=bool)
IS_NINEGRID:   Final[np.ndarray] = np.array(_is_ninegrid,   dtype=bool)
IS_CAMP:       Final[np.ndarray] = np.array(_is_camp,       dtype=bool)
IS_STRONGHOLD: Final[np.ndarray] = np.array(_is_stronghold, dtype=bool)
CELL_OWNER:    Final[np.ndarray] = np.array(_owner,         dtype=np.int8)   # -1 = NineGrid / off
SEAT_INDEX:    Final[np.ndarray] = np.array(_seat_index,    dtype=np.int8)


RAIL_CELLS: Final[tuple[tuple[int, int], ...]] = tuple(
    (_flat_to_xy(f)) for f in range(NC) if _is_railway[f]
)
RAIL_CELL_FLATS: Final[np.ndarray] = np.array(
    [_xy_to_flat(x, y) for (x, y) in RAIL_CELLS], dtype=np.int16,
)
NUM_RAIL_CELLS: Final[int] = len(RAIL_CELLS)
assert NUM_RAIL_CELLS == 73, (
    f"Expected 73 rail cells (64 outer-ring + 9 NineGrid), got {NUM_RAIL_CELLS}"
)


# ---------------------------------------------------------------------------
# Curve-rail identifier per cell (legacy InitCurveRail).
# ---------------------------------------------------------------------------
#
# for seat i in [0..3]:
#     for slot j in [0..29]:
#         if j%5 == 4:
#             ChessPos[i][j].eCurveRail           = i + 1
#             ChessPos[(i+1)%4][j-4].eCurveRail   = i + 1
#
# This produces a set of 12-cell "L-shaped" groups at the 4 inner corners.
#
# CAUTION — only 10 of those 12 cells are railway.  Slots 0..24 form the 5x5
# grid whose perimeter carries the rail; slots 25..29 are the headquarters row
# and carry none.  The legacy loop runs to j < 30, so `j % 5 == 4` also picks
# up j = 29, and its partner `j - 4` picks up slot 25 — two headquarters cells
# per curve.  Reproducing that is deliberate: this module mirrors the legacy
# engine and must not silently diverge from it.
#
# It is harmless for move generation, which checks `is_railway` on both
# endpoints before it ever consults the curve id (see
# `junqi_core.move_gen.is_legal_move`).  Any *other* consumer must apply the
# same filter, or it will pick up 48 cells where the board only has 40.
#
# Note also what this id does and does not mean geometrically.  It groups the
# two straight rail runs that meet at a board corner so that
# `_same_curve_rail` can let a non-engineer travel between them.  It does not
# say those cells are curved: the arc itself is a single edge per corner, and
# the cells adjoining one are `CURVE_ARC_CELLS` further down.

_curve_of: list[int] = [CURVE_RAIL_NONE] * NC


def _set_curve(dir_: int, slot: int, cid: int) -> None:
    x, y = _seat_pos(dir_, slot)
    f = _xy_to_flat(x, y)
    # If a cell appears on two curves (does not happen in practice), last wins.
    _curve_of[f] = cid


for _cid in range(1, 5):
    _seat_a = _cid - 1
    _seat_b = _cid % 4
    for _j in range(SEAT_SLOTS):
        if _j % 5 == 4:
            _set_curve(_seat_a, _j,     _cid)
            _set_curve(_seat_b, _j - 4, _cid)


CURVE_RAIL_OF: Final[np.ndarray] = np.array(_curve_of, dtype=np.int8)


# ---------------------------------------------------------------------------
# Rail adjacency graph (legacy InitBoardGraph + AddSpcNode).
# ---------------------------------------------------------------------------


def _build_rail_adjacency() -> dict[int, tuple[int, ...]]:
    """Return { flat_cell: (ordered tuple of flat_neighbours) }."""
    adj: dict[int, set[int]] = {
        _xy_to_flat(x, y): set()
        for (x, y) in RAIL_CELLS
    }

    def add(a: int, b: int) -> None:
        adj[a].add(b)
        adj[b].add(a)

    # (a) ortho neighbours that are also rails
    for (x, y) in RAIL_CELLS:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and _is_railway[_xy_to_flat(nx, ny)]:
                add(_xy_to_flat(x, y), _xy_to_flat(nx, ny))

    # (b) NineGrid 2-step ortho jumps — between NineGrid cells only.
    for (x, y) in RAIL_CELLS:
        f = _xy_to_flat(x, y)
        if not _is_ninegrid[f]:
            continue
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            nf = _xy_to_flat(nx, ny)
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE and _is_ninegrid[nf]:
                add(f, nf)

    # (c) AddSpcNode: 4 inner-corner diagonal edges.
    for (ax, ay), (bx, by) in _SPC_EDGES:
        add(_xy_to_flat(ax, ay), _xy_to_flat(bx, by))

    # Return as sorted tuples for determinism.
    return {k: tuple(sorted(v)) for k, v in adj.items()}


# AddSpcNode edges from legacy junqi.c:204-236
_SPC_EDGES: Final[tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = (
    ((10, 11), (11, 10)),   # South–East inner corner
    ((6, 11),  (5, 10)),    # South–West inner corner
    ((6, 5),   (5, 6)),     # West–North inner corner
    ((11, 6),  (10, 5)),    # North–East inner corner
)


# Keyed by flat cell id; empty tuple for non-rail cells (for convenience).
_adj_map = _build_rail_adjacency()
RAIL_ADJ: Final[tuple[tuple[int, ...], ...]] = tuple(
    _adj_map.get(f, ()) for f in range(NC)
)


def _build_curve_arc_cells() -> np.ndarray:
    """Cells that adjoin a curved (arc) rail link — the 8 corner endpoints.

    The arc is the property of an *edge*, not of a cell: each board corner has
    exactly one, the diagonal `_SPC_EDGES` link, and it is the only place a
    non-engineer may leave a straight rail run.  Everything else on the rail
    network is straight, so these 8 cells are the whole of the rail topology
    that ``IS_RAILWAY`` does not already say.

    Derived from the adjacency graph rather than from ``_SPC_EDGES`` directly,
    so it stays true to whatever the graph actually contains.
    """
    out = np.zeros(NC, dtype=bool)
    for f, nbrs in enumerate(RAIL_ADJ):
        fx, fy = f % BOARD_SIZE, f // BOARD_SIZE
        for n in nbrs:
            nx, ny = n % BOARD_SIZE, n // BOARD_SIZE
            if abs(fx - nx) == 1 and abs(fy - ny) == 1:
                out[f] = True
                break
    return out


CURVE_ARC_CELLS: Final[np.ndarray] = _build_curve_arc_cells()


# Sanity check: connectedness
def _check_connected() -> None:
    start = next(iter(_adj_map))
    seen = {start}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in _adj_map[c]:
            if n not in seen:
                seen.add(n)
                q.append(n)
    assert len(seen) == NUM_RAIL_CELLS, (
        f"Rail graph is disconnected: reached {len(seen)} of {NUM_RAIL_CELLS} cells"
    )


_check_connected()


# ---------------------------------------------------------------------------
# Derived tables for move generation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public convenience — for debugging / visualisation.
# ---------------------------------------------------------------------------


def rail_map_str() -> str:
    lines = []
    for y in range(BOARD_SIZE):
        row = []
        for x in range(BOARD_SIZE):
            f = _xy_to_flat(x, y)
            if _is_ninegrid[f]:
                row.append('N')
            elif _is_railway[f]:
                row.append('R')
            elif _is_on_board[f]:
                row.append('.')
            else:
                row.append(' ')
        lines.append(f"  y={y:2d}: " + ' '.join(row))
    return '\n'.join(lines)


if __name__ == "__main__":
    print(rail_map_str())
    print(f"\nTotal rail cells: {NUM_RAIL_CELLS}")
    print(f"Total undirected edges: {sum(len(v) for v in _adj_map.values()) // 2}")
    deg: dict[int, int] = {}
    for nbrs in _adj_map.values():
        deg[len(nbrs)] = deg.get(len(nbrs), 0) + 1
    print(f"Degree distribution: {sorted(deg.items())}")
