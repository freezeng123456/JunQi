"""Static lookup tables for move generation (Phase 0.4 M4 / ADR-125).

Everything here is **pure function of the board topology and of
``rules.py``** — built once at import time, shared by every
``GameState`` at zero runtime cost.  Runtime move-generation reduces to
vectorized masked lookups against these tables.

Authoritative topology source
-----------------------------
Rail data (`IS_RAIL_FLAT`, `IS_NINE_GRID_FLAT`, `RAIL_ADJ`, …) is re-exported
from :mod:`junqi_core.rail_topology`, which is the *single source of truth*
decoded directly from the legacy C engine (``legacy_engine/src/junqi.c``).
NineGrid cells ARE railways (legacy ``InitNineGrid`` sets ``isRailway=1``) and
the rail graph includes:

* Orthogonal rail↔rail edges.
* NineGrid ↔ NineGrid 2-step orthogonal "jump" edges.
* The 4 ``AddSpcNode`` diagonal edges at the inner-corner junctions.

Consequently:

* The engineer rail graph is **one** connected component of 73 cells — not
  four 16-cycles as the earlier (buggy) implementation assumed.
* Nodes can have degree up to 4 (curve corners + NineGrid hubs).
* A straight-line rail move follows a *graph walk constrained to the same
  row or column* — NineGrid 2-step jumps extend the longest ray to 12 cells.

Tables summary
--------------
    ADJ_STRAIGHT[flat, 4]              int16  orthogonal neighbours
        (U/D/L/R); -1 for off-board entries.
    ADJ_DIAG_INTO_CAMP[flat] -> list   int16  flats that are diagonal
        neighbours **and** at least one endpoint is a camp.  Stored as a
        per-src ndarray of variable length (usually 0-4 entries).
    STRAIGHT_RAIL_RAYS[flat, 4] -> ndarray[L] int16
        along +x/-x/+y/-y; each entry is the ORDERED sequence of rail
        flats starting from the first step away from ``flat``, walking
        the rail adjacency graph while staying on the same row/column.
        ``L`` varies (≤ 6).
    ENGINEER_RAIL_NEIGHBORS[flat] -> tuple[int, …]
        the rail-graph neighbours of ``flat`` used by the engineer BFS
        (includes NineGrid 2-step jumps and the 4 SpcNode corners).
        Empty for non-rail flats.
    CURVE_ID_OF[flat]                  int8   curve-rail id (>= 1); 0 =
        no curve.
    CURVE_CELLS[curve_id] -> ndarray[K] int16
        the flats that share a given curve id; used for curve BFS.
    CURVE_NEIGHBORS[flat] -> tuple[int, …]
        the rail-graph neighbours of ``flat`` that belong to the SAME
        curve-rail as ``flat`` (empty if ``flat`` has no curve).

Bitmasks (all ndarray[289] bool):
    IS_CAMP_FLAT, IS_STRONGHOLD_FLAT, IS_RAIL_FLAT,
    IS_NINE_GRID_FLAT, IS_ON_BOARD_FLAT.

PieceType lookups (ndarray[14] bool, indexed by PieceType.value):
    IS_IMMOBILE_TYPE, IS_ENGINEER_TYPE.

Invariant
---------
For every ``(src, dst)`` pair,
    move_gen.is_legal_move(pieces, src, dst, seat)
equals the result of consulting only these tables plus the current
``(cell_piece_id, piece_seat_arr)`` occupancy vectors — verified by the
bit-identity fuzz in ``tests/test_move_gen_parity.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Final

import numpy as np

from .board import (
    BOARD_SIZE,
    CELL_TABLE,
    NUM_CELLS,
    cell_info,
    is_camp,
    is_nine_grid,
    is_on_board,
    is_railway,
    is_stronghold,
    orthogonal_neighbors,
    xy_to_flat,
)
from .rail_topology import (
    CURVE_RAIL_OF as _CURVE_RAIL_OF_ARR,
    IS_NINEGRID as _IS_NG_ARR,
    IS_RAILWAY as _IS_RAIL_ARR,
    RAIL_ADJ as _RAIL_ADJ,
)
from .rules import ALL_PLACEABLE_PIECES, PieceType


# ===========================================================================
# Cell-indexed bool masks
# ===========================================================================

def _build_cell_bitmasks() -> dict[str, np.ndarray]:
    is_camp_arr = np.zeros(NUM_CELLS, dtype=bool)
    is_stronghold_arr = np.zeros(NUM_CELLS, dtype=bool)
    is_on_arr = np.zeros(NUM_CELLS, dtype=bool)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            flat = xy_to_flat(x, y)
            is_on_arr[flat] = is_on_board(x, y)
            if not is_on_arr[flat]:
                continue
            is_camp_arr[flat] = is_camp(x, y)
            is_stronghold_arr[flat] = is_stronghold(x, y)
    return {
        "camp": is_camp_arr,
        "stronghold": is_stronghold_arr,
        # Rail / NineGrid are re-exported verbatim from the SSoT module so
        # every consumer (CPU + CUDA) reads the same array.
        "rail": _IS_RAIL_ARR.copy(),
        "nine": _IS_NG_ARR.copy(),
        "on_board": is_on_arr,
    }


_BITMASKS = _build_cell_bitmasks()
IS_CAMP_FLAT:        Final[np.ndarray] = _BITMASKS["camp"]
IS_STRONGHOLD_FLAT:  Final[np.ndarray] = _BITMASKS["stronghold"]
IS_RAIL_FLAT:        Final[np.ndarray] = _BITMASKS["rail"]
IS_NINE_GRID_FLAT:   Final[np.ndarray] = _BITMASKS["nine"]
IS_ON_BOARD_FLAT:    Final[np.ndarray] = _BITMASKS["on_board"]


# ===========================================================================
# PieceType bitmasks (indexed by PieceType.value, size = 14)
# ===========================================================================

_NUM_PIECE_TYPES: Final[int] = 14   # all PieceType values 0..13 inclusive

def _build_piecetype_masks() -> tuple[np.ndarray, np.ndarray]:
    immobile = np.zeros(_NUM_PIECE_TYPES, dtype=bool)
    engineer = np.zeros(_NUM_PIECE_TYPES, dtype=bool)
    for pt in ALL_PLACEABLE_PIECES:
        immobile[pt.value] = pt.is_immobile
        engineer[pt.value] = pt.is_engineer
    return immobile, engineer


IS_IMMOBILE_TYPE, IS_ENGINEER_TYPE = _build_piecetype_masks()


# ===========================================================================
# ADJ_STRAIGHT[flat, 4] : int16, -1 for off-board
# order: +x, -x, +y, -y
# ===========================================================================

_DIRS_STRAIGHT: Final[tuple[tuple[int, int], ...]] = ((1, 0), (-1, 0), (0, 1), (0, -1))

def _build_adj_straight() -> np.ndarray:
    arr = np.full((NUM_CELLS, 4), -1, dtype=np.int16)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            flat = xy_to_flat(x, y)
            if not is_on_board(x, y):
                continue
            for k, (dx, dy) in enumerate(_DIRS_STRAIGHT):
                nx, ny = x + dx, y + dy
                if is_on_board(nx, ny):
                    arr[flat, k] = xy_to_flat(nx, ny)
    return arr


ADJ_STRAIGHT: Final[np.ndarray] = _build_adj_straight()


# ===========================================================================
# ADJ_DIAG_INTO_CAMP[flat] : per-src int16 array of diagonal neighbours
# that are legal 1-step endpoints (either src or dst is a camp).
# Empty arrays for cells that have no camp-involving diagonal neighbours.
# ===========================================================================

_DIRS_DIAG: Final[tuple[tuple[int, int], ...]] = ((1, 1), (1, -1), (-1, 1), (-1, -1))

def _build_adj_diag_into_camp() -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            entries: list[int] = []
            if is_on_board(x, y):
                src_is_camp = is_camp(x, y)
                for dx, dy in _DIRS_DIAG:
                    nx, ny = x + dx, y + dy
                    if not is_on_board(nx, ny):
                        continue
                    if src_is_camp or is_camp(nx, ny):
                        entries.append(xy_to_flat(nx, ny))
            out.append(tuple(entries))
    return out


ADJ_DIAG_INTO_CAMP: Final[tuple[tuple[int, ...], ...]] = tuple(_build_adj_diag_into_camp())


# ===========================================================================
# STRAIGHT_RAIL_RAYS[flat][k]  (k ∈ 0..3, matching _DIRS_STRAIGHT):
#
#   Graph-constrained linear rays: walk the rail adjacency graph from
#   ``flat`` while staying on the same row (for k=2,3) or same column
#   (for k=0,1) *and* moving monotonically in the declared direction.
#
#   Because within a single row or column every rail node has at most one
#   rail-graph neighbour in each signed direction, this walk is always a
#   simple chain (no branching).  The chain can span up to 6 cells in one
#   direction — e.g. column x=6 stretches (6,1)→(6,11) using NineGrid
#   2-step jumps and ortho edges.
#
#   Non-rail flats get four empty tuples.
# ===========================================================================


def _build_straight_rail_rays() -> list[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
]:
    """Compute per-cell ordered rail rays (+x, -x, +y, -y) using graph walk."""
    out: list[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = []
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            flat = xy_to_flat(x, y)
            if not _IS_RAIL_ARR[flat]:
                out.append(((), (), (), ()))
                continue

            rays: list[tuple[int, ...]] = []
            for dx, dy in _DIRS_STRAIGHT:
                cells: list[int] = []
                prev_xy = (x, y)
                while True:
                    px, py = prev_xy
                    # Find the unique rail-graph neighbour of prev that
                    # lies on the same constrained axis and sits strictly
                    # further along (dx, dy).
                    next_cell: int | None = None
                    for nb in _RAIL_ADJ[xy_to_flat(px, py)]:
                        nx = nb % BOARD_SIZE
                        ny = nb // BOARD_SIZE
                        if dx != 0:
                            # Moving along x: same y; nx must be on the
                            # dx-side of px.
                            if ny != y:
                                continue
                            if (nx - px) * dx <= 0:
                                continue
                        else:
                            if nx != x:
                                continue
                            if (ny - py) * dy <= 0:
                                continue
                        # Additional guard: the neighbour must also differ
                        # from src (it always will thanks to the monotone
                        # check, but keep explicit for clarity).
                        if nb == flat:
                            continue
                        # We require at most one matching neighbour — the
                        # chain must be linear within an axis.
                        if next_cell is not None:
                            raise AssertionError(
                                f"unexpected rail branch from ({px},{py}) "
                                f"dir=({dx},{dy}) — got {next_cell} and {nb}"
                            )
                        next_cell = nb
                    if next_cell is None:
                        break
                    cells.append(next_cell)
                    prev_xy = (next_cell % BOARD_SIZE, next_cell // BOARD_SIZE)
                rays.append(tuple(cells))

            out.append((rays[0], rays[1], rays[2], rays[3]))
    return out


STRAIGHT_RAIL_RAYS: Final[
    tuple[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]
] = tuple(_build_straight_rail_rays())


# ===========================================================================
# ENGINEER_RAIL_NEIGHBORS[flat] : rail-graph neighbours used by the engineer
# BFS. Empty for non-rail flats.  Includes NineGrid 2-step jumps and the
# 4 ``AddSpcNode`` diagonal corner edges (legacy InitBoardGraph).
# ===========================================================================


def _build_engineer_rail_neighbors() -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for flat in range(NUM_CELLS):
        if _IS_RAIL_ARR[flat]:
            out.append(tuple(int(n) for n in _RAIL_ADJ[flat]))
        else:
            out.append(())
    return tuple(out)


ENGINEER_RAIL_NEIGHBORS: Final[tuple[tuple[int, ...], ...]] = _build_engineer_rail_neighbors()


# ===========================================================================
# Engineer BFS precomputed tables (Phase 1a vectorized BFS)
#
# The rail graph is ONE connected component of 73 cells (not four 16-cycles
# as the older broken build assumed).  Nodes can have degree up to 4 at
# curve corners and NineGrid hubs.  We therefore cannot encode BFS as "2
# directed chains"; instead we publish the raw ragged adjacency, padded to
# ``_MAX_RAIL_ENG_NBRS`` per row, and keep a per-cell rail-index mapping so
# the runtime can use either flat-cell or rail-index space as convenient.
#
# ENG_RAIL_CELLS:    (R,) int32   flat cell ids of rail cells
# ENG_RAIL_TO_IDX:   (289,) int32 flat -> rail_idx mapping; -1 for non-rail
# ENG_RAIL_DEG:      (R,) int32   number of neighbours of each rail cell
# ===========================================================================


def _build_engineer_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rail_cells = np.nonzero(IS_RAIL_FLAT)[0].astype(np.int32)
    R = len(rail_cells)
    rail_to_idx = np.full(NUM_CELLS, -1, dtype=np.int32)
    rail_to_idx[rail_cells] = np.arange(R, dtype=np.int32)
    degrees = np.array(
        [len(_RAIL_ADJ[int(f)]) for f in rail_cells.tolist()],
        dtype=np.int32,
    )
    return rail_cells, rail_to_idx, degrees


ENG_RAIL_CELLS: Final[np.ndarray]
ENG_RAIL_TO_IDX: Final[np.ndarray]
ENG_RAIL_DEG: Final[np.ndarray]
ENG_RAIL_CELLS, ENG_RAIL_TO_IDX, ENG_RAIL_DEG = _build_engineer_tables()


# ===========================================================================
# Curve rails: CURVE_ID_OF[flat] + CURVE_CELLS[curve_id] + CURVE_NEIGHBORS[flat]
#
# CURVE_NEIGHBORS now walks the full rail graph (not just orthogonal
# neighbours), so curve rails that include NineGrid hubs or SpcNode diagonals
# are handled correctly.
# ===========================================================================


def _build_curve_tables() -> tuple[np.ndarray, dict[int, np.ndarray], tuple[tuple[int, ...], ...]]:
    curve_id_of = _CURVE_RAIL_OF_ARR.astype(np.int8, copy=True)
    cells_by_id: dict[int, list[int]] = {}
    for flat in range(NUM_CELLS):
        cid = int(curve_id_of[flat])
        if cid <= 0:
            continue
        cells_by_id.setdefault(cid, []).append(flat)

    cells_nd: dict[int, np.ndarray] = {
        cid: np.asarray(sorted(cells), dtype=np.int16)
        for cid, cells in cells_by_id.items()
    }

    # Per-flat neighbours within the same curve: use the rail graph so that
    # curve BFS works across NineGrid jumps / corners if any curve happens to
    # span those edges (it doesn't for current legacy data, but this keeps
    # the helper in lock-step with the SSoT rail graph).
    nbr_by_flat: list[tuple[int, ...]] = []
    for flat in range(NUM_CELLS):
        cid = int(curve_id_of[flat])
        if cid == 0:
            nbr_by_flat.append(())
            continue
        nbrs = tuple(
            int(n)
            for n in _RAIL_ADJ[flat]
            if int(curve_id_of[int(n)]) == cid
        )
        nbr_by_flat.append(nbrs)

    return curve_id_of, cells_nd, tuple(nbr_by_flat)


CURVE_ID_OF, _CURVE_CELLS_DICT, CURVE_NEIGHBORS = _build_curve_tables()
CURVE_CELLS: Final[dict[int, np.ndarray]] = dict(_CURVE_CELLS_DICT)


# ===========================================================================
# Curve-rail chain linearization for vectorized BFS (T-02 optimization)
#
# Each of the 4 curve rails is a simple chain (max degree 2 per node).
# We linearize each chain from one endpoint to the other, then for each
# cell on the curve provide two rays: "forward" and "backward" along the
# chain, analogous to STRAIGHT_RAIL_RAYS.  This enables the same
# prefix-product vectorization used for straight rails.
#
# CURVE_CHAIN_RAYS_PAD[flat, 2, 11] int16  (-1 padded)
#   For a cell on a curve with chain position k out of L active cells:
#     ray[0] = cells at positions k+1, k+2, ..., L-1  (forward)
#     ray[1] = cells at positions k-1, k-2, ..., 0    (backward)
#   Non-curve cells have all -1 entries.
#   Max ray length = 11 (a 12-cell chain has at most 11 steps from one end,
#   but only 10 active cells since 2 endpoints have degree 0).
# ===========================================================================

_MAX_CURVE_RAY_LEN: Final[int] = 11


def _linearize_curve(curve_id: int) -> list[int]:
    """Return the flats of curve ``curve_id`` in chain-walk order.

    Only includes cells with degree > 0 (connected to the chain).
    Starts from an endpoint (degree 1) and walks to the other endpoint.
    """
    cells = CURVE_CELLS[curve_id]
    # Filter to cells with degree > 0
    active = [int(f) for f in cells if len(CURVE_NEIGHBORS[int(f)]) > 0]
    if not active:
        return []
    # Find an endpoint (degree 1)
    endpoints = [f for f in active if len(CURVE_NEIGHBORS[f]) == 1]
    if not endpoints:
        # No endpoint found (shouldn't happen for our data)
        return active
    start = endpoints[0]
    # Walk the chain
    chain: list[int] = [start]
    visited = {start}
    while True:
        cur = chain[-1]
        found_next = False
        for nb in CURVE_NEIGHBORS[cur]:
            if nb not in visited:
                chain.append(nb)
                visited.add(nb)
                found_next = True
                break
        if not found_next:
            break
    return chain


def _build_curve_chain_rays_pad() -> np.ndarray:
    """Build per-cell forward/backward rays along the curve chain."""
    out = np.full((NUM_CELLS, 2, _MAX_CURVE_RAY_LEN), -1, dtype=np.int16)
    for cid in CURVE_CELLS:
        chain = _linearize_curve(cid)
        L = len(chain)
        # Build a position map: flat -> index in chain
        pos_of = {flat: idx for idx, flat in enumerate(chain)}
        for idx, flat in enumerate(chain):
            # Forward ray: idx+1, idx+2, ..., L-1
            fwd = chain[idx + 1:]
            for j, f in enumerate(fwd):
                if j < _MAX_CURVE_RAY_LEN:
                    out[flat, 0, j] = f
            # Backward ray: idx-1, idx-2, ..., 0
            bwd = list(reversed(chain[:idx]))
            for j, f in enumerate(bwd):
                if j < _MAX_CURVE_RAY_LEN:
                    out[flat, 1, j] = f
    return out


CURVE_CHAIN_RAYS_PAD: Final[np.ndarray] = _build_curve_chain_rays_pad()

# Bool mask: True for cells that sit on a curve rail AND have degree > 0
IS_CURVE_ACTIVE: Final[np.ndarray] = np.array([
    CURVE_ID_OF[f] > 0 and len(CURVE_NEIGHBORS[f]) > 0
    for f in range(NUM_CELLS)
], dtype=bool)


# ===========================================================================
# 2-D padded lookup tables (Phase 0.4 M4 "plan D" vectorization)
#
# The SoA batch hot path in ``move_gen.generate_legal_action_ids`` works on
# arrays of size K (the number of mobile same-seat pieces).  It needs every
# candidate-destination table in a FIXED-SHAPE padded int16 ndarray so
# fancy indexing works in a single shot:
#
#   * ADJ_STRAIGHT           — shape (289, 4)        already 2-D  (see above)
#   * ADJ_DIAG_INTO_CAMP_PAD — shape (289, 4) int16  -1 for empty slot
#   * STRAIGHT_RAIL_RAYS_PAD — shape (289, 4, _L) int16  rays of len <= _L
#                               -1 pads the tail of each ray
#   * ENGINEER_RAIL_NEIGHBORS_PAD — shape (289, 4) int16  rail graph nbrs
#
# Out-of-bounds / empty slots are encoded as -1 and are masked out at query
# time by combining with the live ``landable`` / ``empty`` arrays and a
# ``>= 0`` indexer safety check.  A conceptually simpler alternative is to
# map -1 to a sentinel cell ``NUM_CELLS`` (289) with ``landable`` padded to
# len 290 and forced to False — this removes the ``>= 0`` check, and the
# batch paths use that trick (see ``move_gen``).
# ===========================================================================

_MAX_DIAG_INTO_CAMP: Final[int] = 4

# Longest linear rail ray in a single direction.  Derived from the
# legacy-correct rail graph: column x=6 reaches 12 cells when starting at
# (6,1) and walking +y — the walk traverses NORTH's column-4 rail cells,
# then crosses the NineGrid via 2-step jumps at (6,6)→(6,8)→(6,10), and
# finally continues through SOUTH's column-4 rail cells to (6,15).  The
# symmetric case starting at (6,15) reaches 12 cells in -y.  Any other
# starting cell reaches at most 12 cells in any given direction.  Kept as
# a module-level constant so down-stream CUDA code can shadow the same
# width.
_MAX_STRAIGHT_RAIL_RAY_LEN: Final[int] = 12

# Max rail degree is 4 (curve corners + NineGrid hub).  Kept as a constant
# so the CUDA side can allocate identical padding.
_MAX_RAIL_ENG_NBRS: Final[int] = 4


def _pad_to_2d(
    variable: tuple[tuple[int, ...], ...],
    width: int,
) -> np.ndarray:
    """Convert tuple-of-tuples into (len(variable), width) int16 ndarray,
    padding missing entries with ``-1``."""
    out = np.full((len(variable), width), -1, dtype=np.int16)
    for i, tup in enumerate(variable):
        n = len(tup)
        assert n <= width, f"row {i} length {n} exceeds width {width}"
        if n:
            out[i, :n] = tup
    return out


ADJ_DIAG_INTO_CAMP_PAD: Final[np.ndarray] = _pad_to_2d(
    ADJ_DIAG_INTO_CAMP, _MAX_DIAG_INTO_CAMP
)
ENGINEER_RAIL_NEIGHBORS_PAD: Final[np.ndarray] = _pad_to_2d(
    ENGINEER_RAIL_NEIGHBORS, _MAX_RAIL_ENG_NBRS
)


def _build_straight_rail_rays_pad() -> np.ndarray:
    out = np.full(
        (NUM_CELLS, 4, _MAX_STRAIGHT_RAIL_RAY_LEN), -1, dtype=np.int16
    )
    for flat, rays in enumerate(STRAIGHT_RAIL_RAYS):
        for k, ray in enumerate(rays):
            n = len(ray)
            assert n <= _MAX_STRAIGHT_RAIL_RAY_LEN, (
                f"ray ({flat}, {k}) length {n} exceeds "
                f"{_MAX_STRAIGHT_RAIL_RAY_LEN}"
            )
            if n:
                out[flat, k, :n] = ray
    return out


STRAIGHT_RAIL_RAYS_PAD: Final[np.ndarray] = _build_straight_rail_rays_pad()


# ===========================================================================
# Totals / self-check
# ===========================================================================

def _total_bytes() -> int:
    total = 0
    total += IS_CAMP_FLAT.nbytes + IS_STRONGHOLD_FLAT.nbytes
    total += IS_RAIL_FLAT.nbytes + IS_NINE_GRID_FLAT.nbytes + IS_ON_BOARD_FLAT.nbytes
    total += IS_IMMOBILE_TYPE.nbytes + IS_ENGINEER_TYPE.nbytes
    total += ADJ_STRAIGHT.nbytes
    # tuple-of-int tables: rough count (each int ~28 B in CPython but we
    # only care for a sanity envelope).
    for t in ADJ_DIAG_INTO_CAMP:
        total += 32 * len(t)
    for rays in STRAIGHT_RAIL_RAYS:
        for r in rays:
            total += 32 * len(r)
    for t in ENGINEER_RAIL_NEIGHBORS:
        total += 32 * len(t)
    total += CURVE_ID_OF.nbytes
    for a in CURVE_CELLS.values():
        total += a.nbytes
    for t in CURVE_NEIGHBORS:
        total += 32 * len(t)
    return total


def _self_check() -> None:
    # 1) ADJ_STRAIGHT symmetry.
    for flat in range(NUM_CELLS):
        if not IS_ON_BOARD_FLAT[flat]:
            continue
        for k_fwd, k_bwd in ((0, 1), (2, 3)):
            nb = ADJ_STRAIGHT[flat, k_fwd]
            if nb >= 0:
                assert ADJ_STRAIGHT[nb, k_bwd] == flat, (
                    f"ADJ_STRAIGHT asymmetric at {flat} k_fwd={k_fwd}"
                )

    # 2) Rail flats connect to SOMETHING (either straight-ray or a 2-step
    #    NineGrid jump reachable only through the BFS graph).
    for flat in range(NUM_CELLS):
        if IS_RAIL_FLAT[flat]:
            has_ray = sum(len(r) for r in STRAIGHT_RAIL_RAYS[flat]) > 0
            has_nbr = len(ENGINEER_RAIL_NEIGHBORS[flat]) > 0
            assert has_ray or has_nbr, (
                f"rail flat {flat} has neither rays nor rail neighbours"
            )
            # Every rail cell MUST have at least one rail neighbour (single
            # connected component invariant).
            assert has_nbr, f"rail flat {flat} isolated in graph"

    # 3) ENGINEER_RAIL_NEIGHBORS empty for non-rail flats.
    for flat in range(NUM_CELLS):
        if not IS_RAIL_FLAT[flat]:
            assert len(ENGINEER_RAIL_NEIGHBORS[flat]) == 0

    # 4) PieceType table: at least JUNQI, DILEI immobile; GONGB engineer.
    assert IS_IMMOBILE_TYPE[PieceType.JUNQI.value]
    assert IS_IMMOBILE_TYPE[PieceType.DILEI.value]
    assert IS_ENGINEER_TYPE[PieceType.GONGB.value]

    # 5) Connectivity check: every rail cell is reachable from every other
    #    via the engineer BFS (single connected component, 73 cells).
    start = int(np.nonzero(IS_RAIL_FLAT)[0][0])
    seen: set[int] = {start}
    q: deque[int] = deque([start])
    while q:
        c = q.popleft()
        for n in ENGINEER_RAIL_NEIGHBORS[c]:
            if n not in seen:
                seen.add(n)
                q.append(n)
    num_rails = int(IS_RAIL_FLAT.sum())
    assert len(seen) == num_rails, (
        f"rail graph disconnected: reached {len(seen)} of {num_rails} cells"
    )
    assert num_rails == 73, (
        f"expected 73 rail cells (legacy parity), got {num_rails}"
    )

    # 6) 4 curve rails, each with exactly 12 cells (legacy InitCurveRail).
    assert set(CURVE_CELLS.keys()) == {1, 2, 3, 4}, CURVE_CELLS.keys()
    for cid, cells in CURVE_CELLS.items():
        assert len(cells) == 12, (
            f"curve rail {cid}: expected 12 cells, got {len(cells)}"
        )


_self_check()


if __name__ == "__main__":
    _self_check()
    print(
        f"junqi_core._movegen_tables self-check: OK "
        f"(total {_total_bytes() / 1024:.1f} KiB; "
        f"{len(CURVE_CELLS)} curve rails; "
        f"{int(IS_RAIL_FLAT.sum())} rail cells; "
        f"max ray len = {_MAX_STRAIGHT_RAIL_RAY_LEN}; "
        f"max rail degree = {_MAX_RAIL_ENG_NBRS})"
    )
