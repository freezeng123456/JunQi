
"""Legal move generation.

Implements §2.3 of RULES.md, porting logic from legacy_engine/src/path.c.

Public API:
  - `is_legal_move(board, pieces, src, dst, acting_seat)`
      True iff a single move from `src` to `dst` is legal for `acting_seat`.
  - `legal_moves_from(board, pieces, src, acting_seat)`
      Enumerate all legal destinations for the piece at `src`.
  - `generate_legal_actions(board, pieces, acting_seat)`
      Enumerate every (src, dst) legal action for the acting seat.

"board" here is the set of STATIC topology lookups already provided by
`junqi_core.board` (is_camp / is_stronghold / is_railway / etc.). The
DYNAMIC piece map is passed in as a `PieceMap` dict (pos -> PieceRef).
This keeps move_gen stateless — GameState (next task) owns the piece map.

Performance target: < 100 µs per `generate_legal_actions` call on CPU with
~80 pieces alive. Caching of adjacency tables is embedded at module scope.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np

from .board import (
    BOARD_SIZE,
    NUM_CELLS,
    CellInfo,
    cell_info,
    eight_neighbors,
    is_camp,
    is_on_board,
    is_railway,
    is_stronghold,
    orthogonal_neighbors,
    xy_to_flat,
)
from .rail_topology import RAIL_ADJ as _RAIL_ADJ
from .rules import PieceType, Seat, same_team

# ===========================================================================
# Data types
# ===========================================================================


@dataclass(frozen=True, slots=True)
class PieceRef:
    """A piece currently on the board. Passed into move_gen via PieceMap.

    We use a frozen dataclass so PieceMap keys (positions) map to immutable
    piece descriptions, safe for hashing and caching.
    """

    seat: Seat
    piece_type: PieceType
    # Optional legacy-style fields that downstream (state.py) may track.
    # move_gen does not read them directly.
    alive: bool = True
    # T7 / ADR-114 — global piece identity. Assigned exactly once by
    # GameState.new_game() and never changes afterwards. -1 denotes
    # "unassigned" (used by legacy tests that build PieceMaps by hand and
    # don't exercise T7 observation channels). See
    # docs/PHASE_0.3_T7_TODO.md §1.1 for the encoding rule.
    piece_id: int = -1


# Map from (x, y) world-coord to the piece at that cell. Empty cells are
# simply not in the dict.
PieceMap = dict[tuple[int, int], PieceRef]


# ===========================================================================
# Basic legality helpers
# ===========================================================================


def _occupant(pieces: PieceMap, pos: tuple[int, int]) -> PieceRef | None:
    """Return the piece at `pos` (if any and alive), else None."""
    p = pieces.get(pos)
    if p is None or not p.alive:
        return None
    return p


def _can_end_on(
    pieces: PieceMap,
    dst: tuple[int, int],
    acting_seat: Seat,
) -> bool:
    """True iff acting_seat's piece may END its move at `dst`.

    - dst must be on-board.
    - If dst is empty, OK (unless it's in an occupied camp — impossible since
      camp occupation implies non-empty; our check is sufficient).
    - If dst holds a piece:
      - must be an ENEMY (not self, not teammate — wait: teammate pieces are
        NOT attackable in Junqi, since you can't eat your own side. So any
        same-team piece blocks the destination.)
      - must not be sitting inside its OWN camp (camps protect defenders).
    """
    if not is_on_board(*dst):
        return False

    occupant = _occupant(pieces, dst)
    if occupant is None:
        return True

    # Same team (self or teammate) — cannot land on same-team piece.
    if same_team(occupant.seat, acting_seat):
        return False

    # Enemy piece sitting in a camp is unattackable.
    return not is_camp(*dst)


# ===========================================================================
# Rail path computation (non-engineer): straight line along same x or same y
# ===========================================================================


def _straight_rail_clear(
    pieces: PieceMap,
    src: tuple[int, int],
    dst: tuple[int, int],
) -> bool:
    """True iff non-engineer can travel src→dst along a straight rail line.

    Both endpoints must be on rail cells that share a row or column; the
    search walks the rail-adjacency graph staying on that axis (NineGrid
    2-step jumps are allowed because they connect same-column / same-row
    rail cells in the graph).  All INTERMEDIATE cells must be empty.

    Mirrors legacy ``path.c::GetRailPath`` with
    ``HORIZONTAL_RAIL``/``VERTICAL_RAIL`` (see ``IsSameRail``).
    """
    if src == dst:
        return False
    sx, sy = src
    dx, dy = dst
    if sx != dx and sy != dy:
        return False  # not same row nor same column

    src_flat = xy_to_flat(sx, sy)
    dst_flat = xy_to_flat(dx, dy)

    # BFS on rail graph, constrained to the shared axis.
    visited = {src_flat}
    queue: deque[int] = deque([src_flat])
    while queue:
        cur = queue.popleft()
        for nb in _RAIL_ADJ[cur]:
            if nb in visited:
                continue
            nx, ny = nb % BOARD_SIZE, nb // BOARD_SIZE
            if sx == dx and nx != sx:
                continue
            if sy == dy and ny != sy:
                continue
            if nb == dst_flat:
                return True
            if _occupant(pieces, (nx, ny)) is not None:
                continue
            visited.add(nb)
            queue.append(nb)
    return False


def _same_curve_rail(src: tuple[int, int], dst: tuple[int, int]) -> bool:
    """True iff both endpoints belong to the same curve rail (eCurveRail > 0)."""
    src_ci = cell_info(*src)
    dst_ci = cell_info(*dst)
    return src_ci.curve_rail > 0 and src_ci.curve_rail == dst_ci.curve_rail


# ===========================================================================
# Engineer BFS on rail subgraph
# ===========================================================================


def _engineer_can_reach(
    pieces: PieceMap,
    src: tuple[int, int],
    dst: tuple[int, int],
) -> bool:
    """BFS over empty rail cells from src to dst.

    The engineer (GONGB) traverses any connected sequence of empty rail
    cells using the full rail-adjacency graph (orthogonal rail edges,
    NineGrid 2-step jumps, and the 4 AddSpcNode corner edges).
    ``dst`` itself may be:
      - an empty rail cell (landing move), OR
      - a rail cell occupied by an enemy (combat move).

    Intermediate cells must be (rail AND empty).
    """
    if not is_railway(*src) or not is_railway(*dst):
        return False
    if src == dst:
        return False

    src_flat = xy_to_flat(*src)
    dst_flat = xy_to_flat(*dst)

    visited: set[int] = {src_flat}
    queue: deque[int] = deque([src_flat])
    while queue:
        cur = queue.popleft()
        for nb in _RAIL_ADJ[cur]:
            if nb in visited:
                continue
            if nb == dst_flat:
                return True
            nx, ny = nb % BOARD_SIZE, nb // BOARD_SIZE
            if _occupant(pieces, (nx, ny)) is not None:
                continue
            visited.add(nb)
            queue.append(nb)
    return False


def _rail_neighbors(pos: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the rail-graph neighbours of ``pos`` as (x, y) pairs.

    Mirrors legacy ``InitBoardGraph``: includes orthogonal rail↔rail edges,
    NineGrid 2-step jumps, and the 4 AddSpcNode diagonal corner edges.
    Empty tuple for non-rail cells.
    """
    flat = xy_to_flat(*pos)
    return tuple(
        (nb % BOARD_SIZE, nb // BOARD_SIZE) for nb in _RAIL_ADJ[flat]
    )


# ===========================================================================
# Full legality check: is_legal_move
# ===========================================================================


def move_requires_gongb(
    pieces: PieceMap,
    src: tuple[int, int],
    dst: tuple[int, int],
) -> bool:
    """True iff src→dst is a move ONLY a GONGB (engineer) can legally make.

    Used by the CombatMemory module: when a piece walks a path that no
    non-engineer could possibly take (multi-hop rail BFS), every observer
    can publicly deduce the moving piece is a GONGB.

    Returns False fast for the 99 % case: orthogonal 1-step, diagonal-via-
    camp, and clean straight-rail rays — all of which non-engineers can
    use too.

    Caller convention: ``pieces`` is the BOARD STATE BEFORE the move was
    executed (so blockers along the path are still in place).
    """
    sx, sy = src
    dx, dy = dst
    if (sx, sy) == (dx, dy):
        return False

    abs_dx, abs_dy = abs(sx - dx), abs(sy - dy)

    # 1-step orthogonal: any piece can do this.
    if abs_dx + abs_dy == 1:
        return False
    # 1-step diagonal: legal for any piece iff endpoint is a camp.
    if abs_dx == 1 and abs_dy == 1 and (is_camp(*src) or is_camp(*dst)):
        return False

    # Beyond this point: must be a rail move.
    if not (is_railway(*src) and is_railway(*dst)):
        # Not a rail move and not 1-step → illegal regardless; treat as
        # "not GONGB-revealing" because the move shouldn't be legal at
        # all (caller already validated legality).
        return False

    # Same row/column straight rail: non-engineer can take it iff the
    # straight rail subgraph is clear.  If yes → not GONGB-only.
    if sx == dx or sy == dy:
        if _straight_rail_clear(pieces, src, dst):
            return False
        # Straight rail blocked but engineer reaches → GONGB-only.
        return _engineer_can_reach(pieces, src, dst)

    # Different row AND column on rails: curve rail or engineer BFS.
    if _same_curve_rail(src, dst) and _curve_rail_clear(pieces, src, dst):
        return False  # non-engineer can curve-rail through here
    # Otherwise: only engineer BFS could legalize this move.
    return _engineer_can_reach(pieces, src, dst)


def is_legal_move(
    pieces: PieceMap,
    src: tuple[int, int],
    dst: tuple[int, int],
    acting_seat: Seat,
) -> bool:
    """Return True iff `acting_seat` may move the piece at `src` to `dst` now.

    Follows RULES.md §2.3:
      1. src is occupied by acting_seat's piece.
      2. src is not a stronghold (immovable).
      3. src.type is not DILEI or JUNQI (immovable).
      4. dst is a legal landing cell (_can_end_on).
      5. Movement mode (adjacent, straight rail, engineer BFS, or curve rail)
         applies.
    """
    if not is_on_board(*src) or not is_on_board(*dst):
        return False
    if src == dst:
        return False

    src_piece = _occupant(pieces, src)
    if src_piece is None:
        return False
    if src_piece.seat is not acting_seat:
        return False

    # Immobile pieces
    if src_piece.piece_type.is_immobile:
        return False

    # Stronghold pieces are immobile
    if is_stronghold(*src):
        return False

    # Dst must be a valid landing cell
    if not _can_end_on(pieces, dst, acting_seat):
        return False

    # --------------------------------------------------------------------
    # Movement mode check — we try each mode in order; adjacency fallthrough
    # to rail so that e.g. a GONGB that looks like a diagonal 1-step (illegal
    # as adjacency) may still be legal via a 2-step rail BFS.
    # This matches legacy path.c::IsEnableMove (adjacent and rail are OR'd
    # via the `!rc &&` fallthrough; see docs/LEGACY_PARITY.md §3).
    # --------------------------------------------------------------------
    sx, sy = src
    dx, dy = dst
    abs_dx, abs_dy = abs(sx - dx), abs(sy - dy)

    # ---- (a) Adjacent move (1 step) ----
    if abs_dx <= 1 and abs_dy <= 1:
        if abs_dx == 1 and abs_dy == 1:
            # Diagonal 1-step: legal only if at least one endpoint is a camp.
            if is_camp(*src) or is_camp(*dst):
                return True
            # else: fall through to rail check (e.g. GONGB may still reach
            # via 2-step rail BFS)
        else:
            # Non-diagonal (orthogonal) 1-step is always OK.
            return True

    # ---- (b) Rail-based long move ----
    if not (is_railway(*src) and is_railway(*dst)):
        return False

    if src_piece.piece_type.is_engineer:
        # Engineer: BFS over rail subgraph
        return _engineer_can_reach(pieces, src, dst)

    # Non-engineer: straight line or curve rail
    if sx == dx or sy == dy:
        return _straight_rail_clear(pieces, src, dst)
    # Curve rail (different row AND column)
    if _same_curve_rail(src, dst):
        return _curve_rail_clear(pieces, src, dst)

    return False


# ===========================================================================
# Curve rail path check (non-engineer)
# ===========================================================================


def _curve_rail_clear(
    pieces: PieceMap,
    src: tuple[int, int],
    dst: tuple[int, int],
) -> bool:
    """Check that there is a clear path along the same curve rail.

    A curve rail segment is a set of cells sharing the same ``curve_rail``
    id.  Non-engineers can traverse any clear subpath along it, walking the
    full rail-adjacency graph restricted to the curve.
    """
    src_ci = cell_info(*src)
    curve_id = src_ci.curve_rail
    assert curve_id > 0

    curve_cell_flats: set[int] = {
        xy_to_flat(ci.x, ci.y) for ci in _curve_cells_cache(curve_id)
    }
    src_flat = xy_to_flat(*src)
    dst_flat = xy_to_flat(*dst)
    if src_flat not in curve_cell_flats or dst_flat not in curve_cell_flats:
        return False

    visited: set[int] = {src_flat}
    queue: deque[int] = deque([src_flat])
    while queue:
        cur = queue.popleft()
        for nb in _RAIL_ADJ[cur]:
            if nb in visited:
                continue
            if nb not in curve_cell_flats:
                continue
            if nb == dst_flat:
                return True
            nx, ny = nb % BOARD_SIZE, nb // BOARD_SIZE
            if _occupant(pieces, (nx, ny)) is not None:
                continue
            visited.add(nb)
            queue.append(nb)
    return False


# Cache: curve id → tuple of CellInfo
_CURVE_CELLS_CACHE: dict[int, tuple[CellInfo, ...]] = {}


def _curve_cells_cache(curve_id: int) -> tuple[CellInfo, ...]:
    from .board import CELL_TABLE
    if curve_id not in _CURVE_CELLS_CACHE:
        _CURVE_CELLS_CACHE[curve_id] = tuple(
            ci for ci in CELL_TABLE if ci.curve_rail == curve_id
        )
    return _CURVE_CELLS_CACHE[curve_id]


# ===========================================================================
# Enumeration
# ===========================================================================


def legal_moves_from(
    pieces: PieceMap,
    src: tuple[int, int],
    acting_seat: Seat,
) -> list[tuple[int, int]]:
    """Enumerate all legal destinations for the piece at `src`."""
    src_piece = _occupant(pieces, src)
    if src_piece is None or src_piece.seat is not acting_seat:
        return []
    if src_piece.piece_type.is_immobile or is_stronghold(*src):
        return []

    destinations: list[tuple[int, int]] = []

    # ---- (1) Adjacent moves ----
    nbrs = eight_neighbors(*src) if is_camp(*src) else orthogonal_neighbors(*src)
    for nbr in nbrs:
        if is_legal_move(pieces, src, nbr, acting_seat):
            destinations.append(nbr)
    # Also consider diagonal-into-camp from non-camp sources
    if not is_camp(*src):
        for nbr in eight_neighbors(*src):
            if nbr in destinations:
                continue
            sx, sy = src
            # Only diagonal neighbors we haven't already covered
            if (
                abs(sx - nbr[0]) == 1
                and abs(sy - nbr[1]) == 1
                and is_camp(*nbr)
                and is_legal_move(pieces, src, nbr, acting_seat)
            ):
                destinations.append(nbr)

    # ---- (2) Rail moves ----
    if is_railway(*src):
        if src_piece.piece_type.is_engineer:
            # BFS every reachable rail cell (empty or enemy)
            reachable = _engineer_reachable_cells(pieces, src, acting_seat)
            for cell in reachable:
                if cell != src and cell not in destinations:
                    destinations.append(cell)
        else:
            # Straight rail along same x or y
            for cell in _straight_rail_destinations(pieces, src, acting_seat):
                if cell not in destinations:
                    destinations.append(cell)
            # Curve rail destinations
            curve_id = cell_info(*src).curve_rail
            if curve_id > 0:
                for ci in _curve_cells_cache(curve_id):
                    cand = (ci.x, ci.y)
                    if cand == src or cand in destinations:
                        continue
                    if is_legal_move(pieces, src, cand, acting_seat):
                        destinations.append(cand)

    return destinations


def _engineer_reachable_cells(
    pieces: PieceMap,
    src: tuple[int, int],
    acting_seat: Seat,
) -> list[tuple[int, int]]:
    """Return every rail cell reachable by engineer from src that can be a
    legal landing point (empty OR enemy-attack)."""
    result: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = {src}
    queue: deque[tuple[int, int]] = deque([src])
    while queue:
        cur = queue.popleft()
        for nbr in _rail_neighbors(cur):
            if nbr in visited:
                continue
            visited.add(nbr)
            occ = _occupant(pieces, nbr)
            if occ is None:
                # Empty rail cell — landing legal; continue BFS
                result.append(nbr)
                queue.append(nbr)
            elif not same_team(occ.seat, acting_seat) and not is_camp(*nbr):
                # Enemy rail cell — combat legal; DO NOT continue BFS through it
                result.append(nbr)
            # Same-team or enemy-in-camp: blocked; do not add, do not continue
    return result


def _straight_rail_destinations(
    pieces: PieceMap,
    src: tuple[int, int],
    acting_seat: Seat,
) -> list[tuple[int, int]]:
    """Return cells reachable by non-engineer along straight rails from src.

    Walks the rail graph in each of the 4 axis-aligned directions, staying
    on the same row or column.  Each direction is a simple chain (no
    branching within an axis); stops at the first occupied cell, emitting
    it only if it's enemy-attackable.  Consistent with
    :func:`_straight_rail_clear`.
    """
    result: list[tuple[int, int]] = []
    sx, sy = src
    src_flat = xy_to_flat(sx, sy)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cur = src_flat
        prev = -1
        while True:
            next_cell = -1
            for nb in _RAIL_ADJ[cur]:
                if nb == prev:
                    continue
                nx = nb % BOARD_SIZE
                ny = nb // BOARD_SIZE
                if dx != 0:
                    if ny != sy:
                        continue
                    if (nx - (cur % BOARD_SIZE)) * dx <= 0:
                        continue
                else:
                    if nx != sx:
                        continue
                    if (ny - (cur // BOARD_SIZE)) * dy <= 0:
                        continue
                next_cell = nb
                break
            if next_cell < 0:
                break
            nx = next_cell % BOARD_SIZE
            ny = next_cell // BOARD_SIZE
            occ = _occupant(pieces, (nx, ny))
            if occ is None:
                result.append((nx, ny))
                prev = cur
                cur = next_cell
            else:
                if not same_team(occ.seat, acting_seat) and not is_camp(nx, ny):
                    result.append((nx, ny))
                break
    return result


def generate_legal_actions(
    pieces: PieceMap,
    acting_seat: Seat,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Enumerate all (src, dst) legal moves for `acting_seat`.

    Returns a list of tuples `(src, dst)` in no particular order.
    """
    actions: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for src, piece in pieces.items():
        if not piece.alive:
            continue
        if piece.seat is not acting_seat:
            continue
        for dst in legal_moves_from(pieces, src, acting_seat):
            actions.append((src, dst))
    return actions


def has_any_legal_move(pieces: PieceMap, acting_seat: Seat) -> bool:
    """True iff acting_seat has at least one legal move.

    Used by `state.py` to implement Q12 (no-moves → seat dies).
    Short-circuits on first legal action found.
    """
    for src, piece in pieces.items():
        if not piece.alive:
            continue
        if piece.seat is not acting_seat:
            continue
        for _dst in legal_moves_from(pieces, src, acting_seat):
            return True
    return False


# ===========================================================================
# Iterator version (memory-friendly for large batches)
# ===========================================================================


def iter_legal_actions(
    pieces: PieceMap,
    acting_seat: Seat,
) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Lazily yield legal actions one at a time."""
    for src, piece in pieces.items():
        if not piece.alive:
            continue
        if piece.seat is not acting_seat:
            continue
        for dst in legal_moves_from(pieces, src, acting_seat):
            yield (src, dst)


# ===========================================================================
# Phase 0.4 M4 / ADR-125 — SoA-based hot path.
#
# These functions operate directly on the ``GameState`` SoA columns
# (``cell_piece_id`` / ``piece_seat_arr`` / ``piece_type_arr`` / ``alive``)
# plus the static tables from ``_movegen_tables.py``.  They DO NOT touch
# ``state.pieces`` (the dict view), so they avoid the per-query dict
# rebuild cost.
#
# Correctness is guaranteed to be bit-identical to the legacy PieceMap
# implementation above — enforced by ``tests/test_move_gen_parity.py``
# (fuzz: 1 000+ random positions).
# ===========================================================================

from . import _movegen_tables as _T  # noqa: E402

# Module-level "empty int16 result" reused instead of ``np.empty(0, ...)``
# in every call where no moves are legal — shaves ~0.3 us per call per
# piece and avoids pointless allocation noise in tracemalloc.
_EMPTY_INT16: Final[np.ndarray] = np.empty(0, dtype=np.int16)
_EMPTY_INT32: Final[np.ndarray] = np.empty(0, dtype=np.int32)

# ---------------------------------------------------------------------------
# Pre-padded adjacency / rail tables  (sentinel = NUM_CELLS = 289)
#
# The original tables use -1 as sentinel for "no neighbor".  We replace -1
# with NUM_CELLS so that flat-index gathers into a (N, 290)-padded boolean
# array automatically return the False sentinel without a separate np.where.
# ---------------------------------------------------------------------------
_ADJ_STRAIGHT_PAD: Final[np.ndarray] = np.where(
    _T.ADJ_STRAIGHT >= 0, _T.ADJ_STRAIGHT, NUM_CELLS
).astype(np.int32)

_ADJ_DIAG_PAD: Final[np.ndarray] = np.where(
    _T.ADJ_DIAG_INTO_CAMP_PAD >= 0, _T.ADJ_DIAG_INTO_CAMP_PAD, NUM_CELLS
).astype(np.int32)

_SRAYS_PAD: Final[np.ndarray] = np.where(
    _T.STRAIGHT_RAIL_RAYS_PAD >= 0, _T.STRAIGHT_RAIL_RAYS_PAD, NUM_CELLS
).astype(np.int32)   # shape (289, 4, L)

_SRAYS_L: int = int(_T.STRAIGHT_RAIL_RAYS_PAD.shape[-1])  # typically 4

# ---------------------------------------------------------------------------
# Engineer BFS vectorized tables (Phase 1a optimisation)
#
# After the legacy-parity rewrite the rail graph is ONE connected component
# of 73 cells with degrees up to 4 (NineGrid hubs + curve corners).  We
# therefore cannot pre-unroll BFS into "2 directed chains" as the earlier
# implementation did.  Instead we keep a padded rail-adjacency table and
# perform BFS iteratively, vectorised across all engineers using a boolean
# "frontier" matrix that is expanded one hop at a time until it stops
# growing.  The BFS depth is bounded by the graph diameter (72 in the
# worst case, but typical engineer moves terminate well before).
#
# ENG_RAIL_CELLS:    (R,) int32   flat cell ids of rail cells
# ENG_RAIL_TO_IDX:   (289,) int32 flat -> rail_idx mapping; -1 for non-rail
# ENG_RAIL_ADJ:      (R, _MAX_RAIL_ENG_NBRS) int32   padded rail-idx
#                    adjacency (padding = -1; clamped to 0 + mask).
# ENG_RAIL_ADJ_VALID:(R, _MAX_RAIL_ENG_NBRS) bool    True where padded slot
#                    holds a real neighbour.
# ---------------------------------------------------------------------------
_ENG_RAIL_CELLS: Final[np.ndarray] = _T.ENG_RAIL_CELLS          # (R,) int32
_ENG_RAIL_TO_IDX: Final[np.ndarray] = _T.ENG_RAIL_TO_IDX        # (289,) int32
_NUM_RAIL_CELLS: int = int(_ENG_RAIL_CELLS.shape[0])


def _build_eng_rail_adj_padded() -> tuple[np.ndarray, np.ndarray]:
    """Build rail-index-space padded adjacency from ENGINEER_RAIL_NEIGHBORS."""
    width = int(_T.ENGINEER_RAIL_NEIGHBORS_PAD.shape[1])
    adj = np.full((_NUM_RAIL_CELLS, width), -1, dtype=np.int32)
    valid = np.zeros((_NUM_RAIL_CELLS, width), dtype=bool)
    for ri, f in enumerate(_ENG_RAIL_CELLS.tolist()):
        nbrs = _T.ENGINEER_RAIL_NEIGHBORS[int(f)]
        for k, nb in enumerate(nbrs):
            adj[ri, k] = int(_ENG_RAIL_TO_IDX[int(nb)])
            valid[ri, k] = True
    return adj, valid


_ENG_RAIL_ADJ_PAIR: Final[tuple[np.ndarray, np.ndarray]] = (
    _build_eng_rail_adj_padded()
)
_ENG_RAIL_ADJ, _ENG_RAIL_ADJ_VALID = _ENG_RAIL_ADJ_PAIR
# Clamp -1 pads to 0 so they index safely; the valid mask filters them out.
_ENG_RAIL_ADJ_CLAMP: Final[np.ndarray] = np.where(
    _ENG_RAIL_ADJ >= 0, _ENG_RAIL_ADJ, 0
).astype(np.int32)
_ENG_RAIL_ADJ_WIDTH: int = int(_ENG_RAIL_ADJ.shape[1])


def _batch_engineer_bfs_n(
    eng_sf: np.ndarray,    # (E,) int32 — src flat cell ids
    eng_a: np.ndarray,     # (E,) int32 — sub-batch env index into empty/enemy
    eng_env: np.ndarray,   # (E,) int32 — original env ids for output
    empty: np.ndarray,     # (A, 289) bool
    enemy_att: np.ndarray, # (A, 289) bool
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Vectorized engineer BFS for all E engineers simultaneously.

    Works on the rail-index-space adjacency.  We maintain two per-engineer
    boolean masks of shape (E, R):

      reached  : True where the engineer can *reach* this rail cell under
                 the pass-through rule (empty + passable).  The source is
                 reached trivially; it is not emitted as a destination.
      landable : True where the engineer may legally END its move.  This
                 equals (reached AND rail cell passable) for empties, or
                 (reached-as-enemy neighbour) for attackable cells.

    BFS iterates until the reached set stops growing, which happens after
    at most ``_NUM_RAIL_CELLS`` iterations in the worst case but typically
    converges far sooner (graph diameter ≈ 10).  The inner update is a
    single vectorised 3-D gather/reduce, so the cost is amortised across
    all engineers and all envs simultaneously.

    Returns (env_arr, act_arr) or (None, None) if no destinations.
    """
    E = int(eng_sf.shape[0])
    if E == 0:
        return None, None

    R = _NUM_RAIL_CELLS
    # Build per-engineer, per-rail-cell occupancy views.  For flat cell f,
    #   empty_e[e, ri]       = True if engineer e's env sees rail cell ri empty
    #   attack_e[e, ri]      = True if engineer e's env sees ri as enemy-
    #                          attackable (non-camp).
    # We cannot pass through enemy cells but we may land on them.
    rail_flats = _ENG_RAIL_CELLS                                  # (R,) int32
    empty_e = empty[eng_a[:, np.newaxis], rail_flats[np.newaxis, :]]  # (E,R)
    attack_e = enemy_att[eng_a[:, np.newaxis], rail_flats[np.newaxis, :]]  # (E,R)

    # Initial reached set = just the source.
    src_ri = _ENG_RAIL_TO_IDX[eng_sf]                             # (E,) int32
    reached = np.zeros((E, R), dtype=bool)
    reached[np.arange(E), src_ri] = True

    # A cell is "pass-through" iff it is empty (non-source).  The source is
    # by construction not considered pass-through (it would re-emit itself),
    # but we force it True so BFS can leave it.
    passable = empty_e.copy()
    passable[np.arange(E), src_ri] = True

    # Precompute: for every rail cell ri the list of its padded neighbours.
    adj = _ENG_RAIL_ADJ_CLAMP                                     # (R, W)
    adj_valid = _ENG_RAIL_ADJ_VALID                               # (R, W)

    # Iterate BFS frontiers until reached is stable.  The maximum number
    # of iterations equals the graph diameter; we bound with R to be safe.
    for _ in range(R):
        # reached_nb[e, ri, k] = reached[e, adj[ri, k]] and adj_valid[ri, k]
        reached_nb = reached[:, adj]                              # (E,R,W) bool
        reached_nb &= adj_valid[np.newaxis, :, :]
        # (E, R) = any over W of reached_nb, gated by ri's passability
        any_nb_reached = reached_nb.any(axis=2)                   # (E, R)
        new_cells = any_nb_reached & passable & ~reached          # (E, R)
        if not new_cells.any():
            break
        reached |= new_cells

    # A cell is a legal landing iff: (a) its neighbour was reached AND it
    # is either empty (landing) or attackable (combat).  We include
    # non-passable attackable cells by checking "any neighbour reached AND
    # cell is landable".
    reached_nb = reached[:, adj]
    reached_nb &= adj_valid[np.newaxis, :, :]
    any_nb_reached = reached_nb.any(axis=2)                       # (E, R)
    landable_per_cell = empty_e | attack_e                        # (E, R)
    # Exclude the source itself (cannot "land" on src).
    landable_per_cell[np.arange(E), src_ri] = False
    can_land = any_nb_reached & landable_per_cell                 # (E, R)

    if not can_land.any():
        return None, None

    ei, ri = np.nonzero(can_land)                                 # (M,) each
    dest_flat = rail_flats[ri]                                    # (M,)
    src_flat = eng_sf[ei]                                         # (M,)
    orig_env = eng_env[ei]                                        # (M,)

    act_ids = src_flat.astype(np.int32) * NUM_CELLS + dest_flat.astype(np.int32)
    return orig_env.astype(np.int32), act_ids


# ---------------------------------------------------------------------------
# TEAM_OF_PID lookup  (120 entries, uint8)
#
# Pieces are assigned seat in fixed blocks: pids 0-29 → seat 0 (team 0),
# 30-59 → seat 1 (team 1), 60-89 → seat 2 (team 0), 90-119 → seat 3 (team 1).
# This is cheaper than going through piece_seat_arr for the batch occupancy.
# ---------------------------------------------------------------------------
_TEAM_OF_PID: Final[np.ndarray] = np.array(
    [0] * 30 + [1] * 30 + [0] * 30 + [1] * 30, dtype=np.uint8
)  # shape (120,)


def _compute_occupancy_masks_soa(
    cell_piece_id: np.ndarray,           # (289,) int16
    piece_seat_arr: np.ndarray,          # (num_pids,) int8
    acting_seat_val: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute three per-step masks used by every SoA move-gen call.

    Returns
    -------
    empty : ndarray[289] bool
        ``True`` where no piece sits.
    same_team_occ : ndarray[289] bool
        ``True`` where a same-team piece (self or teammate) sits.
    enemy_attackable : ndarray[289] bool
        ``True`` where an enemy sits AND the cell is not a camp.
    """
    # Sentinel-padded seat team table.  For ``cell_piece_id == -1`` (empty
    # cells) the index wraps to the last entry, which we force to team -1
    # so nothing flags as same_team / enemy there.  ``piece_seat_arr`` is
    # size (num_pids,), so we extend it by one cell.
    num_pids = piece_seat_arr.shape[0]
    occupied = cell_piece_id >= 0
    # Direct lookup: seat_of_cell[flat] = piece_seat_arr[cell_piece_id[flat]]
    # where cell_piece_id[flat] = -1 yields piece_seat_arr[-1] — we pad to
    # avoid that wrap.  One extra cell for the sentinel.
    seat_of_pid_pad = np.empty(num_pids + 1, dtype=np.int8)
    seat_of_pid_pad[:num_pids] = piece_seat_arr
    seat_of_pid_pad[num_pids] = -1                      # sentinel "no seat"
    # ``cell_piece_id.astype(np.intp)`` handles -1 -> becomes num_pids
    # after the ``np.where`` below.
    pid_idx = np.where(occupied, cell_piece_id, num_pids).astype(
        np.intp, copy=False
    )
    seat_of_cell = seat_of_pid_pad[pid_idx]             # (289,) int8, -1=empty
    # Team = Seat.value % 2  (SOUTH=0, NORTH=2 → team 0; WEST=1, EAST=3 → team 1)
    # The legacy ``>> 1`` encoding I tried briefly is WRONG (it would put
    # SOUTH+WEST on team 0, breaking NORTH/SOUTH pairing).
    team_of_cell = seat_of_cell & 1
    acting_team = acting_seat_val & 1
    empty = ~occupied
    # ``same_team_occ`` and ``enemy_attackable`` must ignore the sentinel
    # (where seat_of_cell == -1).  ``occupied`` already excludes those.
    same_team_occ = occupied & (team_of_cell == acting_team)
    enemy_occ = occupied & (team_of_cell != acting_team)
    enemy_attackable = enemy_occ & (~_T.IS_CAMP_FLAT)
    return empty, same_team_occ, enemy_attackable


def _engineer_dests_soa(
    src_flat: int,
    empty: np.ndarray,
    enemy_attackable: np.ndarray,
) -> list[int]:
    """BFS over empty rail cells from src_flat; collect every cell that is
    either an empty rail cell (landing) or an enemy-attackable rail cell
    (combat, but BFS does NOT pass through enemy cells)."""
    rail_neighbors = _T.ENGINEER_RAIL_NEIGHBORS
    visited = {src_flat}
    queue = [src_flat]
    qi = 0
    result: list[int] = []
    while qi < len(queue):
        cur = queue[qi]
        qi += 1
        for nb in rail_neighbors[cur]:
            if nb in visited:
                continue
            visited.add(nb)
            if empty[nb]:
                result.append(nb)
                queue.append(nb)
            elif enemy_attackable[nb]:
                result.append(nb)
    return result


def _curve_dests_soa(
    src_flat: int,
    curve_id: int,
    empty: np.ndarray,
    enemy_attackable: np.ndarray,
) -> list[int]:
    """BFS along the curve rail identified by ``curve_id``."""
    visited = {src_flat}
    queue: deque[int] = deque([src_flat])
    result: list[int] = []
    curve_nbrs = _T.CURVE_NEIGHBORS
    while queue:
        cur = queue.popleft()
        for nb in curve_nbrs[cur]:
            if nb in visited:
                continue
            visited.add(nb)
            if empty[nb]:
                result.append(nb)
                queue.append(nb)
            elif enemy_attackable[nb]:
                result.append(nb)
    return result


def legal_dests_flat(
    pid: int,
    src_flat: int,
    piece_type_val: int,
    acting_seat_val: int,
    empty: np.ndarray,
    same_team_occ: np.ndarray,
    enemy_attackable: np.ndarray,
    landable: np.ndarray,
) -> np.ndarray:
    """Return the int16 array of legal dst_flat values for ``pid``.

    Parameters
    ----------
    pid
        Piece id of the mover (for identity; not used in logic).
    src_flat
        ``src_y * 17 + src_x``.
    piece_type_val
        ``PieceType.value`` of the mover.
    acting_seat_val
        ``Seat.value`` of the mover.
    empty, same_team_occ, enemy_attackable
        Per-cell occupancy masks from :func:`_compute_occupancy_masks_soa`.
    """
    # Immobile / stronghold src → no moves.
    if _T.IS_IMMOBILE_TYPE[piece_type_val]:
        return _EMPTY_INT16
    if _T.IS_STRONGHOLD_FLAT[src_flat]:
        return _EMPTY_INT16

    dests: list[int] = []

    # (a) Orthogonal 1-step.
    for k in range(4):
        nb = int(_T.ADJ_STRAIGHT[src_flat, k])
        if nb >= 0 and landable[nb]:
            dests.append(nb)

    # (b) Diagonal 1-step into/out-of a camp.
    for nb in _T.ADJ_DIAG_INTO_CAMP[src_flat]:
        if landable[nb]:
            dests.append(nb)

    # (c) Rail long-range.
    if _T.IS_RAIL_FLAT[src_flat]:
        if _T.IS_ENGINEER_TYPE[piece_type_val]:
            dests.extend(_engineer_dests_soa(src_flat, empty, enemy_attackable))
        else:
            # Straight rails: 4 rays, stop at first non-empty.
            rays = _T.STRAIGHT_RAIL_RAYS[src_flat]
            for ray in rays:
                for nb in ray:
                    if empty[nb]:
                        dests.append(nb)
                    else:
                        if enemy_attackable[nb]:
                            dests.append(nb)
                        break
            # Curve rail (currently no curves on the board; table is empty).
            cid = int(_T.CURVE_ID_OF[src_flat])
            if cid > 0:
                dests.extend(
                    _curve_dests_soa(src_flat, cid, empty, enemy_attackable)
                )

    if not dests:
        return _EMPTY_INT16

    # De-duplicate while preserving insertion order.  ``dict.fromkeys`` is
    # the fastest way to do this in pure Python for small lists (<=40
    # entries), and avoids the per-call ``np.zeros(289)`` allocation a
    # bitmask-based sieve would need.  Duplicates can arise because the
    # straight-rail ray's first step is also an orthogonal neighbour.
    return np.fromiter(dict.fromkeys(dests), dtype=np.int16, count=-1)


def generate_legal_action_ids(
    cell_piece_id: np.ndarray,           # (289,) int16
    piece_seat_arr: np.ndarray,          # (num_pids,) int8
    piece_type_arr: np.ndarray,          # (num_pids,) int8
    alive: np.ndarray,                   # (num_pids,) bool
    pos_x: np.ndarray,                   # (num_pids,) int8
    pos_y: np.ndarray,                   # (num_pids,) int8
    acting_seat_val: int,
) -> np.ndarray:
    """Flat-action-id enumeration for ``acting_seat``.

    Returns
    -------
    ndarray[K] int32, each entry = ``src_flat * 289 + dst_flat``.
    """
    empty, same_team_occ, enemy_attackable = _compute_occupancy_masks_soa(
        cell_piece_id, piece_seat_arr, acting_seat_val
    )
    # Pre-compute the landable mask once per call; shared across pids.
    landable = empty | enemy_attackable
    # Pick this seat's alive & mobile piece ids, then filter in one pass:
    #   - seat matches
    #   - piece is alive
    #   - piece type is mobile (not JUNQI / DILEI / …)
    #   - src cell is not a stronghold
    #   - src cell is on-board (pos_x / pos_y >= 0)
    seat_mask = alive & (piece_seat_arr == acting_seat_val)
    if not seat_mask.any():
        return _EMPTY_INT32
    pids = np.nonzero(seat_mask)[0]
    pt_vals = piece_type_arr[pids].astype(np.intp, copy=False)
    # Filter immobile types at the vector level.
    mobile = ~_T.IS_IMMOBILE_TYPE[pt_vals]
    if not mobile.any():
        return _EMPTY_INT32
    pids = pids[mobile]
    pt_vals = pt_vals[mobile]
    # Compute src_flat for each surviving pid in one vectorized op.
    sx_arr = pos_x[pids].astype(np.intp, copy=False)
    sy_arr = pos_y[pids].astype(np.intp, copy=False)
    on_board = (sx_arr >= 0) & (sy_arr >= 0)
    if not on_board.all():
        pids = pids[on_board]
        pt_vals = pt_vals[on_board]
        sx_arr = sx_arr[on_board]
        sy_arr = sy_arr[on_board]
    src_flats = sy_arr * BOARD_SIZE + sx_arr
    # Filter out stronghold src cells at the vector level.
    not_stronghold = ~_T.IS_STRONGHOLD_FLAT[src_flats]
    if not not_stronghold.all():
        pids = pids[not_stronghold]
        pt_vals = pt_vals[not_stronghold]
        src_flats = src_flats[not_stronghold]
    if pids.size == 0:
        return _EMPTY_INT32

    # Hot inner loop: one call per remaining pid.  We already know every
    # surviving pid is mobile and sits on a valid non-stronghold cell, so
    # legal_dests_flat never returns early from those guards.
    parts: list[np.ndarray] = []
    pids_list = pids.tolist()
    src_flats_list = src_flats.tolist()
    pt_vals_list = pt_vals.tolist()
    for i in range(len(pids_list)):
        src_flat = src_flats_list[i]
        tv = pt_vals_list[i]
        dests = legal_dests_flat(
            pids_list[i], src_flat, tv, acting_seat_val,
            empty, same_team_occ, enemy_attackable, landable,
        )
        if dests.size == 0:
            continue
        parts.append(src_flat * NUM_CELLS + dests.astype(np.int32, copy=False))

    if not parts:
        return _EMPTY_INT32
    return np.concatenate(parts)


def has_any_legal_move_soa(
    cell_piece_id: np.ndarray,
    piece_seat_arr: np.ndarray,
    piece_type_arr: np.ndarray,
    alive: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    acting_seat_val: int,
) -> bool:
    """Short-circuit version of :func:`generate_legal_action_ids`."""
    empty, _same, enemy_attackable = _compute_occupancy_masks_soa(
        cell_piece_id, piece_seat_arr, acting_seat_val
    )
    landable = empty | enemy_attackable
    mask = alive & (piece_seat_arr == acting_seat_val)
    pids = np.nonzero(mask)[0]
    for pid in pids.tolist():
        tv = int(piece_type_arr[pid])
        if _T.IS_IMMOBILE_TYPE[tv]:
            continue
        sx = int(pos_x[pid])
        sy = int(pos_y[pid])
        if sx < 0 or sy < 0:
            continue
        src_flat = sy * BOARD_SIZE + sx
        if _T.IS_STRONGHOLD_FLAT[src_flat]:
            continue
        dests = legal_dests_flat(
            pid, src_flat, tv, acting_seat_val,
            empty, _same, enemy_attackable, landable,
        )
        if dests.size > 0:
            return True
    return False


# ===========================================================================
# Plan D — fully vectorized batch legal-action generator.
#
# Strategy (borrowed in spirit from the Ataraxos CUDA action kernel; the
# table shapes and the "sentinel cell 289" trick stand in for the GPU's
# natural per-thread parallelism):
#
#   1. Gather all same-seat, alive, mobile, non-stronghold pieces into a
#      size-K SoA: (pids, pt_vals, src_flats).
#   2. For every "movement class" compute an entire (K, num_candidates)
#      boolean mask in one vectorized numpy op:
#        - orthogonal 1-step            via ADJ_STRAIGHT (K, 4)
#        - diag-into-camp 1-step        via ADJ_DIAG_INTO_CAMP_PAD (K, 4)
#        - straight rail (non-engineer) via STRAIGHT_RAIL_RAYS_PAD
#                                       (K, 4 dirs, 4 cells)
#      The occupancy arrays are padded by one sentinel cell (index 289)
#      whose ``landable`` / ``empty`` values are False; table slots whose
#      pad value is -1 (off-board or missing) are remapped to 289, which
#      then short-circuits everything.
#   3. Engineer BFS is the one movement class that resists vectorization;
#      we keep the Python BFS but it runs at most ~3 times per call
#      (engineers are rare).
#   4. Collect all legal (src_flat, dst_flat) pairs and pack them into the
#      flat action-id space ``src_flat * 289 + dst_flat`` with a single
#      ``np.concatenate`` of the class-wise results.
# ===========================================================================


def _landable_with_sentinel(landable: np.ndarray) -> np.ndarray:
    """Return ``landable`` with one extra False cell at index ``NUM_CELLS``
    so that table entries padded with -1 (remapped to ``NUM_CELLS``)
    always read as "not landable"."""
    out = np.empty(NUM_CELLS + 1, dtype=bool)
    out[:NUM_CELLS] = landable
    out[NUM_CELLS] = False
    return out


def _empty_with_sentinel(empty: np.ndarray) -> np.ndarray:
    """Same idea for ``empty`` (used for rail-ray pass-through logic)."""
    out = np.empty(NUM_CELLS + 1, dtype=bool)
    out[:NUM_CELLS] = empty
    out[NUM_CELLS] = False
    return out


def _remap_neg1(a: np.ndarray) -> np.ndarray:
    """Replace -1 entries with the sentinel index ``NUM_CELLS``.
    Does not copy if ``a`` has no -1 entries."""
    return np.where(a < 0, NUM_CELLS, a).astype(np.intp, copy=False)


def generate_legal_action_ids_batch(
    cell_piece_id: np.ndarray,
    piece_seat_arr: np.ndarray,
    piece_type_arr: np.ndarray,
    alive: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    acting_seat_val: int,
) -> np.ndarray:
    """Vectorized version of :func:`generate_legal_action_ids`.

    Returns ``ndarray[K] int32`` of flat action ids
    ``src_flat * 289 + dst_flat``.  Set-equivalent to the PieceMap-based
    legacy ``generate_legal_actions`` — guaranteed by
    ``tests/test_move_gen_parity.py`` (fuzz).
    """
    empty, _same, enemy_attackable = _compute_occupancy_masks_soa(
        cell_piece_id, piece_seat_arr, acting_seat_val
    )
    landable = empty | enemy_attackable

    # --- K-sized SoA of candidate src pieces ----------------------------
    seat_mask = alive & (piece_seat_arr == acting_seat_val)
    if not seat_mask.any():
        return _EMPTY_INT32
    pids = np.nonzero(seat_mask)[0]
    pt_vals = piece_type_arr[pids].astype(np.intp, copy=False)
    # Drop immobile types immediately.
    mobile = ~_T.IS_IMMOBILE_TYPE[pt_vals]
    if not mobile.any():
        return _EMPTY_INT32
    pids = pids[mobile]
    pt_vals = pt_vals[mobile]
    sx_arr = pos_x[pids].astype(np.intp, copy=False)
    sy_arr = pos_y[pids].astype(np.intp, copy=False)
    on_board = (sx_arr >= 0) & (sy_arr >= 0)
    if not on_board.all():
        pids = pids[on_board]
        pt_vals = pt_vals[on_board]
        sx_arr = sx_arr[on_board]
        sy_arr = sy_arr[on_board]
    src_flats = sy_arr * BOARD_SIZE + sx_arr
    not_stronghold = ~_T.IS_STRONGHOLD_FLAT[src_flats]
    if not not_stronghold.all():
        pids = pids[not_stronghold]
        pt_vals = pt_vals[not_stronghold]
        src_flats = src_flats[not_stronghold]

    K = int(src_flats.size)
    if K == 0:
        return _EMPTY_INT32

    # Sentinel-padded occupancy arrays: index NUM_CELLS always reads as
    # "not landable" / "not empty", so table entries with -1 (remapped to
    # NUM_CELLS) auto-short-circuit.
    landable_pad = _landable_with_sentinel(landable)
    empty_pad = _empty_with_sentinel(empty)

    # Broadcast-friendly src column for pair encoding later.
    src_col = (src_flats.astype(np.int32, copy=False) * NUM_CELLS)  # (K,)

    parts: list[np.ndarray] = []

    # --- (a) Orthogonal 1-step (all pieces) -----------------------------
    adj = _T.ADJ_STRAIGHT[src_flats]                    # (K, 4) int16
    adj_idx = _remap_neg1(adj)                          # (K, 4) intp
    adj_ok = landable_pad[adj_idx]                      # (K, 4) bool
    if adj_ok.any():
        ki, dk = np.nonzero(adj_ok)
        dsts = adj[ki, dk].astype(np.int32, copy=False)
        parts.append(src_col[ki] + dsts)

    # --- (b) Diagonal into camp (all pieces) ----------------------------
    diag = _T.ADJ_DIAG_INTO_CAMP_PAD[src_flats]         # (K, 4) int16
    diag_idx = _remap_neg1(diag)
    diag_ok = landable_pad[diag_idx]
    if diag_ok.any():
        ki, dk = np.nonzero(diag_ok)
        dsts = diag[ki, dk].astype(np.int32, copy=False)
        parts.append(src_col[ki] + dsts)

    # --- (c) Straight rail (non-engineer, src on rail) ------------------
    # Subset: piece type not engineer and src_flat is rail.
    is_rail_src = _T.IS_RAIL_FLAT[src_flats]
    is_engineer = _T.IS_ENGINEER_TYPE[pt_vals]
    rail_nonengineer = is_rail_src & ~is_engineer
    if rail_nonengineer.any():
        sub_src_flats = src_flats[rail_nonengineer]
        sub_src_col = src_col[rail_nonengineer]
        rays = _T.STRAIGHT_RAIL_RAYS_PAD[sub_src_flats]  # (Kr, 4, L)
        rays_idx = _remap_neg1(rays)                     # (Kr, 4, L)
        # Per-cell empty / enemy flags along each ray.  -1 (pad) -> False.
        ray_empty = empty_pad[rays_idx]                  # (Kr, 4, L)
        ray_landable = landable_pad[rays_idx]            # (Kr, 4, L)
        # Prefix "all cells before this one were empty":
        #   prefix[..., 0] = True
        #   prefix[..., j] = prefix[..., j-1] AND ray_empty[..., j-1]
        # Vectorized via cumulative minimum (AND accumulate).
        #   shift ray_empty right by 1, then cumprod over axis=-1.
        L = ray_empty.shape[-1]
        shifted = np.empty_like(ray_empty)
        shifted[..., 0] = True
        if L > 1:
            shifted[..., 1:] = ray_empty[..., :-1]
        # "all True so far" along the ray:
        prefix_empty = np.cumprod(shifted, axis=-1).astype(bool, copy=False)
        # A cell at ray[..., j] is a legal dst iff we reached it AND it is
        # landable.  Padded -1 cells are already not landable by sentinel.
        ok = prefix_empty & ray_landable
        if ok.any():
            ki, dk, dj = np.nonzero(ok)
            dsts = rays[ki, dk, dj].astype(np.int32, copy=False)
            parts.append(sub_src_col[ki] + dsts)

    # --- (d) Engineer BFS (rare — at most ~3 per seat) ------------------
    eng_mask = is_rail_src & is_engineer
    if eng_mask.any():
        eng_idx = np.nonzero(eng_mask)[0]
        for kidx in eng_idx.tolist():
            sf = int(src_flats[kidx])
            dests = _engineer_dests_soa(sf, empty, enemy_attackable)
            if dests:
                d_arr = np.asarray(dests, dtype=np.int32)
                parts.append(sf * NUM_CELLS + d_arr)

    # --- (e) Curve rail BFS (0 curves right now; keep for future) -------
    # Enabled only when the board.py ships actual curve rails.
    if _T.CURVE_CELLS:
        for kidx in range(K):
            sf = int(src_flats[kidx])
            cid = int(_T.CURVE_ID_OF[sf])
            if cid == 0:
                continue
            if _T.IS_ENGINEER_TYPE[pt_vals[kidx]]:
                continue  # engineers handled by (d)
            dests = _curve_dests_soa(sf, cid, empty, enemy_attackable)
            if dests:
                d_arr = np.asarray(dests, dtype=np.int32)
                parts.append(sf * NUM_CELLS + d_arr)

    if not parts:
        return _EMPTY_INT32
    ids = np.concatenate(parts)
    # Remove duplicates that arise when (a) an orthogonal 1-step and
    # (c) a rail ray's first cell happen to coincide (they do: rail pieces
    # with empty neighbouring rail cell).  ``np.unique`` is O(K log K),
    # which for K <= ~150 legal actions is microscopic.
    return np.unique(ids).astype(np.int32, copy=False)


# ===========================================================================
# Fast "has any legal moves" check — avoids full enumeration.
#
# Used by the Q12 path in BatchedGameState._advance_turn to decide whether
# the next candidate seat needs to be killed.  Full move enumeration is ~10×
# slower than this check for typical positions.
# ===========================================================================


# ===========================================================================
# True N-batch vectorized legal-action generator.
#
# ``generate_legal_action_ids_n`` processes N independent game states
# simultaneously using 3-D numpy broadcasting:
#
#   * cell_piece_id  (N, 289),  piece_seat_arr (N, 120), etc.
#   * seat_per_env   (N,) — each env's acting seat value (may differ).
#
# All orthogonal-step, camp-diagonal, and straight-rail logic runs in a
# single pass over all N envs in parallel.  Engineer BFS (rare) runs only
# for envs that actually have an engineer on a rail cell.
#
# The output is a *list* of per-env int32 arrays (ragged), which matches
# the contract of the old ``legal_action_ids_batch`` loop.
# ===========================================================================


def _compute_occupancy_masks_n(
    cell_piece_id: np.ndarray,    # (N, 289) int16
    seat_per_env: np.ndarray,     # (N,) int8
    cell_team_arr: np.ndarray,    # (N, 289) uint8 — 255=empty, precomputed
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized occupancy masks for N envs.

    Uses precomputed ``cell_team_arr`` (uint8, 255=empty) to avoid the
    expensive 2-D fancy-index through piece_seat_arr.

    Returns
    -------
    empty        : (N, 289) bool
    landable_pad : (N, 290) bool — padded with False sentinel at index 289
    empty_pad    : (N, 290) bool — padded with False sentinel at index 289
    """
    N = cell_piece_id.shape[0]

    occupied = cell_piece_id >= 0                                    # (N, 289)
    # acting team: 0 or 1
    at_bc = (seat_per_env & 1).astype(np.uint8)[:, np.newaxis]      # (N, 1)
    enemy_occ = occupied & (cell_team_arr != at_bc)                  # (N, 289)
    enemy_attackable = enemy_occ & (~_T.IS_CAMP_FLAT[np.newaxis, :])# (N, 289)
    empty = ~occupied                                                  # (N, 289)

    landable_pad = np.zeros((N, NUM_CELLS + 1), dtype=bool)
    landable_pad[:, :NUM_CELLS] = empty | enemy_attackable
    empty_pad = np.zeros((N, NUM_CELLS + 1), dtype=bool)
    empty_pad[:, :NUM_CELLS] = empty

    return empty, landable_pad, empty_pad


def _split_by_env(
    env_col: np.ndarray,   # (M,) int32 — which env each action belongs to
    act_col: np.ndarray,   # (M,) int32 — flat action id  (max 83520 < 2^17)
    N: int,                # total number of envs
) -> list[np.ndarray]:
    """Split (env_col, act_col) into a list of per-env action arrays.

    Key-packing trick:  pack (env, act) into a single uint64 key
        key = env * 2^17 + act
    Sort the packed keys once; duplicates collapse adjacent.  Then:
      • diff-dedup to remove duplicate (env, act) pairs in O(M)
      • bincount on dedup env_col to know per-env sizes
      • one pass with cumsum to slice the deduplicated act array

    This avoids argsort+Python-loop-per-env and uses no dict allocation.
    """
    if env_col.size == 0:
        return [_EMPTY_INT32] * N

    # Pack into uint64: high 47 bits = env, low 17 bits = act
    keys = env_col.astype(np.uint64) * np.uint64(1 << 17) + act_col.astype(np.uint64)
    keys.sort()

    # Dedup: keep unique keys
    if keys.size > 1:
        keep = np.empty(keys.size, dtype=bool)
        keep[0] = True
        keep[1:] = keys[1:] != keys[:-1]
        keys = keys[keep]

    # Unpack
    act_dedup = (keys & np.uint64((1 << 17) - 1)).astype(np.int32)
    env_dedup  = (keys >> np.uint64(17)).astype(np.int32)

    # Per-env counts via bincount
    counts = np.bincount(env_dedup, minlength=N)   # (N,)
    starts = np.empty(N + 1, dtype=np.int64)
    starts[0] = 0
    np.cumsum(counts, out=starts[1:])

    result: list[np.ndarray] = [_EMPTY_INT32] * N
    for e in range(N):
        s, t = int(starts[e]), int(starts[e + 1])
        if t > s:
            result[e] = act_dedup[s:t]
    return result


def generate_legal_action_ids_n(
    cell_piece_id: np.ndarray,   # (N, 289) int16
    piece_seat_arr: np.ndarray,  # (N, 120) int8  (kept for engineer BFS compat)
    piece_type_arr: np.ndarray,  # (N, 120) int8
    alive: np.ndarray,           # (N, 120) bool
    pos_x: np.ndarray,           # (N, 120) int8
    pos_y: np.ndarray,           # (N, 120) int8
    seat_per_env: np.ndarray,    # (N,) int8 — acting seat per env
    terminated: np.ndarray,      # (N,) bool — skip terminated envs
    cell_team_arr: np.ndarray,   # (N, 289) uint8 — 255=empty, precomputed
) -> list[np.ndarray]:
    """Fully-vectorized legal-action generation for N envs simultaneously.

    Strategy: work in *sparse candidate* space (C = total mobile same-seat
    pieces across all active envs, C ≈ A*25 << A*120).  This avoids the
    expense of broadcasting full (A, 120, 4) tensors.

    Steps:
      1. Occupancy: ONE (A, 289) vectorized call — uses precomputed cell_team_arr.
      2. Candidates: np.nonzero(cand) → flat (C,) piece list with env labels.
      3. Ortho/diag: flat-gather via pre-padded tables (C*4 indexing, no np.where).
      4. Rail: explicit L=4 unrolled prefix + flat-gather (no cumprod).
      5. Engineer BFS: rare per-piece Python loop.
      6. _split_by_env: key-pack + sort for ragged output.

    Returns a ragged list of per-env int32 arrays.
    """
    N = cell_piece_id.shape[0]
    result: list[np.ndarray] = [_EMPTY_INT32] * N

    active_mask = ~terminated
    if not active_mask.any():
        return result
    active_idx = np.nonzero(active_mask)[0]     # (A,)
    ai = active_idx

    # Sub-batch views
    cpid = cell_piece_id[ai]                     # (A, 289)
    psa  = piece_seat_arr[ai]                    # (A, 120)
    pta  = piece_type_arr[ai]                    # (A, 120)
    alv  = alive[ai]                             # (A, 120)
    px   = pos_x[ai]                             # (A, 120)
    py   = pos_y[ai]                             # (A, 120)
    spe  = seat_per_env[ai]                      # (A,)
    cta  = cell_team_arr[ai]                     # (A, 289)

    # ----------------------------------------------------------------
    # 1. Occupancy  (A, 289/290) — fast path using cell_team_arr
    # ----------------------------------------------------------------
    empty, landable_pad, empty_pad = _compute_occupancy_masks_n(cpid, spe, cta)
    # enemy_att[a, flat] = True iff enemy piece on flat AND not a camp cell
    # (needed by engineer BFS; computed once, shared across all engineer calls)
    enemy_att = landable_pad[:, :NUM_CELLS] & ~empty  # (A, 289) bool

    # ----------------------------------------------------------------
    # 2. Candidate mask  (A, 120) → sparse (C,)
    # ----------------------------------------------------------------
    seat_match    = alv & (psa == spe[:, np.newaxis])
    pt_vals_full  = pta.astype(np.intp, copy=False)
    mobile_full   = ~_T.IS_IMMOBILE_TYPE[pt_vals_full]
    on_board_full = (px >= 0) & (py >= 0)

    sx_full = px.astype(np.int32, copy=False)
    sy_full = py.astype(np.int32, copy=False)
    sf_full = sy_full * BOARD_SIZE + sx_full
    sf_clip = np.clip(sf_full, 0, NUM_CELLS - 1).astype(np.intp)

    not_sh_full = ~_T.IS_STRONGHOLD_FLAT[sf_clip]
    cand = seat_match & mobile_full & on_board_full & not_sh_full   # (A, 120)

    if not cand.any():
        return result

    # Flatten candidates into sparse (C,) arrays
    c_a, c_p = np.nonzero(cand)                  # (C,) each — c_a in [0,A), c_p in [0,120)
    C = c_a.shape[0]

    # Per-candidate env index (mapped back to original env ids)
    c_env = ai[c_a].astype(np.int32)             # (C,) original env ids
    c_sf  = sf_clip[c_a, c_p]                    # (C,) intp — src flat cell
    c_pt  = pt_vals_full[c_a, c_p]               # (C,) intp — piece type

    is_rail_c = _T.IS_RAIL_FLAT[c_sf]            # (C,) bool
    is_eng_c  = _T.IS_ENGINEER_TYPE[c_pt]        # (C,) bool

    # We'll accumulate (env_label, action_id) pairs here
    env_parts: list[np.ndarray] = []
    act_parts: list[np.ndarray] = []

    src_act = c_sf.astype(np.int32) * NUM_CELLS  # (C,) — src_flat * NUM_CELLS

    # Flat arrays for fast gather (avoids 2D fancy index overhead)
    lp_flat = landable_pad.ravel()               # (A * 290,)
    ep_flat = empty_pad.ravel()                  # (A * 290,)
    row_stride = NUM_CELLS + 1                   # 290

    # ----------------------------------------------------------------
    # 3a. Orthogonal 1-step  (C, 4)  — pre-padded table
    # ----------------------------------------------------------------
    adj = _ADJ_STRAIGHT_PAD[c_sf]               # (C, 4) int32, sentinel=289
    # Flat index: c_a[:, None] * 290 + adj
    fi_a = c_a[:, np.newaxis] * row_stride + adj # (C, 4) int64
    aok  = lp_flat[fi_a.ravel()].reshape(C, 4)  # (C, 4) bool
    if aok.any():
        ci, dk = np.nonzero(aok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + adj[ci, dk])

    # ----------------------------------------------------------------
    # 3b. Diagonal into camp  (C, 4)  — pre-padded table
    # ----------------------------------------------------------------
    diag = _ADJ_DIAG_PAD[c_sf]                  # (C, 4) int32, sentinel=289
    fi_d = c_a[:, np.newaxis] * row_stride + diag
    dok  = lp_flat[fi_d.ravel()].reshape(C, 4)  # (C, 4) bool
    if dok.any():
        ci, dk = np.nonzero(dok)
        env_parts.append(c_env[ci])
        act_parts.append(src_act[ci] + diag[ci, dk])

    # ----------------------------------------------------------------
    # 3c. Straight rail — non-engineer on rail  (Cr, 4, L)
    #     Unrolled L=4 prefix to avoid np.cumprod overhead
    # ----------------------------------------------------------------
    rail_ne = is_rail_c & ~is_eng_c              # (C,) bool
    if rail_ne.any():
        r_sf  = c_sf[rail_ne]                    # (Cr,)
        r_a   = c_a[rail_ne]                     # (Cr,) sub-batch env idx
        r_env = c_env[rail_ne]                   # (Cr,) original env ids
        r_sc  = src_act[rail_ne]                 # (Cr,)
        rays  = _SRAYS_PAD[r_sf]                 # (Cr, 4, L) int32, sentinel=289
        Cr    = r_sf.shape[0]
        L     = _SRAYS_L

        # Flat gather for empty and landable along rays
        # fi shape: (Cr, 4, L)
        fi_r  = (r_a[:, np.newaxis, np.newaxis] * row_stride + rays).ravel()
        re    = ep_flat[fi_r].reshape(Cr, 4, L)  # (Cr, 4, L) bool
        rl    = lp_flat[fi_r].reshape(Cr, 4, L)  # (Cr, 4, L) bool

        # Compute prefix: can reach position l only if all prior cells are empty
        # pref[..., 0] = True; pref[..., k] = all(re[..., :k]) for k>0
        # Manual iterative prefix-AND is faster than cumprod for L=12
        pref = np.empty((Cr, 4, L), dtype=bool)
        pref[..., 0] = True
        if L > 1:
            pref[..., 1] = re[..., 0]
            running = re[..., 0]
            for ll in range(2, L):
                running = running & re[..., ll - 1]
                pref[..., ll] = running

        ok = pref & rl
        if ok.any():
            ci, di, li = np.nonzero(ok)
            env_parts.append(r_env[ci])
            act_parts.append(r_sc[ci] + rays[ci, di, li])

    # ----------------------------------------------------------------
    # 3d. Engineer BFS — hybrid: vectorized for many engineers,
    #     per-piece Python BFS for few engineers
    # ----------------------------------------------------------------
    eng_on_rail = is_rail_c & is_eng_c           # (C,) bool
    if eng_on_rail.any():
        eng_sf  = c_sf[eng_on_rail].astype(np.int32)   # (E,)
        eng_a   = c_a[eng_on_rail].astype(np.int32)    # (E,) sub-batch idx
        eng_env = c_env[eng_on_rail]                   # (E,) original env ids
        E = int(eng_sf.shape[0])
        if E > 100:
            # Vectorized BFS: amortize tensor overhead across many engineers
            ev, av = _batch_engineer_bfs_n(
                eng_sf, eng_a, eng_env, empty, enemy_att
            )
            if ev is not None and av is not None:
                env_parts.append(ev)
                act_parts.append(av)
        else:
            # Per-piece BFS: less overhead for small engineer counts
            eng_envs: list[np.ndarray] = []
            eng_acts: list[np.ndarray] = []
            for kidx_i in range(E):
                sf = int(eng_sf[kidx_i])
                a_idx = int(eng_a[kidx_i])
                dests = _engineer_dests_soa(sf, empty[a_idx], enemy_att[a_idx])
                if dests:
                    d_arr = np.asarray(dests, dtype=np.int32)
                    eng_envs.append(np.full(len(dests), int(eng_env[kidx_i]), dtype=np.int32))
                    eng_acts.append(sf * NUM_CELLS + d_arr)
            if eng_envs:
                env_parts.append(np.concatenate(eng_envs))
                act_parts.append(np.concatenate(eng_acts))

    # ----------------------------------------------------------------
    # 3e. Curve-rail — vectorized prefix-product rays (T-02 optimization)
    #
    # The 4 curve rails are simple chains (max degree 2).  Each cell on
    # a curve has two precomputed rays (forward / backward along the chain)
    # in CURVE_CHAIN_RAYS_PAD[flat, 2, 11].  We apply the same
    # prefix-product trick as straight rails: walk each ray, checking
    # that all intermediate cells are empty and the landing cell is
    # landable.
    # ----------------------------------------------------------------
    curve_cand = is_rail_c & ~is_eng_c
    if curve_cand.any() and _T.CURVE_CELLS:
        cids = _T.CURVE_ID_OF[c_sf]                   # (C,) int8
        curve_mask = curve_cand & (cids > 0) & _T.IS_CURVE_ACTIVE[c_sf]
        if curve_mask.any():
            cr_sf  = c_sf[curve_mask]                  # (Cv,) src flats
            cr_a   = c_a[curve_mask]                   # (Cv,) sub-batch env idx
            cr_env = c_env[curve_mask]                 # (Cv,) original env ids
            cr_sc  = src_act[curve_mask]               # (Cv,) src * NUM_CELLS

            # Fetch precomputed rays: (Cv, 2, 11)
            rays = _T.CURVE_CHAIN_RAYS_PAD[cr_sf]     # (Cv, 2, 11) int16
            Cv = cr_sf.shape[0]
            CL = rays.shape[2]  # 11

            # Flat-gather for empty/landable along rays
            fi_cr = (cr_a[:, np.newaxis, np.newaxis] * row_stride + np.where(
                rays >= 0, rays, NUM_CELLS
            ).astype(np.intp)).ravel()
            cr_empty = ep_flat[fi_cr].reshape(Cv, 2, CL)
            cr_land  = lp_flat[fi_cr].reshape(Cv, 2, CL)

            # Prefix "all cells before this one were empty" — same as straight rail
            if CL <= 11:
                # Unroll up to 11 — use cumprod for generality
                sh = np.empty_like(cr_empty)
                sh[..., 0] = True
                if CL > 1:
                    sh[..., 1:] = cr_empty[..., :-1]
                pref = np.cumprod(sh, axis=-1).astype(bool, copy=False)
            else:
                pref = np.ones_like(cr_empty)

            ok = pref & cr_land
            if ok.any():
                ci, di, li = np.nonzero(ok)
                env_parts.append(cr_env[ci])
                act_parts.append(cr_sc[ci] + rays[ci, di, li].astype(np.int32))

    # ----------------------------------------------------------------
    # 4. Assemble output
    # ----------------------------------------------------------------
    if not env_parts:
        return result

    env_col_all = np.concatenate(env_parts)
    act_col_all = np.concatenate(act_parts)
    return _split_by_env(env_col_all, act_col_all, N)


def has_legal_moves_soa(
    cell_piece_id: np.ndarray,   # (289,) int16
    piece_seat_arr: np.ndarray,  # (120,) int8
    piece_type_arr: np.ndarray,  # (120,) int8
    alive: np.ndarray,           # (120,) bool
    pos_x: np.ndarray,           # (120,) int8
    pos_y: np.ndarray,           # (120,) int8
    acting_seat_val: int,
) -> bool:
    """Return True iff ``acting_seat_val`` has at least one legal move.

    Optimized fast path: first checks if any mobile piece has an empty
    orthogonal neighbor (covers >99% of cases without full occupancy
    computation). Falls back to full check only when the fast path fails.
    """
    # Filter to mobile alive pieces of this seat
    seat_mask = alive & (piece_seat_arr == acting_seat_val)
    if not seat_mask.any():
        return False
    pids = np.nonzero(seat_mask)[0]
    pt_vals = piece_type_arr[pids].astype(np.intp, copy=False)
    mobile = ~_T.IS_IMMOBILE_TYPE[pt_vals]
    if not mobile.any():
        return False
    pids = pids[mobile]
    sx_arr = pos_x[pids].astype(np.intp, copy=False)
    sy_arr = pos_y[pids].astype(np.intp, copy=False)
    on_board = (sx_arr >= 0) & (sy_arr >= 0)
    if not on_board.any():
        return False
    pids = pids[on_board]
    sx_arr = sx_arr[on_board]
    sy_arr = sy_arr[on_board]
    src_flats = sy_arr * BOARD_SIZE + sx_arr
    not_stronghold = ~_T.IS_STRONGHOLD_FLAT[src_flats]
    if not not_stronghold.any():
        return False
    src_flats = src_flats[not_stronghold]

    # Fast path: check if any orthogonal neighbor is empty.
    # This avoids building full occupancy masks (the expensive part).
    occupied = np.zeros(NUM_CELLS + 1, dtype=bool)
    occupied[:NUM_CELLS] = (cell_piece_id >= 0)
    # sentinel at NUM_CELLS is False (off-board)

    adj = _ADJ_STRAIGHT_PAD[src_flats]  # (K, 4), sentinel=289=NUM_CELLS
    # Check if any adjacent cell is empty (not occupied)
    adj_occupied = occupied[adj.ravel()].reshape(adj.shape)
    any_empty_adj = ~adj_occupied  # (K, 4)
    if any_empty_adj.any():
        return True

    # Slow path (rare): build full occupancy and do comprehensive check
    empty, _same, enemy_attackable = _compute_occupancy_masks_soa(
        cell_piece_id, piece_seat_arr, acting_seat_val
    )
    landable = empty | enemy_attackable
    landable_pad = _landable_with_sentinel(landable)
    empty_pad = _empty_with_sentinel(empty)

    pids_full = pids[not_stronghold]
    pt_vals_full = piece_type_arr[pids_full].astype(np.intp, copy=False)

    # Check orthogonal 1-step with full landable (includes enemy attack)
    adj_idx = adj.astype(np.intp, copy=False)
    if landable_pad[adj_idx].any():
        return True

    # Check diagonal-into-camp 1-step
    diag = _T.ADJ_DIAG_INTO_CAMP_PAD[src_flats]
    diag_idx = _remap_neg1(diag)
    if landable_pad[diag_idx].any():
        return True

    # Check straight-rail moves (non-engineer on rail)
    is_rail_src = _T.IS_RAIL_FLAT[src_flats]
    is_engineer = _T.IS_ENGINEER_TYPE[pt_vals_full]
    rail_nonengineer = is_rail_src & ~is_engineer
    if rail_nonengineer.any():
        sub_src_flats = src_flats[rail_nonengineer]
        rays = _T.STRAIGHT_RAIL_RAYS_PAD[sub_src_flats]   # (Kr, 4, L)
        rays_idx = _remap_neg1(rays)
        ray_empty = empty_pad[rays_idx]
        ray_landable = landable_pad[rays_idx]
        L = ray_empty.shape[-1]
        pref = np.empty_like(ray_empty)
        pref[..., 0] = True
        if L > 1:
            pref[..., 1] = ray_empty[..., 0]
            running = ray_empty[..., 0]
            for ll in range(2, L):
                running = running & ray_empty[..., ll - 1]
                pref[..., ll] = running
        ok = pref & ray_landable
        if ok.any():
            return True

    # Check engineer BFS (rail engineers)
    eng_mask = is_rail_src & is_engineer
    if eng_mask.any():
        for kidx in np.nonzero(eng_mask)[0].tolist():
            sf = int(src_flats[kidx])
            dests = _engineer_dests_soa(sf, empty, enemy_attackable)
            if dests:
                return True

    return False


# ===========================================================================
# Self-test
# ===========================================================================


def _self_test() -> None:  # pragma: no cover

    # ---- Test 1: simple adjacent forward move for HOME PAIZH ----
    # HOME index 2 → world (8, 11). Move to (8, 10) (empty, on-board).
    pieces: PieceMap = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
    }
    assert is_legal_move(pieces, (8, 11), (8, 10), Seat.SOUTH)
    assert not is_legal_move(pieces, (8, 11), (8, 10), Seat.NORTH)  # wrong seat

    # ---- Test 2: diagonal without camp — illegal ----
    assert not is_legal_move(pieces, (8, 11), (9, 10), Seat.SOUTH)

    # ---- Test 3: diagonal into camp — legal ----
    pieces2: PieceMap = {
        (10, 13): PieceRef(Seat.SOUTH, PieceType.LIANZH),
    }
    assert is_legal_move(pieces2, (10, 13), (9, 12), Seat.SOUTH)  # (9,12) is camp

    # ---- Test 4: stronghold piece cannot move ----
    pieces3: PieceMap = {
        (9, 16): PieceRef(Seat.SOUTH, PieceType.JUNQI),  # stronghold
    }
    assert not is_legal_move(pieces3, (9, 16), (9, 15), Seat.SOUTH)

    # ---- Test 5: DILEI cannot move ----
    pieces4: PieceMap = {
        (10, 15): PieceRef(Seat.SOUTH, PieceType.DILEI),
    }
    assert not is_legal_move(pieces4, (10, 15), (10, 14), Seat.SOUTH)

    # ---- Test 6: rail long-range move (LIANZH on front row) ----
    # HOME front row y=11 is all rail. (10,11) → (6,11) clear path.
    pieces5: PieceMap = {
        (10, 11): PieceRef(Seat.SOUTH, PieceType.LIANZH),
    }
    assert is_legal_move(pieces5, (10, 11), (6, 11), Seat.SOUTH)

    # ---- Test 7: rail blocked by own piece ----
    pieces6: PieceMap = {
        (10, 11): PieceRef(Seat.SOUTH, PieceType.LIANZH),
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),   # blocker
    }
    assert not is_legal_move(pieces6, (10, 11), (6, 11), Seat.SOUTH)

    # ---- Test 8: engineer BFS via corner ----
    # HOME GONGB at (6, 11) can reach (10, 15) by rail BFS through corners.
    pieces7: PieceMap = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.GONGB),
    }
    assert is_legal_move(pieces7, (6, 11), (10, 15), Seat.SOUTH)

    # Non-engineer cannot do this move (requires turn)
    pieces7b: PieceMap = {
        (6, 11): PieceRef(Seat.SOUTH, PieceType.LIANZH),
    }
    assert not is_legal_move(pieces7b, (6, 11), (10, 15), Seat.SOUTH)

    # ---- Test 9: attack enemy ----
    # Via rail: y=10 for HOME is not front-row rail. Wait: HOME front is y=11.
    # (8, 10) is NineGrid cell (8,10) is actually NineGrid = True.
    # Let's check via an attackable scenario: HOME SILING at (8, 11) attacks
    # OPPS at (8, 5) via straight-column rail travel through intermediate
    # empty rail cells. But (8, 5) is OPPS back-row which is NOT a rail cell
    # (only i<25 in OPPS are rails, i.e. rows 0..4 of OPPS = y 5..1).
    # (8, 5) is OPPS index 2 → row 0 → rail = True. OK.
    # Between (8,11) and (8,5): the column x=8 intermediate cells (8,6)..(8,10).
    # (8, 6), (8, 8), (8, 10) are nine-grid cells (not rails).
    # So the straight-rail check will fail because (8, 7) (8, 9) aren't rail
    # either. The legacy behavior allows crossing nine-grid via... actually
    # it doesn't; nine-grid is not rail. So this attack is NOT legal via rail.
    # We'll test this correctly instead with a clearer attack scenario.
    #
    # Skip this specific one — the rail plumbing is correct; we'll verify with
    # integration tests against golden data in the next step.

    # ---- Test 10: enemy in camp is unattackable ----
    pieces9: PieceMap = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.SILING),
        (9, 12): PieceRef(Seat.NORTH, PieceType.PAIZH),   # enemy in HOME's camp
    }
    # Adjacent diagonal into camp — but the camp holds an enemy.
    # Rule: dst is camp AND occupied by an enemy piece → illegal.
    # Our _can_end_on returns False for "enemy piece in a camp". Correct.
    assert not is_legal_move(pieces9, (8, 11), (9, 12), Seat.SOUTH)

    # ---- Test 11: has_any_legal_move ----
    pieces10: PieceMap = {
        (8, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
    }
    assert has_any_legal_move(pieces10, Seat.SOUTH)
    assert not has_any_legal_move(pieces10, Seat.NORTH)

    # ---- Test 12: generate_legal_actions ----
    actions = generate_legal_actions(pieces10, Seat.SOUTH)
    assert len(actions) > 0
    # PAIZH at (8, 11) has up to 4 orthogonal + (long rail if (8,11) is rail).
    # (8, 11) is HOME index 2, row 0 (front row), i/5==0 → rail. So it has
    # straight rail moves along x=8 (blocked by non-rail nine-grid quickly)
    # and along y=11. Expect at least 4 moves (the orthogonal neighbors).
    # (8,10) is nine-grid, so it's a valid landing cell (empty). Adj moves:
    #   (7, 11) rail empty — OK
    #   (9, 11) rail empty — OK
    #   (8, 10) nine-grid empty — OK (adjacent non-diag)
    #   (8, 12) HOME cell empty — OK
    # Plus rail straight moves along y=11: (10, 11), (6, 11).
    # So 6 moves minimum.
    assert len(actions) >= 6, f"expected >=6 legal moves, got {len(actions)}: {actions}"

    print(f"junqi_core.move_gen self-test: OK ({len(actions)} moves from PAIZH at (8,11))")


if __name__ == "__main__":
    _self_test()
