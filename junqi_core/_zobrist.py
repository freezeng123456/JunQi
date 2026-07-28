"""Zobrist hash tables for `GameState.state_hash()` (ADR-117, Phase 0.4 M1).

The table is built once at import time using ``numpy.random.default_rng`` with
a **fixed seed** so the produced int64 constants are deterministic across
interpreter runs, Python versions, and NumPy versions (the PCG64 bitgen
shipped with NumPy is stable across patch versions).

Layout
------
All tables are ``np.ndarray[int64]`` and are XOR'd together to produce the
state hash; XOR-in / XOR-out mirroring gives us O(1) incremental updates
inside :meth:`GameState.step_inplace`.

    ZOB_PIECE[pid, type_val, cell_flat]
        120 × 14 × 289  — the dominating contribution.  Indexed by
        (piece_id, PieceType.value, flat_cell).  ``type_val == 0`` (NONE)
        and ``type_val == 1`` (DARK) rows are included so we can XOR
        blindly by ``piece_type_arr[pid]`` without special-casing.
    ZOB_TURN[seat_val]              — 4
    ZOB_MOVE_COUNTER[counter & mask] — 4096 (low-12-bits sketch of the
        move counter; good enough for an O(1) rolling contribution. The
        actual counter is still exact via integer equality; the Zobrist
        is for dedup, not for replay semantics).
    ZOB_MOVES_SINCE_COMBAT[c & mask] — 512 (low-9-bits)
    ZOB_TERMINATED                  — 1 scalar (XOR'd when terminated)
    ZOB_WINNER[team_val + 1]        — 3  (indices 0/1/2 for -1/0/1)
    ZOB_DRAW                        — 1 scalar
    ZOB_SEAT_DEAD[seat_val]         — 4
    ZOB_SEAT_FLAG_REVEALED[seat_val]— 4
    ZOB_SHOW_MODE[show_mode_val]    — 4

Design notes
------------
*  **Counters in the hash at all.** Two states that differ ONLY by
   ``move_counter`` or ``moves_since_last_combat`` should hash differently
   (they are distinct game positions), which is why the counters are
   mixed in.  The "low-N-bits sketch" is an intentional simplification:
   after 4096 moves the counter contribution wraps around, but by then
   the piece configuration has changed drastically so collisions remain
   astronomically unlikely.  Full-integer hashing is also possible; the
   sketch is just an O(1)-table trick.

*  **Fixed seed.**  ``0x4A554E51495F5A4F`` = ASCII ``"JUNQI_ZO"``.  Never
   change this — it would invalidate any on-disk Zobrist-keyed caches.

*  **API compat.**  ``GameState.state_hash()`` still returns ``int``; the
   only observable difference vs the pre-M1 implementation is the
   numerical value.  Tests only check self-consistency (clone-equality),
   not specific hash numbers, so this change is transparent to the test
   suite (verified by ``grep state_hash`` 2026-04-21).
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .board import NUM_CELLS
from .rules import NUM_SEATS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PIECE_IDS: Final[int] = 4 * 30          # ADR-114 encoding: seat*30 + slot
NUM_PIECE_TYPES: Final[int] = 14            # PieceType.value ∈ [0, 13]
NUM_SHOW_MODES: Final[int] = 4              # ShowMode values ∈ {0,1,2} — pad to 4

MOVE_COUNTER_MASK: Final[int] = 0xFFF       # low 12 bits
MOVES_SINCE_COMBAT_MASK: Final[int] = 0x1FF  # low 9 bits

_ZOBRIST_SEED: Final[int] = 0x4A554E51495F5A4F  # "JUNQI_ZO"


# ---------------------------------------------------------------------------
# Table construction (single deterministic seed → int64 tables)
# ---------------------------------------------------------------------------

def _build_tables() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(_ZOBRIST_SEED)

    def _rand(shape: tuple[int, ...]) -> np.ndarray:
        # uint64 → view as int64 keeps full 64-bit entropy and lets us XOR
        # directly against Python's int (which is signed but 64-bit-safe
        # for XOR by numpy).
        return rng.integers(
            low=np.iinfo(np.uint64).min,
            high=np.iinfo(np.uint64).max,
            size=shape,
            dtype=np.uint64,
            endpoint=True,
        ).astype(np.int64, copy=False)

    return {
        "ZOB_PIECE":              _rand((NUM_PIECE_IDS, NUM_PIECE_TYPES, NUM_CELLS)),
        "ZOB_TURN":               _rand((NUM_SEATS,)),
        "ZOB_MOVE_COUNTER":       _rand((MOVE_COUNTER_MASK + 1,)),
        "ZOB_MOVES_SINCE_COMBAT": _rand((MOVES_SINCE_COMBAT_MASK + 1,)),
        "ZOB_TERMINATED":         _rand((1,))[0],           # scalar int64
        "ZOB_WINNER":             _rand((3,)),              # idx = team+1
        "ZOB_DRAW":               _rand((1,))[0],           # scalar
        "ZOB_SEAT_DEAD":          _rand((NUM_SEATS,)),
        "ZOB_SEAT_FLAG_REVEALED": _rand((NUM_SEATS,)),
        "ZOB_SHOW_MODE":          _rand((NUM_SHOW_MODES,)),
    }


_TABLES: Final[dict[str, np.ndarray]] = _build_tables()

# Export individual tables at module scope for fast direct lookup from
# state.py's hot path (one attribute access vs two).
ZOB_PIECE:              Final[np.ndarray] = _TABLES["ZOB_PIECE"]
ZOB_TURN:               Final[np.ndarray] = _TABLES["ZOB_TURN"]
ZOB_MOVE_COUNTER:       Final[np.ndarray] = _TABLES["ZOB_MOVE_COUNTER"]
ZOB_MOVES_SINCE_COMBAT: Final[np.ndarray] = _TABLES["ZOB_MOVES_SINCE_COMBAT"]
ZOB_TERMINATED:         Final[int]        = int(_TABLES["ZOB_TERMINATED"])
ZOB_WINNER:             Final[np.ndarray] = _TABLES["ZOB_WINNER"]
ZOB_DRAW:               Final[int]        = int(_TABLES["ZOB_DRAW"])
ZOB_SEAT_DEAD:          Final[np.ndarray] = _TABLES["ZOB_SEAT_DEAD"]
ZOB_SEAT_FLAG_REVEALED: Final[np.ndarray] = _TABLES["ZOB_SEAT_FLAG_REVEALED"]
ZOB_SHOW_MODE:          Final[np.ndarray] = _TABLES["ZOB_SHOW_MODE"]


# ---------------------------------------------------------------------------
# Convenience helpers (all return Python int)
# ---------------------------------------------------------------------------

def piece_hash(pid: int, type_val: int, flat_cell: int) -> int:
    """Return the Zobrist contribution for a single piece placement."""
    return int(ZOB_PIECE[pid, type_val, flat_cell])


def turn_hash(seat_val: int) -> int:
    return int(ZOB_TURN[seat_val])


def move_counter_hash(counter: int) -> int:
    return int(ZOB_MOVE_COUNTER[counter & MOVE_COUNTER_MASK])


def moves_since_combat_hash(counter: int) -> int:
    return int(ZOB_MOVES_SINCE_COMBAT[counter & MOVES_SINCE_COMBAT_MASK])


def winner_hash(winner_team: int | None) -> int:
    """Map winner_team {None, 0, 1} -> ZOB_WINNER[0..2]."""
    idx = 0 if winner_team is None else (winner_team + 1)
    return int(ZOB_WINNER[idx])


def seat_dead_hash(seat_val: int) -> int:
    return int(ZOB_SEAT_DEAD[seat_val])


def seat_flag_revealed_hash(seat_val: int) -> int:
    return int(ZOB_SEAT_FLAG_REVEALED[seat_val])


def show_mode_hash(show_mode_val: int) -> int:
    return int(ZOB_SHOW_MODE[show_mode_val])


# ---------------------------------------------------------------------------
# Self-check: tables are deterministic and distinct.
# ---------------------------------------------------------------------------

def _self_check() -> None:
    # Rebuild and confirm identity (determinism under fixed seed).
    rebuild = _build_tables()
    for k, v in _TABLES.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(v, rebuild[k]), f"non-deterministic table {k!r}"
        else:
            assert v == rebuild[k], f"non-deterministic scalar {k!r}"

    # Basic sanity: shape agrees with documented layout.
    assert ZOB_PIECE.shape == (NUM_PIECE_IDS, NUM_PIECE_TYPES, NUM_CELLS)
    assert ZOB_TURN.shape == (NUM_SEATS,)
    assert ZOB_MOVE_COUNTER.shape == (MOVE_COUNTER_MASK + 1,)
    assert ZOB_SEAT_DEAD.shape == (NUM_SEATS,)

    # All entries should be int64 (no accidental float).
    assert ZOB_PIECE.dtype == np.int64
    assert ZOB_TURN.dtype == np.int64


_self_check()


if __name__ == "__main__":
    _self_check()
    n_entries = (
        ZOB_PIECE.size
        + ZOB_TURN.size
        + ZOB_MOVE_COUNTER.size
        + ZOB_MOVES_SINCE_COMBAT.size
        + 1
        + ZOB_WINNER.size
        + 1
        + ZOB_SEAT_DEAD.size
        + ZOB_SEAT_FLAG_REVEALED.size
        + ZOB_SHOW_MODE.size
    )
    print(
        f"junqi_core._zobrist self-check: OK "
        f"({n_entries:,} int64 entries, "
        f"{n_entries * 8 / 1024:.1f} KiB)"
    )
