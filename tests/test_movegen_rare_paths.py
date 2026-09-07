"""Exercise move-existence and large-batch paths against the PieceMap rules."""

from __future__ import annotations

import numpy as np
import pytest

from junqi_core import _movegen_tables as tables
from junqi_core import move_gen
from junqi_core.rules import PieceType, Seat


def position(source, *, engineer=False, blocked=False):
    cells = np.full(289, -1, dtype=np.int16)
    seats = np.repeat(np.arange(4, dtype=np.int8), 30)
    types = np.full(120, PieceType.DILEI.value, dtype=np.int8)
    alive = np.zeros(120, dtype=bool)
    x = np.full(120, -1, dtype=np.int8)
    y = x.copy()
    pieces = {}
    placements = [(0, source, PieceType.GONGB if engineer else PieceType.PAIZH)]
    if blocked:
        neighbors = set(int(f) for f in tables.ADJ_STRAIGHT[source] if f >= 0)
        neighbors.update(int(f) for f in tables.ADJ_DIAG_INTO_CAMP_PAD[source] if f >= 0)
        neighbors.update(int(f) for f in tables.ENGINEER_RAIL_NEIGHBORS[source])
        placements.extend((i, f, PieceType.DILEI) for i, f in enumerate(sorted(neighbors), 1))
    for pid, f, kind in placements:
        cells[f] = pid
        types[pid] = kind.value
        alive[pid] = True
        x[pid] = f % 17
        y[pid] = f // 17
        pieces[(int(x[pid]), int(y[pid]))] = move_gen.PieceRef(Seat.SOUTH, kind, piece_id=pid)
    return (cells, seats, types, alive, x, y, 0), pieces


def reference(pieces):
    return {
        (s[1] * 17 + s[0]) * 289 + d[1] * 17 + d[0]
        for s, d in move_gen.generate_legal_actions(pieces, Seat.SOUTH)
    }


SOURCES = [int(f) for f in np.flatnonzero(tables.IS_ON_BOARD_FLAT & ~tables.IS_STRONGHOLD_FLAT)]


@pytest.mark.parametrize("engineer", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_move_existence_matches_rules_at_every_playable_source(engineer, blocked):
    for source in SOURCES:
        args, pieces = position(source, engineer=engineer, blocked=blocked)
        expected = reference(pieces)
        assert move_gen.has_legal_moves_soa(*args) == bool(expected), (source, engineer, blocked)
        assert set(move_gen.generate_legal_action_ids_batch(*args)) == expected


@pytest.mark.parametrize("count", [100, 101, 128])
def test_large_engineer_batch_matches_independent_rules(count):
    source = int(tables.ENG_RAIL_CELLS[0])
    states = [position(source, engineer=True, blocked=i % 2 == 0) for i in range(count)]
    columns = [np.stack([args[j] for args, _ in states]) for j in range(6)]
    result = move_gen.generate_legal_action_ids_n(
        *columns,
        np.zeros(count, dtype=np.int8),
        np.zeros(count, dtype=bool),
        np.where(columns[0] < 0, 255, 0).astype(np.uint8),
    )
    for actual, (_, pieces) in zip(result, states, strict=True):
        assert set(actual) == reference(pieces)


def test_n_batch_curve_paths_match_rules_at_every_source():
    states = [position(source, blocked=blocked) for source in SOURCES for blocked in (False, True)]
    columns = [np.stack([args[j] for args, _ in states]) for j in range(6)]
    count = len(states)
    result = move_gen.generate_legal_action_ids_n(
        *columns,
        np.zeros(count, dtype=np.int8),
        np.zeros(count, dtype=bool),
        np.where(columns[0] < 0, 255, 0).astype(np.uint8),
    )
    for actual, (_, pieces) in zip(result, states, strict=True):
        assert set(actual) == reference(pieces)
