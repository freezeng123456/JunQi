"""Facts-only identity constraints, independent of any learned belief model.

The old v1 probability-based adapter lives in ``legacy_joint``. Its tensors
and checkpoints are incompatible with this public-history API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from junqi_core.identity import TRACKED_TYPES
from junqi_core.knowledge import KNOWLEDGE_SCHEMA, KnowledgeSnapshot, PublicEvent
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import CAMP_INDICES, PIECE_COUNTS, ShowMode

IDENTITY_INPUT_SCHEMA = "junqi.hidden-identity-facts.v1"
_SLOTS = np.asarray([slot for slot in range(30) if slot not in CAMP_INDICES])
_COUNTS = np.asarray([PIECE_COUNTS[t] for t in TRACKED_TYPES], dtype=np.int64)


@dataclass(frozen=True)
class IdentityConstraintInput:
    schema: str
    observer_seats: np.ndarray
    piece_ids: np.ndarray
    cells: np.ndarray
    initial_cells: np.ndarray
    owners: np.ndarray
    alive: np.ndarray
    known_types: np.ndarray
    type_domains: np.ndarray
    initial_inventory: np.ndarray
    public_counts: np.ndarray
    observer_known_types: np.ndarray
    public_history: tuple[tuple[PublicEvent, ...], ...]


def build_joint_input(snapshots: Sequence[KnowledgeSnapshot]) -> IdentityConstraintInput:
    """Package exact observer facts; probabilities and simulator types are absent.

    Contains all 25 identities in each of the other three armies, including
    known and unknown dead pieces. Initial inventory is fixed game knowledge,
    never the simulator's hidden surviving counts. Public combat relations are
    retained separately from per-piece domains; both constrain future samples.
    """
    snapshots = tuple(snapshots)
    if not snapshots:
        raise ValueError("At least one observer snapshot is required")
    for snapshot in snapshots:
        if not isinstance(snapshot, KnowledgeSnapshot) or snapshot.schema != KNOWLEDGE_SCHEMA:
            raise ValueError("Identity input requires facts-only observer snapshots")
        if snapshot.show_mode is not ShowMode.DARK:
            raise ValueError("The three-hidden-army identity contract requires DARK mode")
    ids = np.stack([np.concatenate([
        ((int(s.observer) + relative_owner) % 4) * 30 + _SLOTS
        for relative_owner in (1, 2, 3)
    ]) for s in snapshots])
    for s, pids in zip(snapshots, ids):
        if (s.initial_cells[pids] < 0).any():
            raise ValueError("Identity input requires a complete standard opening")

    def gather(name):
        return np.stack([getattr(s, name)[pids] for s, pids in zip(snapshots, ids)])

    cells, initial = gather('cells'), gather('initial_cells')
    for row, s in enumerate(snapshots):
        for values in (cells, initial):
            for i in np.flatnonzero(values[row] >= 0):
                x, y = world_to_canonical(int(values[row, i] % 17),
                                         int(values[row, i] // 17), s.observer)
                values[row, i] = y * 17 + x
    inventory = np.zeros((len(snapshots), 4, 12), dtype=np.int64)
    inventory[:, 1:] = _COUNTS
    counts = np.stack([gather(name) for name in (
        'move_counts', 'active_eat_counts', 'passive_survive_counts')], axis=-1)
    domains = gather('type_domains')
    if not domains.any(-1).all():
        raise ValueError("Contradictory public identity restrictions")
    arrays = (np.asarray([int(s.observer) for s in snapshots], dtype=np.int8),
              ids, cells, initial,
              np.tile(np.repeat(np.arange(1, 4), 25), (len(snapshots), 1)),
              gather('alive'), gather('known_types'), domains, inventory, counts,
              np.stack([s.known_types for s in snapshots]))
    for array in arrays:
        array.setflags(write=False)
    return IdentityConstraintInput(IDENTITY_INPUT_SCHEMA, *arrays,
                                   tuple(s.events for s in snapshots))
