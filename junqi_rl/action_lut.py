"""junqi_rl.action_lut — Pre-computed action-id rotation look-up tables.

Building and indexing rotation LUTs avoids O(K) Python-level
``rotate_action_id`` / ``unrotate_action_id`` calls per environment step
(where K ~ 100 legal actions).  Instead we do a single NumPy fancy-index
read of a pre-allocated int32 array.

Compact indexing
----------------
Actions use compact cell indices [0, 129) over on-board cells only, giving
an action space of 129 × 129 = 16,641 (down from 289 × 289 = 83,521).

Tables
------
ROTATE_LUT[seat_idx]   : ndarray int32 shape (FLAT_ACTION_DIM,)
    world_compact_id  → canonical_compact_id  for seat ``Seat(seat_idx)``

UNROTATE_LUT[seat_idx] : ndarray int32 shape (FLAT_ACTION_DIM,)
    canonical_compact_id → world_compact_id   for seat ``Seat(seat_idx)``

Usage
-----
    from junqi_rl.action_lut import ROTATE_LUT, UNROTATE_LUT
    from junqi_core.rules import Seat

    # Batch-rotate a world-frame int32 array to canonical frame
    can_ids = ROTATE_LUT[Seat.WEST.value][world_ids]

    # Batch-unrotate a canonical-frame int32 array to world frame
    world_ids = UNROTATE_LUT[Seat.WEST.value][can_ids]

Both operations are O(K) but performed entirely in NumPy (no Python loop).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from junqi_core.board import (
    BOARD_SIZE,
    NUM_CELLS,
    NUM_ON_BOARD_CELLS,
    COMPACT_ACTION_DIM,
    FLAT_TO_COMPACT,
    COMPACT_TO_FLAT,
)
from junqi_core.rotation import canonical_to_world, world_to_canonical
from junqi_core.rules import ALL_SEATS, Seat

if TYPE_CHECKING:
    from junqi_rl.env import VectorJunqiEnv

# ---------------------------------------------------------------------------
# Compact action space constants
# ---------------------------------------------------------------------------

FLAT_ACTION_DIM: int = COMPACT_ACTION_DIM   # 16,641 (was 83,521)


def _build_cell_rotation_lut(seat: Seat) -> np.ndarray:
    """Return int32 array of length NUM_CELLS mapping world_flat → can_flat."""
    lut = np.empty(NUM_CELLS, dtype=np.int32)
    for flat in range(NUM_CELLS):
        x, y = flat % BOARD_SIZE, flat // BOARD_SIZE
        xc, yc = world_to_canonical(x, y, seat)
        lut[flat] = yc * BOARD_SIZE + xc
    return lut


def _build_cell_unrotation_lut(seat: Seat) -> np.ndarray:
    """Return int32 array of length NUM_CELLS mapping can_flat → world_flat."""
    lut = np.empty(NUM_CELLS, dtype=np.int32)
    for flat in range(NUM_CELLS):
        xc, yc = flat % BOARD_SIZE, flat // BOARD_SIZE
        xw, yw = canonical_to_world(xc, yc, seat)
        lut[flat] = yw * BOARD_SIZE + xw
    return lut


def _build_compact_action_lut(cell_lut: np.ndarray) -> np.ndarray:
    """Build compact action-id LUT from a cell-id LUT (world→can or can→world).

    Maps compact_action_id → compact_action_id using on-board cells only.

    compact_action_id = compact_src * NUM_ON_BOARD + compact_dst
    For each (compact_src, compact_dst):
      1. Map to (world_flat_src, world_flat_dst) via COMPACT_TO_FLAT
      2. Rotate via cell_lut: (can_flat_src, can_flat_dst)
      3. Map back to compact via FLAT_TO_COMPACT
      4. Result = compact_can_src * NUM_ON_BOARD + compact_can_dst
    """
    N = NUM_ON_BOARD_CELLS  # 129
    lut = np.full(FLAT_ACTION_DIM, -1, dtype=np.int32)

    for compact_src in range(N):
        world_src = COMPACT_TO_FLAT[compact_src]
        rotated_src = cell_lut[world_src]
        compact_rot_src = FLAT_TO_COMPACT[rotated_src]
        if compact_rot_src < 0:
            continue  # off-board after rotation (shouldn't happen for on-board cells)
        for compact_dst in range(N):
            world_dst = COMPACT_TO_FLAT[compact_dst]
            rotated_dst = cell_lut[world_dst]
            compact_rot_dst = FLAT_TO_COMPACT[rotated_dst]
            if compact_rot_dst < 0:
                continue
            old_idx = compact_src * N + compact_dst
            new_idx = int(compact_rot_src) * N + int(compact_rot_dst)
            lut[old_idx] = new_idx

    return lut


# Build once — 4 × 2 arrays of shape (16641,) int32
# ROTATE_LUT[i]   : world_compact_id  → canonical_compact_id  for Seat(i)
# UNROTATE_LUT[i] : canonical_compact_id → world_compact_id   for Seat(i)
ROTATE_LUT: list[np.ndarray] = []
UNROTATE_LUT: list[np.ndarray] = []

for _seat in ALL_SEATS:
    _cell_rot = _build_cell_rotation_lut(_seat)
    _cell_unrot = _build_cell_unrotation_lut(_seat)
    ROTATE_LUT.append(_build_compact_action_lut(_cell_rot))
    UNROTATE_LUT.append(_build_compact_action_lut(_cell_unrot))


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------


def batch_rotate_action_ids(
    world_ids: np.ndarray,
    seat: Seat,
) -> np.ndarray:
    """Rotate an int32 array of world-frame action ids to canonical frame.

    Parameters
    ----------
    world_ids
        1-D int32 array of world-frame action ids.
    seat
        Acting seat (canonical frame anchor).

    Returns
    -------
    can_ids : int32 ndarray of same shape as ``world_ids``.
    """
    return ROTATE_LUT[seat.value][world_ids]


def batch_unrotate_action_ids(
    can_ids: np.ndarray,
    seat: Seat,
) -> np.ndarray:
    """Rotate an int32 array of canonical-frame action ids to world frame.

    Parameters
    ----------
    can_ids
        1-D int32 array of canonical-frame action ids.
    seat
        Acting seat whose canonical frame we are leaving.

    Returns
    -------
    world_ids : int32 ndarray of same shape as ``can_ids``.
    """
    return UNROTATE_LUT[seat.value][can_ids]


def build_legal_mask_batch(
    env: "VectorJunqiEnv",
    current_seats: list[Seat],
) -> np.ndarray:
    """Fast legal-mask construction via compact cell indexing.

    World-frame action ids from the CPU env are converted to compact
    canonical-frame action ids (129-based) and scattered into the mask.

    Returns
    -------
    mask : bool ndarray, shape (num_envs, FLAT_ACTION_DIM=16641)
    """
    N = env.num_envs
    mask = np.zeros((N, FLAT_ACTION_DIM), dtype=bool)

    for i in range(N):
        if env.done[i]:
            continue
        seat = current_seats[i]
        # World-frame 289-based action ids from CPU game engine
        world_ids = env.envs[i].legal_action_ids(seat)   # int32 array

        # Convert each world action to compact canonical:
        # 1. Decompose: src_world_flat = id // 289, dst_world_flat = id % 289
        # 2. Rotate world → canonical (289-based)
        # 3. Map 289 → compact via FLAT_TO_COMPACT
        cell_rot = _build_cell_rotation_lut(seat)  # world_flat → can_flat
        src_world = world_ids // 289
        dst_world = world_ids % 289
        src_can = cell_rot[src_world]
        dst_can = cell_rot[dst_world]
        src_compact = FLAT_TO_COMPACT[src_can]
        dst_compact = FLAT_TO_COMPACT[dst_can]
        # Filter out any off-board mappings (shouldn't happen for legal actions)
        valid = (src_compact >= 0) & (dst_compact >= 0)
        compact_ids = src_compact[valid].astype(np.int32) * NUM_ON_BOARD_CELLS + dst_compact[valid].astype(np.int32)
        mask[i, compact_ids] = True
    return mask
