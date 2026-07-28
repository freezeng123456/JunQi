
"""World ↔ canonical coordinate rotation.

When a neural network is queried for seat `s`, we first rotate the board so
seat `s` appears at the SOUTH position (bottom of the canonical image). After
rotation:
  - Acting seat at bottom (canonical y ∈ [11, 16]).
  - Teammate at top (canonical y ∈ [0, 5]).
  - `s.left_side_enemy`  (seat `(s+1)%4`) at canonical x ∈ [0,  5]  (image LEFT  half).
  - `s.right_side_enemy` (seat `(s+3)%4`) at canonical x ∈ [11, 16] (image RIGHT half).

The network outputs actions in the canonical frame; we then unrotate back to
the world frame before executing. See `docs/ARCHITECTURE.md` §3.

Naming note (Phase 0.3 T7 rename, ADR-111): the four seats are named after
their compass position on the world-frame board (SOUTH/WEST/NORTH/EAST),
and the two enemies-of-the-acting-seat are named after which image half
they land on in the canonical frame (`left_side_enemy` / `right_side_enemy`).
This is deliberately different from the legacy first-person-view naming
("my left enemy", which ends up on the image *right* after rotation). The
geometric names make canonical-frame debug dumps match intuition — the
piece you see on the image left is, indeed, `left_side_enemy`.

All rotations are around the board center (8, 8). Pure math; no state.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .board import BOARD_SIZE
from .rules import Seat

# ===========================================================================
# Core rotation primitives
# ===========================================================================

# The last-index that both x and y can reach: 16 when BOARD_SIZE == 17.
_MAX: Final[int] = BOARD_SIZE - 1


def world_to_canonical(x: int, y: int, acting_seat: Seat) -> tuple[int, int]:
    """Rotate a single world-coordinate point into the acting seat's canonical frame.

    The mapping (see ARCHITECTURE.md §3.3):
        SOUTH (0): identity                (x, y) -> (x, y)
        WEST  (1): +90° CCW                (x, y) -> (y, 16-x)
        NORTH (2): 180°                    (x, y) -> (16-x, 16-y)
        EAST  (3): -90° CW                 (x, y) -> (16-y, x)
    """
    if acting_seat is Seat.SOUTH:
        return (x, y)
    if acting_seat is Seat.WEST:
        return (y, _MAX - x)
    if acting_seat is Seat.NORTH:
        return (_MAX - x, _MAX - y)
    if acting_seat is Seat.EAST:
        return (_MAX - y, x)
    raise AssertionError(f"unreachable seat {acting_seat!r}")


def canonical_to_world(x: int, y: int, acting_seat: Seat) -> tuple[int, int]:
    """Inverse of `world_to_canonical`.

        SOUTH (0): identity                (x, y) -> (x, y)
        WEST  (1): inverse of +90° CCW     (x, y) -> (16-y, x)
        NORTH (2): 180° (self-inverse)     (x, y) -> (16-x, 16-y)
        EAST  (3): inverse of -90° CW      (x, y) -> (y, 16-x)
    """
    if acting_seat is Seat.SOUTH:
        return (x, y)
    if acting_seat is Seat.WEST:
        return (_MAX - y, x)
    if acting_seat is Seat.NORTH:
        return (_MAX - x, _MAX - y)
    if acting_seat is Seat.EAST:
        return (y, _MAX - x)
    raise AssertionError(f"unreachable seat {acting_seat!r}")


# ===========================================================================
# Batched rotation on numpy arrays
# ===========================================================================


def rotate_plane(plane: np.ndarray, acting_seat: Seat) -> np.ndarray:
    """Rotate a 2D plane (shape [H, W]) into the acting seat's canonical frame.

    Args:
        plane: np.ndarray with shape (H, W); we assume indexing [y, x] (first
               axis is y as is standard for image tensors).
        acting_seat: seat whose canonical frame we are rotating into.

    Returns:
        np.ndarray of same shape, rotated.

    Implementation uses np.rot90(plane, k, axes=(0, 1)) where axes=(0,1) means
    rotation in the (y, x) plane. With our indexing convention plane[y, x]:
      np.rot90(a, k=+1)  ⇒ out[y, x] = a[x, W-1-y]
      np.rot90(a, k=-1)  ⇒ out[y, x] = a[H-1-x, y]

    Required point-level mapping (see ARCHITECTURE.md §3.3):
      SOUTH : (x, y) -> (x, y)          ⇒ k=0
      WEST  : (x, y) -> (y, 16-x)       ⇒ k=+1   (verified: rotated[16-x, y] = plane[y, x])
      NORTH : (x, y) -> (16-x, 16-y)    ⇒ k=2
      EAST  : (x, y) -> (16-y, x)       ⇒ k=-1
    """
    if plane.ndim != 2:
        raise ValueError(f"expected 2D plane, got shape {plane.shape}")
    if plane.shape[0] != plane.shape[1]:
        raise ValueError(
            f"rotate_plane only supports square planes, got {plane.shape}"
        )
    k = {
        Seat.SOUTH: 0,
        Seat.WEST: 1,
        Seat.NORTH: 2,
        Seat.EAST: -1,
    }[acting_seat]
    return np.rot90(plane, k=k)


def rotate_planes(planes: np.ndarray, acting_seat: Seat) -> np.ndarray:
    """Rotate a stack of 2D planes. Shape (C, H, W) -> (C, H, W)."""
    if planes.ndim != 3:
        raise ValueError(f"expected (C,H,W), got shape {planes.shape}")
    if planes.shape[1] != planes.shape[2]:
        raise ValueError(
            f"rotate_planes only supports square planes, got {planes.shape}"
        )
    k = {
        Seat.SOUTH: 0,
        Seat.WEST: 1,
        Seat.NORTH: 2,
        Seat.EAST: -1,
    }[acting_seat]
    return np.rot90(planes, k=k, axes=(1, 2))


def unrotate_plane(plane: np.ndarray, acting_seat: Seat) -> np.ndarray:
    """Inverse of rotate_plane."""
    if plane.ndim != 2:
        raise ValueError(f"expected 2D plane, got shape {plane.shape}")
    # Negate the rotation amount to invert
    k = {
        Seat.SOUTH: 0,
        Seat.WEST: -1,   # inverse of +1
        Seat.NORTH: 2,   # 180 is self-inverse
        Seat.EAST: 1,    # inverse of -1
    }[acting_seat]
    return np.rot90(plane, k=k)


# ===========================================================================
# Action (src, dst) rotation
# ===========================================================================


def rotate_action(
    src: tuple[int, int],
    dst: tuple[int, int],
    acting_seat: Seat,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Rotate a world-frame action to the canonical frame for the given seat."""
    return (
        world_to_canonical(*src, acting_seat),
        world_to_canonical(*dst, acting_seat),
    )


def unrotate_action(
    src_c: tuple[int, int],
    dst_c: tuple[int, int],
    acting_seat: Seat,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Rotate a canonical-frame action back to world frame."""
    return (
        canonical_to_world(*src_c, acting_seat),
        canonical_to_world(*dst_c, acting_seat),
    )


# ===========================================================================
# Round-trip self-check
# ===========================================================================


def _self_check() -> None:
    """Verify rotation bijection for every seat and every cell."""
    for seat in Seat:
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                c = world_to_canonical(x, y, seat)
                back = canonical_to_world(*c, seat)
                assert back == (x, y), (
                    f"round-trip failed for seat={seat.name} "
                    f"world=({x},{y}) canonical={c} back={back}"
                )

    # Verify seat centers all map to SOUTH center (8, 13) under their own rotation.
    # (This is the key invariant: every acting seat sees itself at the bottom.)
    # SOUTH center is any cell around (8, 13); we pick (8, 13) explicitly as an
    # emblematic SOUTH cell (seat-local index 12, the central camp).
    # For the other 3 seats, the cell that ROTATES TO (8, 13) should be
    # that seat's central camp in world coords.
    # Actually the cleaner invariant is: each seat's "center camp" ends up at
    # canonical (8, 13). Let's verify.
    seat_centers_world = {
        Seat.SOUTH: (8, 13),
        Seat.WEST: (3, 8),
        Seat.NORTH: (8, 3),
        Seat.EAST: (13, 8),
    }
    for seat, world_center in seat_centers_world.items():
        canonical = world_to_canonical(*world_center, seat)
        assert canonical == (8, 13), (
            f"seat {seat.name} center {world_center} maps to {canonical}, "
            f"expected (8, 13)"
        )

    # Check numpy plane rotation round-trip on a 17×17 grid of unique values.
    plane = np.arange(BOARD_SIZE * BOARD_SIZE).reshape(BOARD_SIZE, BOARD_SIZE)
    for seat in Seat:
        rotated = rotate_plane(plane, seat)
        restored_plane = unrotate_plane(rotated, seat)
        assert np.array_equal(restored_plane, plane), (
            f"plane round-trip failed for {seat.name}"
        )

    # Check that point rotation is consistent with plane rotation.
    # plane[y, x] indexes the value at cell (x, y). After rotate_plane, the
    # value that was at world (x, y) should appear at canonical coord
    # (xc, yc) = world_to_canonical(x, y, seat), so rotated[yc, xc] should
    # equal plane[y, x].
    for seat in Seat:
        rotated = rotate_plane(plane, seat)
        # Sample a few points
        for (x, y) in [(0, 0), (3, 5), (8, 8), (16, 16), (10, 12)]:
            xc, yc = world_to_canonical(x, y, seat)
            assert rotated[yc, xc] == plane[y, x], (
                f"plane consistency failed: seat={seat.name} "
                f"world=({x},{y}) canonical=({xc},{yc}) "
                f"rotated={rotated[yc, xc]} original={plane[y, x]}"
            )



_self_check()


if __name__ == "__main__":
    _self_check()
    # Print a visual confirmation: the four "center camps" all end up at (8, 13)
    for seat in Seat:
        world_center = {
            Seat.SOUTH: (8, 13),
            Seat.WEST: (3, 8),
            Seat.NORTH: (8, 3),
            Seat.EAST: (13, 8),
        }[seat]
        print(
            f"seat={seat.name:5s} world={world_center} "
            f"-> canonical={world_to_canonical(*world_center, seat)}"
        )
