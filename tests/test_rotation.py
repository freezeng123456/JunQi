
"""Tests for junqi_core.rotation — canonical coordinate rotation.

Verifies:
  - Round-trip bijection world→canonical→world for all cells and all seats.
  - Seat centers all map to canonical SOUTH position.
  - Teammate ends up at top of canonical view for every acting seat.
  - Numpy plane rotation is consistent with point rotation.
  - Camps/strongholds/rails retain their role after rotation (i.e. rotating
    the static topology gives the same board — a 4-fold symmetry check).
"""

from __future__ import annotations

import numpy as np
import pytest

from junqi_core.board import (
    BOARD_SIZE,
    CELL_TABLE,
    cell_info,
    index_to_pos,
    is_on_board,
)
from junqi_core.rotation import (
    canonical_to_world,
    rotate_action,
    rotate_plane,
    rotate_planes,
    unrotate_action,
    unrotate_plane,
    world_to_canonical,
)
from junqi_core.rules import Seat


@pytest.mark.parametrize("seat", list(Seat))
def test_point_round_trip(seat: Seat) -> None:
    """world → canonical → world should recover the original cell."""
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            c = world_to_canonical(x, y, seat)
            back = canonical_to_world(*c, seat)
            assert back == (x, y), (
                f"seat={seat.name} world=({x},{y}) canonical={c} back={back}"
            )


@pytest.mark.parametrize(
    "seat,world_center",
    [
        (Seat.SOUTH, (8, 13)),
        (Seat.WEST, (3, 8)),
        (Seat.NORTH, (8, 3)),
        (Seat.EAST, (13, 8)),
    ],
)
def test_seat_center_maps_to_canonical_home(
    seat: Seat, world_center: tuple[int, int]
) -> None:
    """Each seat's central camp should map to canonical (8, 13).

    This is THE invariant of canonical rotation: the acting seat always
    appears in the SOUTH (bottom) position after rotation.
    """
    canonical = world_to_canonical(*world_center, seat)
    assert canonical == (8, 13), (
        f"seat {seat.name} center {world_center} should map to (8,13), got {canonical}"
    )


@pytest.mark.parametrize("seat", list(Seat))
def test_teammate_ends_up_at_top(seat: Seat) -> None:
    """For every acting seat, the teammate's pieces should all land in the
    top zone (y ∈ [0, 5]) of the canonical frame."""
    teammate = seat.teammate
    # Take teammate's index-0 corner cell as representative.
    world_tm = index_to_pos(teammate, 0)
    canonical_tm = world_to_canonical(*world_tm, seat)
    _, yc = canonical_tm
    assert yc <= 5, (
        f"acting seat {seat.name}: teammate {teammate.name} corner "
        f"world={world_tm} canonical={canonical_tm} should have y<=5"
    )


@pytest.mark.parametrize("seat", list(Seat))
def test_enemies_end_up_on_sides(seat: Seat) -> None:
    """For every acting seat, enemy seats end up on the left/right of canonical."""
    for enemy in (seat.left_side_enemy, seat.right_side_enemy):
        # Take enemy's corner cell
        world_e = index_to_pos(enemy, 0)
        xc, yc = world_to_canonical(*world_e, seat)
        # Enemies should have y in [6, 10] (middle row band) in canonical frame
        assert 6 <= yc <= 10, (
            f"acting {seat.name}: enemy {enemy.name} corner world={world_e} "
            f"canonical=({xc},{yc}) should have y in [6,10]"
        )
        # And x should be at either left (<= 5) or right (>= 11) side
        assert xc <= 5 or xc >= 11, (
            f"acting {seat.name}: enemy {enemy.name} corner canonical "
            f"x={xc} should be <=5 or >=11"
        )


@pytest.mark.parametrize("seat", list(Seat))
def test_right_side_enemy_lands_on_canonical_right(seat: Seat) -> None:
    """`right_side_enemy` (= seat `(s+3)%4`) lands at canonical x >= 11.

    This pins down the geometric meaning of the name: after `rotate_plane`
    with this seat as acting seat, `right_side_enemy`'s pieces appear on
    the RIGHT half of the canonical image (x ∈ [11, 16]). The algebraic
    identity `right_side_enemy == Seat((self+3)%4)` therefore matches the
    visual identity "on the right side of my canonical image".
    """
    world_e = index_to_pos(seat.right_side_enemy, 0)
    xc, _ = world_to_canonical(*world_e, seat)
    assert xc >= 11, (
        f"acting {seat.name}: right_side_enemy {seat.right_side_enemy.name} corner "
        f"world={world_e} canonical x={xc}, expected >= 11"
    )


@pytest.mark.parametrize("seat", list(Seat))
def test_left_side_enemy_lands_on_canonical_left(seat: Seat) -> None:
    """`left_side_enemy` (= seat `(s+1)%4`) lands at canonical x <= 5.

    Symmetric to `test_right_side_enemy_lands_on_canonical_right`: after
    rotation, `left_side_enemy`'s pieces appear on the LEFT half of the
    canonical image.
    """
    world_e = index_to_pos(seat.left_side_enemy, 0)
    xc, _ = world_to_canonical(*world_e, seat)
    assert xc <= 5, (
        f"acting {seat.name}: left_side_enemy {seat.left_side_enemy.name} corner "
        f"world={world_e} canonical x={xc}, expected <= 5"
    )


@pytest.mark.parametrize("seat", list(Seat))
def test_plane_rotation_consistency(seat: Seat) -> None:
    """rotate_plane must agree with world_to_canonical for every cell."""
    # Build a plane where plane[y, x] = unique int encoding (x, y)
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            plane[y, x] = y * 100 + x

    rotated = rotate_plane(plane, seat)

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            xc, yc = world_to_canonical(x, y, seat)
            assert rotated[yc, xc] == plane[y, x], (
                f"seat={seat.name} world=({x},{y}) canonical=({xc},{yc}) "
                f"rotated[{yc},{xc}]={rotated[yc, xc]} plane[{y},{x}]={plane[y, x]}"
            )


@pytest.mark.parametrize("seat", list(Seat))
def test_plane_round_trip(seat: Seat) -> None:
    """rotate_plane then unrotate_plane should recover the original plane."""
    rng = np.random.default_rng(seed=42)
    plane = rng.standard_normal((BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    back = unrotate_plane(rotate_plane(plane, seat), seat)
    assert np.array_equal(back, plane)


@pytest.mark.parametrize("seat", list(Seat))
def test_planes_batched(seat: Seat) -> None:
    """rotate_planes on a (C, H, W) stack should rotate each plane independently."""
    rng = np.random.default_rng(seed=7)
    planes = rng.standard_normal((5, BOARD_SIZE, BOARD_SIZE)).astype(np.float32)
    rotated = rotate_planes(planes, seat)
    assert rotated.shape == planes.shape
    for c in range(planes.shape[0]):
        expected = rotate_plane(planes[c], seat)
        assert np.array_equal(rotated[c], expected)


@pytest.mark.parametrize("seat", list(Seat))
def test_action_round_trip(seat: Seat) -> None:
    """An action rotated to canonical and back should equal the original."""
    src = (8, 11)
    dst = (8, 10)
    c_src, c_dst = rotate_action(src, dst, seat)
    back_src, back_dst = unrotate_action(c_src, c_dst, seat)
    assert back_src == src and back_dst == dst


def test_topology_symmetry() -> None:
    """Board topology is 4-fold rotation-symmetric: rotating the board should
    produce the same topology (with seat labels permuted)."""
    # Build a plane encoding cell role:
    #   0=off-board, 1=camp, 2=stronghold, 3=rail, 4=ninegrid, 5=plain seat cell
    role = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            ci = cell_info(x, y)
            if not ci.is_on_board:
                role[y, x] = 0
            elif ci.is_camp:
                role[y, x] = 1
            elif ci.is_stronghold:
                role[y, x] = 2
            elif ci.is_nine_grid:
                role[y, x] = 4
            elif ci.is_railway:
                role[y, x] = 3
            else:
                role[y, x] = 5

    # Rotating by each seat should produce the same plane, because the
    # board is symmetric under 90° rotation (4 seats are interchangeable).
    for seat in Seat:
        rotated = rotate_plane(role, seat)
        assert np.array_equal(rotated, role), (
            f"topology not symmetric under seat={seat.name} rotation"
        )
