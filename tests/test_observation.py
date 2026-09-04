"""Tests for junqi_core.observation — observation tensor builder (Phase 0.3 T7).

Covers the frozen 69 spatial + 28 global layout (ADR-106, ARCHITECTURE §4)
under scheme D (Prob-teammate unified mode-agnostic channels).

Invariant categories:
  A. Shape & dtype compliance
  B. World-frame correctness (conservation sums, probability normalisation)
  C. Canonical-frame geometry (own@bottom, left_side@left, right_side@right)
  D. 4-fold canonical symmetry
  E. Global features
  F. ShowMode coverage (BRIGHT / HALF_DARK / DARK)
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor, NUM_TRACKED_TYPES, TRACKED_TYPES
from junqi_core.observation import (
    CHANNEL_LAYOUT,
    GLOBAL_LAYOUT,
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationTensor,
    build_observation,
    channel_name,
)
from junqi_core.rules import ALL_SEATS, PIECE_COUNTS, PieceType, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


# ===========================================================================
# Fixtures
# ===========================================================================


def _random_opening(
    seed: int = 42,
    show_mode: ShowMode = ShowMode.HALF_DARK,
) -> GameState:
    rng = random.Random(seed)
    return GameState.new_game(generate_random_setup(rng), show_mode=show_mode)


def _obs_for(state: GameState, observer: Seat) -> ObservationTensor:
    belief = BeliefTensor.initial(state, observer)
    return build_observation(state, belief, observer)


# ===========================================================================
# A. Shape & dtype compliance
# ===========================================================================


@pytest.mark.parametrize("observer", list(ALL_SEATS))
def test_observation_shape(observer: Seat) -> None:
    st = _random_opening()
    obs = _obs_for(st, observer)
    assert obs.spatial.shape == (OBS_CHANNELS, 17, 17)
    assert obs.global_.shape == (OBS_GLOBAL_DIMS,)
    assert obs.spatial.dtype == np.float32
    assert obs.global_.dtype == np.float32
    assert obs.observer is observer


def test_channel_layout_totals() -> None:
    running = 0
    for name, sl in CHANNEL_LAYOUT.items():
        assert sl.start == running, f"gap at {name}"
        running = sl.stop
    # T7 / ADR-116: tail 32 channels for Ataraxos parity.
    # ADR-128 v4: tail 50 channels for CombatMemory.
    # ADR-129 v5: +46 channels for per-pid identity tail.
    assert running == OBS_CHANNELS == 412

    running = 0
    for name, sl in GLOBAL_LAYOUT.items():
        assert sl.start == running
        running = sl.stop
    assert running == OBS_GLOBAL_DIMS == 28

    # T7 tail groups must all be present and correctly sized.
    # Sizes track observation.py's 256-channel layout (see commit 3413281):
    # death_reason = 3 reasons × 4 perspectives (me/teammate/left/right);
    # dead_at_zero = 2 (me + teammate).
    tail_groups = {
        "move_bucket": 8,
        "active_eat_bucket": 8,
        "passive_survive_bucket": 8,
        "death_reason": 12,
        "dead_at_zero": 2,
    }
    for name, size in tail_groups.items():
        assert name in CHANNEL_LAYOUT, f"missing T7 channel group {name!r}"
        sl = CHANNEL_LAYOUT[name]
        assert sl.stop - sl.start == size, (
            f"{name} expected size {size}, got {sl.stop - sl.start}"
        )


# ===========================================================================
# B. World-frame correctness (conservation / normalisation)
# ===========================================================================


def test_piece_own_sum_equals_live_own_pieces() -> None:
    st = _random_opening()
    obs = _obs_for(st, Seat.SOUTH)
    live_own = sum(
        1 for pr in st.pieces.values() if pr.seat is Seat.SOUTH and pr.alive
    )
    assert int(obs.channel("piece_own").sum()) == live_own == 25


def test_belief_side_channels_probability_conservation() -> None:
    st = _random_opening()
    obs = _obs_for(st, Seat.SOUTH)
    for enemy, group in (
        (Seat.SOUTH.left_side_enemy, "belief_left_side"),
        (Seat.SOUTH.right_side_enemy, "belief_right_side"),
    ):
        live = sum(
            1 for pr in st.pieces.values() if pr.seat is enemy and pr.alive
        )
        mass = float(obs.channel(group).sum())
        assert abs(mass - live) < 1e-4, (
            f"belief mass {mass} != {live} for {enemy.name}"
        )


def test_board_static_is_constant_across_games() -> None:
    st1 = _random_opening(seed=1)
    st2 = _random_opening(seed=2)
    a = _obs_for(st1, Seat.SOUTH).channel("board_static")
    b = _obs_for(st2, Seat.SOUTH).channel("board_static")
    assert np.array_equal(a, b)


# Plane order of the board_static group, with the exact number of cells each
# one must light up.  Pinning the counts is what makes this test non-vacuous:
# the curve_rail plane was silently all-zero for a long time because it was
# read through ``getattr(info, "curve_rail_id", 0)`` while the attribute is
# named ``curve_rail``, and every other check on this group (cross-observer
# equality, CPU/GPU parity) is satisfied by two zero planes.
_BOARD_STATIC_PLANES: tuple[tuple[str, int], ...] = (
    ("camp", 20),
    ("stronghold", 8),
    ("railway", 73),
    ("nine_grid", 9),
    # Two arc endpoints per board corner.  Not 40 (CURVE_RAIL_OF filtered to
    # rail) and not 48 (CURVE_RAIL_OF raw): the arc is a property of an edge,
    # and the 32 further cells CURVE_RAIL_OF groups with it are ordinary
    # straight rail that the railway plane already marks.
    ("curve_arc", 8),
    ("reserved", 0),
)
_CH_RAILWAY = 2
_CH_CURVE_ARC = 4


def test_board_static_planes_have_expected_occupancy() -> None:
    static = _obs_for(_random_opening(), Seat.SOUTH).channel("board_static")
    assert static.shape[0] == len(_BOARD_STATIC_PLANES)
    for idx, (name, expected) in enumerate(_BOARD_STATIC_PLANES):
        assert int(static[idx].sum()) == expected, (
            f"board_static plane {idx} ({name}) lit {int(static[idx].sum())} "
            f"cells, expected {expected}"
        )


def test_curve_arc_plane_marks_exactly_the_arc_endpoints() -> None:
    """The arc plane must carry only what the railway plane does not.

    An arc is a property of an edge: each board corner has one diagonal rail
    link, and its two endpoints are the only cells where a non-engineer may
    leave a straight rail run. ``CURVE_RAIL_OF`` is a coarser thing — it
    groups the two straight runs meeting at a corner so ``_same_curve_rail``
    can join them — so using it here would restate 32 ordinary rail cells the
    railway plane has already marked, and 8 headquarters cells that carry no
    rail at all.
    """
    static = _obs_for(_random_opening(), Seat.SOUTH).channel("board_static")
    arc = static[_CH_CURVE_ARC] > 0
    railway = static[_CH_RAILWAY] > 0

    leaked = arc & ~railway
    assert not leaked.any(), (
        f"{int(leaked.sum())} arc cells are not railway cells: "
        f"{[(int(x), int(y)) for y, x in zip(*np.where(leaked))]}"
    )

    # Cross-check against the rail graph itself: an arc cell is exactly a rail
    # cell with a diagonal rail neighbour.
    from junqi_core.rail_topology import RAIL_ADJ

    expected = set()
    for f, nbrs in enumerate(RAIL_ADJ):
        for n in nbrs:
            if abs(f % 17 - n % 17) == 1 and abs(f // 17 - n // 17) == 1:
                expected.add((f % 17, f // 17))
    got = {(int(x), int(y)) for y, x in zip(*np.where(arc))}
    assert got == expected, f"arc plane {sorted(got)} != graph {sorted(expected)}"


@pytest.mark.parametrize("observer", list(ALL_SEATS))
def test_board_static_planes_are_rotation_invariant(observer: Seat) -> None:
    """Each plane must be unchanged by 90-degree rotation.

    The stack is built in the world frame and then rotated into the
    observer's canonical frame, so a plane that is not rotation-invariant
    encodes the observer's seat identity.  A one-hot-per-curve encoding fails
    here: rot90 permutes the four corner curves in a 4-cycle.
    """
    static = _obs_for(_random_opening(), observer).channel("board_static")
    for idx, (name, _) in enumerate(_BOARD_STATIC_PLANES):
        for k in (1, 2, 3):
            assert np.array_equal(np.rot90(static[idx], k), static[idx]), (
                f"board_static plane {idx} ({name}) changes under rot90(k={k}); "
                f"it would leak the observer's seat into a canonical feature"
            )


# ===========================================================================
# C. Canonical-frame geometry (core promise of ADR-111 rename)
# ===========================================================================


@pytest.mark.parametrize("observer", list(ALL_SEATS))
def test_canonical_own_at_bottom(observer: Seat) -> None:
    """Observer's own pieces at canonical y in [11, 16]."""
    st = _random_opening()
    obs = _obs_for(st, observer)
    piece_own = obs.channel("piece_own")
    top = float(piece_own[:, :11, :].sum())
    bottom = float(piece_own[:, 11:, :].sum())
    assert top == 0.0, f"{observer.name}: own piece leaked to top half"
    assert bottom == 25.0, f"{observer.name}: expected 25 own pieces at bottom"


@pytest.mark.parametrize("observer", list(ALL_SEATS))
def test_canonical_left_side_enemy_on_image_left(observer: Seat) -> None:
    """left_side_enemy at canonical x in [0, 5]."""
    st = _random_opening()
    obs = _obs_for(st, observer)
    mask = obs.channel("piece_left_side_enemy")
    left = float(mask[:, :, :6].sum())
    right = float(mask[:, :, 11:].sum())
    assert right == 0.0, f"{observer.name}: left_side_enemy leaked to right"
    assert left == 25.0, f"{observer.name}: expected 25 left_side_enemy on left"


@pytest.mark.parametrize("observer", list(ALL_SEATS))
def test_canonical_right_side_enemy_on_image_right(observer: Seat) -> None:
    """right_side_enemy at canonical x in [11, 16]."""
    st = _random_opening()
    obs = _obs_for(st, observer)
    mask = obs.channel("piece_right_side_enemy")
    left = float(mask[:, :, :6].sum())
    right = float(mask[:, :, 11:].sum())
    assert left == 0.0, f"{observer.name}: right_side_enemy leaked to left"
    assert right == 25.0, f"{observer.name}: expected 25 right_side_enemy on right"


# ===========================================================================
# D. 4-fold canonical symmetry
# ===========================================================================


def test_canonical_symmetry_across_4_seats() -> None:
    """Board-static identical across observers; piece cardinalities identical."""
    st = _random_opening(show_mode=ShowMode.BRIGHT)
    obss = {s: _obs_for(st, s) for s in ALL_SEATS}

    ref = obss[Seat.SOUTH].channel("board_static")
    for s in ALL_SEATS:
        assert np.array_equal(ref, obss[s].channel("board_static")), (
            f"board_static differs for observer {s.name}"
        )

    for s in ALL_SEATS:
        o = obss[s]
        assert int(o.channel("piece_own").sum()) == 25
        assert abs(float(o.channel("prob_teammate").sum()) - 25.0) < 1e-4
        assert int(o.channel("piece_left_side_enemy").sum()) == 25
        assert int(o.channel("piece_right_side_enemy").sum()) == 25


# ===========================================================================
# E. Global features
# ===========================================================================


def test_global_remaining_counts_match_piece_counts() -> None:
    """Fresh opening: every enemy has PIECE_COUNTS-matching inventory."""
    st = _random_opening()
    obs = _obs_for(st, Seat.SOUTH)

    left = obs.global_slice("remaining_left_side")
    right = obs.global_slice("remaining_right_side")
    assert left.shape == (NUM_TRACKED_TYPES,)
    assert right.shape == (NUM_TRACKED_TYPES,)

    expected = np.array(
        [float(PIECE_COUNTS[t]) for t in TRACKED_TYPES], dtype=np.float32
    )
    assert np.array_equal(left, expected)
    assert np.array_equal(right, expected)


def test_global_flag_revealed_scalars_zero_at_opening() -> None:
    st = _random_opening()
    obs = _obs_for(st, Seat.SOUTH)
    fr = obs.global_slice("flag_revealed")
    assert fr.shape == (4,)
    assert (fr == 0.0).all()


# ===========================================================================
# F. ShowMode coverage
# ===========================================================================


def test_prob_teammate_is_onehot_under_half_dark() -> None:
    """HALF_DARK: every teammate cell has pure one-hot distribution.

    Bit-exact equivalence contract between scheme D's unified Prob-teammate
    and a hard-coded Piece-teammate channel.
    """
    st = _random_opening(show_mode=ShowMode.HALF_DARK)
    observer = Seat.SOUTH
    belief = BeliefTensor.initial(st, observer)

    teammate = observer.teammate
    for pos, piece_ref in st.pieces.items():
        if piece_ref.seat is not teammate:
            continue
        vec = belief.get(pos)
        assert vec.sum() == pytest.approx(1.0)
        assert vec.max() == 1.0, f"not one-hot at {pos}"
        assert int((vec > 0).sum()) == 1


def test_dark_teammate_channel_activates_only_under_dark() -> None:
    for mode in (ShowMode.BRIGHT, ShowMode.HALF_DARK):
        st = _random_opening(show_mode=mode)
        obs = _obs_for(st, Seat.SOUTH)
        assert float(obs.channel("dark_teammate").sum()) == 0.0, (
            f"dark_teammate non-zero under {mode.name}"
        )

    st_dark = _random_opening(show_mode=ShowMode.DARK)
    obs_dark = _obs_for(st_dark, Seat.SOUTH)
    assert int(obs_dark.channel("dark_teammate").sum()) == 25


def test_prob_teammate_becomes_distribution_under_dark() -> None:
    """DARK: at least one teammate cell has non-degenerate belief (entropy > 0)."""
    st = _random_opening(show_mode=ShowMode.DARK)
    observer = Seat.SOUTH
    belief = BeliefTensor.initial(st, observer)

    teammate = observer.teammate
    any_nondegenerate = False
    for pos, piece_ref in st.pieces.items():
        if piece_ref.seat is not teammate:
            continue
        vec = belief.get(pos)
        assert vec.sum() == pytest.approx(1.0)
        if int((vec > 0).sum()) > 1:
            any_nondegenerate = True
    assert any_nondegenerate, "DARK: expected at least one non-degenerate teammate belief"


# ===========================================================================
# Misc: debug utility
# ===========================================================================


def test_channel_name_covers_all_channels() -> None:
    for i in range(OBS_CHANNELS):
        name = channel_name(i)
        assert isinstance(name, str) and name
    with pytest.raises(IndexError):
        channel_name(OBS_CHANNELS)
    with pytest.raises(IndexError):
        channel_name(-1)
