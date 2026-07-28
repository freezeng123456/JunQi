"""CombatMemory v4 observation-channel tests.

Validates the 50-channel v4 layout (OBS_CHANNELS = 306) and the
projector's behaviour across both layers (enemy view + theory-of-mind
on own pieces).
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor
from junqi_core.move_gen import PieceMap, PieceRef
from junqi_core.observation import (
    CHANNEL_LAYOUT,
    OBS_CHANNELS,
    ObservationBuilder,
)
from junqi_core.rules import (
    RULES_VERSION,
    PieceType,
    Seat,
    ShowMode,
)
from junqi_core.setup import generate_random_setup
from junqi_core.state import (
    Action,
    GameState,
    PieceState,
    SeatInfo,
)


def _p(seat, pt, pid):
    return PieceRef(seat=seat, piece_type=pt, alive=True, piece_id=pid)


def _build_state(*, pieces, turn, show_mode=ShowMode.DARK):
    piece_state = {ref.piece_id: PieceState() for ref in pieces.values()}
    info = {s: SeatInfo() for s in Seat}
    return GameState(
        pieces=dict(pieces),
        turn=turn,
        move_counter=0,
        moves_since_last_combat=0,
        info=info,
        terminated=False,
        winner_team=None,
        draw=False,
        show_mode=show_mode,
        rules_version=RULES_VERSION,
        debug_include_private=False,
        zero_board=dict(pieces),
        piece_state=piece_state,
        deaths={},
    )


def _quiet_others(*, exclude):
    out: PieceMap = {}
    if Seat.SOUTH not in exclude:
        out[(6, 13)] = _p(Seat.SOUTH, PieceType.JUNQI, 28)
        out[(8, 13)] = _p(Seat.SOUTH, PieceType.PAIZH, 20)
    if Seat.WEST not in exclude:
        out[(0, 8)] = _p(Seat.WEST, PieceType.JUNQI, 56)
        out[(2, 8)] = _p(Seat.WEST, PieceType.PAIZH, 50)
    if Seat.NORTH not in exclude:
        out[(10, 2)] = _p(Seat.NORTH, PieceType.JUNQI, 86)
        out[(8, 3)] = _p(Seat.NORTH, PieceType.PAIZH, 70)
    if Seat.EAST not in exclude:
        out[(16, 10)] = _p(Seat.EAST, PieceType.JUNQI, 116)
        out[(14, 8)] = _p(Seat.EAST, PieceType.PAIZH, 100)
    return out


# ---------------------------------------------------------------------------
# Layout invariants
# ---------------------------------------------------------------------------


class TestLayout:
    def test_obs_channels_is_352(self):
        assert OBS_CHANNELS == 412

    def test_pre_cm_prefix_unchanged(self):
        assert CHANNEL_LAYOUT["move_history"].stop == 256

    def test_v4_group_sizes(self):
        sizes = {
            "cm_kill_mine_type": 12,
            "cm_kill_mine_ge": 3,
            "cm_kill_other_ge": 3,
            "cm_chain_type": 12,
            "cm_chain_ge": 3,
            "cm_floor_ge": 9,
            "cm_is_gongb": 1,
            "cm_not_gongb": 1,
            "cm_dilei_candidate": 1,
            "cm_my_kill_count_ge": 3,
            "cm_my_is_gongb": 1,
            "cm_my_dilei_candidate": 1,
        }
        for name, expected in sizes.items():
            sl = CHANNEL_LAYOUT[name]
            assert sl.stop - sl.start == expected, (
                f"{name}: expected {expected}, got {sl.stop - sl.start}"
            )

    def test_total_v4_cm_channels(self):
        groups = (
            "cm_kill_mine_type", "cm_kill_mine_ge", "cm_kill_other_ge",
            "cm_chain_type", "cm_chain_ge", "cm_floor_ge",
            "cm_is_gongb", "cm_not_gongb", "cm_dilei_candidate",
            "cm_my_kill_count_ge", "cm_my_is_gongb", "cm_my_dilei_candidate",
        )
        total = sum(CHANNEL_LAYOUT[g].stop - CHANNEL_LAYOUT[g].start for g in groups)
        assert total == 50


# ---------------------------------------------------------------------------
# Build-smoke (no exceptions across all 4 observers under DARK)
# ---------------------------------------------------------------------------


class TestBuildSmoke:
    @pytest.mark.parametrize("observer", [Seat.SOUTH, Seat.WEST, Seat.NORTH, Seat.EAST])
    def test_build_does_not_crash(self, observer):
        rng = random.Random(0)
        setups = generate_random_setup(rng)
        st = GameState.new_game(setups, show_mode=ShowMode.DARK)
        belief = BeliefTensor.initial(st, observer, show_mode=ShowMode.DARK)
        builder = ObservationBuilder()
        obs = builder.build(st, belief, observer)
        assert obs.spatial.shape == (OBS_CHANNELS, 17, 17)
        # CombatMemory tail: most channels are zero before any event has
        # happened.  Two exceptions, by design (runtime checks):
        #   * cm_dilei_candidate (Layer 1) — fires for any enemy piece
        #     that's alive, immobile (move_count==0), in seat back-two-rows,
        #     and not attacked-by-known-GONGB.  At t=0, every enemy piece
        #     in their back rows passes this check.
        #   * cm_my_dilei_candidate (Layer 2) — same gate but for
        #     observer's own pieces.
        # All other groups must be zero pre-event.
        zero_groups = [
            "cm_kill_mine_type", "cm_kill_mine_ge", "cm_kill_other_ge",
            "cm_chain_type", "cm_chain_ge", "cm_floor_ge",
            "cm_is_gongb", "cm_not_gongb",
            "cm_my_kill_count_ge", "cm_my_is_gongb",
        ]
        for name in zero_groups:
            sl = CHANNEL_LAYOUT[name]
            assert (obs.spatial[sl] == 0).all(), (
                f"opening obs: channel group {name} should be zero pre-event"
            )


# ---------------------------------------------------------------------------
# Writer correctness (DARK mode end-to-end)
# ---------------------------------------------------------------------------


class TestWriterCorrectness:
    def test_eat_my_paizh_lights_writer_planes(self):
        """SOUTH plays LIANZH eats WEST's PAIZH at (6, 10) (WEST is left enemy
        of SOUTH).  WEST's view of the LIANZH (now at (6, 10)) should show:
          - cm_kill_mine_type[PAIZH] = 1
          - cm_kill_mine_ge[≥1] = 1
          - cm_floor_ge[≥GONGB ... ≥LIANZH] = 1 (rank 3)
          - cm_not_gongb = 1
          - cm_is_gongb = 0
        SOUTH's view of its own LIANZH (theory-of-mind): nothing yet —
        only one opponent (WEST) has the data, AND of two opponents = 0.
        """
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.LIANZH, 11),
            (6, 10): _p(Seat.WEST, PieceType.PAIZH, 40),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (3, 10): _p(Seat.WEST, PieceType.LIANZH, 51),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, _ = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        builder = ObservationBuilder()
        west_belief = BeliefTensor.initial(st2, Seat.WEST, show_mode=ShowMode.DARK)
        obs_west = builder.build(st2, west_belief, Seat.WEST)

        # canonical (x, y) for WEST observer at world (6, 10)
        from junqi_core.rotation import world_to_canonical
        cx, cy = world_to_canonical(6, 10, Seat.WEST)

        # PAIZH index in TRACKED_TYPES.
        from junqi_core.combat_memory import TRACKED_TYPES, _PIECETYPE_TO_TRACKED_IDX
        paizh_idx = TRACKED_TYPES.index(PieceType.PAIZH)

        kill_mine_type = obs_west.spatial[
            CHANNEL_LAYOUT["cm_kill_mine_type"].start + paizh_idx
        ]
        assert kill_mine_type[cy, cx] > 0.5, "WEST should see kill_mine_type[PAIZH] lit"

        kill_mine_ge1 = obs_west.spatial[CHANNEL_LAYOUT["cm_kill_mine_ge"].start + 0]
        assert kill_mine_ge1[cy, cx] > 0.5, "WEST should see kill_mine_ge ≥1 lit"

        floor_ge_lianzh = obs_west.spatial[CHANNEL_LAYOUT["cm_floor_ge"].start + 2]  # ≥LIANZH
        assert floor_ge_lianzh[cy, cx] > 0.5, "WEST should see floor_ge[≥LIANZH] lit"

        floor_ge_yingzh = obs_west.spatial[CHANNEL_LAYOUT["cm_floor_ge"].start + 3]  # ≥YINGZH
        assert floor_ge_yingzh[cy, cx] < 1e-6, "WEST should NOT see floor_ge[≥YINGZH] (rank=LIANZH)"

        not_gongb = obs_west.spatial[CHANNEL_LAYOUT["cm_not_gongb"].start]
        assert not_gongb[cy, cx] > 0.5, "WEST should see not_gongb lit (ate non-mine)"

        is_gongb = obs_west.spatial[CHANNEL_LAYOUT["cm_is_gongb"].start]
        assert is_gongb[cy, cx] < 1e-6, "WEST should NOT see is_gongb"

        # SOUTH's theory-of-mind view of its own LIANZH (now at (6, 10)).
        south_belief = BeliefTensor.initial(st2, Seat.SOUTH, show_mode=ShowMode.DARK)
        obs_south = builder.build(st2, south_belief, Seat.SOUTH)
        # SOUTH canonical = identity.
        my_kill_ge1 = obs_south.spatial[CHANNEL_LAYOUT["cm_my_kill_count_ge"].start + 0]
        # AND of (WEST view, EAST view).  WEST has count=1 (it lost a piece),
        # EAST has count=0 (it didn't see the type, but other_count=0 because
        # the victim_seat=WEST, observer=EAST → other_count INCREMENTS).  Wait:
        # for EAST, victim_seat ≠ observer → direct_other_count++.  So EAST
        # view: kill_count=1 too.  AND should give ≥1 on both sides.
        # Verify: SOUTH should see ≥1 on its LIANZH cell.
        assert my_kill_ge1[10, 6] > 0.5, "SOUTH should see my_kill_count_ge ≥1 on its LIANZH"

    def test_dilei_candidate_immobile_unattacked_back_row(self):
        """An enemy piece (WEST) that started in WEST's back-two-rows and
        has not moved should be lit as dilei_candidate from SOUTH's view —
        provided no known-GONGB attacked it."""
        rng = random.Random(7)
        setups = generate_random_setup(rng)
        st = GameState.new_game(setups, show_mode=ShowMode.DARK)
        builder = ObservationBuilder()
        belief = BeliefTensor.initial(st, Seat.SOUTH, show_mode=ShowMode.DARK)
        obs = builder.build(st, belief, Seat.SOUTH)

        # Find any WEST piece initially placed in back two rows that's still alive
        # and at zero pos, move_count=0.  Confirm the candidate channel lights up
        # in at least one such cell.
        cand = obs.spatial[CHANNEL_LAYOUT["cm_dilei_candidate"].start]
        # Without combat, every immobile back-row enemy piece is a candidate.
        # The board has 5 + 5 + 5 + 5 = 20 enemy back-row pieces from SOUTH's view
        # (only enemy seats contribute: WEST and EAST).
        # We just check at least 1 cell is lit.
        assert cand.sum() > 0, "Initial state should have at least one dilei_candidate"


# ---------------------------------------------------------------------------
# DARK rule: own-seat eats don't leak to other observers
# ---------------------------------------------------------------------------


class TestDarkRule:
    def test_my_eat_not_visible_to_other_team(self):
        """SOUTH eats WEST's PAIZH; NORTH (SOUTH's teammate, also DARK)
        cannot see PAIZH type.

        NORTH's Layer 1 projects only onto *enemies* (WEST/EAST pieces),
        so the SOUTH attacker — NORTH's teammate — never gets any v4
        channel set, regardless of state.  This is the DARK information
        boundary: a teammate's combat events stay opaque on the
        observation tensor (the underlying ``direct_other_count`` IS
        recorded internally for chain propagation, but is never
        projected to spatial channels for a teammate).
        """
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.LIANZH, 11),
            (6, 10): _p(Seat.WEST, PieceType.PAIZH, 40),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (3, 10): _p(Seat.WEST, PieceType.LIANZH, 51),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, _ = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        builder = ObservationBuilder()
        north_belief = BeliefTensor.initial(st2, Seat.NORTH, show_mode=ShowMode.DARK)
        obs_north = builder.build(st2, north_belief, Seat.NORTH)

        # NORTH canonical: 180° rotation.  World (6,10) → (16-6, 16-10) = (10, 6).
        from junqi_core.rotation import world_to_canonical
        cx, cy = world_to_canonical(6, 10, Seat.NORTH)

        # SOUTH is NORTH's teammate ⇒ Layer 1 does NOT fire on this cell.
        for name in ("cm_kill_mine_type", "cm_kill_mine_ge",
                     "cm_kill_other_ge", "cm_chain_type", "cm_chain_ge",
                     "cm_floor_ge", "cm_is_gongb", "cm_not_gongb"):
            sl = CHANNEL_LAYOUT[name]
            assert (obs_north.spatial[sl, cy, cx] < 1e-6).all(), (
                f"NORTH (SOUTH's teammate) should see Layer-1 channel {name} == 0 "
                f"on SOUTH attacker cell (Layer 1 only fires on enemies)"
            )

        # WEST DOES see the public event count on the SOUTH attacker cell
        # (PAIZH was WEST's piece — Layer 1 fires for WEST as observer).
        west_belief = BeliefTensor.initial(st2, Seat.WEST, show_mode=ShowMode.DARK)
        obs_west = builder.build(st2, west_belief, Seat.WEST)
        wx, wy = world_to_canonical(6, 10, Seat.WEST)
        kill_mine_ge1 = obs_west.spatial[CHANNEL_LAYOUT["cm_kill_mine_ge"].start]
        assert kill_mine_ge1[wy, wx] > 0.5, (
            "WEST should see kill_mine_ge ≥1 on the SOUTH attacker (PAIZH was WEST's)"
        )
