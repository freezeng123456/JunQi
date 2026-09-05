"""CombatMemory v6 (layer-4) reverse-projection channel tests.

Validates the 60 channels of the final cm_eaten_by_pid group:
  cm_eaten_by_pid[60]   for each of observer's 30 own pids, project onto
                        the cell (alive ⇒ current; dead ⇒ zero_pos) the
                        bitmap of which 60 enemy pids have directly OR
                        chain-eaten that mpid.  Channels 0..29 = left
                        enemy ((obs+1)%4); 30..59 = right enemy
                        ((obs+3)%4).

Information-boundary tests: the layer-4 channels MUST NOT leak any
non-DARK fact.  cm.eaten_by_pid is only ever populated for
observer-own mpids — projection across all 4 observers must show the
event ONLY in the observer whose own mpid was eaten.
"""

from __future__ import annotations

import numpy as np

from junqi_core.combat_memory import (
    CombatMemoryState,
    apply_combat_event,
    _OBS_PID_MASK_LO,
    _OBS_PID_MASK_HI,
)
from junqi_core.info_model import BeliefTensor
from junqi_core.move_gen import PieceMap, PieceRef
from junqi_core.observation import (
    CHANNEL_LAYOUT,
    OBS_CHANNELS,
    ObservationBuilder,
)
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import (
    RULES_VERSION,
    PieceType,
    Seat,
    ShowMode,
)
from junqi_core.state import (
    GameState,
    PieceState,
    SeatInfo,
)


def _p(seat, pt, pid):
    return PieceRef(seat=seat, piece_type=pt, alive=True, piece_id=pid)


def _build_state(*, pieces, turn, show_mode=ShowMode.DARK, move_counter=0):
    piece_state = {ref.piece_id: PieceState() for ref in pieces.values()}
    info = {s: SeatInfo() for s in Seat}
    return GameState(
        pieces=dict(pieces),
        turn=turn,
        move_counter=move_counter,
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


# ---------------------------------------------------------------------------
# Layout invariants
# ---------------------------------------------------------------------------


class TestLayerFourLayout:
    def test_layer4_is_the_final_group(self):
        assert CHANNEL_LAYOUT["cm_eaten_by_pid"].stop == OBS_CHANNELS

    def test_layer4_offsets_and_sizes(self):
        # Asserted relative to the neighbouring groups, not as absolute
        # indices: the layout shifted once already (piece_id's 120 planes
        # compacted to piece_slot's 25) and magic offsets meant editing
        # four test files to restate the same invariant. What matters is
        # that the group is the documented size and sits where the layout
        # order says it does; observation.py self-checks contiguity.
        sl = CHANNEL_LAYOUT["cm_eaten_by_pid"]
        assert sl.start == CHANNEL_LAYOUT["cm_recency"].stop
        assert sl.stop == OBS_CHANNELS
        assert sl.stop - sl.start == 60


# ---------------------------------------------------------------------------
# DARK-mask invariants on the SoA itself
# ---------------------------------------------------------------------------


class TestObsPidMasks:
    def test_obs_pid_mask_covers_30_pids(self):
        # Each observer's mask must cover exactly 30 bits (their own pids).
        for obs in range(4):
            popcount = (
                int(_OBS_PID_MASK_LO[obs]).bit_count()
                + int(_OBS_PID_MASK_HI[obs]).bit_count()
            )
            assert popcount == 30, f"obs={obs} mask has {popcount} bits set, want 30"

    def test_obs_pid_masks_disjoint(self):
        # No two observers share a pid bit.
        all_lo = np.uint64(0)
        all_hi = np.uint64(0)
        for obs in range(4):
            assert (all_lo & _OBS_PID_MASK_LO[obs]) == 0
            assert (all_hi & _OBS_PID_MASK_HI[obs]) == 0
            all_lo |= _OBS_PID_MASK_LO[obs]
            all_hi |= _OBS_PID_MASK_HI[obs]
        # All 4 together should cover [0, 120).  Check bit count = 120.
        assert int(all_lo).bit_count() + int(all_hi).bit_count() == 120


# ---------------------------------------------------------------------------
# Direct-EAT semantics: K (EAST pid 110) ate V (SOUTH pid 20)
# ---------------------------------------------------------------------------


def _hand_state_after_eat():
    """SOUTH pid=20 (PAIZH) eaten by EAST pid=110 (LIANZH) at (10,16)."""
    pieces: PieceMap = {}
    pieces[(8, 16)] = _p(Seat.SOUTH, PieceType.JUNQI, 28)
    pieces[(10, 16)] = _p(Seat.SOUTH, PieceType.PAIZH, 20)
    pieces[(7, 16)] = _p(Seat.SOUTH, PieceType.PAIZH, 21)
    pieces[(6, 16)] = _p(Seat.SOUTH, PieceType.LIANZH, 22)
    pieces[(9, 13)] = _p(Seat.SOUTH, PieceType.SILING, 0)
    pieces[(0, 8)] = _p(Seat.WEST, PieceType.JUNQI, 56)
    pieces[(8, 0)] = _p(Seat.NORTH, PieceType.JUNQI, 86)
    pieces[(16, 10)] = _p(Seat.EAST, PieceType.JUNQI, 116)
    pieces[(13, 8)] = _p(Seat.EAST, PieceType.LIANZH, 110)
    state = _build_state(pieces=pieces, turn=Seat.EAST, move_counter=10)
    apply_combat_event(
        state.combat_memory,
        event_is_eat=True,
        attacker_pid=110, defender_pid=20,
        attacker_seat=Seat.EAST.value, defender_seat=Seat.SOUTH.value,
        attacker_type=PieceType.LIANZH, defender_type=PieceType.PAIZH,
        defender_pos_flat=10 * 17 + 16, death_step=8,
    )
    return state


def test_eaten_by_pid_writes_killer_bit_for_observer_own_pid():
    """SoA: SOUTH's eaten_by_pid[20] must contain bit 110."""
    state = _hand_state_after_eat()
    cm = state.combat_memory
    south = Seat.SOUTH.value
    # Killer pid 110 → high half (110 >= 64), bit position = 110 - 64 = 46.
    expected_bit = np.uint64(1) << np.uint64(110 - 64)
    assert (cm.eaten_by_pid_hi[south, 20] & expected_bit) == expected_bit, (
        f"SOUTH eaten_by_pid_hi[20]={int(cm.eaten_by_pid_hi[south, 20]):x}, "
        f"missing bit for pid=110"
    )
    # Lo half should be empty.
    assert cm.eaten_by_pid_lo[south, 20] == 0


def test_eaten_by_pid_dark_safe_other_observers_zero():
    """Other observers must NOT have any eaten_by_pid bit set for pid=20.

    Because pid=20 is SOUTH's pid; non-SOUTH observers have no claim on
    its eaten history (DARK boundary).
    """
    state = _hand_state_after_eat()
    cm = state.combat_memory
    for obs in (Seat.WEST.value, Seat.NORTH.value, Seat.EAST.value):
        assert cm.eaten_by_pid_lo[obs, 20] == 0
        assert cm.eaten_by_pid_hi[obs, 20] == 0


def test_eaten_by_pid_channel_lights_at_victim_cell_for_south():
    """SOUTH observer should see the EAST-killer bit lit at pid=20's cell."""
    state = _hand_state_after_eat()
    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    obs = builder.build(state, belief, Seat.SOUTH)

    cm_eaten = obs.channel("cm_eaten_by_pid")  # (60, 17, 17)
    # SOUTH: left_opp = WEST = (0+1)%4 = 1; right_opp = EAST = (0+3)%4 = 3.
    # EAST pid=110 → EAST's slot 110-90 = 20; channel = 30 + 20 = 50.
    # mpid=20 is alive in the test state (we mutated only cm), so the
    # projection cell is pos_x=10, pos_y=16.  SOUTH's pos (10,16) →
    # canonical via world_to_canonical.
    cx, cy = world_to_canonical(10, 16, Seat.SOUTH)
    assert cm_eaten[50, cy, cx] == 1.0, (
        f"channel 50 (right enemy slot 20 = EAST pid 110) should be 1 at "
        f"victim's cell ({cx},{cy}); got {cm_eaten[50, cy, cx]}"
    )
    # No other channel should be lit at this cell.
    other_sum = float(cm_eaten[:, cy, cx].sum()) - float(cm_eaten[50, cy, cx])
    assert other_sum == 0.0, f"unexpected leakage at victim cell: sum={other_sum}"


def test_eaten_by_pid_no_signal_at_innocent_pid_cells():
    """Other SOUTH pids must show ZERO on all 60 enemy channels."""
    state = _hand_state_after_eat()
    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    obs = builder.build(state, belief, Seat.SOUTH)

    cm_eaten = obs.channel("cm_eaten_by_pid")
    # SOUTH pid 21 (innocent PAIZH) at (7, 16)
    cx, cy = world_to_canonical(7, 16, Seat.SOUTH)
    assert float(cm_eaten[:, cy, cx].sum()) == 0.0
    # SOUTH pid 22 (LIANZH) at (6, 16)
    cx, cy = world_to_canonical(6, 16, Seat.SOUTH)
    assert float(cm_eaten[:, cy, cx].sum()) == 0.0
    # SOUTH pid 0 (SILING) at (9, 13)
    cx, cy = world_to_canonical(9, 13, Seat.SOUTH)
    assert float(cm_eaten[:, cy, cx].sum()) == 0.0


def test_eaten_by_pid_observer_perspective_invariant():
    """Each observer's view should ONLY show their own pids' eaten history.

    For the EAT event SOUTH pid=20 ← EAST pid=110:
      * SOUTH's view: cm_eaten_by_pid[50, victim_cell] == 1 (channel 50 = right enemy slot 20)
      * WEST's view: zero everywhere (pid=20 not WEST's, no signal)
      * NORTH's view: zero everywhere (pid=20 not NORTH's; teammate of SOUTH, but DARK hides this)
      * EAST's view: zero everywhere (pid=20 not EAST's)
    """
    state = _hand_state_after_eat()
    builder = ObservationBuilder()
    for observer in (Seat.WEST, Seat.NORTH, Seat.EAST):
        belief = BeliefTensor.initial(state, observer)
        obs = builder.build(state, belief, observer)
        cm_eaten = obs.channel("cm_eaten_by_pid")
        total = float(cm_eaten.sum())
        assert total == 0.0, (
            f"observer={observer.name}: cm_eaten_by_pid total={total}, "
            f"expected 0 (DARK boundary violation — leak across observers!)"
        )


# ---------------------------------------------------------------------------
# Chain reverse projection: K' kills K who ate V → K' should also be
# recorded as having "chain-eaten" V (via K).
# ---------------------------------------------------------------------------


def test_eaten_by_pid_chain_propagates_to_new_killer():
    """Setup:
      Step 1: EAST pid=110 EATs SOUTH pid=20 (PAIZH).
      Step 2: SOUTH pid=22 (LIANZH) gets KILLED by EAST pid=111 (TUANZH)
              — wait, let's flip: EAST pid=111 TUANZH attacks SOUTH pid=22
              LIANZH and gets KILLED (TUANZH > LIANZH? In Junqi, LARGER
              rank = STRONGER... Actually TUANZH > LIANZH so TUANZH wins).
      Let's use: SOUTH pid=22 LIANZH attacks EAST pid=110 LIANZH.  Mutual.
      Better: Step 2: SOUTH pid=0 SILING EATs EAST pid=110.
              Now via chain, eaten_by_pid for SOUTH's pid=20 should
              contain pid=0 (the new chain-bridger to V=20).
    """
    cm = CombatMemoryState.zeros()
    south = Seat.SOUTH.value

    # Step 1: EAST pid=110 EATs SOUTH pid=20.
    apply_combat_event(
        cm,
        event_is_eat=True,
        attacker_pid=110, defender_pid=20,
        attacker_seat=Seat.EAST.value, defender_seat=Seat.SOUTH.value,
        attacker_type=PieceType.LIANZH, defender_type=PieceType.PAIZH,
        defender_pos_flat=10 * 17 + 16, death_step=5,
    )
    # Check direct write.
    bit_110 = np.uint64(1) << np.uint64(110 - 64)
    assert (cm.eaten_by_pid_hi[south, 20] & bit_110) == bit_110

    # Step 2: SOUTH pid=0 SILING attacks EAST pid=110 → defender (110) dies.
    # Event.EAT with attacker=0 (SOUTH SILING), defender=110 (EAST LIANZH).
    apply_combat_event(
        cm,
        event_is_eat=True,
        attacker_pid=0, defender_pid=110,
        attacker_seat=Seat.SOUTH.value, defender_seat=Seat.EAST.value,
        attacker_type=PieceType.SILING, defender_type=PieceType.LIANZH,
        defender_pos_flat=12 * 17 + 8, death_step=12,
    )
    # Now SOUTH pid=20's eaten_by_pid should contain BOTH pid=110
    # (direct) AND pid=0 (chain via 110).
    bit_0 = np.uint64(1) << np.uint64(0)
    assert (cm.eaten_by_pid_lo[south, 20] & bit_0) == bit_0, (
        f"chain-bridger pid=0 should be recorded in SOUTH's eaten_by_pid[20]; "
        f"got lo={int(cm.eaten_by_pid_lo[south, 20]):x}"
    )
    # Direct bit (110) still present.
    assert (cm.eaten_by_pid_hi[south, 20] & bit_110) == bit_110

    # Other observers must remain zero for SOUTH's pid=20.
    for obs in (Seat.WEST.value, Seat.NORTH.value, Seat.EAST.value):
        assert cm.eaten_by_pid_lo[obs, 20] == 0
        assert cm.eaten_by_pid_hi[obs, 20] == 0


# ---------------------------------------------------------------------------
# DARK regression: a non-observer-own pid never gets eaten_by_pid signal.
# ---------------------------------------------------------------------------


def test_dark_boundary_no_eaten_by_pid_for_non_observer_own():
    """A WEST pid being eaten by EAST should write WEST's eaten_by_pid
    only — SOUTH/NORTH/EAST cm.eaten_by_pid for that WEST pid stays zero.
    """
    cm = CombatMemoryState.zeros()
    # WEST pid=40 eaten by EAST pid=110.
    apply_combat_event(
        cm,
        event_is_eat=True,
        attacker_pid=110, defender_pid=40,
        attacker_seat=Seat.EAST.value, defender_seat=Seat.WEST.value,
        attacker_type=PieceType.LIANZH, defender_type=PieceType.PAIZH,
        defender_pos_flat=4 * 17 + 0, death_step=5,
    )
    bit_110 = np.uint64(1) << np.uint64(110 - 64)
    # WEST's eaten_by_pid[40] should have bit 110.
    assert (cm.eaten_by_pid_hi[Seat.WEST.value, 40] & bit_110) == bit_110
    # Other observers' eaten_by_pid[40] must be empty.
    for obs in (Seat.SOUTH.value, Seat.NORTH.value, Seat.EAST.value):
        assert cm.eaten_by_pid_lo[obs, 40] == 0
        assert cm.eaten_by_pid_hi[obs, 40] == 0
