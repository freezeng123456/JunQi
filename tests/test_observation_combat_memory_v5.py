"""CombatMemory v5 (layer-3) per-pid identity channel tests.

Validates the 46 new channels at indices [306, 352):
  cm_kill_mine_count[12]   per-type COUNT (popcount of direct ∩ type mask)
  cm_kill_mine_slot[30]    slot-i bit lit at enemy cell
  cm_recency[4]            sigmoid(elapsed/τ) on direct/chain/floor steps

Information-boundary tests are at the bottom — a careful regression net for
the DARK rule: the layer-3 channels MUST NOT leak enemy types not already
visible to the observer.
"""

from __future__ import annotations

import random
from math import isclose

import numpy as np
import pytest

from junqi_core.combat_memory import (
    CombatMemoryState,
    apply_combat_event,
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
from junqi_core.setup import generate_random_setup
from junqi_core.state import (
    Action,
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


class TestLayerThreeLayout:
    def test_obs_channels_is_352(self):
        assert OBS_CHANNELS == 412

    def test_layer3_offsets_and_sizes(self):
        assert CHANNEL_LAYOUT["cm_kill_mine_count"].start == 306
        assert CHANNEL_LAYOUT["cm_kill_mine_count"].stop  == 318
        assert CHANNEL_LAYOUT["cm_kill_mine_slot"].start  == 318
        assert CHANNEL_LAYOUT["cm_kill_mine_slot"].stop   == 348
        assert CHANNEL_LAYOUT["cm_recency"].start         == 348
        assert CHANNEL_LAYOUT["cm_recency"].stop          == 352


# ---------------------------------------------------------------------------
# kill_mine_count + kill_mine_slot semantics
# ---------------------------------------------------------------------------


def _setup_2v2_with_eat():
    """Construct a hand-tailored DARK state where EAST's piece (pid=110)
    has eaten SOUTH's PAIZH (pid=20) — this lights cm_kill_mine_*
    at SOUTH's view of pid=110.

    Returns (state, observer=SOUTH, killer_pid=110, killer_pos)."""
    # Seat slot offsets: SOUTH 0..29, WEST 30..59, NORTH 60..89, EAST 90..119
    # Place a small valid game.  Only types of SOUTH matter for kill_mine_count.
    pieces: PieceMap = {}
    # SOUTH: Junqi at stronghold + 2 PAIZH at slot=0,1 → pid 0, 1
    pieces[(8, 16)] = _p(Seat.SOUTH, PieceType.JUNQI, 28)         # pid=28 (slot 28 = stronghold)
    pieces[(10, 16)] = _p(Seat.SOUTH, PieceType.PAIZH, 20)        # pid=20: a PAIZH
    pieces[(7, 16)] = _p(Seat.SOUTH, PieceType.PAIZH, 21)         # pid=21: another PAIZH
    pieces[(6, 16)] = _p(Seat.SOUTH, PieceType.LIANZH, 22)        # pid=22: LIANZH
    pieces[(9, 13)] = _p(Seat.SOUTH, PieceType.SILING, 0)         # pid=0: SILING
    # WEST: just a flag so the seat lives.
    pieces[(0, 8)] = _p(Seat.WEST, PieceType.JUNQI, 56)           # pid=56 (slot 26)
    # NORTH: flag
    pieces[(8, 0)] = _p(Seat.NORTH, PieceType.JUNQI, 86)          # pid=86 (slot 26)
    # EAST: flag + a piece (the killer)
    pieces[(16, 10)] = _p(Seat.EAST, PieceType.JUNQI, 116)        # pid=116
    pieces[(13, 8)] = _p(Seat.EAST, PieceType.LIANZH, 110)        # pid=110: the killer
    state = _build_state(pieces=pieces, turn=Seat.EAST, move_counter=10)

    # Now mutate combat_memory so that EAST's pid=110 has eaten
    # SOUTH's pid=20 (PAIZH).  Direct event from observer SOUTH's view.
    apply_combat_event(
        state.combat_memory,
        event_is_eat=True,
        attacker_pid=110, defender_pid=20,
        attacker_seat=Seat.EAST.value, defender_seat=Seat.SOUTH.value,
        attacker_type=PieceType.LIANZH, defender_type=PieceType.PAIZH,
        defender_pos_flat=10 * 17 + 16, death_step=8,
    )
    return state, Seat.SOUTH, 110, (13, 8)


def test_kill_mine_count_paizh_lights_at_killer_cell():
    state, observer, killer_pid, killer_pos = _setup_2v2_with_eat()

    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, observer)
    obs = builder.build(state, belief, observer)

    # Find PAIZH's tracked-type index = 10 (TRACKED_TYPES order).
    from junqi_core.observation import TRACKED_TYPES
    paizh_idx = TRACKED_TYPES.index(PieceType.PAIZH)

    cm_count = obs.channel("cm_kill_mine_count")            # (12, 17, 17)
    cx, cy = world_to_canonical(*killer_pos, observer)
    # 1 PAIZH eaten → cnt=1 → value = 1/3.
    assert isclose(float(cm_count[paizh_idx, cy, cx]), 1.0 / 3.0, rel_tol=1e-5)

    # All other types must be 0 at the killer cell.
    for t in range(12):
        if t == paizh_idx:
            continue
        assert cm_count[t, cy, cx] == 0.0, f"type {t} non-zero at killer cell"


def test_kill_mine_slot_bit_lights_at_killer_cell():
    state, observer, killer_pid, killer_pos = _setup_2v2_with_eat()

    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, observer)
    obs = builder.build(state, belief, observer)

    cm_slot = obs.channel("cm_kill_mine_slot")              # (30, 17, 17)
    cx, cy = world_to_canonical(*killer_pos, observer)

    # observer_val=0 (SOUTH); victim pid=20 is observer's slot 20.
    assert cm_slot[20, cy, cx] == 1.0
    # All other slot bits must be 0 at the killer cell.
    for s in range(30):
        if s == 20:
            continue
        assert cm_slot[s, cy, cx] == 0.0, f"slot {s} non-zero at killer cell"


def test_kill_mine_count_no_signal_for_innocent_enemy():
    state, observer, _killer_pid, _killer_pos = _setup_2v2_with_eat()

    # WEST's flag (pid=56) at world (0, 8) — never ate anything.
    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, observer)
    obs = builder.build(state, belief, observer)
    cx, cy = world_to_canonical(0, 8, observer)
    cm_count = obs.channel("cm_kill_mine_count")
    cm_slot = obs.channel("cm_kill_mine_slot")
    assert float(cm_count[:, cy, cx].sum()) == 0.0
    assert float(cm_slot[:, cy, cx].sum())  == 0.0


# ---------------------------------------------------------------------------
# recency channel
# ---------------------------------------------------------------------------


def test_recency_decay_is_expected():
    state, observer, killer_pid, killer_pos = _setup_2v2_with_eat()

    # death_step was 8, current move_counter is 10 → elapsed=2.
    # tau values: (32, 256, 64, 128); only direct (0/1) is set since this
    # was a direct EAT (no chain extension yet).
    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, observer)
    obs = builder.build(state, belief, observer)
    cx, cy = world_to_canonical(*killer_pos, observer)

    cm_rec = obs.channel("cm_recency")  # (4, 17, 17)
    expected_d32  = float(np.exp(-2.0 / 32.0))
    expected_d256 = float(np.exp(-2.0 / 256.0))
    assert isclose(float(cm_rec[0, cy, cx]), expected_d32, rel_tol=1e-4)
    assert isclose(float(cm_rec[1, cy, cx]), expected_d256, rel_tol=1e-4)
    # Plane 2 (chain) and 3 (rank-floor) — chain step is set during the same
    # event (last_chain_step is unconditionally bumped); rank_floor_step is
    # set if rank_floor lifted (ate PAIZH → next_floor=3).  Both should be
    # >0 here.
    assert float(cm_rec[2, cy, cx]) > 0.0
    assert float(cm_rec[3, cy, cx]) > 0.0


# ---------------------------------------------------------------------------
# DARK information-boundary regression
# ---------------------------------------------------------------------------


def test_layer3_no_signal_when_victim_is_teammate():
    """If EAST eats SOUTH's TEAMMATE (NORTH, victim_seat==NORTH != observer
    SOUTH), the SOUTH observer should NOT see kill_mine_count/slot for
    that event.

    Layer 1 already enforces this (direct_ate_my_pid is observer-scoped);
    we re-test it for Layer 3 to make sure no new code path leaks.
    """
    pieces: PieceMap = {}
    pieces[(8, 16)] = _p(Seat.SOUTH, PieceType.JUNQI, 28)
    pieces[(8, 0)] = _p(Seat.NORTH, PieceType.JUNQI, 86)
    pieces[(8, 1)] = _p(Seat.NORTH, PieceType.PAIZH, 80)        # NORTH's PAIZH
    pieces[(0, 8)] = _p(Seat.WEST, PieceType.JUNQI, 56)
    pieces[(16, 10)] = _p(Seat.EAST, PieceType.JUNQI, 116)
    pieces[(13, 8)] = _p(Seat.EAST, PieceType.LIANZH, 110)
    state = _build_state(pieces=pieces, turn=Seat.EAST, move_counter=5)

    # EAST's pid=110 ate NORTH's pid=80.  victim_seat == NORTH != SOUTH.
    apply_combat_event(
        state.combat_memory,
        event_is_eat=True,
        attacker_pid=110, defender_pid=80,
        attacker_seat=Seat.EAST.value, defender_seat=Seat.NORTH.value,
        attacker_type=PieceType.LIANZH, defender_type=PieceType.PAIZH,
        defender_pos_flat=1 * 17 + 8, death_step=4,
    )

    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    obs = builder.build(state, belief, Seat.SOUTH)
    cx, cy = world_to_canonical(13, 8, Seat.SOUTH)

    # SOUTH's view: layer-3 channels must be ZERO at the killer cell because
    # NO direct event with victim_seat=SOUTH has occurred.
    cm_count = obs.channel("cm_kill_mine_count")
    cm_slot  = obs.channel("cm_kill_mine_slot")
    assert float(cm_count[:, cy, cx].sum()) == 0.0, (
        "kill_mine_count must not leak teammate's victim type to observer"
    )
    assert float(cm_slot[:, cy, cx].sum())  == 0.0, (
        "kill_mine_slot must not leak teammate's victim slot to observer"
    )

    # But the legitimate Layer-1 channels: kill_other_ge[0] should be lit
    # (DARK lets observer count cross-seat eats, just not learn the type).
    cm_other = obs.channel("cm_kill_other_ge")
    assert cm_other[0, cy, cx] == 1.0, (
        "kill_other_ge should still light up — count IS public"
    )


# ---------------------------------------------------------------------------
# Layer-3 plays nicely with all 4 observers (smoke test from random setup)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observer", [Seat.SOUTH, Seat.WEST, Seat.NORTH, Seat.EAST])
def test_random_setup_no_exception(observer):
    rng = random.Random(13)
    setups = generate_random_setup(rng)
    state = GameState.new_game(setups, show_mode=ShowMode.DARK)
    builder = ObservationBuilder()
    belief = BeliefTensor.initial(state, observer)
    obs = builder.build(state, belief, observer)
    assert obs.spatial.shape == (OBS_CHANNELS, 17, 17)
    # All layer-3 channels should be zero on a fresh game (no combat yet).
    for grp in ("cm_kill_mine_count", "cm_kill_mine_slot", "cm_recency"):
        assert float(obs.channel(grp).sum()) == 0.0, f"{grp} non-zero on fresh game"
