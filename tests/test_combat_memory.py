"""CombatMemory v4 unit tests (CPU reference, DARK mode).

Covers the design points in docs/COMBAT_MEMORY_DESIGN.md (v4):

  1. EAT victim=DILEI ⇒ killer is GONGB (A1)
  2. EAT victim=non-DILEI ⇒ killer is NOT GONGB (B1)
  3. KILLED with attacker visible ⇒ floor lift on defender + dilei_candidate hint
  4. KILLED with non-GONGB attacker observer-visible: defender survives → may be
     ordinary high rank or DILEI (handled by floor + dilei_candidate)
  5. Chain propagation ALL observers (operates on public piece_ids only)
  6. DARK rule: when victim_seat ≠ observer, only direct_other_count++ (no type info)
  7. attacked_by_known_gongb: KILLED + attacker known GONGB → defender flagged

All tests are CPU-only.
"""

from __future__ import annotations

import numpy as np
import pytest

from junqi_core.combat_memory import (
    NUM_OBSERVERS,
    NUM_PIDS,
    NUM_TRACKED_TYPES,
    TRACKED_TYPES,
    CombatMemoryState,
    _PIECETYPE_TO_TRACKED_IDX,
    _RANK_OF_TYPE,
    apply_combat_event,
    apply_path_revealed_gongb,
    is_in_seat_back_two_rows,
    next_floor_after_eat,
)
from junqi_core.observation import _pid_bit_masks, _popcount_pid_pair
from junqi_core.rules import PieceType, Seat


def _bit(t: PieceType) -> int:
    return 1 << int(_PIECETYPE_TO_TRACKED_IDX[t.value])


# ===========================================================================
# 1. EAT + DILEI victim ⇒ killer is GONGB (A1)
# ===========================================================================


class TestRule_A1_EatDilei:
    def test_eat_my_dilei_marks_killer_as_gongb(self):
        cm = CombatMemoryState.zeros()
        # WEST 's piece (pid=37) ate SOUTH's DILEI (pid=12).
        apply_combat_event(
            cm,
            event_is_eat=True,
            attacker_pid=37, defender_pid=12,
            attacker_seat=Seat.WEST.value, defender_seat=Seat.SOUTH.value,
            attacker_type=PieceType.GONGB,
            defender_type=PieceType.DILEI,
            defender_pos_flat=15 * 17 + 8,  # SOUTH back row
            death_step=10,
        )
        south = Seat.SOUTH.value
        # SOUTH knows: killer 37 is GONGB.
        assert cm.is_gongb[south, 37]
        assert not cm.not_gongb[south, 37]
        # DILEI is a special; no floor lift.
        assert cm.rank_floor[south, 37] == 0
        # Other observers: only count, no type info.
        for other in (Seat.WEST.value, Seat.NORTH.value, Seat.EAST.value):
            assert not cm.is_gongb[other, 37]
            assert cm.direct_other_count[other, 37] == 1


# ===========================================================================
# 2. EAT non-DILEI ⇒ killer is NOT GONGB (B1)
# ===========================================================================


class TestRule_B1_NotGongbOnEat:
    def test_eat_my_paizh_excludes_gongb(self):
        cm = CombatMemoryState.zeros()
        apply_combat_event(
            cm,
            event_is_eat=True,
            attacker_pid=37, defender_pid=12,
            attacker_seat=Seat.WEST.value, defender_seat=Seat.SOUTH.value,
            attacker_type=PieceType.LIANZH,
            defender_type=PieceType.PAIZH,
            defender_pos_flat=10 * 17 + 6,
            death_step=5,
        )
        south = Seat.SOUTH.value
        # SOUTH knows: 37 is NOT GONGB and at least LIANZH+ (rank 3).
        assert cm.not_gongb[south, 37]
        assert not cm.is_gongb[south, 37]
        assert cm.rank_floor[south, 37] == 3   # LIANZH+
        # Direct memory recorded: PAIZH bit + pid bit
        assert (cm.direct_ate_my_type_mask[south, 37] & _bit(PieceType.PAIZH)) != 0
        assert cm.direct_ate_my_pid_lo[south, 37] == np.uint64(1) << np.uint64(12)


# ===========================================================================
# 3. KILLED + ordinary attacker visible ⇒ floor lift on defender
# ===========================================================================


class TestRule_KILLED_FloorLift:
    def test_my_paizh_dies_attacking_lifts_floor_to_lianzh(self):
        cm = CombatMemoryState.zeros()
        # SOUTH's PAIZH (pid=12) attacks WEST's piece (pid=37) and dies.
        # Attacker died → KILLED event. Defender 37 is in WEST's back-two-rows.
        apply_combat_event(
            cm,
            event_is_eat=False,
            attacker_pid=12, defender_pid=37,
            attacker_seat=Seat.SOUTH.value, defender_seat=Seat.WEST.value,
            attacker_type=PieceType.PAIZH,
            defender_type=PieceType.SILING,  # actual; observer doesn't know
            defender_pos_flat=8 * 17 + 0,  # WEST back row
            death_step=20,
        )
        south = Seat.SOUTH.value
        # SOUTH knows attacker (its own PAIZH) → floor lift on defender 37.
        assert cm.rank_floor[south, 37] == 3   # LIANZH+
        # Direct memory recorded for SOUTH (its piece died): PAIZH bit set.
        assert (cm.direct_ate_my_type_mask[south, 37] & _bit(PieceType.PAIZH)) != 0
        # WEST (defender's own seat) only sees direct_other_count.
        assert cm.direct_other_count[Seat.WEST.value, 37] == 1


# ===========================================================================
# 4. attacked_by_known_gongb flag for dilei_candidate downstream test
# ===========================================================================


class TestAttackedByKnownGongb:
    def test_known_gongb_attack_flags_defender(self):
        cm = CombatMemoryState.zeros()
        # 1) Reveal pid=18 as GONGB to all observers via path-reveal.
        apply_path_revealed_gongb(cm, 18)
        for obs in range(NUM_OBSERVERS):
            assert cm.is_gongb[obs, 18]
        # 2) pid=18 attacks pid=44 and dies (KILLED event).
        apply_combat_event(
            cm,
            event_is_eat=False,
            attacker_pid=18, defender_pid=44,
            attacker_seat=Seat.SOUTH.value, defender_seat=Seat.WEST.value,
            attacker_type=PieceType.GONGB,
            defender_type=PieceType.LIANZH,
            defender_pos_flat=8 * 17 + 0,
            death_step=15,
        )
        # Now 44 should have attacked_by_known_gongb=True for ALL observers
        # (path-reveal made it public, so the GONGB attack is public).
        for obs in range(NUM_OBSERVERS):
            assert cm.attacked_by_known_gongb[obs, 44], (
                f"observer {obs}: 44 should be flagged as attacked-by-known-GONGB"
            )

    def test_unknown_attacker_does_not_flag(self):
        cm = CombatMemoryState.zeros()
        # SOUTH 's PAIZH (not a GONGB, no path-reveal) attacks pid=44 and dies.
        apply_combat_event(
            cm,
            event_is_eat=False,
            attacker_pid=12, defender_pid=44,
            attacker_seat=Seat.SOUTH.value, defender_seat=Seat.WEST.value,
            attacker_type=PieceType.PAIZH,
            defender_type=PieceType.LIANZH,
            defender_pos_flat=8 * 17 + 0,
            death_step=20,
        )
        # No observer should have attacked_by_known_gongb set.
        for obs in range(NUM_OBSERVERS):
            assert not cm.attacked_by_known_gongb[obs, 44]


# ===========================================================================
# 5. Chain propagation across all observers
# ===========================================================================


class TestChainPropagation:
    def test_two_step_chain(self):
        cm = CombatMemoryState.zeros()
        # Step 1: pid=37 ate SOUTH's PAIZH (pid=12).  Direct + floor + not_gongb.
        apply_combat_event(
            cm,
            event_is_eat=True,
            attacker_pid=37, defender_pid=12,
            attacker_seat=Seat.WEST.value, defender_seat=Seat.SOUTH.value,
            attacker_type=PieceType.LIANZH,
            defender_type=PieceType.PAIZH,
            defender_pos_flat=10 * 17 + 6,
            death_step=10,
        )
        # Step 2: pid=87 ate pid=37.
        apply_combat_event(
            cm,
            event_is_eat=True,
            attacker_pid=87, defender_pid=37,
            attacker_seat=Seat.NORTH.value, defender_seat=Seat.WEST.value,
            attacker_type=PieceType.YINGZH,
            defender_type=PieceType.LIANZH,
            defender_pos_flat=10 * 17 + 6,
            death_step=15,
        )
        south = Seat.SOUTH.value
        # SOUTH's chain memory: 87 inherits PAIZH lineage (pid=12 + type bit).
        assert cm.chain_pid_lo[south, 87] & (np.uint64(1) << np.uint64(12))
        assert (cm.chain_ate_my_type_mask[south, 87] & _bit(PieceType.PAIZH)) != 0
        # 87's rank_floor should be at least YINGZH+ (37 was LIANZH+, +1 = YINGZH+).
        assert cm.rank_floor[south, 87] >= 4
        # Chain propagation works for all observers (chain_pid bitmap).
        for obs in (Seat.WEST.value, Seat.NORTH.value, Seat.EAST.value):
            # 87's chain bitmap includes 37 (the public victim_pid).
            assert cm.chain_pid_lo[obs, 87] & (np.uint64(1) << np.uint64(37))


# ===========================================================================
# 6. DARK rule: non-victim observer only counts, no type info
# ===========================================================================


class TestDarkRule_OtherObserverNoType:
    def test_my_eat_does_not_leak_type_to_others(self):
        cm = CombatMemoryState.zeros()
        # SOUTH's piece (pid=11) eats WEST's PAIZH (pid=40).
        apply_combat_event(
            cm,
            event_is_eat=True,
            attacker_pid=11, defender_pid=40,
            attacker_seat=Seat.SOUTH.value, defender_seat=Seat.WEST.value,
            attacker_type=PieceType.LIANZH,
            defender_type=PieceType.PAIZH,
            defender_pos_flat=10 * 17 + 6,
            death_step=8,
        )
        west = Seat.WEST.value
        # WEST sees full direct memory.
        assert (cm.direct_ate_my_type_mask[west, 11] & _bit(PieceType.PAIZH)) != 0
        assert cm.rank_floor[west, 11] == 3
        # SOUTH (attacker's own seat — but defender is enemy from this seat):
        # only direct_other_count.
        south = Seat.SOUTH.value
        assert cm.direct_ate_my_type_mask[south, 11] == 0
        assert cm.direct_other_count[south, 11] == 1
        assert cm.rank_floor[south, 11] == 0


# ===========================================================================
# 7. Helpers / smoke
# ===========================================================================


class TestHelpers:
    def test_next_floor_after_eat(self):
        assert next_floor_after_eat(PieceType.GONGB) == 2
        assert next_floor_after_eat(PieceType.PAIZH) == 3
        assert next_floor_after_eat(PieceType.SHIZH) == 8
        assert next_floor_after_eat(PieceType.SILING) == 9
        for sp in (PieceType.JUNQI, PieceType.DILEI, PieceType.ZHADAN):
            assert next_floor_after_eat(sp) == 0

    def test_back_two_rows(self):
        assert is_in_seat_back_two_rows(15 * 17 + 8, Seat.SOUTH.value)
        assert is_in_seat_back_two_rows(16 * 17 + 8, Seat.SOUTH.value)
        assert not is_in_seat_back_two_rows(14 * 17 + 8, Seat.SOUTH.value)
        assert is_in_seat_back_two_rows(0, Seat.NORTH.value)
        assert is_in_seat_back_two_rows(8 * 17 + 0, Seat.WEST.value)
        assert is_in_seat_back_two_rows(8 * 17 + 16, Seat.EAST.value)

    def test_zeros_init_correct_shapes(self):
        cm = CombatMemoryState.zeros()
        assert cm.direct_ate_my_pid_lo.shape == (4, 120)
        assert cm.direct_ate_my_type_mask.dtype == np.uint16
        assert cm.is_gongb.dtype == bool
        assert cm.attacked_by_known_gongb.dtype == bool
        assert (cm.last_direct_step == -1).all()
        assert (cm.rank_floor == 0).all()

    def test_path_revealed_gongb_writes_all_observers(self):
        cm = CombatMemoryState.zeros()
        apply_path_revealed_gongb(cm, 88)
        for obs in range(NUM_OBSERVERS):
            assert cm.is_gongb[obs, 88]
            assert not cm.is_gongb[obs, 0]   # spot check uninvolved pid


# ===========================================================================
# Vectorised helpers behind the observation projection
# ===========================================================================
#
# _write_combat_memory writes 156 of the observation channels and is the
# dominant cost of an observation build. Batching its popcounts collapsed
# sixteen NumPy
# calls per build into five and a 30-iteration Python loop into one block,
# which is an easy place to change results by accident. These pin the two
# helpers that batching introduced against straightforward references.


def test_popcount_pid_pair_matches_bit_count() -> None:
    """Shape-preserving popcount must equal int.bit_count() elementwise."""
    rng = np.random.default_rng(0)
    for shape in [(0,), (1,), (17,), (34,), (12, 34), (4, 120)]:
        lo = rng.integers(0, 2**64, size=shape, dtype=np.uint64)
        hi = rng.integers(0, 2**64, size=shape, dtype=np.uint64)
        got = _popcount_pid_pair(lo, hi)
        want = np.array(
            [int(a).bit_count() + int(b).bit_count()
             for a, b in zip(lo.reshape(-1), hi.reshape(-1), strict=True)],
            dtype=np.int16,
        ).reshape(shape)
        assert got.shape == want.shape, shape
        assert got.dtype == np.int16, got.dtype
        np.testing.assert_array_equal(got, want, err_msg=f"shape={shape}")

    # Saturated words: 64 bits per half.
    full = np.array([2**64 - 1], dtype=np.uint64)
    assert int(_popcount_pid_pair(full, full)[0]) == 128


def test_pid_bit_masks_are_single_bits_at_the_right_offset() -> None:
    """One bit per pid, in the lo half below 64 and the hi half above."""
    for pid_lo in (0, 30, 60, 90):
        lo, hi = _pid_bit_masks(pid_lo, 30)
        assert lo.dtype == np.uint64 and hi.dtype == np.uint64
        # Exactly one bit set across the pair, per slot.
        np.testing.assert_array_equal(
            _popcount_pid_pair(lo, hi), np.ones(30, dtype=np.int16)
        )
        for s in range(30):
            gp = pid_lo + s
            if gp < 64:
                assert lo[s] == np.uint64(1) << np.uint64(gp), (pid_lo, s)
                assert hi[s] == np.uint64(0), (pid_lo, s)
            else:
                assert hi[s] == np.uint64(1) << np.uint64(gp - 64), (pid_lo, s)
                assert lo[s] == np.uint64(0), (pid_lo, s)
