"""tests/test_bug_n_belief_label_position.py — regression for BUG-N.

BUG-N (2026-05-11): RevealTracker._emit used to label cells based on the
INITIAL slot-to-cell mapping (``_SLOT_TO_CELL[seat, slot]``), so labels
described where the enemy pieces *started* the game. The corresponding
obs, however, is the END-OF-GAME observation showing pieces at their
CURRENT positions. After 500 random plies, only ~22% of labels matched
the piece actually present in that cell; the other ~78% were either
labels at empty cells (piece moved away or died) or labels at cells
holding a different piece (a different enemy piece moved in).

This regression test asserts the FIXED behaviour:
  * RevealTracker reads the current GPU SoA at termination time.
  * Labels are placed at each alive enemy piece's CURRENT cell.
  * No labels are placed at the piece's initial cell (unless the piece
    has not yet moved).
  * ``enemy_mask`` is True exactly on cells holding a current alive
    enemy piece (NOT on the static seat-geometry mask).

The test directly drives RevealTracker with a hand-built mock state
where one enemy piece has visibly moved, so the assertion pinpoints
the position semantics.
"""
from __future__ import annotations
import numpy as np
import pytest
import torch

from junqi_core.rules import PieceType, Seat, ALL_SEATS, CAMP_INDICES, SLOTS_PER_SEAT
from junqi_core.board import index_to_pos
from junqi_core.info_model import TRACKED_TYPES
from junqi_rl.belief.buffer import BeliefBuffer
from junqi_rl.belief.reveal_tracker import RevealTracker, BOARD_SIZE, NUM_CELLS


_PT_TO_BELIEF = {pt: i for i, pt in enumerate(TRACKED_TYPES)}


class _MockState:
    def __init__(self, soa: dict): self._soa = soa
    def copy_to_host(self) -> dict: return self._soa


class _MockRollout:
    def __init__(self, soa: dict, num_envs: int = 1):
        self._soa = soa
        self.num_envs = num_envs

    @property
    def state(self) -> _MockState:
        return _MockState(self._soa)

    def build_acting_seat_observation_torch(self, acting_t):
        # We only need a return-shape compatible obs for the buffer.
        from junqi_core.observation import OBS_CHANNELS
        B = acting_t.shape[0]
        return (torch.zeros(B, OBS_CHANNELS, 17, 17),
                torch.zeros(B, 28))


def _build_initial_soa(num_envs: int = 1) -> dict:
    """Build a SoA where every piece is at its INITIAL slot cell.

    Filled with deterministic types: slot s gets type (s % 12) + JUNQI.
    """
    N = num_envs
    piece_seat = np.full((N, 120), -1, dtype=np.int8)
    piece_type = np.zeros((N, 120), dtype=np.int8)
    pos_x = np.full((N, 120), -1, dtype=np.int8)
    pos_y = np.full((N, 120), -1, dtype=np.int8)
    alive = np.zeros((N, 120), dtype=bool)

    # Use real placement-rule-respecting types: just put SHIZH everywhere
    # except camps (slot ∈ CAMP_INDICES → NONE).
    for env in range(N):
        for seat in ALL_SEATS:
            s = int(seat)
            for slot in range(SLOTS_PER_SEAT):
                pid = s * 30 + slot
                piece_seat[env, pid] = s
                if slot in CAMP_INDICES:
                    piece_type[env, pid] = 0  # NONE
                    alive[env, pid] = False
                    continue
                # Use SHIZH (value=7) for simplicity; tracked.
                piece_type[env, pid] = int(PieceType.SHIZH.value)
                alive[env, pid] = True
                x, y = index_to_pos(seat, slot)
                pos_x[env, pid] = x
                pos_y[env, pid] = y
    return {
        "piece_seat_arr": piece_seat.flatten(),
        "piece_type_arr": piece_type.flatten(),
        "pos_x": pos_x.flatten(),
        "pos_y": pos_y.flatten(),
        "alive": alive.flatten(),
    }


def test_label_uses_current_position_after_piece_moves():
    """A SHIZH that moved from its initial cell to a new cell must be
    labeled at the NEW cell, not the initial cell."""
    soa = _build_initial_soa(num_envs=1)

    # Pick WEST seat (1) slot 0 → its piece. Move it from initial to a
    # new cell that is NOT in WEST's territory.
    seat = 1
    slot = 0
    pid = seat * 30 + slot
    init_x, init_y = index_to_pos(Seat(seat), slot)
    init_cell = init_y * BOARD_SIZE + init_x

    # Move to a centre cell (8, 8) — definitely not initial.
    new_x, new_y = 8, 8
    new_cell = new_y * BOARD_SIZE + new_x
    px = soa["pos_x"].reshape(1, 120)
    py = soa["pos_y"].reshape(1, 120)
    px[0, pid] = new_x
    py[0, pid] = new_y
    soa["pos_x"] = px.flatten()
    soa["pos_y"] = py.flatten()

    # Drive RevealTracker.
    buf = BeliefBuffer(capacity=10)
    tracker = RevealTracker(buf)
    snap = np.zeros((1, 4, 30), dtype=np.int64)   # unused now (BUG-N fix), but signature requires
    cb = tracker.make_callback(snap)

    rollout = _MockRollout(soa, num_envs=1)
    cb(
        fired_t=torch.tensor([True]),
        rewards_t=torch.tensor([1.0]),
        acting_t=torch.tensor([0], dtype=torch.int64),  # observer = SOUTH
        rollout_world=rollout,
    )

    assert len(buf) == 1
    label = buf._label[0]
    enemy = buf._enemy[0]

    # Piece moved: NEW cell should have the label, INITIAL cell should NOT.
    expected_belief = _PT_TO_BELIEF[PieceType.SHIZH]
    assert label[new_cell] == expected_belief, (
        f"label at NEW cell {new_cell} should be {expected_belief}, "
        f"got {label[new_cell]} — BUG-N regression"
    )
    assert label[init_cell] == -1, (
        f"label at INITIAL cell {init_cell} should be -1 (piece moved away), "
        f"got {label[init_cell]} — BUG-N regression"
    )

    # enemy_mask must mark the NEW cell, not the initial.
    assert enemy[new_cell] == True
    assert enemy[init_cell] == False, (
        f"enemy_mask at initial cell {init_cell} should be False after move "
        "(BUG-N: legacy code used static seat-geometry mask, not current positions)"
    )


def test_dead_piece_is_not_labeled():
    """A piece that has died (alive=False) must not appear in the label."""
    soa = _build_initial_soa(num_envs=1)
    seat = 1   # WEST (enemy of SOUTH observer)
    slot = 0
    pid = seat * 30 + slot
    init_x, init_y = index_to_pos(Seat(seat), slot)
    init_cell = init_y * BOARD_SIZE + init_x

    # Mark it dead.
    al = soa["alive"].reshape(1, 120)
    al[0, pid] = False
    soa["alive"] = al.flatten()

    buf = BeliefBuffer(capacity=10)
    tracker = RevealTracker(buf)
    snap = np.zeros((1, 4, 30), dtype=np.int64)
    cb = tracker.make_callback(snap)
    cb(
        fired_t=torch.tensor([True]),
        rewards_t=torch.tensor([1.0]),
        acting_t=torch.tensor([0], dtype=torch.int64),
        rollout_world=_MockRollout(soa),
    )

    assert len(buf) == 1
    label = buf._label[0]
    enemy = buf._enemy[0]
    assert label[init_cell] == -1, "dead piece must not be labeled"
    assert enemy[init_cell] == False, "dead piece's cell must not be in enemy_mask"


def test_teammate_piece_is_not_labeled_or_masked():
    """Pieces of the observer's teammate (same team) must be excluded from
    both label and enemy_mask."""
    soa = _build_initial_soa(num_envs=1)
    # Observer = SOUTH (seat 0); teammate = NORTH (seat 2). NORTH's piece
    # at slot 0 should NOT appear in the label.
    teammate = 2
    slot = 0
    pid = teammate * 30 + slot
    teammate_x, teammate_y = index_to_pos(Seat(teammate), slot)
    teammate_cell = teammate_y * BOARD_SIZE + teammate_x

    buf = BeliefBuffer(capacity=10)
    tracker = RevealTracker(buf)
    snap = np.zeros((1, 4, 30), dtype=np.int64)
    cb = tracker.make_callback(snap)
    cb(
        fired_t=torch.tensor([True]),
        rewards_t=torch.tensor([1.0]),
        acting_t=torch.tensor([0], dtype=torch.int64),
        rollout_world=_MockRollout(soa),
    )

    label = buf._label[0]
    enemy = buf._enemy[0]
    assert label[teammate_cell] == -1, "teammate piece must not be labeled"
    assert enemy[teammate_cell] == False, "teammate cell must not be in enemy_mask"


def test_observer_own_piece_is_not_labeled_or_masked():
    """Observer's own pieces are visible to the policy and must not appear
    in label or enemy_mask (the obs already encodes them via piece_own/piece_id
    channels)."""
    soa = _build_initial_soa(num_envs=1)
    own = 0  # SOUTH
    slot = 0
    own_x, own_y = index_to_pos(Seat(own), slot)
    own_cell = own_y * BOARD_SIZE + own_x

    buf = BeliefBuffer(capacity=10)
    tracker = RevealTracker(buf)
    snap = np.zeros((1, 4, 30), dtype=np.int64)
    cb = tracker.make_callback(snap)
    cb(
        fired_t=torch.tensor([True]),
        rewards_t=torch.tensor([1.0]),
        acting_t=torch.tensor([0], dtype=torch.int64),
        rollout_world=_MockRollout(soa),
    )

    label = buf._label[0]
    enemy = buf._enemy[0]
    assert label[own_cell] == -1
    assert enemy[own_cell] == False


def test_enemy_mask_count_matches_alive_enemy_count():
    """enemy_mask True count == number of alive enemy pieces in current
    state. With initial-state SoA: 25 non-camp pieces × 2 enemy seats = 50."""
    soa = _build_initial_soa(num_envs=1)
    buf = BeliefBuffer(capacity=10)
    tracker = RevealTracker(buf)
    snap = np.zeros((1, 4, 30), dtype=np.int64)
    cb = tracker.make_callback(snap)
    cb(
        fired_t=torch.tensor([True]),
        rewards_t=torch.tensor([1.0]),
        acting_t=torch.tensor([0], dtype=torch.int64),
        rollout_world=_MockRollout(soa),
    )
    enemy = buf._enemy[0]
    # 25 non-camp slots per enemy seat × 2 enemy seats = 50.
    # All initial pieces are SHIZH (alive), so all 50 should be in mask.
    assert int(enemy.sum()) == 50, (
        f"expected 50 enemy cells in mask, got {int(enemy.sum())}"
    )


def test_no_label_at_camp_cells():
    """Camp cells (slots 6,8,12,16,18 of each seat) are never piece-occupied
    so should never appear in label or enemy_mask, regardless of seat."""
    soa = _build_initial_soa(num_envs=1)
    buf = BeliefBuffer(capacity=10)
    tracker = RevealTracker(buf)
    snap = np.zeros((1, 4, 30), dtype=np.int64)
    cb = tracker.make_callback(snap)
    cb(
        fired_t=torch.tensor([True]),
        rewards_t=torch.tensor([1.0]),
        acting_t=torch.tensor([0], dtype=torch.int64),
        rollout_world=_MockRollout(soa),
    )
    label = buf._label[0]
    enemy = buf._enemy[0]
    for seat in ALL_SEATS:
        for slot in CAMP_INDICES:
            x, y = index_to_pos(seat, slot)
            cell = y * BOARD_SIZE + x
            assert label[cell] == -1, f"camp cell {cell} (seat={seat}) labeled"
            assert enemy[cell] == False, f"camp cell {cell} in enemy_mask"


if __name__ == "__main__":
    test_label_uses_current_position_after_piece_moves()
    print("[1/6] label tracks current position: OK")
    test_dead_piece_is_not_labeled()
    print("[2/6] dead piece not labeled: OK")
    test_teammate_piece_is_not_labeled_or_masked()
    print("[3/6] teammate not labeled: OK")
    test_observer_own_piece_is_not_labeled_or_masked()
    print("[4/6] own piece not labeled: OK")
    test_enemy_mask_count_matches_alive_enemy_count()
    print("[5/6] enemy_mask count matches alive: OK")
    test_no_label_at_camp_cells()
    print("[6/6] no label at camps: OK")
    print("\nAll BUG-N regression tests pass.")
