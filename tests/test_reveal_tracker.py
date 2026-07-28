"""Tests for junqi_rl.belief.reveal_tracker."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from junqi_core.observation import OBS_CHANNELS
from junqi_core.rules import CAMP_INDICES, PieceType, SLOTS_PER_SEAT
from junqi_core.setup import generate_random_lineup
from junqi_rl.belief.buffer import BeliefBuffer
from junqi_rl.belief.reveal_tracker import (
    NUM_CELLS,
    RevealTracker,
    _ARR_VOCAB_TO_BELIEF,
    _SLOT_TO_CELL,
    _SEAT_CELLS,
    _build_enemy_mask,
    _lineup_to_labels,
)
from junqi_rl.networks.arrangement_net import PIECE_TYPE_VALUE_TO_VOCAB_IDX
from junqi_rl.networks.belief_net import N_BELIEF_TYPES


# ---------------------------------------------------------------------------
# LUT invariants
# ---------------------------------------------------------------------------


def test_slot_to_cell_lut_shape():
    assert _SLOT_TO_CELL.shape == (4, SLOTS_PER_SEAT)
    assert _SLOT_TO_CELL.dtype == np.int64


def test_slot_to_cell_lut_camps_are_minus_one():
    """All 5 camp slots across all 4 seats must map to -1 (no board cell)."""
    for seat in range(4):
        for slot in CAMP_INDICES:
            assert _SLOT_TO_CELL[seat, slot] == -1, (
                f"seat {seat} slot {slot} (camp) must be -1"
            )


def test_slot_to_cell_lut_non_camps_in_range():
    """Non-camp slots must map to valid flat cell indices in [0, 289)."""
    for seat in range(4):
        for slot in range(SLOTS_PER_SEAT):
            if slot in CAMP_INDICES:
                continue
            cell = _SLOT_TO_CELL[seat, slot]
            assert 0 <= cell < NUM_CELLS, (
                f"seat {seat} slot {slot} -> cell {cell} out of range"
            )


def test_slot_to_cell_lut_no_collisions_within_seat():
    """Within one seat, distinct non-camp slots must map to distinct cells."""
    for seat in range(4):
        cells = [
            _SLOT_TO_CELL[seat, s]
            for s in range(SLOTS_PER_SEAT)
            if s not in CAMP_INDICES
        ]
        assert len(cells) == 25
        assert len(set(cells)) == 25, f"seat {seat}: collision in slot→cell"


def test_slot_to_cell_lut_no_collisions_across_seats():
    """No cell can be owned by two seats (territories are disjoint)."""
    all_cells = []
    for seat in range(4):
        for slot in range(SLOTS_PER_SEAT):
            if slot in CAMP_INDICES:
                continue
            all_cells.append(_SLOT_TO_CELL[seat, slot])
    assert len(all_cells) == 100
    assert len(set(all_cells)) == 100


def test_arr_vocab_to_belief_idx():
    """Arrangement vocab 0 (NONE) → -1; 1..12 → 0..11."""
    assert _ARR_VOCAB_TO_BELIEF[0] == -1
    for arr_idx in range(1, 13):
        belief_idx = _ARR_VOCAB_TO_BELIEF[arr_idx]
        assert 0 <= belief_idx < N_BELIEF_TYPES, (
            f"arr_vocab {arr_idx} -> belief {belief_idx} out of range"
        )
    # JUNQI (arr=1) should map to belief idx 0 (first tracked type).
    assert _ARR_VOCAB_TO_BELIEF[1] == 0
    # GONGB (arr=12) should map to belief idx 11 (last tracked type).
    assert _ARR_VOCAB_TO_BELIEF[12] == N_BELIEF_TYPES - 1


def test_seat_cells_lut_matches_slot_to_cell():
    """_SEAT_CELLS[s] must be the union of _SLOT_TO_CELL[s, non_camp_slots]."""
    for seat in range(4):
        from_slot_lut = sorted(
            _SLOT_TO_CELL[seat, s]
            for s in range(SLOTS_PER_SEAT)
            if s not in CAMP_INDICES
        )
        from_seat_lut = sorted(_SEAT_CELLS[seat].tolist())
        assert from_slot_lut == from_seat_lut


# ---------------------------------------------------------------------------
# _lineup_to_labels
# ---------------------------------------------------------------------------


def _make_vocab_lineup(lineup) -> np.ndarray:
    """Convert junqi_core.setup.generate_random_lineup output to vocab idx."""
    return np.array(
        [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup],
        dtype=np.int64,
    )


def test_lineup_to_labels_shape():
    import random
    rng = random.Random(1)
    lineup = generate_random_lineup(rng)
    vocab = _make_vocab_lineup(lineup)
    labels = _lineup_to_labels(vocab, seat_idx=0)
    assert labels.shape == (NUM_CELLS,)
    assert labels.dtype == np.int64


def test_lineup_to_labels_exactly_25_revealed():
    """A valid lineup has 25 non-NONE pieces in 25 non-camp slots."""
    import random
    rng = random.Random(2)
    for seat_idx in range(4):
        lineup = generate_random_lineup(rng)
        vocab = _make_vocab_lineup(lineup)
        labels = _lineup_to_labels(vocab, seat_idx=seat_idx)
        n_revealed = int((labels >= 0).sum())
        assert n_revealed == 25, (
            f"seat {seat_idx}: expected 25 revealed cells, got {n_revealed}"
        )


def test_lineup_to_labels_all_in_seat_territory():
    """All revealed cells must belong to the same seat's territory."""
    import random
    rng = random.Random(3)
    lineup = generate_random_lineup(rng)
    vocab = _make_vocab_lineup(lineup)
    for seat_idx in range(4):
        labels = _lineup_to_labels(vocab, seat_idx=seat_idx)
        revealed_cells = np.where(labels >= 0)[0]
        seat_cells = set(_SEAT_CELLS[seat_idx].tolist())
        for c in revealed_cells:
            assert c in seat_cells, (
                f"seat {seat_idx}: revealed cell {c} not in seat territory"
            )


def test_lineup_to_labels_camp_vocab_ignored():
    """If a NONE entry (arr_vocab=0) appears in a non-camp slot (invalid but
    edge-case robust), the cell gets -1 rather than crashing."""
    vocab = np.zeros(SLOTS_PER_SEAT, dtype=np.int64)  # all NONE (invalid lineup)
    labels = _lineup_to_labels(vocab, seat_idx=0)
    assert (labels == -1).all()


# ---------------------------------------------------------------------------
# _build_enemy_mask
# ---------------------------------------------------------------------------


def test_enemy_mask_counts():
    """Observer has 2 enemies, each owns 25 cells → 50 True cells total."""
    for observer_seat in range(4):
        mask = _build_enemy_mask(observer_seat)
        assert mask.shape == (NUM_CELLS,)
        assert int(mask.sum()) == 50, (
            f"observer={observer_seat}: expected 50 enemy cells, got {int(mask.sum())}"
        )


def test_enemy_mask_excludes_self_and_teammate():
    """The observer's own cells and teammate's cells must be False."""
    observer = 0   # SOUTH, team = {SOUTH, NORTH}
    mask = _build_enemy_mask(observer)
    # SOUTH (0) and NORTH (2) cells must all be False.
    for own_seat in (0, 2):
        for c in _SEAT_CELLS[own_seat]:
            assert not mask[c], (
                f"observer=0: cell {c} of seat {own_seat} should be False"
            )


def test_enemy_mask_teams_are_complementary():
    """Observer=0 and observer=2 (teammates) have IDENTICAL enemy masks."""
    mask_south = _build_enemy_mask(0)
    mask_north = _build_enemy_mask(2)
    assert np.array_equal(mask_south, mask_north)

    mask_west = _build_enemy_mask(1)
    mask_east = _build_enemy_mask(3)
    assert np.array_equal(mask_west, mask_east)

    # And team-RED's enemy mask must be the bitwise-NOT of team-BLUE's
    # (restricted to the 100 on-territory cells).
    on_territory = np.zeros(NUM_CELLS, dtype=bool)
    for seat in range(4):
        on_territory[_SEAT_CELLS[seat]] = True
    assert np.array_equal(mask_south & on_territory, ~mask_west & on_territory)


# ---------------------------------------------------------------------------
# Callback end-to-end
# ---------------------------------------------------------------------------


class _FakeRolloutWorld:
    """Minimal GpuRollout stand-in for callback testing.

    BUG-N (2026-05-11) note: the production ``RevealTracker._emit`` now
    reads the GPU SoA via ``rollout_world.state.copy_to_host()`` to label
    each enemy piece at its CURRENT cell. To keep these tests focused on
    the callback wiring (not on simulating piece movement), we synthesise
    a ``state`` mock from the same ``env_arr_snapshot`` that the test
    feeds into ``make_callback``: each piece is placed at its INITIAL
    slot cell. Effectively this asserts that, in the no-movement case
    (which is what the test setup represents), the new label-construction
    path returns the same labels the old LUT-based code did. Tests that
    want to exercise piece-movement behaviour should be added separately.
    """

    def __init__(self, num_envs: int, env_arr_snapshot: np.ndarray | None = None):
        self.num_envs = num_envs
        # Build a SoA mock from the lineup, treating each slot's initial
        # cell as the current position. ``env_arr_snapshot`` may be None
        # for tests that don't exercise the label path.
        self._mock_soa = self._build_mock_soa(env_arr_snapshot, num_envs)

    @staticmethod
    def _build_mock_soa(snap: np.ndarray | None, num_envs: int) -> dict:
        """Build a SoA dict mirroring DeviceGameStateBatch.copy_to_host().

        Uses initial-slot positions for piece coordinates — equivalent to
        a brand-new game with no movement yet.
        """
        from junqi_core.rules import ALL_SEATS, CAMP_INDICES, SLOTS_PER_SEAT
        from junqi_core.board import index_to_pos
        from junqi_rl.networks.arrangement_net import VOCAB_IDX_TO_PIECE_TYPE_VALUE

        N = num_envs
        piece_seat = np.full((N, 120), -1, dtype=np.int8)
        piece_type = np.zeros((N, 120), dtype=np.int8)
        pos_x = np.full((N, 120), -1, dtype=np.int8)
        pos_y = np.full((N, 120), -1, dtype=np.int8)
        alive = np.zeros((N, 120), dtype=bool)

        if snap is not None:
            for env in range(N):
                for seat in ALL_SEATS:
                    s = int(seat)
                    for slot in range(SLOTS_PER_SEAT):
                        pid = s * 30 + slot   # canonical pid layout
                        piece_seat[env, pid] = s
                        if slot in CAMP_INDICES:
                            piece_type[env, pid] = 0   # NONE
                            alive[env, pid] = False
                            continue
                        arr_vocab = int(snap[env, s, slot])
                        pt_value = int(VOCAB_IDX_TO_PIECE_TYPE_VALUE[arr_vocab])
                        piece_type[env, pid] = pt_value
                        if pt_value > 1:   # not NONE/DARK
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

    @property
    def state(self) -> "_FakeState":
        return _FakeState(self._mock_soa)

    def build_acting_seat_observation_torch(self, acting_t):
        B = acting_t.shape[0]
        # Return deterministic obs so we can check round-trip.
        obs = torch.arange(B, dtype=torch.float32).reshape(B, 1, 1, 1).expand(
            B, OBS_CHANNELS, 17, 17,
        ).contiguous()
        return obs, torch.zeros(B, 28)


class _FakeState:
    """Mock for ``rollout_world.state``."""

    def __init__(self, soa: dict):
        self._soa = soa

    def copy_to_host(self) -> dict:
        return self._soa


def _make_env_arr_snapshot(num_envs: int, *, seed: int = 0) -> np.ndarray:
    """Build a valid (N, 4, 30) arr snapshot from random lineups."""
    import random
    rng = random.Random(seed)
    snap = np.zeros((num_envs, 4, 30), dtype=np.int64)
    for e in range(num_envs):
        for s in range(4):
            lineup = generate_random_lineup(rng)
            snap[e, s] = _make_vocab_lineup(lineup)
    return snap


def test_callback_no_fired_envs_is_noop():
    buf = BeliefBuffer(capacity=100)
    tracker = RevealTracker(buf)
    snap = _make_env_arr_snapshot(4, seed=10)
    cb = tracker.make_callback(snap)

    fired = torch.zeros(4, dtype=torch.bool)
    acting = torch.zeros(4, dtype=torch.int64)
    cb(
        fired_t=fired,
        rewards_t=torch.zeros(4),
        acting_t=acting,
        rollout_world=_FakeRolloutWorld(4, snap),
    )
    assert len(buf) == 0
    assert tracker.n_events == 0
    assert tracker.n_inserted == 0


def test_callback_fires_single_env():
    buf = BeliefBuffer(capacity=100)
    tracker = RevealTracker(buf)
    snap = _make_env_arr_snapshot(4, seed=11)
    cb = tracker.make_callback(snap)

    fired = torch.tensor([False, True, False, False])
    acting = torch.tensor([0, 2, 0, 0], dtype=torch.int64)
    cb(
        fired_t=fired,
        rewards_t=torch.zeros(4),
        acting_t=acting,
        rollout_world=_FakeRolloutWorld(4, snap),
    )
    assert len(buf) == 1
    # Observer is seat 2 (NORTH); enemies are WEST (1) and EAST (3) → 50 cells.
    stats = buf.stats()
    # reveal_ratio = revealed cells / total cells. 50 revealed out of 289.
    assert 0.15 < stats["belief_buf/reveal_ratio"] < 0.20
    # enemy_density = enemy_mask True cells / total. 50/289.
    assert 0.15 < stats["belief_buf/enemy_density"] < 0.20


def test_callback_fires_multiple_envs():
    buf = BeliefBuffer(capacity=100)
    tracker = RevealTracker(buf)
    snap = _make_env_arr_snapshot(4, seed=12)
    cb = tracker.make_callback(snap)

    fired = torch.tensor([True, False, True, True])
    acting = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    cb(
        fired_t=fired,
        rewards_t=torch.zeros(4),
        acting_t=acting,
        rollout_world=_FakeRolloutWorld(4, snap),
    )
    assert len(buf) == 3
    assert tracker.n_events == 3


def test_callback_labels_match_source_lineup():
    """End-to-end: a terminated env's label must match the lineup we fed in."""
    buf = BeliefBuffer(capacity=100)
    tracker = RevealTracker(buf)
    snap = _make_env_arr_snapshot(4, seed=13)
    cb = tracker.make_callback(snap)

    # Observer = seat 0 (SOUTH). Enemies = seats 1 (WEST) and 3 (EAST).
    fired = torch.tensor([True, False, False, False])
    acting = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
    cb(
        fired_t=fired,
        rewards_t=torch.zeros(4),
        acting_t=acting,
        rollout_world=_FakeRolloutWorld(4, snap),
    )
    assert len(buf) == 1

    # Recover the stored labels.
    batch = next(buf.sample(batch_size=1, n_batches=1))
    stored_labels = batch.true_type_idx[0].numpy()          # (289,) int64
    stored_enemy = batch.enemy_mask[0].numpy()              # (289,) bool
    assert int(batch.seat_idx[0].item()) == 0

    # Expected: labels for enemy seats (1, 3) only; labels for own/teammate = -1.
    expected = np.full(NUM_CELLS, -1, dtype=np.int64)
    for enemy_seat in (1, 3):
        enemy_labels = _lineup_to_labels(snap[0, enemy_seat, :], enemy_seat)
        got = enemy_labels >= 0
        expected[got] = enemy_labels[got]

    assert np.array_equal(stored_labels, expected)
    # enemy_mask must be True exactly on the 50 enemy cells (regardless
    # of whether they were revealed).
    expected_mask = _build_enemy_mask(0)
    assert np.array_equal(stored_enemy, expected_mask)


def test_callback_observer_gets_own_team_not_masked():
    """Teammate pieces must NOT appear in the labels (observer can't see them
    as 'revealed' — they're visible normally, handled elsewhere)."""
    buf = BeliefBuffer(capacity=100)
    tracker = RevealTracker(buf)
    snap = _make_env_arr_snapshot(2, seed=14)
    cb = tracker.make_callback(snap)

    fired = torch.tensor([True, False])
    acting = torch.tensor([0, 0], dtype=torch.int64)
    cb(
        fired_t=fired,
        rewards_t=torch.zeros(2),
        acting_t=acting,
        rollout_world=_FakeRolloutWorld(2, snap),
    )

    batch = next(buf.sample(batch_size=1, n_batches=1))
    labels = batch.true_type_idx[0].numpy()
    # Observer = 0, teammate = 2. Their cells must all be -1.
    for seat in (0, 2):
        for c in _SEAT_CELLS[seat]:
            assert labels[c] == -1


def test_callback_disabled_buffer_is_noop():
    tracker = RevealTracker(belief_buffer=None)
    snap = _make_env_arr_snapshot(2, seed=15)
    cb = tracker.make_callback(snap)
    fired = torch.tensor([True, True])
    cb(
        fired_t=fired,
        rewards_t=torch.zeros(2),
        acting_t=torch.zeros(2, dtype=torch.int64),
        rollout_world=_FakeRolloutWorld(2, snap),
    )
    assert tracker.n_events == 0
    assert tracker.n_inserted == 0


def test_callback_snapshot_shape_validated():
    tracker = RevealTracker(BeliefBuffer(capacity=10))
    bad_snap = np.zeros((4, 3, 30), dtype=np.int64)   # wrong n_seats
    with pytest.raises(ValueError, match="env_arr_snapshot"):
        tracker.make_callback(bad_snap)


def test_stats():
    buf = BeliefBuffer(capacity=100)
    tracker = RevealTracker(buf)
    s0 = tracker.stats()
    assert s0["reveal_tracker/n_events"] == 0.0
    assert s0["reveal_tracker/n_inserted"] == 0.0

    snap = _make_env_arr_snapshot(4, seed=16)
    cb = tracker.make_callback(snap)
    cb(
        fired_t=torch.tensor([True, True, False, False]),
        rewards_t=torch.zeros(4),
        acting_t=torch.zeros(4, dtype=torch.int64),
        rollout_world=_FakeRolloutWorld(4, snap),
    )
    s1 = tracker.stats()
    assert s1["reveal_tracker/n_events"] == 2.0
    assert s1["reveal_tracker/n_inserted"] == 2.0
