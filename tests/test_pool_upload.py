"""Tests for junqi_rl.arrangement.pool_upload."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from junqi_core.rules import PieceType, SLOTS_PER_SEAT
from junqi_core.setup import generate_random_lineup, validate_lineup
from junqi_rl.arrangement.pool_upload import (
    arrangements_to_pool,
    read_env_arrangements_from_state,
    refresh_gpu_setup_pool,
)
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    PIECE_TYPE_VALUE_TO_VOCAB_IDX,
    VOCAB_IDX_TO_PIECE_TYPE_VALUE,
)


# ---------------------------------------------------------------------------
# arrangements_to_pool
# ---------------------------------------------------------------------------


def _make_onehot_from_lineup(lineup) -> torch.Tensor:
    vocab = torch.tensor(
        [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup], dtype=torch.long
    )
    return F.one_hot(vocab, num_classes=N_PIECE_TYPE_WITH_NONE).float()


def _make_n_samples(n: int, *, seed: int = 0):
    import random
    rng = random.Random(seed)
    samples = torch.stack(
        [_make_onehot_from_lineup(generate_random_lineup(rng)) for _ in range(n)]
    )
    seats = torch.arange(n, dtype=torch.long) % 4  # balanced cycle
    return samples, seats


def test_pool_shape_and_dtype():
    samples, seats = _make_n_samples(16, seed=1)
    pool = arrangements_to_pool(samples, seats)
    assert pool.shape == (4, 120)   # 16 samples / 4 seats = 4 combined entries
    assert pool.dtype == np.int8


def test_pool_recovers_original_lineups():
    """Each pool entry's 30-slot seat block must equal one of the original
    lineups (in PieceType.value)."""
    samples, seats = _make_n_samples(8, seed=2)
    # Manually compute expected: group samples by seat and take per-seat slot.
    vocab = samples.argmax(dim=-1).numpy()
    lut = np.asarray(VOCAB_IDX_TO_PIECE_TYPE_VALUE, dtype=np.int8)
    pt_values = lut[vocab]  # (8, 30)

    pool = arrangements_to_pool(samples, seats)
    # pool_size = 2 (8 samples / 4 seats)
    assert pool.shape == (2, 120)

    for pool_entry in range(2):
        for seat in range(4):
            # Source index for seat s, slot k: sample index = k*4 + s where k=pool_entry.
            # seats = [0,1,2,3,0,1,2,3] → per_seat_indices[0] = [0, 4], etc.
            expected_sample_idx = seat + 4 * pool_entry
            expected = pt_values[expected_sample_idx]
            got = pool[pool_entry, seat * SLOTS_PER_SEAT:(seat + 1) * SLOTS_PER_SEAT]
            assert np.array_equal(got, expected), (
                f"pool entry {pool_entry} seat {seat}: "
                f"expected {expected.tolist()}, got {got.tolist()}"
            )


def test_pool_validates_under_junqi_core():
    """Every seat's 30-slot block in a pool entry must pass validate_lineup."""
    samples, seats = _make_n_samples(32, seed=3)
    pool = arrangements_to_pool(samples, seats)
    for entry_idx, entry in enumerate(pool):
        for seat in range(4):
            lineup_values = entry[seat * SLOTS_PER_SEAT:(seat + 1) * SLOTS_PER_SEAT]
            lineup = [PieceType(int(v)) for v in lineup_values]
            r = validate_lineup(lineup)
            assert r.ok, f"entry {entry_idx} seat {seat}: {r.violations}"


def test_pool_size_is_min_across_seats():
    """If seats are imbalanced (2 of seat 0, 5 of seat 1, etc), pool_size
    equals the min count."""
    samples_list = []
    seats_list = []
    import random
    rng = random.Random(7)
    # 2 of seat 0, 3 of seat 1, 4 of seat 2, 5 of seat 3 = 14 total.
    counts = [2, 3, 4, 5]
    for s, c in enumerate(counts):
        for _ in range(c):
            samples_list.append(_make_onehot_from_lineup(generate_random_lineup(rng)))
            seats_list.append(s)
    samples = torch.stack(samples_list)
    seats = torch.tensor(seats_list, dtype=torch.long)
    pool = arrangements_to_pool(samples, seats)
    assert pool.shape[0] == min(counts)


def test_pool_rejects_missing_seat():
    """If some seat has zero samples, we must raise rather than silently
    produce an all-zero pool."""
    samples_list = []
    seats_list = []
    import random
    rng = random.Random(8)
    for _ in range(4):
        samples_list.append(_make_onehot_from_lineup(generate_random_lineup(rng)))
        seats_list.append(0)  # all seat 0
    samples = torch.stack(samples_list)
    seats = torch.tensor(seats_list, dtype=torch.long)
    with pytest.raises(ValueError):
        arrangements_to_pool(samples, seats)


def test_pool_rejects_bad_shape():
    with pytest.raises(ValueError):
        arrangements_to_pool(torch.zeros(4, 30), torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError):
        arrangements_to_pool(torch.zeros(4, 30, 13), torch.zeros(5, dtype=torch.long))


def test_pool_size_zero_keeps_zip_pairing():
    samples, seats = _make_n_samples(8, seed=2)
    zipped = arrangements_to_pool(samples, seats)
    also = arrangements_to_pool(samples, seats, pool_size=0, seed=99)
    assert zipped.shape == (2, 120)
    assert np.array_equal(zipped, also)


def test_expanded_pool_keeps_zip_prefix_and_grows():
    samples, seats = _make_n_samples(8, seed=4)
    zipped = arrangements_to_pool(samples, seats)
    expanded = arrangements_to_pool(samples, seats, pool_size=16, seed=0)
    assert expanded.shape == (16, 120)
    assert np.array_equal(expanded[:2], zipped)
    n_unique = int(np.unique(expanded, axis=0).shape[0])
    assert n_unique >= 8, n_unique
    for entry in expanded:
        for seat in range(4):
            lineup = [PieceType(int(v)) for v in entry[seat * SLOTS_PER_SEAT:(seat + 1) * SLOTS_PER_SEAT]]
            r = validate_lineup(lineup)
            assert r.ok, r.violations


def test_expanded_pool_is_deterministic_in_seed():
    samples, seats = _make_n_samples(16, seed=5)
    a = arrangements_to_pool(samples, seats, pool_size=32, seed=7)
    b = arrangements_to_pool(samples, seats, pool_size=32, seed=7)
    c = arrangements_to_pool(samples, seats, pool_size=32, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_expanded_pool_uses_independent_seat_pairing():
    """Zip pairing freezes 4-tuples; expansion should remix seats."""
    samples, seats = _make_n_samples(32, seed=6)
    zipped = arrangements_to_pool(samples, seats)
    expanded = arrangements_to_pool(samples, seats, pool_size=64, seed=1)
    zip_set = {row.tobytes() for row in zipped}
    extra = expanded[len(zipped):]
    n_new = sum(1 for row in extra if row.tobytes() not in zip_set)
    assert n_new >= 16, n_new
    # Every seat block in the expanded pool must come from that seat's samples.
    vocab = samples.argmax(dim=-1).numpy()
    lut = np.asarray(VOCAB_IDX_TO_PIECE_TYPE_VALUE, dtype=np.int8)
    pt = lut[vocab]
    seats_np = seats.numpy()
    for s in range(4):
        allowed = {pt[i].tobytes() for i in np.nonzero(seats_np == s)[0]}
        for row in expanded:
            block = row[s * SLOTS_PER_SEAT:(s + 1) * SLOTS_PER_SEAT]
            assert block.tobytes() in allowed


# ---------------------------------------------------------------------------
# refresh_gpu_setup_pool — monkey-patched CUDA
# ---------------------------------------------------------------------------


def test_refresh_calls_cuda(monkeypatch):
    called = {}

    class FakeCuda:
        @staticmethod
        def upload_setup_pool(pool):
            called["pool"] = pool.copy()

    monkeypatch.setitem(
        __import__("sys").modules,
        "junqi_cuda",
        FakeCuda,
    )
    pool = np.zeros((3, 120), dtype=np.int8)
    refresh_gpu_setup_pool(pool)
    assert "pool" in called
    assert np.array_equal(called["pool"], pool)


def test_refresh_rejects_bad_shape():
    with pytest.raises(ValueError):
        refresh_gpu_setup_pool(np.zeros((3, 119), dtype=np.int8))


# ---------------------------------------------------------------------------
# read_env_arrangements_from_state — stubbed rollout.state
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal rollout.state stand-in with a copy_to_host() method."""

    def __init__(self, piece_type_arr: np.ndarray):
        # Flatten to match CUDA's ``copy_to_host`` convention: (N*120,) int8.
        self._pta = piece_type_arr.reshape(-1).astype(np.int8)

    def copy_to_host(self):
        return {"piece_type_arr": self._pta}


class _FakeRollout:
    def __init__(self, piece_type_arr: np.ndarray):
        self.num_envs = piece_type_arr.shape[0]
        self.state = _FakeState(piece_type_arr)


def test_read_env_arrangements_shape_and_values():
    # Build an N=2 env with known per-seat lineups.
    import random
    rng = random.Random(11)
    N = 2
    pta = np.zeros((N, 120), dtype=np.int8)
    expected_vocab = np.zeros((N, 4, 30), dtype=np.int64)
    for e in range(N):
        for s in range(4):
            lineup = generate_random_lineup(rng)
            for slot, pt in enumerate(lineup):
                pta[e, s * 30 + slot] = int(pt)
                expected_vocab[e, s, slot] = PIECE_TYPE_VALUE_TO_VOCAB_IDX[int(pt)]
    rollout = _FakeRollout(pta)
    got = read_env_arrangements_from_state(rollout)
    assert got.shape == (N, 4, 30)
    assert np.array_equal(got, expected_vocab)


# ---------------------------------------------------------------------------
# Round-trip: arrangements → pool → snapshot recovers original
# ---------------------------------------------------------------------------


def test_pool_and_snapshot_roundtrip():
    """The pool we upload and the snapshot we read back must agree on piece
    types for each (pool_entry, seat) → env identity."""
    samples, seats = _make_n_samples(12, seed=13)   # 3 pool entries
    pool = arrangements_to_pool(samples, seats)
    # Simulate: each env gets pool[e % pool_size] (what hash-based selection
    # would approximate uniformly). For this test we pick deterministic
    # mapping to verify the decode path.
    P = pool.shape[0]
    N = P * 2  # 2 envs per pool entry
    pta = np.zeros((N, 120), dtype=np.int8)
    for e in range(N):
        pta[e] = pool[e % P]
    rollout = _FakeRollout(pta)
    got = read_env_arrangements_from_state(rollout)

    # For each env, each seat's lineup (as vocab) should match the sample
    # that was fed into the pool for that (entry, seat) slot.
    # Reconstruct expected from the original samples using the grouping
    # convention of arrangements_to_pool.
    vocab = samples.argmax(dim=-1).numpy()
    seats_np = seats.numpy()
    per_seat_indices = [np.nonzero(seats_np == s)[0][:P] for s in range(4)]

    for e in range(N):
        pool_entry = e % P
        for s in range(4):
            src = per_seat_indices[s][pool_entry]
            assert np.array_equal(got[e, s], vocab[src]), (
                f"env {e} seat {s}: snapshot does not recover sample {src}"
            )
