"""Tests for junqi_rl.belief.buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from junqi_rl.belief.buffer import BeliefBuffer, BeliefSample
from junqi_rl.networks.belief_net import N_BELIEF_TYPES
from junqi_core.observation import OBS_CHANNELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_samples(
    N: int,
    *,
    seed: int = 0,
    reveal_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Make N valid (obs, seat, label, enemy_mask) tuples."""
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((N, OBS_CHANNELS, 17, 17)).astype(np.float32)
    seat = rng.integers(0, 4, size=(N,), dtype=np.int64)

    label = np.full((N, 289), -1, dtype=np.int64)
    # Randomly reveal a fraction of cells with valid type indices.
    n_cells_to_reveal = int(reveal_ratio * 289)
    for i in range(N):
        cells = rng.choice(289, size=n_cells_to_reveal, replace=False)
        types = rng.integers(0, N_BELIEF_TYPES, size=n_cells_to_reveal)
        label[i, cells] = types

    # Enemy mask: a superset of revealed cells (revealed ⊆ enemy).
    enemy = (label >= 0)
    # Add some extra enemy cells that haven't been revealed yet.
    n_extra = int(0.1 * 289)
    for i in range(N):
        unrevealed = np.where(label[i] < 0)[0]
        if len(unrevealed) >= n_extra:
            cells = rng.choice(unrevealed, size=n_extra, replace=False)
            enemy[i, cells] = True

    return obs, seat, label, enemy


# ---------------------------------------------------------------------------
# Construction / basic invariants
# ---------------------------------------------------------------------------


def test_construction_defaults():
    buf = BeliefBuffer()
    assert len(buf) == 0
    assert buf.capacity == 12_000
    assert buf._obs.shape[0] == 0
    assert buf.head == 0
    assert not buf.is_full


def test_construction_rejects_bad_capacity():
    with pytest.raises(ValueError):
        BeliefBuffer(capacity=0)
    with pytest.raises(ValueError):
        BeliefBuffer(capacity=-10)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_basic():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(16, seed=1)
    n = buf.add(obs, seat, label, enemy)
    assert n == 16
    assert len(buf) == 16
    assert buf.head == 16


def test_add_multiple_writes_accumulate():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs1, seat1, label1, enemy1 = _fake_samples(30, seed=2)
    obs2, seat2, label2, enemy2 = _fake_samples(20, seed=3)
    buf.add(obs1, seat1, label1, enemy1)
    buf.add(obs2, seat2, label2, enemy2)
    assert len(buf) == 50
    assert buf.head == 50


def test_add_wraps_past_capacity():
    buf = BeliefBuffer(capacity=10, seed=0)
    obs, seat, label, enemy = _fake_samples(7, seed=4)
    buf.add(obs, seat, label, enemy)     # head=7, size=7
    obs, seat, label, enemy = _fake_samples(5, seed=5)
    buf.add(obs, seat, label, enemy)     # head=12 mod 10 = 2, size=10
    assert len(buf) == 10
    assert buf.head == 2
    assert buf.is_full


def test_add_larger_than_capacity_keeps_most_recent():
    """If caller dumps N > capacity, keep only the last `capacity` items."""
    buf = BeliefBuffer(capacity=5, seed=0)
    obs, seat, label, enemy = _fake_samples(20, seed=6)
    n = buf.add(obs, seat, label, enemy)
    assert n == 5
    assert len(buf) == 5
    # The stored seats should be the last 5.
    # Seeking through buffer, seat values are shuffled by ring layout but the
    # SET of seats stored should equal the SET of last 5 seats from source.
    stored_seats_sorted = sorted(buf._seat[:5].tolist())
    expected_sorted = sorted(seat[-5:].tolist())
    assert stored_seats_sorted == expected_sorted


def test_add_empty_batch_noop():
    buf = BeliefBuffer(capacity=100, seed=0)
    empty_obs = np.zeros((0, OBS_CHANNELS, 17, 17), dtype=np.float32)
    empty_seat = np.zeros((0,), dtype=np.int64)
    empty_label = np.zeros((0, 289), dtype=np.int64)
    empty_enemy = np.zeros((0, 289), dtype=bool)
    n = buf.add(empty_obs, empty_seat, empty_label, empty_enemy)
    assert n == 0
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# add — shape / domain validation
# ---------------------------------------------------------------------------


def test_add_rejects_obs_wrong_ndim():
    buf = BeliefBuffer(capacity=10)
    obs = np.zeros((8, OBS_CHANNELS, 17, 17, 1), dtype=np.float32)   # 5D
    seat = np.zeros(8, dtype=np.int64)
    label = np.full((8, 289), -1, dtype=np.int64)
    enemy = np.zeros((8, 289), dtype=bool)
    with pytest.raises(ValueError, match="4D"):
        buf.add(obs, seat, label, enemy)


def test_add_rejects_obs_wrong_channels():
    buf = BeliefBuffer(capacity=10)
    obs = np.zeros((8, 64, 17, 17), dtype=np.float32)
    seat = np.zeros(8, dtype=np.int64)
    label = np.full((8, 289), -1, dtype=np.int64)
    enemy = np.zeros((8, 289), dtype=bool)
    with pytest.raises(ValueError):
        buf.add(obs, seat, label, enemy)


def test_add_rejects_label_out_of_range():
    buf = BeliefBuffer(capacity=10)
    obs, seat, label, enemy = _fake_samples(4, seed=11)
    label[0, 0] = N_BELIEF_TYPES   # invalid
    with pytest.raises(ValueError, match="true_type_idx values"):
        buf.add(obs, seat, label, enemy)


def test_add_accepts_label_negative_one_sentinel():
    buf = BeliefBuffer(capacity=10)
    obs, seat, label, enemy = _fake_samples(4, seed=12)
    label[:] = -1   # all unknown
    buf.add(obs, seat, label, enemy)
    assert len(buf) == 4


def test_add_rejects_seat_out_of_range():
    buf = BeliefBuffer(capacity=10)
    obs, seat, label, enemy = _fake_samples(4, seed=13)
    seat[0] = 4   # invalid (seats 0-3 only)
    with pytest.raises(ValueError, match="seat_idx values"):
        buf.add(obs, seat, label, enemy)


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_returns_correct_shape():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(30, seed=21)
    buf.add(obs, seat, label, enemy)
    gen = buf.sample(batch_size=8, n_batches=3)
    batches = list(gen)
    assert len(batches) == 3
    for b in batches:
        assert isinstance(b, BeliefSample)
        assert b.obs_spatial.shape == (8, OBS_CHANNELS, 17, 17)
        assert b.obs_spatial.dtype == torch.float32
        assert b.seat_idx.shape == (8,)
        assert b.true_type_idx.shape == (8, 289)
        assert b.enemy_mask.shape == (8, 289)


def test_sample_empty_buffer_yields_nothing():
    buf = BeliefBuffer(capacity=100, seed=0)
    batches = list(buf.sample(batch_size=8, n_batches=4))
    assert batches == []


def test_sample_with_replacement_when_bufsize_less_than_batch():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(3, seed=22)
    buf.add(obs, seat, label, enemy)
    # batch_size > size → replace=True
    batch = next(buf.sample(batch_size=8, n_batches=1))
    assert batch.obs_spatial.shape == (8, OBS_CHANNELS, 17, 17)


def test_sample_indefinite_generator_stops_on_break():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(30, seed=23)
    buf.add(obs, seat, label, enemy)
    it = buf.sample(batch_size=4)
    seen = 0
    for _ in it:
        seen += 1
        if seen >= 5:
            break
    assert seen == 5


def test_sample_device_cpu():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(16, seed=24)
    buf.add(obs, seat, label, enemy)
    batch = next(buf.sample(batch_size=4, device="cpu", n_batches=1))
    assert batch.obs_spatial.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sample_device_cuda():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(16, seed=25)
    buf.add(obs, seat, label, enemy)
    batch = next(buf.sample(batch_size=4, device="cuda", n_batches=1))
    assert batch.obs_spatial.device.type == "cuda"


# ---------------------------------------------------------------------------
# Round-trip: add → sample recovers values
# ---------------------------------------------------------------------------


def test_roundtrip_label_values_preserved():
    """The exact int labels we inserted must come back out via sample()."""
    buf = BeliefBuffer(capacity=50, seed=0)
    obs, seat, label, enemy = _fake_samples(20, seed=31)
    buf.add(obs, seat, label, enemy)

    # Pull a batch equal to the buffer size and dedup-check labels.
    batch = next(buf.sample(batch_size=100, n_batches=1))  # with replacement
    # Every label tensor entry must match a row from `label` (value-wise;
    # position-wise hard because of random sampling). Instead: confirm
    # every row of batch.true_type_idx is equal to some row of label.
    got_rows = {tuple(row.tolist()) for row in batch.true_type_idx}
    orig_rows = {tuple(row.tolist()) for row in torch.from_numpy(label)}
    assert got_rows.issubset(orig_rows), (
        "Sampled label rows that don't appear in original — data corruption?"
    )


def test_roundtrip_obs_values_preserved_within_fp16_tolerance():
    """obs is stored fp16; round-trip should preserve it within fp16 tolerance."""
    buf = BeliefBuffer(capacity=10, seed=0)
    obs, seat, label, enemy = _fake_samples(4, seed=32)

    buf.add(obs, seat, label, enemy)
    # Sample one with a deterministic RNG to force batch == first 4 rows.
    # Easiest: set the buffer's rng.
    buf._rng = np.random.default_rng(999)
    batch = next(buf.sample(batch_size=4, n_batches=1))

    # Build the set of (obs, seat) rows from both sides; check each sampled
    # row appears with correct fp16 rounding.
    for i in range(4):
        got_obs = batch.obs_spatial[i].numpy().astype(np.float16).astype(np.float32)
        # Find the matching source row by seat+enemy_mask.
        matched = False
        for j in range(4):
            src_seat = seat[j]
            if int(batch.seat_idx[i].item()) == int(src_seat):
                src_obs_fp16 = obs[j].astype(np.float16).astype(np.float32)
                if np.allclose(got_obs, src_obs_fp16, atol=1e-3):
                    matched = True
                    break
        assert matched, f"sampled row {i} didn't match any source row"


# ---------------------------------------------------------------------------
# LRU / eviction semantics
# ---------------------------------------------------------------------------


def test_lru_old_entries_overwritten():
    """After filling capacity and then adding more, the oldest entries
    are evicted (overwritten) — the newest N should always be present."""
    buf = BeliefBuffer(capacity=10, seed=0)

    # Fill with 10 distinct seats (each cycle 0,1,2,3,0,1,2,3,0,1)
    obs, seat, label, enemy = _fake_samples(10, seed=41)
    seat = np.array([i % 4 for i in range(10)], dtype=np.int64)
    buf.add(obs, seat, label, enemy)
    assert buf.is_full

    # Add 5 more with seats = [0, 0, 0, 0, 0] — these should evict 5 oldest.
    obs2, _, label2, enemy2 = _fake_samples(5, seed=42)
    seat2 = np.zeros(5, dtype=np.int64)
    buf.add(obs2, seat2, label2, enemy2)
    assert len(buf) == 10

    # Since we inserted 10+5=15 and capacity=10, the buffer now holds
    # "items 5..14" of the insertion order. The latest 5 are all seat 0,
    # so at least 5 seat-0s must be in the buffer.
    n_seat_zero = int((buf._seat == 0).sum())
    # Originally 3 seat-0s (indices 0, 4, 8) + 5 new = 8, minus however
    # many originals got evicted. Items evicted are indices 0..4, removing
    # 2 seat-0s (idx 0, 4). So: 3 - 2 + 5 = 6 seat-0s remain.
    assert n_seat_zero >= 5, f"expected >=5 seat-0 entries post-eviction, got {n_seat_zero}"


def test_clear_resets_buffer():
    buf = BeliefBuffer(capacity=20, seed=0)
    obs, seat, label, enemy = _fake_samples(15, seed=51)
    buf.add(obs, seat, label, enemy)
    assert len(buf) == 15
    buf.clear()
    assert len(buf) == 0
    assert buf.head == 0
    # Can add again post-clear.
    buf.add(obs, seat, label, enemy)
    assert len(buf) == 15


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_empty_buffer():
    buf = BeliefBuffer(capacity=100, seed=0)
    s = buf.stats()
    assert s["belief_buf/size"] == 0.0
    assert s["belief_buf/fill_ratio"] == 0.0


def test_stats_non_empty_reveals_and_density():
    buf = BeliefBuffer(capacity=100, seed=0)
    obs, seat, label, enemy = _fake_samples(20, seed=61, reveal_ratio=0.3)
    buf.add(obs, seat, label, enemy)
    s = buf.stats()
    assert s["belief_buf/size"] == 20.0
    assert s["belief_buf/fill_ratio"] == 0.2
    # reveal_ratio is (# cells with label >= 0) / total cells.
    # With reveal_ratio=0.3 we revealed ~0.3 * 289 ≈ 86 cells per sample.
    assert 0.25 < s["belief_buf/reveal_ratio"] < 0.35
    # enemy_density includes revealed + extra enemies.
    assert s["belief_buf/enemy_density"] > s["belief_buf/reveal_ratio"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_state_dict_roundtrip():
    buf1 = BeliefBuffer(capacity=50, seed=123)
    obs, seat, label, enemy = _fake_samples(30, seed=71)
    buf1.add(obs, seat, label, enemy)
    sd = buf1.state_dict()

    buf2 = BeliefBuffer(capacity=50, seed=456)
    buf2.load_state_dict(sd)
    assert len(buf2) == len(buf1)
    assert buf2.head == buf1.head
    # The loaded arrays should be byte-equal to the originals.
    np.testing.assert_array_equal(buf1._label[:30], buf2._label[:30])
    np.testing.assert_array_equal(buf1._seat[:30], buf2._seat[:30])


def test_state_dict_capacity_mismatch_raises():
    buf1 = BeliefBuffer(capacity=50, seed=0)
    obs, seat, label, enemy = _fake_samples(10, seed=81)
    buf1.add(obs, seat, label, enemy)
    sd = buf1.state_dict()

    buf2 = BeliefBuffer(capacity=100, seed=0)   # different capacity
    with pytest.raises(ValueError, match="capacity mismatch"):
        buf2.load_state_dict(sd)


def test_state_dict_with_full_buffer():
    """state_dict must work correctly when size == capacity."""
    buf1 = BeliefBuffer(capacity=10, seed=0)
    obs, seat, label, enemy = _fake_samples(15, seed=91)
    buf1.add(obs, seat, label, enemy)
    assert buf1.is_full
    sd = buf1.state_dict()
    assert sd["size"] == 10

    buf2 = BeliefBuffer(capacity=10, seed=0)
    buf2.load_state_dict(sd)
    assert buf2.is_full
    assert buf2._size == 10


# ---------------------------------------------------------------------------
# BeliefSample validation
# ---------------------------------------------------------------------------


def test_sample_rejects_mismatched_shapes():
    obs = torch.zeros(4, OBS_CHANNELS, 17, 17)
    seat = torch.zeros(3, dtype=torch.long)   # mismatched B
    label = torch.zeros(4, 289, dtype=torch.long)
    enemy = torch.zeros(4, 289, dtype=torch.bool)
    with pytest.raises(ValueError, match="seat_idx"):
        BeliefSample(obs, seat, label, enemy)
