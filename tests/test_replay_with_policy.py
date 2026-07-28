"""tests/test_replay_with_policy.py — TrajectoryWithPolicy roundtrip +
record_game_with_policy smoke tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

import random

from junqi_core.replay_with_policy import (
    TrajectoryWithPolicy,
    probs_to_top_k,
)
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup


def test_trajectory_with_policy_save_load_roundtrip():
    rng = random.Random(7)
    setups = generate_random_setup(rng)
    T = 5
    K = 16
    top_ids = np.arange(T * K, dtype=np.int32).reshape(T, K)
    top_probs = np.random.RandomState(0).rand(T, K).astype(np.float32)
    values = np.linspace(-1, 1, T, dtype=np.float32)
    actions = np.zeros((T, 5), dtype=np.int16)
    seats = np.array([0, 1, 2, 3, 0], dtype=np.int8)

    traj = TrajectoryWithPolicy(
        setups=setups,
        actions=actions,
        rng_seed=7,
        final_state_hash=0xDEADBEEF,
        first_seat=Seat.SOUTH,
        show_mode=ShowMode.HALF_DARK,
        top_action_ids=top_ids,
        top_probs=top_probs,
        values=values,
        acting_seats=seats,
        meta={"iteration": 42, "win_rate": 0.75},
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "t.npz"
        traj.save(p)
        loaded = TrajectoryWithPolicy.load(p)
    assert np.array_equal(loaded.top_action_ids, top_ids)
    assert np.allclose(loaded.top_probs, top_probs)
    assert np.allclose(loaded.values, values)
    assert loaded.num_steps == T
    assert loaded.top_k == K
    assert loaded.meta == {"iteration": 42, "win_rate": 0.75}


def test_probs_to_top_k():
    FLAT = 83521
    probs = np.zeros(FLAT, dtype=np.float32)
    probs[10] = 0.5
    probs[20] = 0.3
    probs[30] = 0.2
    mask = np.zeros(FLAT, dtype=bool)
    mask[10] = mask[20] = mask[30] = True
    mask[40] = True  # masked zero-prob
    ids, ps = probs_to_top_k(probs, mask, k=16)
    assert ids.shape == (16,)
    # Top-3 must be the actual probs in descending order
    assert ids[0] == 10 and abs(ps[0] - 0.5) < 1e-6
    assert ids[1] == 20 and abs(ps[1] - 0.3) < 1e-6
    assert ids[2] == 30 and abs(ps[2] - 0.2) < 1e-6


@pytest.mark.skipif(not _HAS_TORCH, reason="torch+cuda not available")
def test_record_game_with_policy_smoke():
    """Run a random untrained net for <=20 steps, save + reload."""
    from junqi_rl.analysis import record_game_with_policy
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    cfg = JunqiNetConfig(
        cnn_channels=16, cnn_layers=2, depth=1,
        embed_dim=32, n_head=2, ff_factor=2, dropout=0.0,
    )
    policy = JunqiNet(cfg).to("cuda").eval()
    traj = record_game_with_policy(
        policy, rng_seed=1, device="cuda", max_steps=20, top_k=8,
    )
    # Must have run ≥ 1 step
    assert traj.num_steps >= 1
    assert traj.top_action_ids.shape == (traj.num_steps, 8)
    # Prob rows should sum (approximately) to at most 1 (some may sum to
    # less because of the top-K truncation).
    row_sums = traj.top_probs.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-3).all()
    assert (row_sums >= 0.0).all()

    # Roundtrip
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "recording.npz"
        traj.save(p)
        loaded = TrajectoryWithPolicy.load(p)
    assert loaded.num_steps == traj.num_steps

    # replay_with_records yields (state, move_result, step_record)
    n_yielded = 0
    for post_state, mr, rec in traj.replay_with_records():
        n_yielded += 1
        assert rec.step == n_yielded - 1
        assert rec.top_action_ids.shape == (8,)
        assert 0 <= rec.acting_seat < 4
    assert n_yielded == traj.num_steps
