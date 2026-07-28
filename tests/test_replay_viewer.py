"""Headless manual/RL replay adapter regressions."""

from __future__ import annotations

import random

import numpy as np

from junqi_core.replay import record_trajectory
from junqi_core.replay_viewer import ReplayViewer, frame_summary
from junqi_core.replay_with_policy import TrajectoryWithPolicy
from junqi_core.rules import Seat
from junqi_core.setup import generate_random_setup


def test_cursor_seek_and_reverse_are_deterministic() -> None:
    traj = record_trajectory(rng_seed=11, max_steps=12)
    viewer = ReplayViewer(traj)
    assert viewer.seek(0).step == 0
    at_three = viewer.seek(3)
    at_end = viewer.seek(traj.num_steps)
    assert at_end.step == traj.num_steps
    assert viewer.previous().step == max(0, traj.num_steps - 1)
    assert viewer.seek(3).state.state_hash() == at_three.state.state_hash()


def test_policy_replay_exposes_step_alignment() -> None:
    setups = generate_random_setup(random.Random(3))
    base = record_trajectory(setups=setups, rng_seed=3, max_steps=4)
    t = base.num_steps
    policy = TrajectoryWithPolicy(
        setups=setups,
        actions=base.actions,
        rng_seed=base.rng_seed,
        final_state_hash=base.final_state_hash,
        top_action_ids=np.zeros((t, 2), dtype=np.int32),
        top_probs=np.zeros((t, 2), dtype=np.float32),
        values=np.arange(t, dtype=np.float32),
        acting_seats=base.actions[:, 0].astype(np.int8),
    )
    viewer = ReplayViewer(policy)
    assert viewer.seek(0).policy is None
    if t:
        assert viewer.seek(1).policy is not None
        assert viewer.seek(1).policy.value == 0.0


def test_frame_summary_is_json_friendly() -> None:
    traj = record_trajectory(rng_seed=2, max_steps=1)
    data = frame_summary(ReplayViewer(traj).seek(0))
    assert data["step"] == 0
    assert set(data["alive_pieces"]) == {s.name for s in Seat}
