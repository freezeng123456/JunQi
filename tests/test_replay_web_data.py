from __future__ import annotations

import json

from junqi_core.replay import record_trajectory
from junqi_core.replay_with_policy import TrajectoryWithPolicy
from junqi_viz.replay_data import ReplayData


def test_manual_replay_payload_is_json_safe(tmp_path) -> None:
    path = tmp_path / "manual.npz"
    trajectory = record_trajectory(rng_seed=44, max_steps=5)
    trajectory.save(path)

    data = ReplayData(path)
    meta = data.metadata()
    frame = data.frame(min(1, data.length))
    assert meta["kind"] == "manual"
    assert meta["length"] == trajectory.num_steps
    assert len(meta["cells"]) == 129
    assert frame["length"] == trajectory.num_steps
    assert frame["pieces"]
    json.dumps(meta, ensure_ascii=False)
    json.dumps(frame, ensure_ascii=False)


def test_policy_replay_payload_exposes_action_source_and_top_moves(tmp_path) -> None:
    path = tmp_path / "policy.npz"
    base = record_trajectory(rng_seed=9, max_steps=2)
    t = base.num_steps
    import numpy as np

    policy = TrajectoryWithPolicy(
        setups=base.setups,
        actions=base.actions,
        rng_seed=base.rng_seed,
        final_state_hash=base.final_state_hash,
        top_action_ids=np.tile(np.array([[0, 1]], dtype=np.int32), (t, 1)),
        top_probs=np.tile(np.array([[0.7, 0.3]], dtype=np.float32), (t, 1)),
        values=np.zeros(t, dtype=np.float32),
        acting_seats=base.actions[:, 0].astype(np.int8),
    )
    policy.save(path)
    data = ReplayData(path)
    assert data.metadata()["kind"] == "policy"
    if t:
        frame = data.frame(1)
        assert frame["policy"]["action_source"] == "policy_sample"
        assert len(frame["policy"]["top_actions"]) == 2
