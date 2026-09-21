from __future__ import annotations

import json

from junqi_core.replay import record_trajectory
from junqi_core.replay_with_policy import TrajectoryWithPolicy
from junqi_viz.replay_data import ReplayData
from junqi_core.rules import Seat, ShowMode


def test_player_frame_masks_hidden_identities_on_board_and_move_detail(tmp_path):
    path = tmp_path / "dark.npz"
    record_trajectory(rng_seed=44, max_steps=8, show_mode=ShowMode.DARK).save(path)
    data = ReplayData(path)
    for observer in Seat:
        opening = data.frame(0, observer=observer)
        for piece in opening["pieces"]:
            assert (piece["type"] != "DARK") == (piece["seat"] == observer.name)
        for step in (1, 6, 2, 0):
            frame = data.frame(step, observer=observer)
            assert frame["observer"] == observer.name
            assert frame["policy"] is None
            if frame["move_detail"]:
                for key in ("src_piece", "dst_piece"):
                    piece = frame["move_detail"][key]
                    if piece and piece["seat"] != observer.name:
                        assert piece["type"] == "DARK"


def test_player_metadata_does_not_include_private_or_future_diagnostics(tmp_path):
    path = tmp_path / "metadata.npz"
    trajectory = record_trajectory(rng_seed=9, max_steps=60, show_mode=ShowMode.DARK)
    trajectory.meta = {"private_setup": ["SILING"], "secret_analysis": "hidden"}
    trajectory.save(path)
    data = ReplayData(path)
    meta = data.metadata(observer=Seat.SOUTH)
    assert meta["meta"] == {}
    assert meta["key_events"] == []
    assert not meta["has_beliefs"]
    assert data.frame(1, observer=Seat.SOUTH)["policy"] is None
    assert all(e["step"] <= 1 for e in data.frame(1, observer=Seat.SOUTH)["key_events"])
    # Full diagnostic replay remains an explicit, separate view.
    assert data.metadata()["meta"]["private_setup"] == ["SILING"]


def test_mutual_removal_label_does_not_claim_a_bomb_was_seen():
    from junqi_viz.replay_data import EVENT_LABELS
    # Equal ordinary ranks also produce Event.BOMB; the public event does not
    # establish that either combatant was a bomb.
    assert EVENT_LABELS["BOMB"] == "同归于尽"


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
    assert "key_events" in meta
    assert frame["length"] == trajectory.num_steps
    assert frame["pieces"]
    assert set(frame["seat_info"]) == {"SOUTH", "WEST", "NORTH", "EAST"}
    assert "move_counter" in frame["state"]
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
    meta = data.metadata()
    assert meta["kind"] == "policy"
    assert "key_events" in meta
    if t:
        frame = data.frame(1)
        assert frame["policy"]["action_source"] == "policy_sample"
        assert frame["policy"]["source_label"] == "策略采样"
        assert len(frame["policy"]["top_actions"]) == 2
        assert frame["policy"]["top_actions"][0]["rank"] == 1
