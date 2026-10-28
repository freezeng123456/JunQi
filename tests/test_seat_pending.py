"""Cross-rollout seat transitions retain the original sampling contract."""
import io

import pytest
import torch

from junqi_core.board import BOARD_SIZE, COMPACT_ACTION_DIM
from junqi_core.observation import OBS_CHANNELS
from junqi_rl.training.seat_pending import PendingSeatTransitions


def decision(seats=(0, 2), marker=1.0, global_dims=0):
    n = len(seats)
    spatial = torch.full((n, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), marker, dtype=torch.float32)
    global_obs = torch.full((n, global_dims), marker + 1, dtype=torch.float32)
    legal = torch.zeros(n, COMPACT_ACTION_DIM, dtype=torch.bool)
    legal[:, [3, 17, COMPACT_ACTION_DIM - 1]] = True
    logits = torch.full((n, COMPACT_ACTION_DIM), -torch.inf)
    logits[:, [3, 17, COMPACT_ACTION_DIM - 1]] = torch.tensor([0.000137, 0.734561, -1.234567])
    behavior = logits.log_softmax(-1)
    actions = torch.full((n,), 17, dtype=torch.int32)
    return dict(obs_spatial=spatial, obs_global=global_obs, legal_mask=legal,
                actions=actions, log_probs=behavior[:, 17].clone(),
                values=torch.arange(n, dtype=torch.float32) / 7,
                seats=torch.tensor(seats, dtype=torch.int8), behavior_log_probs=behavior)


def assert_snapshot_equal(left, right):
    for key in PendingSeatTransitions.TENSOR_FIELDS:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_record_updates_only_acting_seats_with_original_behavior_distribution():
    pending = PendingSeatTransitions(2, "cpu", torch.float16, 3)
    first = decision(global_dims=3)
    pending.record(**first)
    assert pending.valid.tolist() == [[True, False], [False, False], [False, True], [False, False]]
    rows, envs = first["seats"].long(), torch.arange(2)
    assert pending.actions.dtype == torch.int32
    assert pending.seats.dtype == torch.int8
    assert pending.obs_spatial.dtype == pending.obs_global.dtype == torch.float16
    torch.testing.assert_close(pending.behavior_log_probs[rows, envs], first["behavior_log_probs"], rtol=0, atol=0)
    torch.testing.assert_close(pending.log_probs[rows, envs], first["log_probs"], rtol=0, atol=0)
    assert not torch.equal(first["behavior_log_probs"], first["behavior_log_probs"].half().float())
    pending.record(**decision(seats=(1, 0), marker=2, global_dims=3))
    assert pending.valid.sum().item() == 4
    assert pending.obs_spatial[0, 0, 0, 0, 0].item() == 1
    assert pending.obs_spatial[1, 0, 0, 0, 0].item() == 2
    # Reusing a simulator output must not overwrite a recorded action.
    first["obs_spatial"].zero_()
    first["behavior_log_probs"].zero_()
    assert pending.obs_spatial[0, 0, 0, 0, 0].item() == 1
    assert pending.log_probs[0, 0].item() < 0


def test_begin_rollout_is_an_independent_four_row_snapshot():
    pending = PendingSeatTransitions(2, "cpu", torch.float16, 0)
    pending.record(**decision())
    snapshot = pending.begin_rollout()
    frozen = {name: value.clone() for name, value in snapshot.items()}
    pending.record(**decision(marker=9))
    pending.terminal(torch.tensor([True, True]))
    assert_snapshot_equal(snapshot, frozen)
    snapshot["obs_spatial"].fill_(12)
    snapshot["valid"].fill_(True)
    assert not pending.valid.any()
    assert pending.obs_spatial[0, 0, 0, 0, 0].item() == 9


def test_only_game_terminal_clears_slots_not_elimination_or_absent_turns():
    pending = PendingSeatTransitions(2, "cpu")
    for seat in range(4):
        pending.record(**decision(seats=(seat, seat), marker=seat + 1))
    # A seat can stop receiving turns after elimination. Its last action
    # remains pending until the game reward is available.
    pending.record(**decision(seats=(2, 3), marker=20))
    pending.terminal(torch.tensor([False, False]))
    assert pending.valid.all()
    assert pending.obs_spatial[1, 0, 0, 0, 0].item() == 2
    pending.terminal(torch.tensor([True, False]))
    assert not pending.valid[:, 0].any()
    assert pending.valid[:, 1].all()
    pending.clear()
    assert not pending.valid.any()


def test_state_dict_round_trip_preserves_pending_exactly_and_owns_storage():
    pending = PendingSeatTransitions(2, "cpu", torch.float16, 2)
    pending.record(**decision(global_dims=2))
    pending.record(**decision(seats=(3, 1), marker=3, global_dims=2))
    saved = pending.state_dict()
    assert saved["schema"] == PendingSeatTransitions.SCHEMA
    assert all(saved[key].device.type == "cpu" for key in PendingSeatTransitions.TENSOR_FIELDS)
    stream = io.BytesIO()
    torch.save(saved, stream)
    stream.seek(0)
    restored_state = torch.load(stream, weights_only=True)
    restored = PendingSeatTransitions(2, "cpu", torch.float16, 2)
    restored.load_state_dict(restored_state)
    assert_snapshot_equal(restored.begin_rollout(), pending.begin_rollout())
    restored_state["obs_spatial"].zero_()
    restored_state["valid"].zero_()
    saved["behavior_log_probs"].zero_()
    saved["valid"].zero_()
    assert restored.valid.sum().item() == pending.valid.sum().item() == 4
    assert_snapshot_equal(restored.begin_rollout(), pending.begin_rollout())
    next_action = decision(seats=(2, 0), marker=7, global_dims=2)
    pending.record(**next_action)
    restored.record(**next_action)
    assert_snapshot_equal(restored.begin_rollout(), pending.begin_rollout())


@pytest.mark.parametrize("corruption", [
    "missing", "schema", "num_envs", "dtype", "shape", "seats", "action",
    "illegal", "chosen_log_prob", "nan", "unnormalized", "illegal_mass",
])
def test_restore_rejects_invalid_state_before_mutating_live_cache(corruption):
    source = PendingSeatTransitions(2, "cpu")
    source.record(**decision())
    bad = source.state_dict()
    if corruption == "missing":
        del bad["behavior_log_probs"]
    elif corruption == "schema":
        bad["schema"] = "unknown"
    elif corruption == "num_envs":
        bad["num_envs"] = 2.0
    elif corruption == "dtype":
        bad["behavior_log_probs"] = bad["behavior_log_probs"].half()
    elif corruption == "shape":
        bad["valid"] = bad["valid"].T
    elif corruption == "seats":
        bad["seats"][0, 0] = 3
    elif corruption == "action":
        bad["actions"][0, 0] = COMPACT_ACTION_DIM
    elif corruption == "illegal":
        bad["legal_mask"][0, 0, 17] = False
    elif corruption == "chosen_log_prob":
        bad["log_probs"][0, 0] += .01
    elif corruption == "nan":
        bad["values"][0, 0] = torch.nan
    elif corruption == "unnormalized":
        bad["behavior_log_probs"][0, 0] += .1
    elif corruption == "illegal_mass":
        bad["behavior_log_probs"][0, 0] = torch.full((COMPACT_ACTION_DIM,), -float(COMPACT_ACTION_DIM)).log_softmax(-1)
    target = PendingSeatTransitions(2, "cpu")
    target.record(**decision(seats=(3, 1), marker=8))
    before = target.begin_rollout()
    with pytest.raises(ValueError, match="pending state"):
        target.load_state_dict(bad)
    assert_snapshot_equal(target.begin_rollout(), before)


def test_empty_and_cleared_state_restore_without_fabricating_pending():
    source = PendingSeatTransitions(2, "cpu")
    source.record(**decision())
    source.clear()
    target = PendingSeatTransitions(2, "cpu")
    target.load_state_dict(source.state_dict())
    assert not target.valid.any()
    source = PendingSeatTransitions(2, "cpu")
    target.load_state_dict(source.state_dict())
    assert_snapshot_equal(target.begin_rollout(), source.begin_rollout())


def test_record_rejects_low_precision_behavior_and_wrong_shapes_atomically():
    pending = PendingSeatTransitions(2, "cpu")
    data = decision()
    data["behavior_log_probs"] = data["behavior_log_probs"].half()
    with pytest.raises(ValueError, match="behavior_log_probs"):
        pending.record(**data)
    assert not pending.valid.any()
    data = decision()
    data["seats"] = torch.zeros(2, 1, dtype=torch.int8)
    with pytest.raises(ValueError, match="seats"):
        pending.record(**data)
    assert not pending.valid.any()
