from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.networks.junqi_net import FLAT_ACTION_DIM, JunqiNet, JunqiNetConfig


def _tiny_model() -> JunqiNet:
    cfg = JunqiNetConfig(
        cnn_channels=8,
        cnn_layers=1,
        depth=1,
        embed_dim=32,
        n_head=2,
        ff_factor=2,
        action_key_dim=8,
    )
    return JunqiNet(cfg).eval()


def test_forward_evaluates_supplied_actions_without_sampling(monkeypatch) -> None:
    model = _tiny_model()
    batch_size = 2
    spatial = torch.zeros(batch_size, OBS_CHANNELS, 17, 17)
    global_ = torch.zeros(batch_size, OBS_GLOBAL_DIMS)
    legal = torch.zeros(batch_size, FLAT_ACTION_DIM, dtype=torch.bool)
    legal[0, [3, 11]] = True
    legal[1, [7, 19]] = True
    actions = torch.tensor([11, 7])

    def fail_if_sampled(*_args, **_kwargs):
        raise AssertionError("training evaluation must not sample an action")

    monkeypatch.setattr(torch.distributions.Categorical, "sample", fail_if_sampled)
    out = model(spatial, global_, legal, actions=actions)

    assert set(out) == {"action", "action_log_prob", "log_probs", "value"}
    assert torch.equal(out["action"], actions.to(torch.int32))
    expected = out["log_probs"].gather(1, actions[:, None]).squeeze(1)
    assert torch.equal(out["action_log_prob"], expected)


def test_supplied_action_path_matches_legacy_forward_and_preserves_rng() -> None:
    model = _tiny_model()
    batch_size = 2
    spatial = torch.zeros(batch_size, OBS_CHANNELS, 17, 17)
    global_ = torch.zeros(batch_size, OBS_GLOBAL_DIMS)
    legal = torch.zeros(batch_size, FLAT_ACTION_DIM, dtype=torch.bool)
    legal[0, [3, 11, 29]] = True
    legal[1, [7, 19, 31]] = True

    # The original three-input API remains a sampling call with the same four
    # outputs. Its selected actions become the fixed inputs for the fast path.
    torch.manual_seed(20260826)
    with torch.inference_mode():
        legacy = model(spatial, global_, legal)

    rng_before = torch.random.get_rng_state().clone()
    with torch.inference_mode():
        evaluated = model(
            spatial,
            global_,
            legal,
            actions=legacy["action"].to(torch.long),
        )
    rng_after = torch.random.get_rng_state()

    assert set(legacy) == {"action", "action_log_prob", "log_probs", "value"}
    assert set(evaluated) == set(legacy)
    assert torch.equal(evaluated["action"], legacy["action"])
    assert torch.equal(evaluated["log_probs"], legacy["log_probs"])
    assert torch.equal(evaluated["action_log_prob"], legacy["action_log_prob"])
    assert torch.equal(evaluated["value"], legacy["value"])
    assert torch.equal(rng_after, rng_before)


def test_state_dict_strict_load_keeps_fast_path_outputs() -> None:
    model = _tiny_model()
    restored = _tiny_model()
    restored.load_state_dict(model.state_dict(), strict=True)

    spatial = torch.zeros(1, OBS_CHANNELS, 17, 17)
    global_ = torch.zeros(1, OBS_GLOBAL_DIMS)
    legal = torch.zeros(1, FLAT_ACTION_DIM, dtype=torch.bool)
    legal[0, [5, 13]] = True
    actions = torch.tensor([13])

    with torch.inference_mode():
        expected = model(spatial, global_, legal, actions=actions)
        actual = restored(spatial, global_, legal, actions=actions)

    assert model.state_dict().keys() == restored.state_dict().keys()
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_forward_rejects_mismatched_supplied_action_batch() -> None:
    model = _tiny_model()
    spatial = torch.zeros(2, OBS_CHANNELS, 17, 17)
    global_ = torch.zeros(2, OBS_GLOBAL_DIMS)
    legal = torch.ones(2, FLAT_ACTION_DIM, dtype=torch.bool)

    with pytest.raises(ValueError, match="actions must have shape"):
        model(spatial, global_, legal, actions=torch.tensor([0]))
