"""Optional execution, numeric parity and checkpoint contracts for obs317 v4."""

from __future__ import annotations

import copy
import sys
import types
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from junqi_core.observation import CHANNEL_LAYOUT, OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.checkpoint_compat import current_checkpoint_metadata, validate_policy_checkpoint
from junqi_rl.networks import junqi_net as net_module
from junqi_rl.networks.feature_context import PolicyFeatureContext
from junqi_rl.networks.junqi_net import FLAT_ACTION_DIM, JunqiNet, JunqiNetConfig
from junqi_rl.training.checkpoint import export_evaluation_policy, load_evaluation_policy


def tiny(**kwargs):
    return JunqiNet(JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1,
        embed_dim=16, n_head=2, ff_factor=2, action_key_dim=8, **kwargs)).eval()


def inputs(batch=3):
    spatial = torch.rand(batch, OBS_CHANNELS, 17, 17)
    global_ = torch.rand(batch, OBS_GLOBAL_DIMS)
    legal = torch.zeros(batch, FLAT_ACTION_DIM, dtype=torch.bool)
    legal[:, [5, 13, 25]] = True
    actions = torch.full((batch,), 13, dtype=torch.long)
    return spatial, global_, legal, actions


class LegacyGatherContext(PolicyFeatureContext):
    """Independent oracle: the previous heads gathered all 317 channels."""

    def __init__(self, spatial, on_board):
        super().__init__(spatial, on_board)
        self.flat = spatial.flatten(-2).index_select(-1, on_board).float()

    def channels(self, name):
        return self.flat[:, CHANNEL_LAYOUT[name]]


@pytest.mark.parametrize("features", [
    {}, {"combat": 1}, {"tactical": 2}, {"relational": 1},
    {"combat": 1, "tactical": 2, "relational": 1},
])
def test_generic_configuration_preserves_legacy_weights_and_outputs(features):
    flags = {"combat_outcome_features": "combat" in features,
             "tactical_features": "tactical" in features,
             "relational_features": "relational" in features}
    torch.manual_seed(317)
    legacy = tiny(**flags)
    torch.manual_seed(317)
    generic = tiny(policy_features=features)
    assert list(legacy.state_dict()) == list(generic.state_dict())
    assert list(dict(legacy.named_parameters())) == list(dict(generic.named_parameters()))
    for k, v in legacy.state_dict().items():
        assert torch.equal(v, generic.state_dict()[k]), k
    sp, gl, legal, actions = inputs()
    with torch.no_grad():
        a, b = legacy(sp, gl, legal, actions=actions), generic(sp, gl, legal, actions=actions)
    for key in a:
        assert torch.equal(a[key], b[key]), key
    assert current_checkpoint_metadata(generic)["policy_features"] == features


def test_disabled_features_allocate_no_context_or_extra_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled features must not be constructed or computed")
    monkeypatch.setattr(net_module, "PolicyFeatureContext", forbidden)
    from junqi_rl.feature_specs import PolicyFeatureSpec
    monkeypatch.setattr(PolicyFeatureSpec, "build", forbidden)
    model = tiny()
    sp, gl, legal, actions = inputs()
    model(sp, gl, legal, actions=actions)
    model.act(sp, gl, legal)
    model.act_greedy(sp, gl, legal)
    assert not any(k.startswith(("combat_head.", "tactical_head.", "relational_head."))
                   for k in model.state_dict())


@pytest.mark.parametrize("features", [{"tactical": 2}, {"relational": 1}])
def test_shared_policy_value_path_supports_features_without_combat(features, monkeypatch):
    model = tiny(policy_features=features)
    with torch.no_grad():
        for attribute, _ in model._feature_plan:
            for p in getattr(model, attribute).parameters():
                p.uniform_(-0.1, 0.1)
    sp, gl, legal, actions = inputs()
    indices = torch.tensor([2, 0])
    with torch.no_grad():
        expected = model(sp[indices], gl[indices], legal[indices], actions=actions[indices])
        actual = model.forward_policy_value_shared(
            sp, gl, indices, legal[indices], actions=actions[indices])
    for key in ("action", "action_log_prob", "log_probs"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-6, rtol=2e-6)
    def forbidden(*args, **kwargs):
        raise AssertionError("value-only rows must not execute policy features")
    monkeypatch.setattr(net_module, "PolicyFeatureContext", forbidden)
    with torch.no_grad():
        torch.testing.assert_close(model.forward_value(sp, gl), actual["value"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("feature", ["combat", "tactical", "relational"])
def test_partial_channel_gather_matches_full_gather_outputs_and_gradients(feature, dtype):
    model = tiny(policy_features={feature: 2 if feature == "tactical" else 1})
    head = getattr(model, feature + "_head")
    with torch.no_grad():
        for p in head.parameters():
            p.uniform_(-0.2, 0.2)  # Exercise learned residuals, not zero init.
    reference = copy.deepcopy(head)
    sp, _, legal, _ = inputs(2)
    sp = sp.to(dtype)
    old_context = LegacyGatherContext(sp, model.on_board_idx)
    context = PolicyFeatureContext(sp, model.on_board_idx,
        channel_slices=model._feature_channel_slices,
        gather_indices=model._feature_gather_indices)
    args = (sp, legal) if feature == "relational" else (sp,)
    expected = reference(*args, context=old_context)
    actual = head(*args, context=context)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for p, q in zip(head.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-7)


def test_context_shares_only_requested_channels_and_never_crosses_forwards():
    model = tiny(policy_features={"tactical": 2, "relational": 1})
    sp, _, legal, _ = inputs(1)
    context = PolicyFeatureContext(sp, model.on_board_idx)
    model.tactical_head(sp, context=context)
    own = context.channels("piece_own")
    pieces = context.pieces()
    model.relational_head(sp, legal, context=context)
    assert context.channels("piece_own") is own
    assert context.pieces() is pieces
    assert set(context._channels) == {
        "piece_own", "piece_left_side_enemy", "piece_right_side_enemy",
        "belief_left_side", "belief_right_side", "prob_teammate",
        "dark_teammate", "cm_my_is_gongb",
    }
    assert sum(t.shape[1] for t in context._channels.values()) == 52
    next_context = PolicyFeatureContext(torch.zeros_like(sp), model.on_board_idx)
    assert torch.count_nonzero(next_context.channels("piece_own")) == 0
    assert torch.count_nonzero(own) > 0


def test_inference_export_roundtrip_and_legacy_raw_weights(tmp_path):
    model = tiny(policy_features={"combat": 1, "tactical": 2})
    path = export_evaluation_policy(model, tmp_path / "policy.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert "optimizer" not in state
    assert not state["capabilities"]["training_resume"]
    restored = load_evaluation_policy(path)
    assert current_checkpoint_metadata(model) == current_checkpoint_metadata(restored)
    for k, v in model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k])
    raw_path = tmp_path / "raw.pt"
    torch.save(model.state_dict(), raw_path)
    with pytest.raises(ValueError, match="missing train_cfg.net"):
        load_evaluation_policy(raw_path)
    with pytest.warns(UserWarning, match="legacy observation"):
        raw = load_evaluation_policy(raw_path, net_config=asdict(model.cfg))
    for k, v in model.state_dict().items():
        assert torch.equal(v, raw.state_dict()[k])


@pytest.mark.parametrize("field,value", [
    ("policy_features", {"tactical": 1}), ("format_version", 999),
    ("input_layout_sha256", "different"), ("observation_global_dims", 29),
    ("architecture_id", "unknown"), ("tactical_feature_version", 1),
])
def test_semantic_mismatch_is_rejected_before_weight_load(field, value):
    model = tiny(policy_features={"tactical": 2})
    metadata = current_checkpoint_metadata(model)
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        validate_policy_checkpoint(model, {"policy": model.state_dict(),
                                           "checkpoint_meta": metadata})


def test_supplied_config_cannot_override_embedded_checkpoint_config(tmp_path):
    model = tiny(tactical_features=True)
    path = export_evaluation_policy(model, tmp_path / "policy.pt")
    equivalent = asdict(model.cfg)
    equivalent.update(tactical_features=False, policy_features={"tactical": 2})
    load_evaluation_policy(path, net_config=equivalent)
    equivalent["policy_features"] = {}
    with pytest.raises(ValueError, match="conflicts"):
        load_evaluation_policy(path, net_config=equivalent)


def test_load_keeps_optimizer_and_source_weights_off_gpu(tmp_path, monkeypatch):
    model = tiny()
    path = export_evaluation_policy(model, tmp_path / "policy.pt")
    calls = []
    original = torch.load
    def tracked(*args, **kwargs):
        calls.append(kwargs.get("map_location"))
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", tracked)
    load_evaluation_policy(path, device="cpu")
    assert calls == ["cpu"]


def test_feature_requests_are_validated_before_model_allocation():
    from junqi_rl.training.config import TrainConfig, validate_config
    cfg = TrainConfig()
    cfg.net.policy_features = {"tactical": 1}
    with pytest.raises(ValueError, match="feature version"):
        validate_config(cfg)


def test_shared_gather_is_built_once_and_not_serialized():
    model = tiny(policy_features={"combat": 1, "tactical": 2, "relational": 1})
    assert model._feature_gather_indices.numel() == 52 * 129
    assert "_feature_gather_indices" not in model.state_dict()
    sp, _, legal, _ = inputs(2)
    context = PolicyFeatureContext(sp, model.on_board_idx,
        channel_slices=model._feature_channel_slices,
        gather_indices=model._feature_gather_indices)
    model.combat_head(sp, context=context)
    shared = context._gathered_raw
    model.tactical_head(sp, context=context)
    model.relational_head(sp, legal, context=context)
    assert context._gathered_raw is shared
    for name in model._feature_channel_slices:
        expected = sp.flatten(-2)[:, CHANNEL_LAYOUT[name]].index_select(-1, model.on_board_idx)
        torch.testing.assert_close(context.raw_channels(name), expected, rtol=0, atol=0)
    assert tiny()._feature_gather_indices is None


def test_facts_learned_extension_is_lazy_trainable_and_reloadable(monkeypatch, tmp_path):
    from junqi_rl import feature_specs
    builds, contexts = [], []

    class LearnedFactsHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            builds.append(self)
            self.projection = torch.nn.Linear(12, 129, bias=False)

        def forward(self, spatial, *, context):
            contexts.append(context)
            own = context.channels("piece_own").transpose(1, 2)
            return self.projection(own).flatten(1)

    module = types.ModuleType("_facts_test_extension")
    module.Head = LearnedFactsHead
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(feature_specs, "_FEATURES", dict(feature_specs._FEATURES))
    feature_specs.register_policy_feature(feature_specs.PolicyFeatureSpec(
        "learned_facts", 1, "learned_facts_head", module.__name__, "Head",
        channels=("piece_own",), supported_identity_encodings=("known_only",)))
    cfg = dict(identity_encoding="known_only", use_global_summary=False)
    disabled = tiny(**cfg)
    assert not builds and disabled._feature_gather_indices is None
    model = tiny(**cfg, policy_features={"learned_facts": 1})
    assert len(builds) == 1
    assert model._feature_gather_indices.numel() == 12 * 129
    spatial, _, legal, actions = inputs(2)
    globals_ = torch.empty(2, 0)
    result = model(spatial, globals_, legal, actions=actions)
    (-result["action_log_prob"].mean()).backward()
    grad = model.learned_facts_head.projection.weight.grad
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    assert set(contexts[0]._channels) == {"piece_own"}
    assert contexts[0]._gathered_raw.shape == (2, 12, 129)
    path = export_evaluation_policy(model, tmp_path / "facts-extension.pt")
    restored = load_evaluation_policy(path)
    assert current_checkpoint_metadata(restored)["policy_features"] == {"learned_facts": 1}
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    with torch.no_grad():
        result = model(spatial, globals_, legal, actions=actions)
        reloaded = restored(spatial, globals_, legal, actions=actions)
    for key in result:
        torch.testing.assert_close(result[key], reloaded[key], rtol=0, atol=0)
