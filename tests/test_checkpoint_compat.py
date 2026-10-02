from __future__ import annotations

import pytest
import torch

from junqi_core.board import COMPACT_ACTION_DIM
from junqi_core.observation import OBS_CHANNELS
from junqi_rl.checkpoint_compat import (
    current_checkpoint_metadata,
    validate_policy_checkpoint,
)
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def _tiny_net(**kwargs) -> JunqiNet:
    return JunqiNet(
        JunqiNetConfig(
            cnn_channels=16,
            cnn_layers=1,
            depth=1,
            embed_dim=16,
            n_head=4,
            ff_factor=2,
            action_key_dim=8,
            **kwargs,
        )
    )


def test_checkpoint_metadata_tracks_runtime_schema() -> None:
    net = _tiny_net()
    metadata = current_checkpoint_metadata(net)

    assert metadata["observation_channels"] == OBS_CHANNELS
    assert metadata["action_dim"] == COMPACT_ACTION_DIM
    assert metadata["stem_class"] == "GraphStem"
    assert metadata["observation_semantics_version"] == 4


def test_matching_legacy_checkpoint_without_metadata_is_accepted() -> None:
    net = _tiny_net()
    validate_policy_checkpoint(net, {"policy": net.state_dict()})


def test_checkpoint_shape_mismatch_fails_before_load_state_dict() -> None:
    net = _tiny_net()
    state = {key: value.clone() for key, value in net.state_dict().items()}
    weight = state["stem.embed.weight"]
    state["stem.embed.weight"] = torch.empty(
        weight.shape[0], weight.shape[1] + 1, dtype=weight.dtype
    )

    with pytest.raises(ValueError, match=r"stem\.embed\.weight"):
        validate_policy_checkpoint(net, {"policy": state}, source="legacy baseline")


def test_checkpoint_metadata_mismatch_is_rejected() -> None:
    net = _tiny_net()
    metadata = current_checkpoint_metadata(net)
    metadata["observation_channels"] = 412

    with pytest.raises(ValueError, match="observation_channels"):
        validate_policy_checkpoint(
            net,
            {"policy": net.state_dict(), "checkpoint_meta": metadata},
        )


def test_observation_semantics_mismatch_is_rejected():
    net = _tiny_net()
    metadata = current_checkpoint_metadata(net)
    metadata["observation_semantics_version"] = 3
    with pytest.raises(ValueError, match="observation_semantics_version"):
        validate_policy_checkpoint(net, {"policy": net.state_dict(), "checkpoint_meta": metadata})


def test_unversioned_weights_report_semantic_uncertainty():
    net = _tiny_net()
    with pytest.warns(UserWarning, match="does not reproduce"):
        validate_policy_checkpoint(net, {"policy": net.state_dict()})


@pytest.fixture
def facts_net() -> JunqiNet:
    return _tiny_net(identity_encoding="known_only", use_global_summary=False)


@pytest.mark.parametrize("missing_field", [
    "format_version", "architecture_id", "identity_encoding", "policy_class",
    "observation_channels", "observation_global_dims", "global_summary_enabled",
    "input_layout_sha256", "policy_features", "action_dim", "num_on_board_cells",
    "observation_semantics_version", "combat_feature_version",
    "tactical_feature_version", "relational_feature_version", "stem_class",
])
def test_facts_checkpoint_requires_complete_metadata(facts_net, missing_field):
    metadata = current_checkpoint_metadata(facts_net)
    del metadata[missing_field]
    with pytest.raises(ValueError, match=f"missing required metadata field '{missing_field}'"):
        validate_policy_checkpoint(facts_net, {
            "policy": facts_net.state_dict(), "checkpoint_meta": metadata,
        })


@pytest.mark.parametrize("metadata", [None, {}, {"identity_encoding": "known_only"}])
def test_identity_marker_alone_cannot_establish_facts_compatibility(facts_net, metadata):
    checkpoint = {"policy": facts_net.state_dict()}
    if metadata is not None:
        checkpoint["checkpoint_meta"] = metadata
    with pytest.raises(ValueError, match="missing required metadata field"):
        validate_policy_checkpoint(facts_net, checkpoint)


@pytest.mark.parametrize(("field", "bad_value"), [
    ("format_version", 2.0), ("format_version", True),
    ("architecture_id", None), ("identity_encoding", None),
    ("observation_channels", 317.0), ("observation_global_dims", False),
    ("global_summary_enabled", 0), ("input_layout_sha256", b"hash"),
    ("policy_features", []), ("policy_features", {"combat": True}),
    ("action_dim", 16641.0), ("num_on_board_cells", 129.0),
    ("observation_semantics_version", 5.0), ("combat_feature_version", False),
    ("tactical_feature_version", 0.0), ("relational_feature_version", False),
    ("policy_class", None), ("stem_class", None),
])
def test_facts_checkpoint_rejects_metadata_type_coercion(facts_net, field, bad_value):
    metadata = current_checkpoint_metadata(facts_net)
    metadata[field] = bad_value
    with pytest.raises(ValueError, match=f"{field}: invalid metadata type"):
        validate_policy_checkpoint(facts_net, {
            "policy": facts_net.state_dict(), "checkpoint_meta": metadata,
        })


def test_facts_export_reload_preserves_weights_and_contract(facts_net, tmp_path):
    from junqi_rl.training.checkpoint import export_evaluation_policy, load_evaluation_policy

    path = export_evaluation_policy(facts_net, tmp_path / "facts.pt")
    loaded = load_evaluation_policy(path)
    assert current_checkpoint_metadata(loaded) == current_checkpoint_metadata(facts_net)
    for key, tensor in facts_net.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[key], rtol=0, atol=0)
    state = torch.load(path, weights_only=True)
    del state["checkpoint_meta"]["input_layout_sha256"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="missing required metadata field 'input_layout_sha256'"):
        load_evaluation_policy(path)


def test_legacy_metadata_cannot_hide_malformed_types():
    net = _tiny_net()
    metadata = current_checkpoint_metadata(net)
    metadata["policy_features"] = {"combat": True}
    with pytest.raises(ValueError, match="policy_features: invalid metadata type"):
        validate_policy_checkpoint(net, {"policy": net.state_dict(), "checkpoint_meta": metadata})
