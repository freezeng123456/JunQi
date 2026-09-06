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


def _tiny_net() -> JunqiNet:
    return JunqiNet(
        JunqiNetConfig(
            cnn_channels=16,
            cnn_layers=1,
            depth=1,
            embed_dim=16,
            n_head=4,
            ff_factor=2,
            action_key_dim=8,
        )
    )


def test_checkpoint_metadata_tracks_runtime_schema() -> None:
    net = _tiny_net()
    metadata = current_checkpoint_metadata(net)

    assert metadata["observation_channels"] == OBS_CHANNELS
    assert metadata["action_dim"] == COMPACT_ACTION_DIM
    assert metadata["stem_class"] == "GraphStem"


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
