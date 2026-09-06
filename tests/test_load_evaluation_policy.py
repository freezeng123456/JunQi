from __future__ import annotations

from junqi_rl.checkpoint_compat import current_checkpoint_metadata
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.checkpoint import load_evaluation_policy, save_checkpoint
from junqi_rl.training.config import TrainConfig


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


class _FakeTrainer:
    def __init__(self, policy: JunqiNet) -> None:
        self._policy = policy

    def state_dict(self) -> dict:
        return {
            "policy": self._policy.state_dict(),
            "checkpoint_meta": current_checkpoint_metadata(self._policy),
        }


def test_save_checkpoint_roundtrip_loads_the_same_evaluation_policy(tmp_path) -> None:
    policy = _tiny_net()
    cfg = TrainConfig(net=policy.cfg)
    path = save_checkpoint(_FakeTrainer(policy), cfg, 1, str(tmp_path))

    loaded = load_evaluation_policy(path, device="cpu")
    expected = policy.state_dict()
    actual = loaded.state_dict()

    assert set(actual) == set(expected)
    for key, tensor in expected.items():
        assert tensor.equal(actual[key]), key
    assert loaded.cfg.embed_dim == policy.cfg.embed_dim
    assert loaded.training is False
