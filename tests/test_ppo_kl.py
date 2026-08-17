"""Regression tests for the sampled PPO KL trust-region path."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import EMAPolicy, PPOConfig, PPOTrainer


def _trainer() -> PPOTrainer:
    net_cfg = JunqiNetConfig(
        cnn_channels=8,
        cnn_layers=1,
        depth=1,
        embed_dim=32,
        n_head=2,
        ff_factor=2,
        action_key_dim=8,
        use_cat_vf=True,
    )
    cfg = PPOConfig(
        net=net_cfg,
        dtype="float32",
        kl_coef=0.2,
        kl_proxy_beta=1.0,
    )
    return PPOTrainer(JunqiNet(net_cfg), cfg, device="cpu")


def test_sampled_kl_penalty_is_finite_and_nonnegative():
    trainer = _trainer()
    log_ratio = torch.tensor([-3.0, 0.0, 2.0], requires_grad=True)
    penalty = trainer._sampled_kl_penalty(log_ratio)
    assert torch.isfinite(penalty)
    assert penalty.item() >= 0.0
    penalty.backward()
    assert torch.isfinite(log_ratio.grad).all()


def test_full_kl_masks_illegal_negative_infinity_entries():
    trainer = _trainer()
    old = torch.log(torch.tensor([[0.7, 0.3, 0.0]]))
    new = old.clone()
    value = trainer._kl_loss(new, old)
    assert torch.isfinite(value)
    assert abs(value.item()) < 1e-6


def test_policy_ratio_does_not_overflow_on_stale_action():
    trainer = _trainer()
    new = torch.tensor([1000.0], requires_grad=True)
    old = torch.zeros(1)
    adv = torch.ones(1)
    loss = trainer._policy_loss(new, old, adv)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(new.grad).all()


def test_ema_copies_batchnorm_buffers_exactly():
    """EMA must remain functionally aligned with the learner network."""
    cfg = JunqiNetConfig(
        cnn_channels=8,
        cnn_layers=1,
        depth=1,
        embed_dim=32,
        n_head=2,
        ff_factor=2,
        action_key_dim=8,
    )
    model = JunqiNet(cfg)
    ema = EMAPolicy(model, decay=0.999)
    with torch.no_grad():
        for index, (_, buf) in enumerate(model.named_buffers()):
            if torch.is_floating_point(buf):
                buf.fill_(float(index + 1))
            else:
                buf.fill_(index + 1)
    ema.update(model)
    shadow_buffers = dict(ema.model.named_buffers())
    for name, model_buffer in model.named_buffers():
        assert torch.equal(shadow_buffers[name], model_buffer), name


def test_policy_active_action_legality_is_checked_directly():
    trainer = _trainer()
    legal = torch.tensor([[True, False, True], [True, False, True]])
    actions = torch.tensor([0, 1])
    active = torch.tensor([True, False])
    # Value-only / inactive rows are intentionally excluded.
    trainer._assert_policy_active_actions_legal(legal, actions, active)

    with pytest.raises(RuntimeError, match="stored action is illegal"):
        trainer._assert_policy_active_actions_legal(
            legal, actions, torch.tensor([True, True])
        )
