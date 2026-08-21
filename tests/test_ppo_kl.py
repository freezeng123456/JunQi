"""Regression tests for the sampled PPO KL trust-region path."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import (
    EMAPolicy,
    PPOConfig,
    PPOTrainer,
    magnet_alpha,
)


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



def test_magnet_alpha_matches_paper_formula():
    assert magnet_alpha(0.05, 1, 0.3) == pytest.approx(0.05)
    assert magnet_alpha(0.05, 900, 0.3) == pytest.approx(0.05 / (900 ** 0.3))
    assert magnet_alpha(0.05, 900, 0.3) > 0.001
    assert magnet_alpha(0.05, 3000, 0.3) == pytest.approx(0.05 / (3000 ** 0.3))


def test_get_temperature_rollout_unit_uses_paper_formula():
    trainer = _trainer()
    trainer.cfg.temperature_schedule_unit = "rollout"
    trainer.cfg.temperature_coef = 0.05
    trainer.cfg.temperature_decay = 0.3
    trainer.num_rollout = 0
    assert trainer._get_temperature() == pytest.approx(0.05)
    trainer.num_rollout = 899
    assert trainer._get_temperature() == pytest.approx(0.05 / (900 ** 0.3))


def test_reverse_kl_zero_on_identical_legal_support():
    trainer = _trainer()
    legal = torch.tensor([[True, True, False], [True, False, True]])
    logits = torch.tensor([[1.0, 0.0, 9.0], [0.4, 9.0, -0.1]])
    logits = logits.masked_fill(~legal, float("-inf"))
    logp = torch.log_softmax(logits, dim=-1)
    kl = trainer._reverse_kl(logp, logp, legal)
    assert torch.isfinite(kl)
    assert abs(kl.item()) < 1e-6


def test_reverse_kl_positive_when_new_is_peakier():
    trainer = _trainer()
    legal = torch.tensor([[True, True, True]])
    old = torch.log_softmax(torch.zeros(1, 3), dim=-1)
    logits = torch.tensor([[5.0, 0.0, 0.0]], requires_grad=True)
    new = torch.log_softmax(logits, dim=-1)
    kl = trainer._reverse_kl(new, old, legal)
    assert torch.isfinite(kl)
    assert kl.item() > 0.0
    kl.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()



def test_magnet_alpha_respects_floor():
    assert magnet_alpha(0.08, 3000, 0.2, floor=0.02) == pytest.approx(0.02)
    assert magnet_alpha(0.08, 1, 0.2, floor=0.02) == pytest.approx(0.08)


def test_piece_then_dest_kl_zero_when_policy_matches_rho():
    trainer = _trainer()
    trainer.cfg.magnet_shape = "piece_then_dest"
    trainer.cfg.uniform_magnet = True
    # 2×2 board: actions src*2+dst. src0 has dests 0 and 1; src1 has dest 0.
    legal = torch.tensor([[True, True, True, False]])
    rho = torch.tensor([[0.25, 0.25, 0.50, 0.0]])
    logp = rho.clamp(min=1e-12).log()
    logp = logp.masked_fill(~legal, float("-inf"))
    loss, entropy = trainer._entropy_loss(logp, legal)
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-5
    assert entropy.item() > 0.0


def test_piece_then_dest_upweights_scarce_piece_vs_uniform():
    trainer = _trainer()
    trainer.cfg.uniform_magnet = True
    legal = torch.tensor([[True, True, True, False]])
    # Peak on the scarce piece's only dest (action 2).
    logits = torch.tensor([[0.0, 0.0, 4.0, -1e9]])
    logp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=-1)
    trainer.cfg.magnet_shape = "uniform_legal"
    loss_u, _ = trainer._entropy_loss(logp, legal)
    trainer.cfg.magnet_shape = "piece_then_dest"
    loss_p, _ = trainer._entropy_loss(logp, legal)
    # ρ_ptd(action 2)=0.5 > ρ_unif=1/3, so a peak there is closer to ptd.
    assert loss_p.item() < loss_u.item()
