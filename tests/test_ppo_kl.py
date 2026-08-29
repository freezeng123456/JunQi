"""Regression tests for the sampled PPO KL trust-region path."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.ppo import (
    EMAPolicy,
    PPOConfig,
    PPOTrainer,
    magnet_alpha,
)
from junqi_rl.training.rollout import RolloutBatch


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
    )
    return PPOTrainer(JunqiNet(net_cfg), cfg, device="cpu")


def _split_value_batch(
    trainer: PPOTrainer,
    *,
    n_policy: int = 2,
    n_value: int = 5,
) -> tuple[RolloutBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    value_spatial = torch.randn(n_value, OBS_CHANNELS, 17, 17)
    value_global = torch.randn(n_value, OBS_GLOBAL_DIMS)
    value_returns = torch.linspace(-0.8, 0.8, n_value)
    policy_value_indices = torch.arange(n_policy) * max(1, n_value // n_policy)
    policy_spatial = value_spatial.index_select(0, policy_value_indices)
    policy_global = value_global.index_select(0, policy_value_indices)
    legal = torch.zeros(n_policy, 129 * 129, dtype=torch.bool)
    legal[0, [3, 11]] = True
    legal[1, [7, 19]] = True
    actions = torch.tensor([11, 7])
    trainer._sync_collect_policy()
    with torch.inference_mode():
        old_log_probs = trainer._collect_policy(
            policy_spatial,
            policy_global,
            legal,
            actions=actions,
        )["action_log_prob"]

    batch = RolloutBatch(
        obs_spatial=policy_spatial,
        obs_global=policy_global,
        legal_mask=legal,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.tensor([0.25, -0.25]),
        returns=torch.zeros(n_policy),
        values=torch.zeros(n_policy),
        adv_mask=torch.ones(n_policy, dtype=torch.bool),
        value_only_mask=torch.zeros(n_policy, dtype=torch.bool),
        value_obs_spatial=value_spatial,
        value_obs_global=value_global,
        value_returns=value_returns,
        policy_value_indices=policy_value_indices,
    )
    return batch, value_spatial, value_global, value_returns


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


def test_trainer_always_allocates_frozen_collection_policy():
    trainer = _trainer()

    assert trainer._collect_policy is not None


def test_ppo_update_evaluates_stored_actions_without_sampling(monkeypatch):
    trainer = _trainer()
    batch_size = 2
    spatial = torch.zeros(batch_size, OBS_CHANNELS, 17, 17)
    global_ = torch.zeros(batch_size, OBS_GLOBAL_DIMS)
    legal = torch.zeros(batch_size, 129 * 129, dtype=torch.bool)
    legal[0, [3, 11]] = True
    legal[1, [7, 19]] = True
    actions = torch.tensor([11, 7])
    trainer._sync_collect_policy()
    with torch.inference_mode():
        old_log_probs = trainer.policy(
            spatial,
            global_,
            legal,
            actions=actions,
        )["action_log_prob"]

    batch = RolloutBatch(
        obs_spatial=spatial,
        obs_global=global_,
        legal_mask=legal,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=torch.tensor([0.25, -0.25]),
        returns=torch.tensor([0.5, -0.5]),
        values=torch.zeros(batch_size),
        adv_mask=torch.ones(batch_size, dtype=torch.bool),
        value_only_mask=torch.zeros(batch_size, dtype=torch.bool),
    )

    def fail_if_sampled(*_args, **_kwargs):
        raise AssertionError("PPO update must evaluate stored actions")

    monkeypatch.setattr(torch.distributions.Categorical, "sample", fail_if_sampled)
    metrics = trainer._update_step(batch)

    assert trainer.num_train_step == 1
    assert trainer._nan_skip_count == 0
    assert trainer._grad_nan_skip_count == 0
    assert "train/kl_proxy" not in metrics
    assert torch.isfinite(metrics["train/kl_loss"])
    for value in metrics.values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
        elif isinstance(value, float):
            assert math.isfinite(value)


def test_split_value_chunks_match_one_full_value_backward():
    """Chunking changes activation memory, not the all-valid value update."""
    split = _trainer()
    reference = _trainer()
    reference.policy.load_state_dict(split.policy.state_dict(), strict=True)
    for trainer in (split, reference):
        trainer.cfg.policy_coef = 0.0
        trainer.cfg.temperature_coef = 0.0
        trainer.cfg.kl_coef = 0.0
        trainer.cfg.value_minibatch_size = 2
        trainer.cfg.max_grad_norm = 1.0e9
        trainer.policy.eval()
        # Adam's first step normalises tiny near-zero gradients by their own
        # magnitude, which can amplify harmless accumulation-order noise.
        # Plain SGD makes parameter deltas directly proportional to gradients.
        trainer.optimizer = torch.optim.SGD(trainer.policy.parameters(), lr=1e-4)

    n_value = 5
    batch, value_spatial, value_global, value_returns = _split_value_batch(
        split, n_value=n_value,
    )

    split_metrics = split._update_step(batch)

    reference.optimizer.zero_grad(set_to_none=True)
    reference_value = reference.policy.forward_value(value_spatial, value_global)
    reference_loss = (
        reference.cfg.vf_coef
        * reference._value_loss(reference_value, value_returns)
    )
    reference_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        reference.policy.parameters(), reference.cfg.max_grad_norm
    )
    reference.optimizer.step()

    assert split_metrics["train/value_batch_size"] == n_value
    assert split_metrics["train/value_loss"].item() == pytest.approx(
        reference_loss.item() / reference.cfg.vf_coef,
        abs=2e-6,
    )
    for (name, actual), (expected_name, expected) in zip(
        split.policy.named_parameters(),
        reference.policy.named_parameters(),
        strict=True,
    ):
        assert name == expected_name
        assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6), name


def test_split_value_encodes_learner_batch_once(monkeypatch):
    """All-valid value and filtered policy must share one learner encode."""
    trainer = _trainer()
    trainer.cfg.value_minibatch_size = 16
    batch, _spatial, _global, _returns = _split_value_batch(trainer)

    encode_calls = 0
    original_encode = trainer.policy._encode

    def counted_encode(*args, **kwargs):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(trainer.policy, "_encode", counted_encode)
    trainer._update_step(batch)

    assert encode_calls == 1


def test_shared_encoder_update_matches_separate_split_update():
    shared = _trainer()
    separate = _trainer()
    separate.policy.load_state_dict(shared.policy.state_dict(), strict=True)
    separate._sync_collect_policy()
    for trainer, chunk_size in ((shared, 16), (separate, 2)):
        trainer.cfg.value_minibatch_size = chunk_size
        trainer.cfg.max_grad_norm = 1.0e9
        trainer.policy.eval()
        trainer.optimizer = torch.optim.SGD(trainer.policy.parameters(), lr=1e-4)

    batch, _spatial, _global, _returns = _split_value_batch(shared)
    with torch.inference_mode():
        shared_pre = shared.policy.forward_policy_value_shared(
            batch.value_obs_spatial,
            batch.value_obs_global,
            batch.policy_value_indices,
            batch.legal_mask,
            actions=batch.actions,
        )
        separate_pre = separate.policy(
            batch.obs_spatial,
            batch.obs_global,
            batch.legal_mask,
            actions=batch.actions,
        )
        shared_old = shared._collect_policy(
            batch.obs_spatial,
            batch.obs_global,
            batch.legal_mask,
            actions=batch.actions,
        )
        separate_old = separate._collect_policy(
            batch.obs_spatial,
            batch.obs_global,
            batch.legal_mask,
            actions=batch.actions,
        )
    torch.testing.assert_close(
        shared_pre["log_probs"], separate_pre["log_probs"], atol=2e-6, rtol=2e-6,
    )
    torch.testing.assert_close(
        shared_old["log_probs"], separate_old["log_probs"], atol=2e-6, rtol=2e-6,
    )
    shared_metrics = shared._update_step(batch)
    separate_metrics = separate._update_step(batch)

    for key in (
        "train/policy_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/kl_loss",
        "train/total_loss",
    ):
        assert shared_metrics[key].item() == pytest.approx(
            separate_metrics[key].item(), abs=2e-6,
        ), key
    for (name, actual), (expected_name, expected) in zip(
        shared.policy.named_parameters(),
        separate.policy.named_parameters(),
        strict=True,
    ):
        assert name == expected_name
        assert torch.allclose(actual, expected, atol=3e-6, rtol=3e-6), name


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
