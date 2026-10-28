"""Pending actions retain their real sampling policy across learner updates."""
import pytest
import torch

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.gpu_collector import _policy_act
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout import RolloutBatch


def network():
    cfg = JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1, embed_dim=32,
                        n_head=2, ff_factor=2, action_key_dim=8, use_cat_vf=True)
    return JunqiNet(cfg).eval(), cfg


def inputs(n=5):
    spatial = torch.randn(n, OBS_CHANNELS, 17, 17)
    global_ = torch.randn(n, OBS_GLOBAL_DIMS)
    legal = torch.zeros(n, 129 * 129, dtype=torch.bool)
    for row in range(n):
        legal[row, [3 + row, 30 + row, 100 + row]] = True
    return spatial, global_, legal


@pytest.mark.parametrize("chunk_size", [0, 2, 8])
def test_behavior_distribution_matches_sampled_logp_and_chunk_order(chunk_size):
    policy, _ = network()
    policy.track_behavior_policy = True
    policy.track_first_policy = True
    spatial, global_, legal = inputs()
    actions, sampled_logp, _ = _policy_act(policy, spatial, global_, legal, chunk_size)
    behavior = policy._last_behavior_log_probs
    assert behavior.shape == legal.shape
    assert behavior.dtype == torch.float32 and not behavior.requires_grad
    # The network currently uses a finite illegal-logit sentinel; its FP32
    # probabilities are zero. Preserve those exact logits rather than rewrite.
    assert (behavior[~legal].exp() == 0).all()
    torch.testing.assert_close(behavior.gather(1, actions.long()[:, None]).squeeze(1),
                               sampled_logp, rtol=0, atol=0)
    torch.testing.assert_close(behavior.logsumexp(-1), torch.zeros(len(actions)), rtol=0, atol=2e-6)
    with torch.no_grad():
        expected = policy(spatial, global_, legal)["log_probs"]
    torch.testing.assert_close(behavior, expected, rtol=1e-5, atol=2e-6)
    assert policy._last_entropy_t.shape == (len(actions),)
    assert policy._last_first_policy_stats.shape == (len(actions), 10)


def test_behavior_capture_is_opt_in():
    policy, _ = network()
    _policy_act(policy, *inputs(2), chunk_size=1)
    assert not hasattr(policy, "_last_behavior_log_probs")


def trainer_and_batch(*, carried):
    policy, net_cfg = network()
    trainer = PPOTrainer(policy, PPOConfig(net=net_cfg, dtype="float32", kl_coef=0.2), device="cpu")
    spatial, global_, legal = inputs(2)
    actions = torch.tensor([3, 4])
    with torch.no_grad():
        current = policy(spatial, global_, legal, actions=actions)["log_probs"]
    if carried:
        # An older behavior policy with a very different legal distribution.
        behavior = torch.full_like(current, float("-inf"))
        for row in range(2):
            behavior[row, legal[row]] = torch.log(torch.tensor([0.85, 0.10, 0.05]))
        behavior.requires_grad_(True)
        old_logp = behavior.detach().gather(1, actions[:, None]).squeeze(1)
    else:
        behavior = None
        old_logp = current.gather(1, actions[:, None]).squeeze(1)
    batch = RolloutBatch(obs_spatial=spatial, obs_global=global_, legal_mask=legal,
                         actions=actions, old_log_probs=old_logp,
                         advantages=torch.tensor([0.02, -0.03]), returns=torch.zeros(2),
                         values=torch.zeros(2), adv_mask=torch.ones(2, dtype=torch.bool),
                         behavior_log_probs=behavior)
    return trainer, batch, current


def test_ppo_uses_carried_probabilities_without_evaluating_new_frozen_snapshot(monkeypatch):
    trainer, batch, current = trainer_and_batch(carried=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("The new frozen model did not generate this pending action")

    monkeypatch.setattr(trainer._collect_policy, "forward", forbidden)
    expected = (current[batch.legal_mask].exp()
                * (current[batch.legal_mask] - batch.behavior_log_probs.detach()[batch.legal_mask])).sum() / 2
    assert expected.item() > 0.05
    metrics = trainer._update_step(batch)
    torch.testing.assert_close(metrics["train/kl_loss"], expected, rtol=1e-5, atol=1e-6)
    assert batch.behavior_log_probs.grad is None
    assert trainer.num_train_step == 1


def test_legacy_batch_still_evaluates_frozen_snapshot(monkeypatch):
    trainer, batch, _ = trainer_and_batch(carried=False)
    original = trainer._collect_policy.forward
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer._collect_policy, "forward", counted)
    metrics = trainer._update_step(batch)
    assert len(calls) == 1
    assert abs(metrics["train/kl_loss"].item()) < 1e-6


def test_carried_distribution_must_cover_full_action_support():
    trainer, batch, _ = trainer_and_batch(carried=True)
    batch.behavior_log_probs = batch.behavior_log_probs[:, :3]
    with pytest.raises(ValueError, match="full legal-mask shape"):
        trainer._update_step(batch)
