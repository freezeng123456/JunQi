"""Actor reward-unit scale and raw selection stay independent of batching."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi_rl.training.rollout_gpu import (
    BOARD_SIZE, FLAT_ACTION_DIM, OBS_CHANNELS, RolloutBufferGPU,
)


def buffer(advantages, *, mode="raw", grouping="timestep", value_scope="policy",
           rate=1.0, threshold=0.0, seats=None, random_opponent=False):
    # Populate just the tensor storage to exercise the real minibatch method
    # on CPU; the production constructor remains CUDA-only.
    raw = torch.tensor(advantages, dtype=torch.float32)
    result = RolloutBufferGPU.__new__(RolloutBufferGPU)
    result.steps_per_env, result.num_envs = raw.shape
    result.device = torch.device("cpu")
    result.uses_compact_history = False
    result.csr_legal_mask = False
    result.random_opponent = random_opponent
    result.train_value_on_random_seats = False
    result.adv_filter_scope = "rollout"
    result.value_sample_scope = value_scope
    result.minibatch_group = grouping
    result.adv_filt_rate = rate
    result.adv_filt_thresh = threshold
    if mode is not None:
        result.advantage_normalization = mode
    result.obs_spatial = torch.zeros((*raw.shape, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE))
    result.obs_global_dims = 0
    result.obs_global = torch.empty((*raw.shape, 0))
    result.legal_mask = torch.ones((*raw.shape, FLAT_ACTION_DIM), dtype=torch.bool)
    # Action IDs identify the emitted row indices independently of advantage values.
    result.actions = torch.arange(raw.numel(), dtype=torch.int32).reshape(raw.shape)
    result.log_probs = torch.zeros_like(raw)
    result.advantages_ = raw.clone()
    result.returns_ = raw + 0.3
    result.values = torch.full_like(raw, 0.2)
    result.seats = torch.zeros(raw.shape, dtype=torch.int8) if seats is None else torch.tensor(seats, dtype=torch.int8)
    return result


@pytest.mark.parametrize("grouping", ["global", "timestep"])
@pytest.mark.parametrize("raw", [[[0.03, 0.03]], [[0.01, 0.04], [-0.02, 0.03]]])
def test_raw_actor_preserves_scale_sign_and_nonzero_constant(raw, grouping):
    rollout = buffer(raw, grouping=grouping)
    original = rollout.advantages_.clone()
    batches = list(rollout.minibatches(16, shuffle=False))
    torch.testing.assert_close(torch.cat([b.advantages for b in batches]), original.flatten(), rtol=0, atol=0)
    torch.testing.assert_close(rollout.advantages_, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat([b.returns for b in batches]), (original + 0.3).flatten())


@pytest.mark.parametrize("mode", [None, "standardize"])
def test_legacy_standardization_still_uses_population_variance(mode):
    rollout = buffer([[0.02, 0.04], [0.06, 0.08]], mode=mode)
    result = torch.cat([b.advantages for b in rollout.minibatches(16, shuffle=False)])
    raw = rollout.advantages_.numpy().flatten()
    np.testing.assert_allclose(result.numpy(), (raw - raw.mean()) / (raw.std() + 1e-8), atol=1e-6)


@pytest.mark.parametrize("mode", ["raw", "standardize"])
@pytest.mark.parametrize("value_scope", ["policy", "all_valid"])
def test_raw_rollout_filter_has_same_survivors_for_both_actor_scales(mode, value_scope):
    rollout = buffer([[0.02, 0.04, 0.06, 0.08], [0.001, 0.002, 0.003, 0.004]],
                     mode=mode, value_scope=value_scope, rate=0.25, threshold=0.01)
    batches = list(rollout.minibatches(16, shuffle=False))
    assert torch.cat([b.actions for b in batches]).tolist() == [2, 3]
    assert rollout._last_thresh_used == pytest.approx(0.045)
    expected = rollout.advantages_.flatten()
    if mode == "standardize":
        expected = (expected - expected.mean()) / (expected.std(unbiased=False) + 1e-8)
    torch.testing.assert_close(torch.cat([b.advantages for b in batches]), expected[[2, 3]])
    if value_scope == "all_valid":
        assert [b.actions.numel() for b in batches] == [2, 0]
        torch.testing.assert_close(torch.cat([b.value_returns for b in batches]), rollout.returns_.flatten())


@pytest.mark.parametrize("grouping", ["global", "timestep"])
def test_raw_minimum_filter_is_not_applied_in_standard_deviation_units(grouping):
    rollout = buffer([[0.003, 0.02]], grouping=grouping, threshold=0.01)
    batches = list(rollout.minibatches(16, shuffle=False))
    assert torch.cat([b.actions for b in batches]).tolist() == [1]
    torch.testing.assert_close(batches[0].advantages, torch.tensor([0.02]))


def test_random_seat_exclusion_does_not_center_raw_actor_advantages():
    rollout = buffer([[0.02, 0.0, 0.06, 0.0]], seats=[[0, 1, 2, 3]], random_opponent=True)
    batch, = list(rollout.minibatches(16, shuffle=False))
    assert batch.actions.tolist() == [0, 2]
    torch.testing.assert_close(batch.advantages, torch.tensor([0.02, 0.06]), rtol=0, atol=0)


def test_ppo_actor_gradient_keeps_raw_reward_scale():
    from junqi_rl.training.ppo import PPOTrainer

    rollout = buffer([[0.02, -0.03]])
    batch, = list(rollout.minibatches(16, shuffle=False))
    trainer = PPOTrainer.__new__(PPOTrainer)
    trainer.cfg = SimpleNamespace(clip_range=0.2)
    new_log_prob = torch.zeros(2, requires_grad=True)
    loss = trainer._policy_loss(new_log_prob, torch.zeros(2), batch.advantages)
    loss.backward()
    torch.testing.assert_close(new_log_prob.grad, torch.tensor([-0.01, 0.015]), rtol=0, atol=1e-8)
