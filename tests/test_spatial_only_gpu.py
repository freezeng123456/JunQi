"""Native zero-global observations and PPO gates for the target CUDA host."""

import pytest
import torch

cuda = pytest.importorskip('junqi_cuda')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout_gpu import RolloutBufferGPU


def world(identity_encoding="legacy_probabilities"):
    assert getattr(cuda, 'OPTIONAL_GLOBAL_OBSERVATIONS', False), 'Rebuild junqi_cuda for zero-global observation support'
    env = GpuRollout(2, include_global=False, identity_encoding=identity_encoding)
    env.reset(seed_base=317)
    return env


def test_native_no_global_output_and_history_preserve_spatial():
    env = world()
    legacy = GpuRollout(2)
    legacy.reset(seed_base=317)
    acting = env.turn_torch().clone()
    spatial, global_ = env.build_acting_seat_observation_torch(acting)
    expected, _ = legacy.build_acting_seat_observation_torch(acting)
    torch.testing.assert_close(spatial, expected, rtol=0, atol=0)
    assert global_.shape == (2, 0) and env.obs_single.d_global_ptr == 0
    all_sp, all_gl = env.build_all_seat_observations_torch()
    assert all_gl.shape == (2, 4, 0) and env.obs.d_global_ptr == 0
    _, host_gl = env.build_all_seat_observations()
    assert host_gl.shape == (2, 4, 0)
    history = env.create_rollout_history(1)
    history.snapshot(env.state, acting, 0)
    reconstructed, rebuilt_gl, legal = history.reconstruct(
        torch.arange(2, device='cuda', dtype=torch.int64), acting, dtype=torch.float32)
    torch.testing.assert_close(reconstructed, spatial, rtol=0, atol=0)
    assert rebuilt_gl.shape == (2, 0) and legal.any(-1).all()


@pytest.mark.parametrize('identity_encoding', ['legacy_probabilities', 'known_only'])
@pytest.mark.parametrize('storage', ['full_obs', 'compact_history'])
def test_gpu_collect_and_ppo_update_without_global(storage, identity_encoding):
    env = world(identity_encoding)
    cfg = JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1, embed_dim=16,
                        n_head=2, ff_factor=2, action_key_dim=8, use_global_summary=False,
                        identity_encoding=identity_encoding)
    policy = JunqiNet(cfg).cuda()
    ppo = PPOConfig(net=cfg, dtype='float32', minibatch_size=4,
                    num_epochs_per_rollout=1, adv_filt_rate=1, adv_filt_thresh=0)
    trainer = PPOTrainer(policy, ppo, device='cuda')
    rollout = RolloutBufferGPU(num_envs=2, steps_per_env=4, obs_global_dims=0,
        storage_mode=storage, history=env.create_rollout_history(4) if storage == 'compact_history' else None,
        random_opponent=False, adv_filt_rate=1, adv_filt_thresh=0)
    collect_rollout_gpu_v2(env, policy, rollout, device='cuda', random_opponent=False,
                          use_compile=False, autocast_dtype=torch.float32, reward_shaping=False)
    for batch in rollout.minibatches(4):
        assert batch.obs_global.shape[1] == 0
    initial = policy.cls_token.detach().clone()
    trainer.train_epoch(rollout)
    assert trainer.num_train_step > 0
    assert torch.isfinite(policy.cls_token).all() and not torch.equal(policy.cls_token, initial)
