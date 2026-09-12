"""A real collector/supervision/update/publication cycle preserves PPO history."""
import pytest
import torch

pytest.importorskip('junqi_cuda')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def test_joint_cycle_preserves_collected_observations_and_supplies_real_labels():
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
    from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
    from junqi_rl.training.rollout_gpu import RolloutBufferGPU
    from junqi_rl.belief.buffer import BeliefBuffer
    from junqi_rl.belief.sampling import MidgameBeliefSampler
    from junqi_rl.belief.inference import guarded_refresh_beliefs_neural
    from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig
    from junqi_rl.training.belief_ppo import BeliefPPOTrainer, BeliefPPOConfig
    world = GpuRollout(num_envs=4)
    history = world.create_rollout_history(8)
    buf = RolloutBufferGPU(num_envs=4, steps_per_env=8, storage_mode='compact_history',
                          history=history, random_opponent=False)
    policy = JunqiNet(JunqiNetConfig(cnn_channels=16, cnn_layers=1, depth=1,
                                     embed_dim=32, n_head=2, ff_factor=2)).cuda()
    labels = BeliefBuffer(capacity=64, seed=912)
    sampler = MidgameBeliefSampler(labels, every_steps=2, envs_per_sample=2)
    collect_rollout_gpu_v2(world, policy, buf, seed_base=912, use_compile=False,
                          on_observation=sampler)
    assert len(labels) > 0 and sampler.valid_labels > 0
    assert (labels._enemy[:len(labels)].sum(1) > 0).all()
    indices = torch.arange(32, device='cuda', dtype=torch.int64)
    seats = buf.seats.reshape(-1)
    before = tuple(t.clone() for t in history.reconstruct(indices, seats, dtype=torch.float32))
    net = BeliefNet(BeliefNetConfig(n_encoder_layer=1, n_head=2, embed_dim=32,
                                   cnn_channels=16, cnn_layers=1, ff_factor=2)).cuda()
    trainer = BeliefPPOTrainer(net, BeliefPPOConfig(batch_size=4, epochs_per_rollout=2),
                              device='cuda')
    metrics = trainer.train_epoch(labels)
    assert metrics['belief_train/num_updates'] == 2
    refresh = guarded_refresh_beliefs_neural(world, trainer.ema.model, policy, neural_weight=.25,
                                    max_kl=.05, rule_only_input=True, empty_cache=False)
    assert refresh['belief_infer/max_kl_to_rules'] <= .050001
    assert refresh['belief_guard/max_policy_kl'] <= .020001
    after = history.reconstruct(indices, seats, dtype=torch.float32)
    for a,b in zip(before, after, strict=True):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
