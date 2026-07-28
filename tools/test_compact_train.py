"""Training smoke test with compact action space."""
import sys
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig, FLAT_ACTION_DIM
from junqi_rl.training import PPOTrainer, PPOConfig, RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

device = torch.device('cuda')
print(f"FLAT_ACTION_DIM = {FLAT_ACTION_DIM}")

cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device)
rollout = GpuRollout(num_envs=32)
buf = RolloutBufferGPU(num_envs=32, steps_per_env=128, device=device)
ppo_cfg = PPOConfig(net=cfg, minibatch_size=512, num_epochs_per_rollout=2)
trainer = PPOTrainer(policy, ppo_cfg, device=device)

for i in range(3):
    rollout.reset(seed_base=i * 100)
    collect_rollout_gpu_v2(rollout, policy, buf, device='cuda',
                           seed_base=i * 100, reset_at_start=False)
    stats = trainer.train_epoch(buf)
    pl = stats.get('train/policy_loss', 0)
    vl = stats.get('train/value_loss', 0)
    print(f"  Rollout {i+1}: pl={pl:.4f} vl={vl:.4f}")
print("COMPACT TRAINING SMOKE TEST PASSED")
