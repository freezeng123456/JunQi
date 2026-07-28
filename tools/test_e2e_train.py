"""End-to-end training test with optimized config (5 rollouts)."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig, FLAT_ACTION_DIM
from junqi_rl.training import PPOTrainer, PPOConfig, RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

device = torch.device('cuda')
print(f"Action dim: {FLAT_ACTION_DIM}, device: {torch.cuda.get_device_name(0)}")

# Medium config (realistic for T4)
cfg = JunqiNetConfig(cnn_channels=64, cnn_layers=2, depth=4, embed_dim=128,
                     n_head=4, ff_factor=4, action_key_dim=32)
policy = JunqiNet(cfg).to(device)
n_params = policy.num_parameters()
print(f"Network: {n_params:,} params (depth={cfg.depth}, embed={cfg.embed_dim})")

N, T = 256, 128
rollout = GpuRollout(num_envs=N)
buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, gamma=1.0, gae_lambda=0.5,
                       td_lambda=0.8, device=device)
ppo_cfg = PPOConfig(net=cfg, minibatch_size=512, num_epochs_per_rollout=3,
                    dtype="float16", torch_compile=True)
trainer = PPOTrainer(policy, ppo_cfg, device=device)

print(f"\nTraining 5 rollouts: N={N}, T={T}, {N*T:,} steps/rollout\n")

for i in range(5):
    t0 = time.time()
    reset = (i == 0)
    if reset:
        rollout.reset(seed_base=42)
    collect_rollout_gpu_v2(rollout, policy, buf, device='cuda',
                           seed_base=42 + i, reset_at_start=reset)
    torch.cuda.synchronize()
    t_collect = time.time() - t0

    t0 = time.time()
    stats = trainer.train_epoch(buf)
    torch.cuda.synchronize()
    t_train = time.time() - t0

    rate = N * T / t_collect
    games_s = rate / 780
    pl = stats.get('train/policy_loss', 0)
    vl = stats.get('train/value_loss', 0)
    el = stats.get('train/entropy_loss', 0)
    print(f"  Rollout {i+1}: collect={t_collect:.2f}s ({rate:,.0f} sps, {games_s:.1f} g/s) "
          f"train={t_train:.2f}s  pl={pl:.4f} vl={vl:.4f} el={el:.4f}")

print(f"\nE2E TRAINING TEST PASSED")
