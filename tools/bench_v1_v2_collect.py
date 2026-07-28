"""Benchmark v1 vs v2 collect loop at N=64."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch, numpy as np
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training import RolloutBuffer, RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu, collect_rollout_gpu_v2

N = 64
T = 128
device = torch.device("cuda")

# Small policy for fast inference
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64, n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

# V1: old collector + CPU buffer
r1 = GpuRollout(num_envs=N)
buf1 = RolloutBuffer(num_envs=N, steps_per_env=T, device=device)
r1.reset(seed_base=0)

torch.cuda.synchronize()
t0 = time.perf_counter()
collect_rollout_gpu(r1, policy, buf1, device="cuda", seed_base=0, reset_at_start=False)
torch.cuda.synchronize()
v1_time = time.perf_counter() - t0

# V2: new collector + GPU buffer
r2 = GpuRollout(num_envs=N)
buf2 = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
r2.reset(seed_base=0)

torch.cuda.synchronize()
t0 = time.perf_counter()
collect_rollout_gpu_v2(r2, policy, buf2, device="cuda", seed_base=0, reset_at_start=False)
torch.cuda.synchronize()
v2_time = time.perf_counter() - t0

total_steps = N * T
print(f"N={N}, T={T}, total_steps={total_steps}")
print(f"  V1 (old): {v1_time:.3f}s = {total_steps/v1_time:,.0f} env-steps/s")
print(f"  V2 (new): {v2_time:.3f}s = {total_steps/v2_time:,.0f} env-steps/s")
print(f"  Speedup: {v1_time/v2_time:.2f}x")
