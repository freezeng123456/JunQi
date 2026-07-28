"""Benchmark V2 collect with compact action space."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig, FLAT_ACTION_DIM
from junqi_rl.training import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

N, T, device = 64, 128, torch.device('cuda')
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

print(f"FLAT_ACTION_DIM = {FLAT_ACTION_DIM}")

r = GpuRollout(num_envs=N)
buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
r.reset(seed_base=0)

# Warmup (includes torch.compile)
collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
torch.cuda.synchronize()

# Benchmark
times = []
for trial in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=trial+1, reset_at_start=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
v2_time = min(times)

total = N * T
print(f"N={N}, T={T}, total={total}")
print(f"  V2 compact (129x129): {v2_time:.3f}s = {total/v2_time:,.0f} env-steps/s")
print(f"  Baseline (pre-compact, 289x289): ~0.69s = ~12,000 steps/s")
print(f"  V2 trials: {[f'{t:.3f}' for t in times]}")
print()

# Also test N=256
for N2 in [128, 256, 512]:
    try:
        torch.cuda.empty_cache()
        r2 = GpuRollout(num_envs=N2)
        buf2 = RolloutBufferGPU(num_envs=N2, steps_per_env=T, device=device)
        r2.reset(seed_base=0)
        collect_rollout_gpu_v2(r2, policy, buf2, device='cuda', seed_base=0, reset_at_start=False)
        torch.cuda.synchronize()
        ts = []
        for trial in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            collect_rollout_gpu_v2(r2, policy, buf2, device='cuda', seed_base=trial+1, reset_at_start=False)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        best = min(ts)
        rate = N2 * T / best
        mem_lm = T * N2 * FLAT_ACTION_DIM / 1e6
        print(f"  N={N2:>4}: {best:.3f}s = {rate:>10,.0f} steps/s  lm_buf={mem_lm:.0f}MB")
        del r2, buf2
    except Exception as e:
        print(f"  N={N2}: FAILED: {e}")
