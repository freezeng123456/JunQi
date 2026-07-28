"""Benchmark V2 collect at different N values to find optimal batch size."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

T = 128
device = torch.device('cuda')
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

print(f"{'N':>6}  {'time':>8}  {'steps/s':>12}  {'speedup':>8}  {'mem_obs_MB':>10}  {'mem_lm_MB':>10}")
print("-" * 70)

base_rate = None

for N in [32, 64, 128, 256, 512]:
    try:
        torch.cuda.empty_cache()
        r = GpuRollout(num_envs=N)
        buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
        r.reset(seed_base=0)

        # Warmup
        collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
        torch.cuda.synchronize()

        # Benchmark
        times = []
        for trial in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=trial+1, reset_at_start=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        best = min(times)
        total = N * T
        rate = total / best
        if base_rate is None:
            base_rate = rate

        mem_obs = T * N * 101 * 17 * 17 * 4 / 1e6
        mem_lm = T * N * 83521 / 1e6

        print(f"{N:>6}  {best:>8.3f}  {rate:>12,.0f}  {rate/base_rate:>8.2f}x  {mem_obs:>10.0f}  {mem_lm:>10.0f}")

        del r, buf
    except Exception as e:
        print(f"{N:>6}  FAILED: {e}")

print()
print("mem_obs_MB = T * N * 101 * 17 * 17 * 4 bytes")
print("mem_lm_MB  = T * N * 83521 * 1 byte")
