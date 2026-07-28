"""Final scaling benchmark with all optimizations."""
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

print(f"{'N':>6}  {'time':>8}  {'steps/s':>12}  {'obs_MB':>8}  {'lm_MB':>8}  {'total_MB':>10}")
print("-" * 66)

for N in [32, 64, 128, 256, 512]:
    try:
        torch.cuda.empty_cache()
        r = GpuRollout(num_envs=N)
        buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
        r.reset(seed_base=0)

        # Warmup (includes torch.compile)
        collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
        torch.cuda.synchronize()

        times = []
        for trial in range(3):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=trial+1, reset_at_start=False)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        best = min(times)
        total = N * T
        rate = total / best
        obs_mb = T * N * 101 * 17 * 17 * 2 / 1e6
        lm_mb = T * N * 83521 / 1e6
        tot_mb = obs_mb + lm_mb

        print(f"{N:>6}  {best:>8.3f}  {rate:>12,.0f}  {obs_mb:>8.0f}  {lm_mb:>8.0f}  {tot_mb:>10.0f}")
        del r, buf
    except Exception as e:
        print(f"{N:>6}  FAILED: {e}")
