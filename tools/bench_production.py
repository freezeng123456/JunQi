"""Benchmark with production-size network."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig, FLAT_ACTION_DIM
from junqi_rl.training import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

T = 128
device = torch.device('cuda')

configs = {
    "small (bench)": JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64, n_head=2, ff_factor=2, action_key_dim=16),
    "medium": JunqiNetConfig(cnn_channels=64, cnn_layers=2, depth=4, embed_dim=128, n_head=4, ff_factor=4, action_key_dim=32),
    "production": JunqiNetConfig(cnn_channels=128, cnn_layers=3, depth=6, embed_dim=256, n_head=8, ff_factor=4, action_key_dim=64),
}

for name, cfg in configs.items():
    policy = JunqiNet(cfg).to(device).eval()
    n_params = policy.num_parameters()
    print(f"\n=== {name}: {n_params:,} params, depth={cfg.depth}, embed={cfg.embed_dim} ===")

    for N in [64, 256]:
        try:
            torch.cuda.empty_cache()
            r = GpuRollout(num_envs=N)
            buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
            r.reset(seed_base=0)

            # Warmup + compile
            collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
            torch.cuda.synchronize()
            # Need to reset compile state for next policy
            policy._compiled_for_collect = False

            # Re-warmup with compile
            policy.act = torch.compile(policy.act, mode="reduce-overhead")
            policy._compiled_for_collect = True
            collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
            torch.cuda.synchronize()

            ts = []
            for trial in range(3):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=trial+1, reset_at_start=False)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            best = min(ts)
            rate = N * T / best
            games_s = rate / 780
            print(f"  N={N:>4}: {best:.3f}s = {rate:>10,.0f} steps/s = {games_s:.1f} games/s")
            del r, buf
        except Exception as e:
            print(f"  N={N}: FAILED: {e}")
    del policy
