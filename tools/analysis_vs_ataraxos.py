"""JunQi game-length and throughput analysis."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch, numpy as np
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

device = torch.device('cuda')

# Measure average game length with random play
print("=== Game Length Analysis (random play) ===")
np.random.seed(42)
r = GpuRollout(num_envs=64)
r.reset(seed_base=0)
total_terms = 0
total_steps = 0
for step in range(4000):
    turn_t = r.turn_torch()
    lm = r.legal_mask_canonical_torch_device(turn_t)
    actions = torch.zeros(64, dtype=torch.int32, device='cuda')
    for i in range(64):
        legal = lm[i].nonzero(as_tuple=False).squeeze(-1)
        if len(legal) > 0:
            idx = int(np.random.randint(len(legal)))
            actions[i] = legal[idx]
    result = r.step_device_torch(actions, turn_t)
    n_term = result['terminated'].sum().item()
    if n_term > 0:
        total_terms += n_term
        r.reset_terminated_device(seed=step)
    total_steps += 64
avg_game_len = total_steps / max(total_terms, 1)
print(f"  Total env-steps: {total_steps:,}")
print(f"  Total terminations: {total_terms}")
print(f"  Avg game length: {avg_game_len:.0f} env-steps")
print(f"  (4 seats take turns, so {avg_game_len:.0f} steps = ~{avg_game_len/4:.0f} moves per seat)")
print()

# Throughput at different N
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()
n_params = sum(p.numel() for p in policy.parameters())

print(f"=== Throughput (small net, {n_params:,} params) ===")
for N in [64, 256, 512]:
    T = 128
    r2 = GpuRollout(num_envs=N)
    buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
    r2.reset(seed_base=0)
    collect_rollout_gpu_v2(r2, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
    torch.cuda.synchronize()
    times = []
    for trial in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        collect_rollout_gpu_v2(r2, policy, buf, device='cuda', seed_base=trial+1, reset_at_start=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    best = min(times)
    rate = N * T / best
    games_per_s = rate / avg_game_len
    print(f"  N={N:>4}: {rate:>10,.0f} steps/s = {games_per_s:.1f} games/s")
    del r2, buf; torch.cuda.empty_cache()

print()
print("=== Dimensions Comparison ===")
print(f"  JunQi:    17x17 board, 4 players, 120 pieces, action_space=83521, obs=101ch")
print(f"  Ataraxos: 10x10 board, 2 players,  80 pieces, action_space=1800,  obs=539ch")
print(f"  Action ratio: {83521/1800:.0f}x larger")
print(f"  Board ratio:  {289/100:.1f}x larger")
print(f"  Players:      4 vs 2 (2x)")
