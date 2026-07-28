"""Profile with full act() compile + reduce-overhead."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig, FLAT_ACTION_DIM
from junqi_rl.training import RolloutBufferGPU
from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2

N = 256; T = 128
device = torch.device('cuda')
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

r = GpuRollout(num_envs=N)
buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
r.reset(seed_base=0)

# Warmup (compile happens here)
collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=0, reset_at_start=False)
torch.cuda.synchronize()

# Full pipeline time
times = []
for trial in range(5):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    collect_rollout_gpu_v2(r, policy, buf, device='cuda', seed_base=trial+1, reset_at_start=False)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
best = min(times)
total = N * T
rate = total / best

# Component isolation — compiled act() as single unit
turn_t = r.turn_torch()
obs_sp, obs_gl = r.build_acting_seat_observation_torch(turn_t)
lm = r.legal_mask_canonical_torch_device(turn_t)

# act() — compiled with reduce-overhead (CUDA graph)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        a, lp, v = policy.act(obs_sp, obs_gl, lm)
torch.cuda.synchronize()
t_act = time.perf_counter() - t0

# obs
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    sp, gl = r.build_acting_seat_observation_torch(turn_t)
torch.cuda.synchronize()
t_obs = time.perf_counter() - t0

# legal mask
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    lm2 = r.legal_mask_canonical_torch_device(turn_t)
torch.cuda.synchronize()
t_lm = time.perf_counter() - t0

# step
acts_z = torch.zeros(N, dtype=torch.int32, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    res = r.step_device_torch(acts_z, turn_t)
torch.cuda.synchronize()
t_step = time.perf_counter() - t0

# buffer writes
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    buf.obs_spatial[0] = obs_sp.half()
    buf.obs_global[0] = obs_gl.half()
    buf.legal_mask[0] = lm
    buf.actions[0] = a
    buf.log_probs[0] = lp.float()
    buf.values[0] = v.float().squeeze(-1) if v.dim() > 1 else v.float()
    buf.rewards[0] = torch.zeros(N, device=device)
    buf.dones[0] = torch.zeros(N, dtype=torch.bool, device=device)
    buf.seats[0] = turn_t
torch.cuda.synchronize()
t_buf = time.perf_counter() - t0

# reset (no-op on non-terminated)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    r.reset_terminated_device(seed=0)
torch.cuda.synchronize()
t_reset = time.perf_counter() - t0

total_iso = t_act + t_obs + t_lm + t_step + t_buf + t_reset
print(f"=== Post-compile profiling (N={N}, T={T}, CUDA Graph capture) ===\n")
print(f"Full pipeline: {best:.3f}s = {rate:,.0f} steps/s")
print(f"  trials: {[f'{t:.3f}' for t in times]}\n")
print(f"Component breakdown ({T} iters each):")
print(f"  act() [compiled]:  {t_act:.3f}s  {t_act/T*1e3:.2f}ms  {t_act/total_iso*100:.1f}%")
print(f"  obs single-seat:   {t_obs:.3f}s  {t_obs/T*1e3:.2f}ms  {t_obs/total_iso*100:.1f}%")
print(f"  legal_mask:        {t_lm:.3f}s  {t_lm/T*1e3:.2f}ms  {t_lm/total_iso*100:.1f}%")
print(f"  step+reward:       {t_step:.3f}s  {t_step/T*1e3:.2f}ms  {t_step/total_iso*100:.1f}%")
print(f"  buffer writes:     {t_buf:.3f}s  {t_buf/T*1e3:.2f}ms  {t_buf/total_iso*100:.1f}%")
print(f"  reset:             {t_reset:.3f}s  {t_reset/T*1e3:.2f}ms  {t_reset/total_iso*100:.1f}%")
print(f"  isolated total:    {total_iso:.3f}s")
print(f"\n  Pipeline overhead: {best - total_iso:.3f}s (Python loop + sync)")

# Games per second
avg_game = 780
games_s = rate / avg_game
print(f"\n=== Games/second: {rate:,.0f} steps/s / {avg_game} steps/game = {games_s:.1f} games/s ===")
