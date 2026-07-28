import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch, numpy as np
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

device = torch.device('cuda')
N = 64
T = 128

cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64, n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

rollout = GpuRollout(num_envs=N)
rollout.reset(seed_base=0)

# Warm up
for _ in range(5):
    turn_t = rollout.turn_torch()
    acting_t = turn_t.clone()
    sp, gl = rollout.build_all_seat_observations_torch()
    lm = rollout.legal_mask_canonical_torch_device(acting_t)
    actions = torch.zeros(N, dtype=torch.int32, device='cuda')
    result = rollout.step_device_torch(actions, acting_t)
torch.cuda.synchronize()

# Profile each component
timings = {}

# 1. Observation building (4 seats)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(T):
    sp, gl = rollout.build_all_seat_observations_torch()
torch.cuda.synchronize()
timings['obs_4seat'] = time.perf_counter() - t0

# 2. Observation slicing (acting seat)
env_idx_t = torch.arange(N, device=device)
acting_long = torch.zeros(N, dtype=torch.long, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(T):
    sp, gl = rollout.build_all_seat_observations_torch()
    obs_sp = sp[env_idx_t, acting_long].contiguous()
    obs_gl = gl[env_idx_t, acting_long].contiguous()
torch.cuda.synchronize()
timings['obs_4seat+slice'] = time.perf_counter() - t0

# 3. Legal mask
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(T):
    lm = rollout.legal_mask_canonical_torch_device(acting_t)
torch.cuda.synchronize()
timings['legal_mask'] = time.perf_counter() - t0

# 4. Policy forward pass
obs_sp = torch.randn(N, 101, 17, 17, device=device)
obs_gl = torch.randn(N, 28, device=device)
lm = torch.ones(N, 83521, dtype=torch.bool, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(T):
        actions, logp, vals = policy.act(obs_sp, obs_gl, lm)
torch.cuda.synchronize()
timings['policy_fwd'] = time.perf_counter() - t0

# 5. Step (rotate + step + reward)
actions_t = torch.zeros(N, dtype=torch.int32, device=device)
seats_t = torch.zeros(N, dtype=torch.int8, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(T):
    result = rollout.step_device_torch(actions_t, seats_t)
torch.cuda.synchronize()
timings['step_device'] = time.perf_counter() - t0

# 6. Reset (unconditional, no-ops for non-terminated)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(T):
    rollout.reset_terminated_device(seed=0)
torch.cuda.synchronize()
timings['reset_device'] = time.perf_counter() - t0

# 7. Buffer add overhead (simulate)
buf_sp = torch.zeros(T, N, 101, 17, 17, device=device)
buf_gl = torch.zeros(T, N, 28, device=device)
buf_lm = torch.zeros(T, N, 83521, dtype=torch.bool, device=device)
buf_act = torch.zeros(T, N, dtype=torch.int32, device=device)
buf_lp = torch.zeros(T, N, device=device)
buf_val = torch.zeros(T, N, device=device)
buf_rew = torch.zeros(T, N, device=device)
buf_done = torch.zeros(T, N, dtype=torch.bool, device=device)
buf_seats = torch.zeros(T, N, dtype=torch.int8, device=device)

torch.cuda.synchronize()
t0 = time.perf_counter()
for t in range(T):
    buf_sp[t] = obs_sp
    buf_gl[t] = obs_gl
    buf_lm[t] = lm
    buf_act[t] = actions_t
    buf_lp[t] = logp if isinstance(logp, torch.Tensor) else torch.zeros(N, device=device)
    buf_val[t] = vals.squeeze(-1) if isinstance(vals, torch.Tensor) else torch.zeros(N, device=device)
    buf_rew[t] = torch.zeros(N, device=device)
    buf_done[t] = torch.zeros(N, dtype=torch.bool, device=device)
    buf_seats[t] = seats_t
torch.cuda.synchronize()
timings['buffer_add'] = time.perf_counter() - t0

total = sum(timings.values())
print(f"\n{'='*60}")
print(f"Per-component timing (N={N}, T={T} iterations each)")
print(f"{'='*60}")
for k, v in sorted(timings.items(), key=lambda x: -x[1]):
    pct = v / total * 100
    per_step_us = v / T * 1e6
    print(f"  {k:25s}: {v:7.3f}s ({pct:5.1f}%)  {per_step_us:8.1f} us/step")
print(f"  {'TOTAL':25s}: {total:7.3f}s")

# Memory analysis
print(f"\n{'='*60}")
print(f"Memory analysis (per step, N={N})")
print(f"{'='*60}")
obs_sp_mb = N * 4 * 101 * 17 * 17 * 4 / 1e6
obs_sp_1seat_mb = N * 101 * 17 * 17 * 4 / 1e6
obs_gl_mb = N * 4 * 28 * 4 / 1e6
legal_mask_mb = N * 83521 / 1e6
step_result_mb = N * (1+1+1+1+1+4) / 1e6
buf_lm_total = T * N * 83521 / 1e6

print(f"  obs spatial (4 seats): {obs_sp_mb:.1f} MB")
print(f"  obs spatial (1 seat):  {obs_sp_1seat_mb:.1f} MB")
print(f"  obs global (4 seats):  {obs_gl_mb:.3f} MB")
print(f"  legal mask:            {legal_mask_mb:.1f} MB")
print(f"  step result:           {step_result_mb:.3f} MB")
print(f"  buffer legal_mask:     {buf_lm_total:.1f} MB (T*N*83521)")
print(f"  obs ratio 4-seat/1:    4.0x wasted bandwidth")

# Compare with Ataraxos scale
print(f"\n{'='*60}")
print(f"Comparison with Ataraxos dimensions")
print(f"{'='*60}")
print(f"  JunQi action space: 83,521 (289x289)")
print(f"  Ataraxos action space: 1,800 (30x60)")
print(f"  Ratio: {83521/1800:.0f}x")
print(f"  JunQi obs channels: 101")
print(f"  JunQi board: 17x17 = 289 cells")
print(f"  Ataraxos board: 10x10 = 100 cells")
print(f"  JunQi obs per env: {101*17*17*4/1024:.1f} KB")
print(f"  Ataraxos obs per env: ~{42*10*10*4/1024:.1f} KB")
