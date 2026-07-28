"""Deep sub-component profiling for optimized V2 pipeline."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch, numpy as np
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from torch.distributions import Categorical

device = torch.device('cuda')
N = 64; T = 128
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()
policy._encode = torch.compile(policy._encode)
policy._policy_logits = torch.compile(policy._policy_logits)

r = GpuRollout(num_envs=N); r.reset(seed_base=0)
turn_t = r.turn_torch()

# Warmup
obs_sp, obs_gl = r.build_acting_seat_observation_torch(turn_t)
lm = r.legal_mask_canonical_torch_device(turn_t)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    a, lp, v = policy.act(obs_sp, obs_gl, lm)
torch.cuda.synchronize()

results = {}

# 1. _encode
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        cls, cells = policy._encode(obs_sp, obs_gl)
torch.cuda.synchronize()
results['encode'] = time.perf_counter() - t0

# 2. _policy_logits
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    cls, cells = policy._encode(obs_sp, obs_gl)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        logits = policy._policy_logits(cells, lm)
torch.cuda.synchronize()
results['policy_logits'] = time.perf_counter() - t0

# 3. log_softmax
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        lp_full = logits.log_softmax(dim=-1)
torch.cuda.synchronize()
results['log_softmax'] = time.perf_counter() - t0

# 4. Categorical sample
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad():
        dist = Categorical(logits=logits.float())
        actions = dist.sample()
torch.cuda.synchronize()
results['categorical_sample'] = time.perf_counter() - t0

# 5. value head
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        val = policy._value(cls)
torch.cuda.synchronize()
results['value_head'] = time.perf_counter() - t0

# 6. obs
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    obs_sp2, obs_gl2 = r.build_acting_seat_observation_torch(turn_t)
torch.cuda.synchronize()
results['obs_single_seat'] = time.perf_counter() - t0

# 7. legal mask
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    lm2 = r.legal_mask_canonical_torch_device(turn_t)
torch.cuda.synchronize()
results['legal_mask'] = time.perf_counter() - t0

# 8. step
acts = torch.zeros(N, dtype=torch.int32, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    res = r.step_device_torch(acts, turn_t)
torch.cuda.synchronize()
results['step'] = time.perf_counter() - t0

# 9. buffer obs write
buf = torch.zeros(T, N, 101, 17, 17, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
for t_i in range(T):
    buf[t_i] = obs_sp
torch.cuda.synchronize()
results['buf_obs'] = time.perf_counter() - t0

# 10. buffer lm write
buf_lm = torch.zeros(T, N, 83521, dtype=torch.bool, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
for t_i in range(T):
    buf_lm[t_i] = lm
torch.cuda.synchronize()
results['buf_lm'] = time.perf_counter() - t0

total = sum(results.values())
print(f"=== Sub-component profiling (N={N}, T={T} iters) ===\n")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    pct = v / total * 100
    ms = v / T * 1e3
    print(f"  {k:25s}: {v:7.3f}s  {ms:6.2f}ms/step  {pct:5.1f}%")
print(f"\n  {'TOTAL':25s}: {total:7.3f}s")
print(f"\n=== Memory (N={N}, T={T}) ===")
print(f"  buf obs_spatial: {T*N*101*17*17*4/1e6:.0f} MB")
print(f"  buf legal_mask:  {T*N*83521/1e6:.0f} MB")
print(f"  At N=1024: obs={T*1024*101*17*17*4/1e9:.1f}GB lm={T*1024*83521/1e9:.1f}GB")
