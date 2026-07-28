"""Post-compact-sequence deep profiling."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch, numpy as np
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig, FLAT_ACTION_DIM

device = torch.device('cuda')
N = 256; T = 128
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

# 1. _encode (CNN + Transformer on 130 tokens)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        cls, cells = policy._encode(obs_sp, obs_gl)
torch.cuda.synchronize()
results['encode(130tok)'] = time.perf_counter() - t0

# 2. _policy_logits (129x129 bmm)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    cls, cells = policy._encode(obs_sp, obs_gl)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        logits = policy._policy_logits(cells, lm)
torch.cuda.synchronize()
results['policy_logits(129x129)'] = time.perf_counter() - t0

# 3. Gumbel sample + logsumexp (16641 dim)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad():
        lf = logits.float()
        u = torch.rand_like(lf).clamp_(1e-10, 1.0)
        g = -torch.log(-torch.log(u))
        acts = (lf + g).argmax(dim=-1)
        lse = lf.logsumexp(dim=-1)
        lp2 = lf.gather(1, acts.unsqueeze(1)).squeeze(1) - lse
torch.cuda.synchronize()
results['gumbel+lse(16641)'] = time.perf_counter() - t0

# 4. value head
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        val = policy._value(cls)
torch.cuda.synchronize()
results['value_head'] = time.perf_counter() - t0

# 5. obs single-seat
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    sp2, gl2 = r.build_acting_seat_observation_torch(turn_t)
torch.cuda.synchronize()
results['obs_single_seat'] = time.perf_counter() - t0

# 6. legal mask
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    lm2 = r.legal_mask_canonical_torch_device(turn_t)
torch.cuda.synchronize()
results['legal_mask(16641)'] = time.perf_counter() - t0

# 7. step
acts_z = torch.zeros(N, dtype=torch.int32, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    res = r.step_device_torch(acts_z, turn_t)
torch.cuda.synchronize()
results['step+reward'] = time.perf_counter() - t0

# 8. buffer writes
buf_sp = torch.zeros(T, N, 101, 17, 17, dtype=torch.float16, device=device)
buf_lm = torch.zeros(T, N, FLAT_ACTION_DIM, dtype=torch.bool, device=device)
torch.cuda.synchronize(); t0 = time.perf_counter()
for t_i in range(T):
    buf_sp[t_i] = obs_sp
    buf_lm[t_i] = lm
torch.cuda.synchronize()
results['buffer_writes'] = time.perf_counter() - t0

# 9. reset
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    r.reset_terminated_device(seed=0)
torch.cuda.synchronize()
results['reset'] = time.perf_counter() - t0

total = sum(results.values())
print(f"=== Post-compact profiling (N={N}, T={T}, action_dim={FLAT_ACTION_DIM}) ===\n")
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    pct = v / total * 100
    ms = v / T * 1e3
    print(f"  {k:30s}: {v:7.3f}s  {ms:6.2f}ms/step  {pct:5.1f}%")
print(f"\n  {'TOTAL':30s}: {total:7.3f}s")

# Memory
print(f"\n=== Buffer memory (N={N}, T={T}) ===")
obs_mb = T*N*101*17*17*2/1e6
lm_mb = T*N*FLAT_ACTION_DIM/1e6
print(f"  obs_spatial (fp16): {obs_mb:.0f} MB")
print(f"  legal_mask (bool):  {lm_mb:.0f} MB")
print(f"  total ~:            {obs_mb+lm_mb:.0f} MB")
