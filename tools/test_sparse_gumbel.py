"""Verify sparse Gumbel-max sampling correctness + benchmark."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

device = torch.device('cuda')
N = 64; T = 128
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

r = GpuRollout(num_envs=N); r.reset(seed_base=0)
turn_t = r.turn_torch()
obs_sp, obs_gl = r.build_acting_seat_observation_torch(turn_t)
lm = r.legal_mask_canonical_torch_device(turn_t)

# Print legal action stats
n_legal = lm.sum(dim=-1)
print(f"Legal actions per env: min={n_legal.min().item()}, "
      f"max={n_legal.max().item()}, mean={n_legal.float().mean().item():.1f}")

# Verify correctness: 20 trials
for trial in range(20):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        actions, log_probs, values = policy.act(obs_sp, obs_gl, lm)
    # All actions must be legal
    legal_check = lm[torch.arange(N, device=device), actions.long()]
    assert legal_check.all(), f"Trial {trial}: illegal actions!"
    # log_probs must be finite and negative
    assert torch.isfinite(log_probs).all(), f"Trial {trial}: non-finite log_probs"
    assert (log_probs <= 0).all(), f"Trial {trial}: positive log_probs"
print("Correctness: 20 trials passed (all legal, valid log_probs)")

# Benchmark: sparse vs dense Gumbel
# First isolate sampling cost
policy2 = JunqiNet(cfg).to(device).eval()
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    cls, cells = policy2._encode(obs_sp, obs_gl)
    logits = policy2._policy_logits(cells, lm)  # (B, 83521)

logits_f = logits.float()

# Dense Gumbel (old)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    u = torch.rand_like(logits_f).clamp_(1e-10, 1.0)
    gumbel = -torch.log(-torch.log(u))
    a_dense = (logits_f + gumbel).argmax(dim=-1)
    lse_d = logits_f.logsumexp(dim=-1)
    lp_d = logits_f.gather(1, a_dense.unsqueeze(1)).squeeze(1) - lse_d
torch.cuda.synchronize()
t_dense = time.perf_counter() - t0

# Sparse Gumbel (new)
K = int(lm.sum(dim=-1).max().item())
_, legal_idx = lm.int().topk(K, dim=-1, sorted=False)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    ll = logits_f.gather(1, legal_idx)
    u = torch.rand_like(ll).clamp_(1e-10, 1.0)
    gumbel = -torch.log(-torch.log(u))
    chosen = (ll + gumbel).argmax(dim=-1)
    a_sparse = legal_idx.gather(1, chosen.unsqueeze(1)).squeeze(1)
    cl = ll.gather(1, chosen.unsqueeze(1)).squeeze(1)
    lse_s = ll.logsumexp(dim=-1)
    lp_s = cl - lse_s
torch.cuda.synchronize()
t_sparse = time.perf_counter() - t0

print(f"\nSampling benchmark (T={T}):")
print(f"  Dense  (83521 dim): {t_dense:.3f}s  {t_dense/T*1e3:.2f}ms/step")
print(f"  Sparse (K={K} dim):  {t_sparse:.3f}s  {t_sparse/T*1e3:.2f}ms/step")
print(f"  Speedup:            {t_dense/t_sparse:.2f}x")

# End-to-end (includes topk extraction cost)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    n_l = lm.sum(dim=-1)
    K2 = int(n_l.max().item())
    _, lidx = lm.int().topk(K2, dim=-1, sorted=False)
    ll = logits_f.gather(1, lidx)
    u = torch.rand_like(ll).clamp_(1e-10, 1.0)
    gumbel = -torch.log(-torch.log(u))
    chosen = (ll + gumbel).argmax(dim=-1)
    a = lidx.gather(1, chosen.unsqueeze(1)).squeeze(1)
    cl = ll.gather(1, chosen.unsqueeze(1)).squeeze(1)
    lse = ll.logsumexp(dim=-1)
    lp = cl - lse
torch.cuda.synchronize()
t_sparse_full = time.perf_counter() - t0
print(f"  Sparse+topk:        {t_sparse_full:.3f}s  {t_sparse_full/T*1e3:.2f}ms/step")
print(f"  vs Dense speedup:   {t_dense/t_sparse_full:.2f}x")
