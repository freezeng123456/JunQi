"""Verify Gumbel-max sampling correctness + benchmark."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from torch.distributions import Categorical

device = torch.device('cuda')
N = 64; T = 128
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to(device).eval()

r = GpuRollout(num_envs=N); r.reset(seed_base=0)
turn_t = r.turn_torch()
obs_sp, obs_gl = r.build_acting_seat_observation_torch(turn_t)
lm = r.legal_mask_canonical_torch_device(turn_t)

# Verify correctness
for trial in range(10):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        actions, log_probs, values = policy.act(obs_sp, obs_gl, lm)
    legal_check = lm[torch.arange(N, device=device), actions.long()]
    assert legal_check.all(), f'Trial {trial}: illegal actions!'
    assert torch.isfinite(log_probs).all(), f'Trial {trial}: non-finite log_probs'
    assert (log_probs <= 0).all(), f'Trial {trial}: positive log_probs'
print('Gumbel-max: all 10 trials passed (legal actions, valid log_probs)')

# Benchmark isolated: old Categorical vs new Gumbel-max
policy2 = JunqiNet(cfg).to(device).eval()
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    cls, cells = policy2._encode(obs_sp, obs_gl)
    logits = policy2._policy_logits(cells, lm)

# Old path
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad():
        dist = Categorical(logits=logits.float())
        a_old = dist.sample()
        lp_full = logits.float().log_softmax(dim=-1)
        lp_old = lp_full.gather(1, a_old.unsqueeze(1)).squeeze(1)
torch.cuda.synchronize()
t_old = time.perf_counter() - t0

# New path
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad():
        u = torch.rand_like(logits.float()).clamp_(1e-10, 1.0)
        gumbel = -torch.log(-torch.log(u))
        a_new = (logits.float() + gumbel).argmax(dim=-1)
        lse = logits.float().logsumexp(dim=-1)
        lp_new = logits.float().gather(1, a_new.unsqueeze(1)).squeeze(1) - lse
torch.cuda.synchronize()
t_new = time.perf_counter() - t0

print(f'Old (Categorical+log_softmax): {t_old:.3f}s ({t_old/T*1e3:.2f}ms/step)')
print(f'New (Gumbel-max+logsumexp):    {t_new:.3f}s ({t_new/T*1e3:.2f}ms/step)')
print(f'Speedup: {t_old/t_new:.2f}x')
