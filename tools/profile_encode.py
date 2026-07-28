"""Profile _encode sub-components: CNN vs Transformer."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

device = torch.device('cuda')
N = 64; T = 256
cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
net = JunqiNet(cfg).to(device).eval()

obs_sp = torch.randn(N, 101, 17, 17, device=device, dtype=torch.float16)
obs_gl = torch.randn(N, 28, device=device, dtype=torch.float16)

# Warmup
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    cls, cells = net._encode(obs_sp, obs_gl)
torch.cuda.synchronize()

# 1. CNN stem
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        feat = net.cnn(obs_sp)
torch.cuda.synchronize()
t_cnn = time.perf_counter() - t0

# 2. Patch proj + global proj + pos embed
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    feat = net.cnn(obs_sp)
    B = obs_sp.size(0)
    feat_flat = feat.permute(0, 2, 3, 1).reshape(B, 289, -1)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        cell_tokens = net.patch_proj(feat_flat)
        cls_token = net.global_proj(obs_gl).unsqueeze(1)
        tokens = torch.cat([cls_token, cell_tokens], dim=1) + net.pos_emb
torch.cuda.synchronize()
t_proj = time.perf_counter() - t0

# 3. Transformer layers
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    feat = net.cnn(obs_sp)
    feat_flat = feat.permute(0, 2, 3, 1).reshape(B, 289, -1)
    cell_tokens = net.patch_proj(feat_flat)
    cls_token = net.global_proj(obs_gl).unsqueeze(1)
    tokens = torch.cat([cls_token, cell_tokens], dim=1) + net.pos_emb

torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        t2 = tokens.clone()
        for layer in net.transformer:
            t2 = layer(t2)
        t2 = net.norm_out(t2)
torch.cuda.synchronize()
t_transformer = time.perf_counter() - t0

total = t_cnn + t_proj + t_transformer
print(f"=== _encode sub-profiling (N={N}, T={T}, depth={cfg.depth}) ===")
print(f"  CNN stem:        {t_cnn:.3f}s  {t_cnn/T*1e3:.2f}ms  {t_cnn/total*100:.1f}%")
print(f"  Proj+PosEmb:     {t_proj:.3f}s  {t_proj/T*1e3:.2f}ms  {t_proj/total*100:.1f}%")
print(f"  Transformer:     {t_transformer:.3f}s  {t_transformer/T*1e3:.2f}ms  {t_transformer/total*100:.1f}%")
print(f"  TOTAL:           {total:.3f}s  {total/T*1e3:.2f}ms")

# Also test larger N
for N_test in [128, 256, 512]:
    obs_sp2 = torch.randn(N_test, 101, 17, 17, device=device, dtype=torch.float16)
    obs_gl2 = torch.randn(N_test, 28, device=device, dtype=torch.float16)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        net._encode(obs_sp2, obs_gl2)
    torch.cuda.synchronize()
    T2 = 64
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(T2):
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            net._encode(obs_sp2, obs_gl2)
    torch.cuda.synchronize()
    t_enc = time.perf_counter() - t0
    total_steps = N_test * T2
    print(f"  N={N_test}: _encode = {t_enc/T2*1e3:.2f}ms/step, throughput = {total_steps/t_enc:,.0f} samples/s")
