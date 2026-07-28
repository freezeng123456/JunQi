"""Test alternative sparse extraction strategies."""
import sys, time
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch

device = torch.device('cuda')
B, D = 64, 83521
K_avg = 50

# Create realistic legal mask
mask = torch.zeros(B, D, dtype=torch.bool, device=device)
for i in range(B):
    n = torch.randint(30, 70, (1,)).item()
    idx = torch.randperm(D, device=device)[:n]
    mask[i, idx] = True

logits = torch.randn(B, D, device=device)
logits.masked_fill_(~mask, float('-inf'))

T = 256

# Method 1: topk on mask.int()
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    K = int(mask.sum(dim=-1).max().item())
    _, idx = mask.int().topk(K, dim=-1, sorted=False)
torch.cuda.synchronize()
print(f"topk(mask.int()): {(time.perf_counter()-t0)/T*1e3:.2f}ms  K={K}")

# Method 2: where + pad (per-batch, slow)
# Skip — Python loop too slow

# Method 3: masked_select + reshape (doesn't work for variable lengths)

# Method 4: Replace the full logits with masked logits directly
# Instead of extracting sparse indices, just do operations on masked logits
# where illegal = -inf. The argmax/logsumexp naturally ignore -inf.
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    # This is the dense approach but cheaper than topk
    u = torch.rand(B, D, device=device).clamp_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))
    # Only add gumbel where legal, skip the rest
    g.masked_fill_(~mask, float('-inf'))  # illegal gumbel = -inf, so argmax ignores
    a = (logits + g).argmax(dim=-1)
    cl = logits.gather(1, a.unsqueeze(1)).squeeze(1)
    lse = logits.logsumexp(dim=-1)
    lp = cl - lse
torch.cuda.synchronize()
print(f"dense gumbel+mask: {(time.perf_counter()-t0)/T*1e3:.2f}ms")

# Method 5: Smaller random — only generate K random numbers
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    K = int(mask.sum(dim=-1).max().item())
    _, idx = mask.int().topk(K, dim=-1, sorted=False)
    ll = logits.gather(1, idx)  # (B, K)
    u = torch.rand(B, K, device=device).clamp_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))
    chosen = (ll + g).argmax(dim=-1)
    a = idx.gather(1, chosen.unsqueeze(1)).squeeze(1)
    cl = ll.gather(1, chosen.unsqueeze(1)).squeeze(1)
    lse = ll.logsumexp(dim=-1)
    lp = cl - lse
torch.cuda.synchronize()
print(f"topk+sparse:       {(time.perf_counter()-t0)/T*1e3:.2f}ms")

# Method 6: Use the mask to generate gumbel only on legal positions
# But fill illegal with -inf BEFORE argmax (same as dense gumbel+mask)
# The key insight: logits already have -inf for illegal actions.
# So logits + Gumbel(0,1) will be -inf + finite = -inf for illegal.
# We don't need to mask the gumbel separately!
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    u = torch.rand(B, D, device=device).clamp_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))
    a = (logits + g).argmax(dim=-1)  # -inf + g = -inf, correct!
    cl = logits.gather(1, a.unsqueeze(1)).squeeze(1)
    lse = logits.logsumexp(dim=-1)
    lp = cl - lse
torch.cuda.synchronize()
print(f"direct gumbel:     {(time.perf_counter()-t0)/T*1e3:.2f}ms (no mask needed)")

# Method 7: Only optimize logsumexp via sparse
# argmax can stay dense (it's fast), just optimize logsumexp
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(T):
    u = torch.rand(B, D, device=device).clamp_(1e-10, 1.0)
    g = -torch.log(-torch.log(u))
    a = (logits + g).argmax(dim=-1)
    # Use masked logits for logsumexp — but logits already have -inf
    # so logsumexp naturally handles it (exp(-inf)=0)
    cl = logits.gather(1, a.unsqueeze(1)).squeeze(1)
    lse = logits.logsumexp(dim=-1)
    lp = cl - lse
torch.cuda.synchronize()
print(f"dense gumbel+lse:  {(time.perf_counter()-t0)/T*1e3:.2f}ms")

print("\nConclusion: topk extraction dominates; dense gumbel with -inf is fastest.")
