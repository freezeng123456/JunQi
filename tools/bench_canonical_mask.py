"""Benchmark new canonical legal mask kernel vs old D2H+scatter path."""
import torch, numpy as np, time
from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.gpu_collector import build_legal_mask_batch_gpu_torch

N = 1024
r = GpuRollout(num_envs=N)
r.reset(seed_base=42)

# Step a few times to diversify board states
for _ in range(5):
    acts = np.zeros(N, dtype=np.int32)
    ids, counts = r.legal_actions_dense(np.zeros(N, dtype=np.int8))
    for i in range(N):
        if counts[i] > 0:
            acts[i] = ids[i, np.random.randint(counts[i])]
    r.step(acts)

turn_np = np.asarray(r.state.copy_turn_to_host(), dtype=np.int8).reshape(N)
term_np = np.zeros(N, dtype=bool)

# Warmup
for _ in range(3):
    r.legal_mask_canonical_torch(turn_np)
    build_legal_mask_batch_gpu_torch(r, turn_np, term_np, torch.device('cuda'))
torch.cuda.synchronize()

REPS = 50

# New path
t0 = time.perf_counter()
for _ in range(REPS):
    lm = r.legal_mask_canonical_torch(turn_np)
torch.cuda.synchronize()
new_ms = (time.perf_counter() - t0) / REPS * 1000

# Old path
t0 = time.perf_counter()
for _ in range(REPS):
    lm_old = build_legal_mask_batch_gpu_torch(r, turn_np, term_np, torch.device('cuda'))
torch.cuda.synchronize()
old_ms = (time.perf_counter() - t0) / REPS * 1000

print(f'N={N}')
print(f'  New (canonical kernel): {new_ms:.2f} ms/call')
print(f'  Old (D2H+scatter+H2D): {old_ms:.2f} ms/call')
print(f'  Speedup: {old_ms/new_ms:.1f}x')
