"""Check available acceleration backends."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Check numba
try:
    import numba
    print(f"numba: {numba.__version__}")
except ImportError:
    print("numba: NOT available")

# Check cython
try:
    import Cython
    print(f"Cython: {Cython.__version__}")
except ImportError:
    print("Cython: NOT available")

# Check torch
try:
    import torch
    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError:
    print("torch: NOT available")

# Check numpy version
import numpy as np
print(f"numpy: {np.__version__}")

# Check cpython version
print(f"Python: {sys.version}")

# Quick micro-benchmark: how many simple np ops/sec?
import time
N = 1024
K = 289
a = np.random.randint(-1, 120, (N, K), dtype=np.int16)
b = np.zeros((N, K), dtype=bool)
t0 = time.perf_counter()
for _ in range(10000):
    b[:] = a >= 0
elapsed = time.perf_counter() - t0
print(f"10k x (N={N}, K={K}) bool broadcast: {elapsed*1000:.1f}ms total, {elapsed/10000*1000:.3f}ms each")
