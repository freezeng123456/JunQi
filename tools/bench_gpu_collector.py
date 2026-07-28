"""tools/bench_gpu_collector.py — Throughput benchmark for the GPU PPO collector.

This benchmark **does not** require torch; it wires a minimal mock policy
that emits random (masked) actions using NumPy.  The point is to measure
end-to-end collector overhead including:

  * per-step full-seat observation build + host copy
  * acting-seat slicing
  * legal-mask construction in canonical frame
  * canonical → world action un-rotation
  * GPU step_batch
  * per-env reward/done bookkeeping

For real PPO this loop additionally uploads obs to GPU for policy forward +
backward, which is typically the dominant cost; here we isolate the
non-learner cost.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from junqi_rl.action_lut import ROTATE_LUT
from junqi_rl.gpu_rollout import GpuRollout


def _load_gpu_collector():
    """Import gpu_collector without touching torch-dependent siblings."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "junqi_rl", "training",
        "gpu_collector.py",
    )
    spec = importlib.util.spec_from_file_location("_bench_gc", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class MockPolicy:
    """Picks a uniformly random legal canonical action per env.

    Mimics the ``JunqiNet.act`` contract: returns (actions, log_probs,
    values) as "tensors" (actually numpy arrays wrapped in a tiny shim
    that exposes ``.detach().cpu().numpy()`` so the collector doesn't
    have to special-case).
    """

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def eval(self) -> None: ...

    def act(self, sp, gl, lm):
        mask = lm.numpy().astype(bool)
        N = mask.shape[0]
        actions = np.zeros(N, dtype=np.int32)
        for i in range(N):
            idxs = np.flatnonzero(mask[i])
            if idxs.size:
                actions[i] = int(self.rng.choice(idxs))
        log_probs = np.zeros(N, dtype=np.float32)
        values = np.zeros(N, dtype=np.float32)
        return _MockTensor(actions), _MockTensor(log_probs), _MockTensor(values)


class _MockTensor:
    """Pretends to be a torch tensor for the collector's .detach().cpu().numpy()
    and .numpy() paths."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def numpy(self):
        return self._arr


class _MockTorchModule:
    """Bare-minimum torch shim for from_numpy / no_grad / device."""

    class device:
        def __init__(self, name): self.name = name

    @staticmethod
    def from_numpy(a):
        return _MockTensor(np.asarray(a))

    class no_grad:
        def __enter__(self): return self
        def __exit__(self, *a): return False


def patch_torch(mod) -> None:
    """Inject the mock torch module so collect_rollout_gpu runs without torch."""
    mod.torch = _MockTorchModule()
    mod._TORCH_AVAILABLE = True


def bench(N: int, T: int) -> None:
    gc = _load_gpu_collector()
    patch_torch(gc)

    # Import RolloutBuffer manually without triggering torch via __init__.
    # We substitute a tiny stand-in that exposes just the fields the collector
    # writes to, because the real RolloutBuffer requires torch.
    class StubBuffer:
        def __init__(self, N, T):
            self.num_envs = N
            self.steps_per_env = T
            self.rewards = np.zeros((T, N), dtype=np.float32)
            self.dones = np.zeros((T, N), dtype=bool)
        def reset(self): ...
        def add(self, **kw): ...
        def compute_returns(self, last_values): ...

    rollout = GpuRollout(num_envs=N)
    buffer = StubBuffer(N, T)
    policy = MockPolicy(seed=0xC0FFEE)

    # Warmup
    gc.collect_rollout_gpu(rollout, policy, buffer, device="cpu", seed_base=0)

    # Timed run
    t0 = time.perf_counter()
    gc.collect_rollout_gpu(rollout, policy, buffer, device="cpu", seed_base=1,
                           reset_at_start=False)
    total = time.perf_counter() - t0
    envsteps = N * T / total
    print(f"N={N:5d}  T={T:4d}  total {total*1e3:7.1f} ms  "
          f"→ {envsteps:11,.0f} env·steps/s")


def main() -> None:
    for N in (256, 1024, 4096):
        T = 16
        bench(N, T)


if __name__ == "__main__":
    main()
