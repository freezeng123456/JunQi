"""tools/bench_gpu_rollout.py — E2E GpuRollout throughput.

Measures a realistic PPO-style rollout pattern:
  * reset()                          (upload)
  * step() + legal_actions_dense()   in a tight loop

Observations are NOT rebuilt every step here (PPO builds obs only for the
acting seat once per env-step; the full-obs build is a separate, optional
phase that can be rate-limited or sharded by worker).
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import junqi_cuda as _cuda
from junqi_rl.gpu_rollout import GpuRollout


def _pick_actions(ids: np.ndarray, counts: np.ndarray, rng) -> np.ndarray:
    N = ids.shape[0]
    out = np.zeros(N, dtype=np.int32)
    has = counts > 0
    idxs = rng.integers(0, 1 << 30, size=N) % np.maximum(counts, 1)
    picked = ids[np.arange(N), idxs.astype(np.int32)]
    out[has] = picked[has]
    return out


def main() -> None:
    steps = 100
    for N in (256, 1024, 4096):
        r = GpuRollout(num_envs=N)
        r.reset(seed_base=0)
        rng = np.random.default_rng(0xBEEF)

        t_legal = 0.0; t_step = 0.0; t_obs = 0.0

        t0 = time.perf_counter()
        for _ in range(steps):
            # Driver keeps acting_seats=0 (benchmark only — real RL reads turn).
            acting_seats = np.zeros(N, dtype=np.int8)
            s = time.perf_counter()
            ids, counts = r.legal_actions_dense(acting_seats)
            t_legal += time.perf_counter() - s
            actions = _pick_actions(ids, counts, rng)
            s = time.perf_counter()
            r.step(actions)
            t_step += time.perf_counter() - s
        total = time.perf_counter() - t0
        envsteps = steps * N / total

        print(f"N={N:5d}  {steps:3d} steps  total {total*1e3:6.1f} ms  "
              f"→ {envsteps:10,.0f} env·steps/s  "
              f"(legal {t_legal*1e3:6.1f} ms, step {t_step*1e3:6.1f} ms)")

        # Optional: also time a full-obs build to show its cost.
        # Use device_synchronize so we measure the kernel, not the lazy queue.
        for _ in range(3):  # warmup
            _cuda.build_observation_batch_resident(r.state, r.obs, np.int8(2))
        _cuda.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            _cuda.build_observation_batch_resident(r.state, r.obs, np.int8(2))
        _cuda.device_synchronize()
        t_obs_kernel = (time.perf_counter() - t0) / 50
        # Full build+D2H time (what RL code actually pays if it needs host-side obs).
        _cuda.device_synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            r.build_all_seat_observations()   # kernel + copy_to_host
        t_obs_full = (time.perf_counter() - t0) / 10
        print(f"         obs kernel  {t_obs_kernel*1e3:6.2f} ms/call  "
              f"({N/t_obs_kernel:,.0f} env·obs/s)")
        print(f"         obs + D2H   {t_obs_full*1e3:6.2f} ms/call  "
              f"(478 MB D2H at N={N} included)")

        del r


if __name__ == "__main__":
    main()
