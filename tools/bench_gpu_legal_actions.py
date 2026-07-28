"""tools/bench_gpu_legal_actions.py — Benchmark M2 GPU legal_action throughput.

Measures throughput of ``junqi_cuda.legal_action_ids_batch`` at several batch
sizes against the Phase 1b target of ≥500 k plays/sec at N=1024.

The benchmark separates:
  1. Pure GPU kernel time (state already on device).
  2. End-to-end time including state pack + H2D + kernel + D2H.

Both numbers matter: (1) is the theoretical GPU ceiling, (2) is the practical
number users will see when calling from Python.
"""

from __future__ import annotations

import os
import random
import sys
import time

# Make the compiled .so at repo root importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

import junqi_cuda as _cuda
from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays, _pack_state_arrays_lite


_NUM_WARMUP = 5
_NUM_ITERS  = 50


def _make_envs(n: int, seed_base: int = 0xABCDEF) -> list[JunqiEnv]:
    envs: list[JunqiEnv] = []
    for i in range(n):
        rng = random.Random(seed_base + i)
        env = JunqiEnv()
        env.reset(seed=seed_base + i)
        steps = i % 50
        for _ in range(steps):
            if env.state.terminated:
                break
            aids = env.legal_action_ids()
            if aids.size == 0:
                break
            action_id = int(rng.choice(aids))
            env._step_game_only(action_id)
        envs.append(env)
    return envs


def bench(N: int) -> None:
    print(f"\n=== N={N} ===")
    envs = _make_envs(N)
    acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

    # Pack once, upload once — this measures the pure kernel time.
    state_dict = _pack_state_arrays(envs)
    gpu_state  = _cuda.DeviceGameStateBatch(N)
    gpu_state.copy_from_host(state_dict)

    # Warm-up kernel.
    for _ in range(_NUM_WARMUP):
        _cuda.legal_action_ids_batch(gpu_state, acting)

    # Pure kernel-only (state already on GPU, results copied back each iter).
    t0 = time.perf_counter()
    for _ in range(_NUM_ITERS):
        ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)
    t1 = time.perf_counter()
    dt_kernel = (t1 - t0) / _NUM_ITERS
    throughput_kernel = N / dt_kernel
    print(f"  kernel-only:  {dt_kernel*1e3:7.3f} ms/batch   "
          f"throughput = {throughput_kernel/1e3:8.1f} k envs/sec  "
          f"sum(counts)={int(counts.sum())}")

    # End-to-end: re-pack + upload every iter, mimicking real use.
    # warm
    for _ in range(_NUM_WARMUP):
        sd = _pack_state_arrays(envs)
        gpu_state.copy_from_host(sd)
        _cuda.legal_action_ids_batch(gpu_state, acting)

    t0 = time.perf_counter()
    for _ in range(_NUM_ITERS):
        sd = _pack_state_arrays(envs)
        gpu_state.copy_from_host(sd)
        _cuda.legal_action_ids_batch(gpu_state, acting)
    t1 = time.perf_counter()
    dt_e2e = (t1 - t0) / _NUM_ITERS
    throughput_e2e = N / dt_e2e
    print(f"  end-to-end :  {dt_e2e*1e3:7.3f} ms/batch   "
          f"throughput = {throughput_e2e/1e3:8.1f} k envs/sec")

    # Breakdown: pack alone
    t0 = time.perf_counter()
    for _ in range(_NUM_ITERS):
        _pack_state_arrays(envs)
    t1 = time.perf_counter()
    dt_pack = (t1 - t0) / _NUM_ITERS
    print(f"  pack alone :  {dt_pack*1e3:7.3f} ms/batch   "
          f"({dt_pack/dt_e2e*100:4.1f}% of e2e)")

    # Upload alone
    sd = _pack_state_arrays(envs)
    t0 = time.perf_counter()
    for _ in range(_NUM_ITERS):
        gpu_state.copy_from_host(sd)
    t1 = time.perf_counter()
    dt_up = (t1 - t0) / _NUM_ITERS
    print(f"  upload alone: {dt_up*1e3:7.3f} ms/batch   "
          f"({dt_up/dt_e2e*100:4.1f}% of e2e)")

    # === Lite-path (fused 6-field upload) ===
    # Warm-up
    for _ in range(_NUM_WARMUP):
        psa, pta, alv, px, py_, cpi = _pack_state_arrays_lite(envs)
        gpu_state.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)
        _cuda.legal_action_ids_batch(gpu_state, acting)

    t0 = time.perf_counter()
    for _ in range(_NUM_ITERS):
        psa, pta, alv, px, py_, cpi = _pack_state_arrays_lite(envs)
        gpu_state.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)
        _cuda.legal_action_ids_batch(gpu_state, acting)
    t1 = time.perf_counter()
    dt_lite = (t1 - t0) / _NUM_ITERS
    throughput_lite = N / dt_lite
    print(f"  LITE e2e   :  {dt_lite*1e3:7.3f} ms/batch   "
          f"throughput = {throughput_lite/1e3:8.1f} k envs/sec  "
          f"(speedup over full: {dt_e2e/dt_lite:.2f}×)")

    # Lite pack alone
    t0 = time.perf_counter()
    for _ in range(_NUM_ITERS):
        _pack_state_arrays_lite(envs)
    t1 = time.perf_counter()
    dt_lite_pack = (t1 - t0) / _NUM_ITERS
    print(f"  LITE pack  :  {dt_lite_pack*1e3:7.3f} ms/batch   "
          f"({dt_lite_pack/dt_lite*100:4.1f}% of lite e2e)")


def main() -> None:
    _cuda.init_tables()
    for N in (32, 128, 512, 1024, 2048):
        bench(N)


if __name__ == "__main__":
    main()
