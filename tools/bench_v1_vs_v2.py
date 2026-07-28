"""tools/bench_v1_vs_v2.py — compare v1 (thread-per-piece) vs v2 (block-per-env).

Runs both kernels back-to-back under the same warm state so the comparison is
apples-to-apples.
"""

from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Allow runtime toggle by re-importing with env var set.
import numpy as np
import random


def run_bench(kernel_version: str) -> dict[int, float]:
    """Run kernel-only bench with the given kernel selected via env var."""
    if kernel_version == "v1":
        os.environ["JUNQI_CUDA_KERNEL_V1"] = "1"
    else:
        os.environ.pop("JUNQI_CUDA_KERNEL_V1", None)

    # Fresh import so the static env-var read picks up the value.
    for mod in list(sys.modules):
        if mod.startswith("junqi_cuda"):
            del sys.modules[mod]
    # NB: the .so already has s_use_v1 initialised at load time; env-var must
    # be set BEFORE first import.  To compare both, we must fork — or run
    # this script twice with different env vars.
    import junqi_cuda as _cuda
    from junqi_rl.env import JunqiEnv
    from junqi_rl.env_gpu import _pack_state_arrays

    _cuda.init_tables()
    print(f"(Note: kernel selection baked at import time; use v1/v2 mode = {kernel_version})")

    result = {}
    for N in (32, 128, 512, 1024, 2048):
        envs = []
        for i in range(N):
            rng = random.Random(i)
            env = JunqiEnv(); env.reset(seed=i)
            for _ in range(i % 40):
                if env.state.terminated: break
                aids = env.legal_action_ids()
                if aids.size == 0: break
                env._step_game_only(int(rng.choice(aids)))
            envs.append(env)

        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        sd = _pack_state_arrays(envs)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(sd)

        # Warm
        for _ in range(5):
            _cuda.legal_action_ids_batch(gs, acting)

        NIT = 100
        t0 = time.perf_counter()
        for _ in range(NIT):
            _cuda.legal_action_ids_batch(gs, acting)
        dt = (time.perf_counter() - t0) / NIT
        result[N] = dt
        print(f"  N={N:>5d}  kernel: {dt*1e3:7.3f} ms  throughput = {N/dt/1e3:7.1f} k envs/s")
    return result


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "v2"
    if mode not in ("v1", "v2"):
        print(f"Usage: {sys.argv[0]} [v1|v2]")
        sys.exit(2)
    print(f"=== Running kernel-only benchmark, mode = {mode} ===")
    run_bench(mode)


if __name__ == "__main__":
    main()
