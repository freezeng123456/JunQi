"""Minimal VectorEnv timing."""
import random
import sys
from time import perf_counter

import numpy as np

from junqi_rl import VectorJunqiEnv


def bench(N: int, steps: int) -> None:
    venv = VectorJunqiEnv(num_envs=N)
    venv.reset(seed_base=0)
    rng = random.Random(0)

    for _ in range(3):  # warmup
        a = np.zeros(N, dtype=np.int32)
        for i, e in enumerate(venv.envs):
            if venv.done[i]:
                continue
            ids = e.legal_action_ids()
            if ids.size:
                a[i] = int(ids[rng.randrange(ids.size)])
        venv.step(a)

    total = 0
    t0 = perf_counter()
    for _ in range(steps):
        a = np.zeros(N, dtype=np.int32)
        for i, e in enumerate(venv.envs):
            if venv.done[i]:
                continue
            ids = e.legal_action_ids()
            if ids.size:
                a[i] = int(ids[rng.randrange(ids.size)])
        _, _, _, d, _ = venv.step(a)
        total += int((~d).sum())
    dt = perf_counter() - t0
    print(f"N={N:3d} steps={steps:3d} : {total/dt:7.0f} plays/s, "
          f"{dt*1000/steps:5.1f} ms/step", flush=True)


def main() -> int:
    # single env
    from junqi_rl import JunqiEnv
    env = JunqiEnv()
    env.reset(seed=0)
    rng = random.Random(0)
    for _ in range(30):  # warmup
        ids = env.legal_action_ids()
        if ids.size == 0:
            env.reset(seed=rng.randrange(1 << 30)); continue
        env.step(int(ids[rng.randrange(ids.size)]))
    total = 0; t0 = perf_counter()
    for _ in range(500):
        ids = env.legal_action_ids()
        if ids.size == 0:
            env.reset(seed=rng.randrange(1 << 30)); continue
        _, _, done, _ = env.step(int(ids[rng.randrange(ids.size)]))
        total += 1
        if done:
            env.reset(seed=rng.randrange(1 << 30))
    dt = perf_counter() - t0
    print(f"single env           : {total/dt:7.0f} plays/s, "
          f"{dt*1000/total:5.1f} ms/step", flush=True)

    # vector envs
    for N, st in ((8, 30), (16, 20), (32, 10), (64, 10)):
        bench(N, st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
