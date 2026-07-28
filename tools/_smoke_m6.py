"""Quick smoke test for JunqiEnv / VectorJunqiEnv."""
from __future__ import annotations

import sys

import numpy as np

from junqi_rl import (
    NUM_ACTIONS,
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    JunqiEnv,
    VectorJunqiEnv,
)


def main() -> int:
    # ---------------- JunqiEnv single ----------------
    env = JunqiEnv(seed=0)
    obs = env.reset()
    print(f"JunqiEnv.reset: seats={[s.name for s in obs.keys()]}")
    print(f"  spatial per-seat shape: {obs[env.current_seat()].shape}")
    print(f"  NUM_ACTIONS={NUM_ACTIONS} OBS_CHANNELS={OBS_CHANNELS} "
          f"OBS_GLOBAL_DIMS={OBS_GLOBAL_DIMS}")
    rng = np.random.default_rng(0)
    done = False
    ply = 0
    while not done and ply < 500:
        ids = env.legal_action_ids()
        if ids.size == 0:
            print(f"no legal actions at ply {ply}"); break
        aid = int(ids[rng.integers(ids.size)])
        obs, r, done, info = env.step(aid)
        ply += 1
    print(f"JunqiEnv: terminated={done} plies={ply} rewards={r} "
          f"winner={info.winner_team}")

    # ---------------- VectorJunqiEnv ----------------
    N = 4
    venv = VectorJunqiEnv(N, seed_base=100)
    obs_s, obs_g = venv.reset()
    print(f"VectorJunqiEnv.reset: spatial={obs_s.shape} global={obs_g.shape}")
    rng2 = np.random.default_rng(0)
    steps_total = 0
    for _ in range(50):
        per_env_ids = venv.legal_action_ids()
        actions = np.zeros(N, dtype=np.int64)
        for i, ids in enumerate(per_env_ids):
            if ids.size == 0:
                # Terminated env; reset it.
                venv.envs[i].reset()
                ids = venv.envs[i].legal_action_ids()
            actions[i] = int(ids[rng2.integers(ids.size)])
        obs_s, obs_g, rewards, done, _ = venv.step(actions)
        steps_total += N
    print(f"VectorJunqiEnv: {steps_total} plies across {N} envs; "
          f"done_count={int(done.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
