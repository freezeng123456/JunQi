"""Phase 0.4 M7 — Trajectory / ExperienceBuffer benchmarks.

Reports:
  * Trajectory record/replay per-step wall-clock.
  * ExperienceBuffer.append_step rate at capacity=1024 in each legal-mask
    mode.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_core.replay import Trajectory, record_trajectory
from junqi_rl import ExperienceBuffer

BOARD = 17


def bench_record_replay() -> None:
    N_GAMES = 20
    MAX_STEPS = 200

    # --- Record
    t0 = perf_counter()
    trajs = [
        record_trajectory(rng_seed=s, max_steps=MAX_STEPS)
        for s in range(N_GAMES)
    ]
    dt_rec = perf_counter() - t0
    steps_rec = sum(t.num_steps for t in trajs)
    print(
        f"record (random-policy, {N_GAMES} games, {steps_rec} steps total) : "
        f"{steps_rec / dt_rec:7.0f} steps/s, "
        f"{dt_rec * 1e6 / steps_rec:6.2f} µs/step",
        flush=True,
    )

    # --- Save to temp dir.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        t0 = perf_counter()
        for i, traj in enumerate(trajs):
            traj.save(tmp_p / f"g{i:03d}.npz")
        dt_save = perf_counter() - t0
        total_kb = sum(
            (tmp_p / f"g{i:03d}.npz").stat().st_size for i in range(N_GAMES)
        ) / 1024
        print(
            f"save (.npz, {N_GAMES} files, {total_kb:.1f} KiB total)         : "
            f"{N_GAMES / dt_save:7.0f} games/s, "
            f"{total_kb / N_GAMES:6.1f} KiB/game",
            flush=True,
        )

        # --- Load.
        t0 = perf_counter()
        loaded = [Trajectory.load(tmp_p / f"g{i:03d}.npz") for i in range(N_GAMES)]
        dt_load = perf_counter() - t0
        print(
            f"load (.npz, {N_GAMES} files)                                 : "
            f"{N_GAMES / dt_load:7.0f} games/s",
            flush=True,
        )

        # --- Replay round-trip.
        t0 = perf_counter()
        for t in loaded:
            t.replay()
        dt_rep = perf_counter() - t0
        print(
            f"replay (engine step-through, {steps_rec} steps)              : "
            f"{steps_rec / dt_rep:7.0f} steps/s, "
            f"{dt_rep * 1e6 / steps_rec:6.2f} µs/step",
            flush=True,
        )


def bench_buffer(mode: str, capacity: int = 1024, n_steps: int = 4096) -> None:
    """Measure append_step throughput under a given legal-mask mode."""
    buf = ExperienceBuffer(capacity=capacity, legal_mask_mode=mode)  # type: ignore[arg-type]
    sp = np.ones((OBS_CHANNELS, BOARD, BOARD), dtype=np.float32)
    gl = np.ones((OBS_GLOBAL_DIMS,), dtype=np.float32)
    legal: np.ndarray | None
    if mode == "dense":
        from junqi_rl import FLAT_ACTION_DIM
        legal = np.zeros(FLAT_ACTION_DIM, dtype=bool)
        legal[:100] = True
    elif mode == "sparse":
        legal = np.arange(100, dtype=np.int32)
    else:
        legal = None

    # Warmup.
    for _ in range(100):
        buf.append_step(
            obs_spatial=sp, obs_global=gl, action_id=1,
            reward=(1, 0, 0, 0), done=False, seat=0, legal=legal,
        )
    buf.reset()

    t0 = perf_counter()
    for _ in range(n_steps):
        buf.append_step(
            obs_spatial=sp, obs_global=gl, action_id=1,
            reward=(1, 0, 0, 0), done=False, seat=0, legal=legal,
        )
    dt = perf_counter() - t0
    print(
        f"ExperienceBuffer.append_step  mode={mode:6s} : "
        f"{n_steps / dt:9.0f} steps/s, "
        f"{dt * 1e6 / n_steps:6.2f} µs/step",
        flush=True,
    )


def main() -> int:
    print("-- Trajectory record / save / load / replay --", flush=True)
    bench_record_replay()
    print("-- ExperienceBuffer.append_step --", flush=True)
    for mode in ("none", "sparse", "dense"):
        bench_buffer(mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
