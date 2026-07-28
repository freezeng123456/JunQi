#!/usr/bin/env python3
"""scripts/profile_env.py — Throughput benchmark for VectorJunqiEnv.

Measures per-component latency so bottlenecks can be tracked over time.

Usage
-----
    python scripts/profile_env.py [--num_envs 32] [--steps 200] [--device cpu]

Outputs
-------
Per-component timing table + estimated env-steps/sec and games/sec.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from typing import Generator

import numpy as np
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

_timings: dict[str, list[float]] = {}


@contextmanager
def timed(name: str) -> Generator[None, None, None]:
    t0 = time.perf_counter()
    yield
    _timings.setdefault(name, []).append(time.perf_counter() - t0)


def print_report(num_envs: int, num_steps: int) -> None:
    total_env_steps = num_envs * num_steps
    print(f"\n{'─' * 62}")
    print(f"{'Component':<35}  {'Mean (ms)':>9}  {'Total (ms)':>10}")
    print(f"{'─' * 62}")
    total_wall = 0.0
    for name, samples in _timings.items():
        mean_ms = 1000.0 * np.mean(samples)
        total_ms = 1000.0 * np.sum(samples)
        print(f"{name:<35}  {mean_ms:>9.3f}  {total_ms:>10.1f}")
        if name == "env.step":
            total_wall += total_ms
    print(f"{'─' * 62}")

    # env.step is the outer loop; use it as wallclock for fps
    if "env.step" in _timings:
        wall_s = np.sum(_timings["env.step"])
        fps = total_env_steps / max(wall_s, 1e-9)
        # average game length assumption
        avg_game_moves = 1200   # typical 四国军棋 game
        games_per_sec = fps / avg_game_moves
        print(f"\n  Total env steps : {total_env_steps:,}")
        print(f"  Wall time (step): {wall_s*1000:.1f} ms")
        print(f"  Env-steps / sec : {fps:,.0f}")
        print(f"  Games / sec     : {games_per_sec:.2f}  (assuming {avg_game_moves} moves/game)")

    if "legal_mask" in _timings:
        mask_ms = 1000.0 * np.mean(_timings["legal_mask"])
        step_ms = 1000.0 * np.mean(_timings.get("env.step", [1.0]))
        print(f"\n  legal_mask share: {100*mask_ms/step_ms:.1f}% of env.step")
    if "policy_inf" in _timings:
        inf_ms = 1000.0 * np.mean(_timings["policy_inf"])
        step_ms = 1000.0 * np.mean(_timings.get("env.step", [1.0]))
        print(f"  policy_inf share: {100*inf_ms/step_ms:.1f}% of env.step")
    print()


# ---------------------------------------------------------------------------
# Env-only benchmark (no policy)
# ---------------------------------------------------------------------------

def bench_env_only(num_envs: int, num_steps: int, seed: int = 0, num_workers: int | None = None) -> None:
    """Benchmark VectorJunqiEnv step loop without a policy."""
    from junqi_rl import VectorJunqiEnv, build_legal_mask_batch

    _workers_desc = num_workers if num_workers is not None else f"auto (≤8)"
    print(f"\n[profile] Env-only benchmark: num_envs={num_envs}, steps={num_steps}, "
          f"num_workers={_workers_desc}")
    env = VectorJunqiEnv(num_envs=num_envs, num_workers=num_workers)
    obs_sp, obs_gl = env.reset(seed_base=seed)

    for i in range(num_steps):
        seats = env.current_seats()

        with timed("legal_mask"):
            mask = build_legal_mask_batch(env, seats)

        # Random legal action selection from mask
        actions_world = np.zeros(num_envs, dtype=np.int32)
        with timed("action_sample"):
            for j in range(num_envs):
                if env.done[j]:
                    continue
                wids = env._envs[j].legal_action_ids(seats[j])
                if len(wids):
                    actions_world[j] = int(np.random.choice(wids))

        with timed("env.step"):
            obs_sp, obs_gl, rewards, done, infos = env.step(actions_world)

        # Auto-reset
        if done.any():
            for j in range(num_envs):
                if done[j]:
                    env._envs[j].reset(seed=seed + j + i)
                    env._done[j] = False
            env._fill_all_obs()
            obs_sp = env.obs_spatial
            obs_gl = env.obs_global

    print_report(num_envs, num_steps)
    env.close()


# ---------------------------------------------------------------------------
# Workers sweep: compare sequential vs multi-threaded
# ---------------------------------------------------------------------------

def bench_workers_sweep(num_envs: int, num_steps: int, seed: int = 0) -> None:
    """Sweep num_workers to find the optimal thread count."""
    import time
    from junqi_rl import VectorJunqiEnv, build_legal_mask_batch

    worker_counts = [1, 2, 4, 8, min(16, num_envs)]
    worker_counts = sorted(set(w for w in worker_counts if w <= num_envs))

    print(f"\n[workers_sweep] num_envs={num_envs}, steps={num_steps}")
    print(f"{'num_workers':>12}  {'env.step mean (ms)':>20}  {'speedup vs seq':>16}")
    print("─" * 54)

    baseline_ms: float | None = None
    for nw in worker_counts:
        env = VectorJunqiEnv(num_envs=num_envs, num_workers=nw)
        env.reset(seed_base=seed)
        times: list[float] = []
        for i in range(num_steps):
            seats = env.current_seats()
            actions = np.zeros(num_envs, dtype=np.int32)
            for j in range(num_envs):
                if not env.done[j]:
                    wids = env._envs[j].legal_action_ids(seats[j])
                    if len(wids):
                        actions[j] = int(wids[0])
            t0 = time.perf_counter()
            env.step(actions)
            times.append(time.perf_counter() - t0)
            if env.done.any():
                for j in range(num_envs):
                    if env.done[j]:
                        env._envs[j].reset(seed=seed + j + i)
                        env._done[j] = False
                env._fill_all_obs()
        env.close()
        mean_ms = 1000.0 * np.mean(times)
        if baseline_ms is None:
            baseline_ms = mean_ms
        speedup = baseline_ms / mean_ms if mean_ms > 0 else float("inf")
        print(f"{nw:>12}  {mean_ms:>20.3f}  {speedup:>15.2f}×")
    print()


# ---------------------------------------------------------------------------
# Full pipeline benchmark (with random-weight policy)
# ---------------------------------------------------------------------------

def bench_full_pipeline(
    num_envs: int,
    num_steps: int,
    device: str = "cpu",
    seed: int = 0,
    num_workers: int | None = None,
) -> None:
    """Benchmark full collect_rollout loop with a randomly-initialised JunqiNet."""
    from junqi_rl import VectorJunqiEnv, build_legal_mask_batch
    from junqi_rl.action_lut import UNROTATE_LUT
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    print(f"\n[profile] Full pipeline: num_envs={num_envs}, steps={num_steps}, device={device}")

    _dev = torch.device(device)
    net_cfg = JunqiNetConfig(
        cnn_channels=64, cnn_layers=2, depth=2, embed_dim=64, n_head=2,
    )
    policy = JunqiNet(net_cfg).to(_dev).eval()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[profile] Policy parameters: {n_params:,}")

    env = VectorJunqiEnv(num_envs=num_envs, num_workers=num_workers)
    obs_sp, obs_gl = env.reset(seed_base=seed)
    _env_range = np.arange(num_envs, dtype=np.intp)

    for i in range(num_steps):
        seats = env.current_seats()
        acting_idx = np.array([s.value for s in seats], dtype=np.intp)

        with timed("legal_mask"):
            mask_np = build_legal_mask_batch(env, seats)

        act_obs_sp = obs_sp[_env_range, acting_idx]
        act_obs_gl = obs_gl[_env_range, acting_idx]

        with timed("to_device"):
            sp_t = torch.from_numpy(act_obs_sp).to(_dev)
            gl_t = torch.from_numpy(act_obs_gl).to(_dev)
            lm_t = torch.from_numpy(mask_np).to(_dev)

        with timed("policy_inf"):
            with torch.no_grad():
                actions_can, log_probs, values = policy.act(sp_t, gl_t, lm_t)

        with timed("to_cpu"):
            actions_can_np = actions_can.cpu().numpy().astype(np.int32)
            _ = log_probs.cpu().numpy()
            _ = values.cpu().numpy()

        # Unrotate actions
        with timed("action_unrotate"):
            actions_world = np.array([
                UNROTATE_LUT[seats[j].value][actions_can_np[j]] if not env.done[j] else 0
                for j in range(num_envs)
            ], dtype=np.int32)

        with timed("env.step"):
            obs_sp, obs_gl, rewards, done, infos = env.step(actions_world)

        # Auto-reset
        if done.any():
            for j in range(num_envs):
                if done[j]:
                    env._envs[j].reset(seed=seed + j + i)
                    env._done[j] = False
            env._fill_all_obs()
            obs_sp = env.obs_spatial
            obs_gl = env.obs_global

    print_report(num_envs, num_steps)


# ---------------------------------------------------------------------------
# LUT correctness / speed micro-bench
# ---------------------------------------------------------------------------

def bench_lut_vs_scalar(num_actions: int = 100, num_reps: int = 10000) -> None:
    """Compare scalar rotate_action_id vs LUT batch rotation."""
    from junqi_rl import rotate_action_id
    from junqi_rl.action_lut import ROTATE_LUT
    from junqi_core.rules import Seat

    rng = np.random.default_rng(0)
    world_ids = rng.integers(0, 83521, size=num_actions, dtype=np.int32)
    seat = Seat.WEST

    # Warm-up
    for _ in range(100):
        _ = ROTATE_LUT[seat.value][world_ids]

    # LUT timing
    t0 = time.perf_counter()
    for _ in range(num_reps):
        can_ids_lut = ROTATE_LUT[seat.value][world_ids]
    t_lut = (time.perf_counter() - t0) / num_reps * 1e6  # µs

    # Scalar timing
    t0 = time.perf_counter()
    for _ in range(num_reps):
        can_ids_scalar = np.array(
            [rotate_action_id(int(w), seat) for w in world_ids], dtype=np.int32
        )
    t_scalar = (time.perf_counter() - t0) / num_reps * 1e6  # µs

    # Correctness check
    np.testing.assert_array_equal(can_ids_lut, can_ids_scalar,
                                   err_msg="LUT and scalar rotation disagree!")

    print(f"\n[lut_bench] K={num_actions} actions, seat={seat.name}")
    print(f"  Scalar loop : {t_scalar:8.2f} µs  (1×)")
    print(f"  LUT batch   : {t_lut:8.2f} µs  ({t_scalar/t_lut:.1f}× speedup)")
    print(f"  Correctness : ✓ matches")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VectorJunqiEnv throughput profiler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel envs")
    parser.add_argument("--steps", type=int, default=200, help="Steps per benchmark run")
    parser.add_argument("--device", default="cpu", help="Device for policy (cpu/cuda)")
    parser.add_argument("--env_only", action="store_true", help="Skip policy inference")
    parser.add_argument("--lut_bench", action="store_true", help="Run LUT micro-benchmark")
    parser.add_argument("--workers_sweep", action="store_true",
                        help="Sweep num_workers to find optimal threading")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Thread-pool size for game stepping (default: auto)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.lut_bench:
        bench_lut_vs_scalar()
        return

    if args.workers_sweep:
        bench_workers_sweep(args.num_envs, args.steps, seed=args.seed)
        return

    if args.env_only:
        bench_env_only(args.num_envs, args.steps, seed=args.seed,
                       num_workers=args.num_workers)
    else:
        if not _TORCH_AVAILABLE:
            print("[profile] PyTorch not found — falling back to env-only benchmark.")
            bench_env_only(args.num_envs, args.steps, seed=args.seed,
                           num_workers=args.num_workers)
        else:
            bench_full_pipeline(args.num_envs, args.steps, device=args.device, seed=args.seed,
                                num_workers=args.num_workers)


if __name__ == "__main__":
    main()
