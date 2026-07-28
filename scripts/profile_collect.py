"""scripts/profile_collect.py — profile the P2/P3 hot path.

Measures per-stage time in `collect_rollout_gpu_v2` at a configurable
(num_envs, steps_per_env) to find the remaining bottleneck and verify
whether scaling envs is compute- or bandwidth-bound.

Run:
    PYTHONPATH=. python3 scripts/profile_collect.py --num_envs 128 --steps 64
    PYTHONPATH=. python3 scripts/profile_collect.py --num_envs 256 --steps 64
    PYTHONPATH=. python3 scripts/profile_collect.py --num_envs 512 --steps 64
"""
from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

import numpy as np
import torch


def _ns_to_ms(ns: int) -> float:
    return ns / 1e6


class GPUTimer:
    def __init__(self) -> None:
        self.times: dict[str, list[float]] = {}

    @contextmanager
    def section(self, name: str):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        yield
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        self.times.setdefault(name, []).append(_ns_to_ms(t1 - t0))

    def summary(self) -> None:
        print(f"\n{'Section':<40s} {'mean(ms)':>10s} {'p50':>10s} {'p95':>10s} {'total(s)':>10s}")
        print("-" * 85)
        total = 0.0
        for k, v in sorted(self.times.items(), key=lambda x: -sum(x[1])):
            arr = np.asarray(v)
            s_ms = arr.sum()
            total += s_ms
            print(
                f"{k:<40s} {arr.mean():>10.3f} {np.percentile(arr,50):>10.3f} "
                f"{np.percentile(arr,95):>10.3f} {s_ms/1000:>10.3f}"
            )
        print("-" * 85)
        print(f"{'TOTAL':<40s} {'':>10s} {'':>10s} {'':>10s} {total/1000:>10.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    device = torch.device("cuda")
    torch.cuda.set_device(0)

    N = args.num_envs
    T = args.steps

    print(f"[profile] num_envs={N} steps={T} amp={args.amp}")

    # Build rollout + policy with the same shape as beat_random_v15_3bin.
    rollout = GpuRollout(num_envs=N)
    cfg = JunqiNetConfig(
        cnn_channels=64, cnn_layers=2,
        depth=4, embed_dim=128, n_head=4,
        ff_factor=4, action_key_dim=32,
        use_cat_vf=True, dropout=0.0,
    )
    policy = JunqiNet(cfg).to(device)
    policy.eval()

    print(f"[profile] net params: {sum(p.numel() for p in policy.parameters()):,}")
    mem0 = torch.cuda.memory_allocated() / 1024**2
    print(f"[profile] GPU mem allocated: {mem0:.0f} MB")

    rollout.reset(seed_base=0)
    done_t = rollout.terminated_torch().clone()

    timer = GPUTimer()

    # Warmup
    for _ in range(args.warmup):
        with torch.no_grad():
            turn_t = rollout.turn_torch()
            acting_t = torch.where(done_t, torch.zeros_like(turn_t), turn_t)
            obs_sp, obs_gl = rollout.build_acting_seat_observation_torch(acting_t)
            lm = rollout.legal_mask_canonical_torch_device(acting_t)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                a, lp, v = policy.act(obs_sp, obs_gl, lm)
            r = rollout.step_device_torch(a.to(torch.int32), acting_t)
            done_t = rollout.terminated_torch().clone()
            rollout.reset_terminated_device(seed=0)
    torch.cuda.synchronize()

    mem_warm = torch.cuda.memory_allocated() / 1024**2
    peak_warm = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[profile] after warmup: alloc={mem_warm:.0f} MB  peak={peak_warm:.0f} MB")

    # Profile
    torch.cuda.reset_peak_memory_stats()
    t_wall0 = time.perf_counter()
    for _ in range(T):
        with torch.no_grad():
            with timer.section("01_turn_torch"):
                turn_t = rollout.turn_torch()
                acting_t = torch.where(done_t, torch.zeros_like(turn_t), turn_t)

            with timer.section("02_build_observation"):
                obs_sp, obs_gl = rollout.build_acting_seat_observation_torch(acting_t)

            with timer.section("03_legal_mask"):
                lm = rollout.legal_mask_canonical_torch_device(acting_t)

            with timer.section("04_policy_forward"):
                with torch.amp.autocast("cuda", dtype=torch.float16) if args.amp else _noop():
                    a, lp, v = policy.act(obs_sp, obs_gl, lm)
                torch.cuda.synchronize()

            with timer.section("05_step_device"):
                r = rollout.step_device_torch(a.to(torch.int32), acting_t)

            with timer.section("06_reset_terminated"):
                rollout.reset_terminated_device(seed=0)

            with timer.section("07_terminated_clone"):
                done_t = rollout.terminated_torch().clone()

    torch.cuda.synchronize()
    t_wall = time.perf_counter() - t_wall0
    peak_hot = torch.cuda.max_memory_allocated() / 1024**2

    timer.summary()
    fps = N * T / t_wall
    print(f"\n[profile] wall={t_wall:.3f}s  fps={fps:.0f}  peak_mem(hot)={peak_hot:.0f} MB")
    print(f"[profile] fps/env = {fps/N:.2f}")


@contextmanager
def _noop():
    yield


if __name__ == "__main__":
    main()
