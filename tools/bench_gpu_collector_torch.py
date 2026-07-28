"""tools/bench_gpu_collector_torch.py — End-to-end GPU collector benchmark
with a real (small) JunqiNet policy.

This is T-01 acceptance bar: exercise ``collect_rollout_gpu`` with a live
torch network on CUDA, and report env·steps/s.

Target: >= 500 k env·steps/s at N=1024 (for a small enough network —
bigger nets will be network-bound).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training import collect_rollout_gpu
from junqi_rl.training.rollout import RolloutBuffer


def _tiny_cfg() -> JunqiNetConfig:
    return JunqiNetConfig(
        cnn_channels=32,
        cnn_layers=2,
        depth=2,
        embed_dim=64,
        n_head=4,
        ff_factor=2,
        dropout=0.0,
    )


def _big_cfg() -> JunqiNetConfig:
    # Representative of the real training config.
    return JunqiNetConfig(
        cnn_channels=128,
        cnn_layers=3,
        depth=6,
        embed_dim=256,
        n_head=8,
        ff_factor=4,
        dropout=0.0,
    )


def bench(N: int, T: int, cfg: JunqiNetConfig, label: str) -> None:
    device = torch.device("cuda")
    world = GpuRollout(num_envs=N)
    policy = JunqiNet(cfg).to(device).eval()
    buf = RolloutBuffer(num_envs=N, steps_per_env=T, device=device)
    n_params = sum(p.numel() for p in policy.parameters())

    # Warmup (also pays the first-time CUDA-kernel JIT cost)
    collect_rollout_gpu(
        rollout_world=world, policy=policy, buffer=buf,
        device=device, seed_base=0,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    collect_rollout_gpu(
        rollout_world=world, policy=policy, buffer=buf,
        device=device, seed_base=1, reset_at_start=False,
    )
    torch.cuda.synchronize()
    total = time.perf_counter() - t0

    envsteps = N * T / total
    print(f"{label:10s}  N={N:5d}  T={T:3d}  params={n_params/1e6:5.2f}M  "
          f"total {total*1e3:7.1f} ms  → {envsteps:11,.0f} env·steps/s")


def main() -> None:
    print(f"# torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")

    print("=== tiny net (32ch, 2 layers) ===")
    for N in (256, 1024, 4096):
        bench(N, T=16, cfg=_tiny_cfg(), label="tiny")

    print("\n=== big net (128ch, 6 layers) — representative training config ===")
    for N in (256, 1024):
        bench(N, T=16, cfg=_big_cfg(), label="big")


if __name__ == "__main__":
    main()
