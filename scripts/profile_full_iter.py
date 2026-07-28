"""scripts/profile_full_iter.py — measure collect vs PPO train phase."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_envs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--compile", action="store_true", default=False)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
    from junqi_rl.training.rollout_gpu import RolloutBufferGPU
    from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
    from junqi_rl.training.ppo import PPOTrainer, PPOConfig

    device = torch.device("cuda")
    torch.cuda.set_device(0)

    N, T = args.num_envs, args.steps
    print(f"[profile] num_envs={N} steps={T} minibatch={args.minibatch}")

    rollout_world = GpuRollout(num_envs=N)
    net_cfg = JunqiNetConfig(
        cnn_channels=64, cnn_layers=2,
        depth=4, embed_dim=128, n_head=4,
        ff_factor=4, action_key_dim=32,
        use_cat_vf=True, dropout=0.0,
    )
    policy = JunqiNet(net_cfg).to(device)

    ppo_cfg = PPOConfig(
        clip_range=0.2, gamma=1.0, gae_lambda=0.5, td_lambda=0.8,
        vf_coef=1.0, policy_coef=1.0, temperature_coef=0.05,
        kl_coef=0.1, lr_ceil=1e-4, lr_coef=0.5,
        minibatch_size=args.minibatch, num_epochs_per_rollout=4,
        dtype=args.dtype, torch_compile=args.compile,
        net=net_cfg,
    )
    trainer = PPOTrainer(policy, ppo_cfg, device=device)

    buffer = RolloutBufferGPU(
        num_envs=N, steps_per_env=T,
        gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda,
        td_lambda=ppo_cfg.td_lambda,
        adv_filt_thresh=ppo_cfg.adv_filt_thresh,
        adv_filt_rate=ppo_cfg.adv_filt_rate,
        device=device,
    )

    rng = np.random.default_rng(42)

    # Warmup
    print("[profile] warmup iter...")
    collect_rollout_gpu_v2(
        rollout_world, trainer.ema.model, buffer,
        device="cuda", seed_base=0, reset_at_start=True,
        random_opponent=True,
    )
    trainer.train_epoch(buffer, rng=rng)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[profile] warmup done. peak_mem={peak_mb:.0f} MB")

    # Measure
    collect_s = []
    train_s = []
    for i in range(args.iters):
        torch.cuda.synchronize()
        t0 = time.time()
        collect_rollout_gpu_v2(
            rollout_world, trainer.ema.model, buffer,
            device="cuda", seed_base=1000 + i, reset_at_start=False,
            random_opponent=True,
        )
        torch.cuda.synchronize()
        t_c = time.time() - t0

        t0 = time.time()
        trainer.train_epoch(buffer, rng=rng)
        torch.cuda.synchronize()
        t_t = time.time() - t0

        collect_s.append(t_c)
        train_s.append(t_t)
        total = t_c + t_t
        fps = N * T / total
        print(f"  iter {i}: collect={t_c:.3f}s  train={t_t:.3f}s  "
              f"total={total:.3f}s  fps={fps:.0f}")

    c = np.asarray(collect_s)
    t = np.asarray(train_s)
    tot = c + t
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n[summary] N={N} T={T}")
    print(f"  collect mean = {c.mean():.3f}s")
    print(f"  train   mean = {t.mean():.3f}s")
    print(f"  total   mean = {tot.mean():.3f}s")
    print(f"  fps          = {N*T/tot.mean():.0f}")
    print(f"  collect/train ratio = {c.mean()/t.mean():.2f}")
    print(f"  peak_mem     = {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
