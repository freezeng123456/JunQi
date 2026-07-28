"""scripts/train_toy.py — Tiny PPO self-play training on a single GPU.

Designed for T4 / T-class inference GPUs where the real training config
would take hours.  Uses:

  * tiny JunqiNet (32ch, depth=2, d=64) — ~140 k params
  * N=64 envs, T=32 steps per rollout
  * 50 rollouts (≈ 100 k env·steps total)

Produces:
  * a checkpoint at exps/toy/ckpt.pt
  * a per-rollout log (mean reward, value loss, policy loss, entropy)
  * a final-game recording at exps/toy/recording.npz (with policy logits)

Run:
    python3 scripts/train_toy.py --num_rollouts 50 --save_dir exps/toy
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training import (
    PPOConfig, PPOTrainer, RolloutBufferGPU, collect_rollout_gpu,
)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_rollouts", type=int, default=50)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps_per_env", type=int, default=32)
    p.add_argument("--save_dir", default="exps/toy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--record_game", action="store_true",
                   help="After training, record a single game with policy logits")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=10,
                   help="Run eval_vs_random every N rollouts (0 to disable)")
    p.add_argument("--eval_games", type=int, default=64)
    p.add_argument("--eval_max_steps", type=int, default=1500)
    p.add_argument("--target_win_rate", type=float, default=0.55,
                   help="Stop early once trained team win-rate >= this")
    p.add_argument("--reward_shaping", action="store_true",
                   help="Enable dense reward shaping (+0.05 eat, -0.05 killed)")
    return p.parse_args()


def tiny_net_cfg() -> JunqiNetConfig:
    return JunqiNetConfig(
        cnn_channels=32, cnn_layers=2, depth=2,
        embed_dim=64, n_head=4, ff_factor=2, dropout=0.0,
    )


def tiny_ppo_cfg() -> PPOConfig:
    cfg = PPOConfig()
    # Bring the training-step cost down to match the toy rollout.
    cfg.num_epochs_per_rollout = 2
    cfg.minibatch_size = 64
    cfg.clip_range = 0.2
    cfg.value_coef = 0.5
    cfg.ent_coef = 0.01
    cfg.max_grad_norm = 1.0
    cfg.gamma = 1.0
    cfg.gae_lambda = 0.95
    return cfg


def main() -> None:
    args = parse()
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda")
    print(f"[toy] device={torch.cuda.get_device_name(0)} torch={torch.__version__}")

    # World + policy + trainer
    world = GpuRollout(num_envs=args.num_envs)
    policy = JunqiNet(tiny_net_cfg()).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[toy] policy params={n_params/1e3:.1f}k")

    ppo_cfg = tiny_ppo_cfg()
    ppo_cfg.net = tiny_net_cfg()
    trainer = PPOTrainer(policy, ppo_cfg, device=device)

    buf = RolloutBufferGPU(
        num_envs=args.num_envs, steps_per_env=args.steps_per_env,
        gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda,
        device=device,
    )

    # Training loop
    rng = np.random.default_rng(args.seed)
    t_start = time.time()
    log_f = open(save_dir / "train.log", "w")
    from junqi_rl.analysis import eval_vs_random
    best_win_rate = -1.0

    for r in range(args.num_rollouts):
        t0 = time.time()
        collect_rollout_gpu(
            rollout_world=world,
            policy=trainer.ema.model,
            buffer=buf,
            device=device,
            seed_base=args.seed + r,
            reset_at_start=(r == 0),
            reward_shaping=args.reward_shaping,
        )
        t_collect = time.time() - t0

        t0 = time.time()
        metrics = trainer.train_epoch(buf, rng=rng)
        t_train = time.time() - t0

        steps_done = (r + 1) * args.num_envs * args.steps_per_env
        msg = (
            f"[r={r:3d}/{args.num_rollouts}] "
            f"env_steps={steps_done:>7,d}  "
            f"collect={t_collect*1e3:5.0f}ms  train={t_train*1e3:5.0f}ms  "
            f"policy_loss={metrics.get('train/policy_loss', 0):+.4f}  "
            f"value_loss={metrics.get('train/value_loss', 0):.4f}  "
            f"entropy_loss={metrics.get('train/entropy_loss', 0):+.4f}  "
            f"buf_reward_mean={float(buf.rewards.mean().item()):+.4f}"
        )
        if r % args.log_every == 0 or r == args.num_rollouts - 1:
            print(msg)
        log_f.write(msg + "\n"); log_f.flush()

        # ---- Periodic eval vs random ----
        if args.eval_every > 0 and (
            r % args.eval_every == 0 or r == args.num_rollouts - 1
        ):
            t0 = time.time()
            stats = eval_vs_random(
                trainer.ema.model,
                num_games=args.eval_games,
                trained_team=0,
                max_steps=args.eval_max_steps,
                device=device,
                seed_base=args.seed + r * 10007,
                greedy=False,
            )
            t_eval = time.time() - t0
            eval_msg = (
                f"  EVAL[r={r:3d}]  win={stats['trained_win_rate']:.3f}  "
                f"loss={stats['trained_loss_rate']:.3f}  "
                f"draw={stats['draw_rate']:.3f}  "
                f"ongoing={stats['ongoing_rate']:.3f}  "
                f"mean_len={stats['mean_length']:.1f}  "
                f"({t_eval:.1f}s)"
            )
            print(eval_msg)
            log_f.write(eval_msg + "\n"); log_f.flush()
            if stats["trained_win_rate"] > best_win_rate:
                best_win_rate = stats["trained_win_rate"]
                best_path = save_dir / "ckpt_best.pt"
                torch.save({
                    "policy": policy.state_dict(),
                    "ema":    trainer.ema.model.state_dict(),
                    "eval_stats": stats,
                    "rollout": r,
                }, best_path)
                print(f"  [best] saved {best_path} (win={best_win_rate:.3f})")

            if stats["trained_win_rate"] >= args.target_win_rate:
                print(f"[toy] TARGET REACHED at r={r}: "
                      f"win_rate={stats['trained_win_rate']:.3f} "
                      f">= {args.target_win_rate:.3f}")
                break

    log_f.close()
    total_time = time.time() - t_start
    total_steps = args.num_rollouts * args.num_envs * args.steps_per_env
    print(f"[toy] DONE in {total_time:.1f}s — {total_steps/total_time:,.0f} env·steps/s")

    # Save checkpoint
    ckpt_path = save_dir / "ckpt.pt"
    torch.save({
        "policy": policy.state_dict(),
        "ema":    trainer.ema.model.state_dict(),
        "config": {
            "net": vars(tiny_net_cfg()),
            "ppo": vars(ppo_cfg),
            "num_rollouts": args.num_rollouts,
        },
    }, ckpt_path)
    print(f"[toy] saved {ckpt_path}")

    # Record a final game with policy logits
    if args.record_game:
        from junqi_rl.analysis import record_game_with_policy
        print("[toy] recording final demo game ...")
        traj = record_game_with_policy(
            trainer.ema.model, rng_seed=args.seed + 9999,
            device=device, max_steps=400, top_k=8, greedy=False,
        )
        rec_path = save_dir / "recording.npz"
        traj.save(rec_path)
        print(f"[toy] saved recording to {rec_path}  ({traj.num_steps} steps)")


if __name__ == "__main__":
    main()
