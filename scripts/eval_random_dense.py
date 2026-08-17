#!/usr/bin/env python3
"""Dense vs-random evaluation with Wilson 95% confidence intervals.

Use this script (not a single 128-game eval) to judge whether the policy
has crossed the 90% win-rate bar against a uniform-random opponent.

Example::

    python scripts/eval_random_dense.py \\
        --ckpt exps/v42_planB_T_vs_random/ckpt_best.pt \\
        --config configs/v42_planB_T_vs_random.yaml \\
        --num-games 2048

By default this follows the main training score exactly: the raw learner
policy, both team assignments on paired seeds, greedy actions, and a fixed
held-out GPU setup pool.  ``--single-team`` and ``--checkpoint-kind ema`` are
available only for targeted diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
_JUNQI_RL = ROOT / "junqi_rl"
for _p in (str(ROOT), str(_JUNQI_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Policy checkpoint (.pt)")
    ap.add_argument("--config", required=True, help="Training yaml (net section)")
    ap.add_argument("--num-games", type=int, default=2048)
    ap.add_argument(
        "--single-team",
        action="store_true",
        help="Evaluate only --trained-team instead of the paired two-team protocol",
    )
    ap.add_argument("--trained-team", type=int, default=0, choices=[0, 1])
    ap.add_argument(
        "--checkpoint-kind",
        choices=["policy", "ema"],
        default="policy",
        help="Checkpoint weights to evaluate (default: raw PPO learner)",
    )
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument(
        "--setup-seed",
        type=int,
        default=20_260_818,
        help="Fixed uniform setup-pool seed; use a new seed for held-out checks",
    )
    action_mode = ap.add_mutually_exclusive_group()
    action_mode.add_argument(
        "--greedy",
        dest="greedy",
        action="store_true",
        default=True,
        help="Use greedy actions (default; matches the main training score)",
    )
    action_mode.add_argument(
        "--sample",
        dest="greedy",
        action="store_false",
        help="Sample policy actions instead; diagnostic only",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for eval_random_dense")

    from junqi_rl.analysis.random_eval import (
        evaluate_paired_vs_random,
        evaluate_vs_random_gpu,
    )
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    with open(args.config, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    net_cfg = cfg_yaml.get("net") or cfg_yaml.get("ppo", {}).get("net")
    if net_cfg is None:
        raise SystemExit(f"No net section in {args.config}")

    ppo_cfg = cfg_yaml.get("ppo") or {}
    dtype_name = str(ppo_cfg.get("dtype", "bfloat16"))
    autocast_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if autocast_dtype is None:
        raise SystemExit(f"Unsupported ppo.dtype {dtype_name!r}")

    policy = JunqiNet(JunqiNetConfig(**net_cfg)).to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if args.checkpoint_kind == "ema":
        if not isinstance(sd, dict) or "ema" not in sd:
            raise SystemExit("Checkpoint has no EMA state")
        ema_sd = sd["ema"]
        # Training checkpoints store EMAPolicy.state_dict() → {"decay", "shadow"}.
        weights = ema_sd.get("shadow") if isinstance(ema_sd, dict) else ema_sd
        if weights is None:
            raise SystemExit("Checkpoint EMA state has no shadow weights")
    elif isinstance(sd, dict) and "policy" in sd:
        weights = sd["policy"]
    elif isinstance(sd, dict) and "net" in sd:
        weights = sd["net"]
    else:
        weights = sd
    policy.load_state_dict(weights)
    policy.eval()

    if args.setup_seed is not None:
        # The CUDA setup pool is process-global. Bootstrap it before the
        # evaluator creates/reuses its bounded batch so this CLI can provide a
        # genuinely held-out, reproducible setup distribution.
        device_id = torch.device(args.device).index or 0
        bootstrap = GpuRollout(
            num_envs=min(64, args.num_games),
            device_id=device_id,
        )
        pool_size = bootstrap.upload_fixed_evaluation_setup_pool(
            seed=args.setup_seed,
        )
    else:
        pool_size = 0

    if args.single_team:
        stats = evaluate_vs_random_gpu(
            policy,
            num_envs=min(64, args.num_games),
            num_games=args.num_games,
            trained_team=args.trained_team,
            max_moves=args.max_steps,
            device=args.device,
            seed=args.seed_base,
            autocast_dtype=autocast_dtype,
            greedy=args.greedy,
        )
        protocol = f"single team {args.trained_team}"
    else:
        stats = evaluate_paired_vs_random(
            policy,
            num_games=args.num_games,
            num_envs=min(64, args.num_games),
            use_gpu=True,
            device=args.device,
            seed=args.seed_base,
            max_moves=args.max_steps,
            autocast_dtype=autocast_dtype,
            greedy=args.greedy,
        )
        protocol = "paired teams 0+1"

    requested = int(stats["eval/requested_games"])
    completed = int(stats["eval/num_games"])
    low = stats["eval/win_rate_ci95_low"]
    high = stats["eval/win_rate_ci95_high"]

    print(f"ckpt:     {args.ckpt}")
    print(f"weights:  {args.checkpoint_kind}")
    print(f"protocol: {protocol}")
    if pool_size:
        print(f"setups:   fixed seed={args.setup_seed} pool={pool_size}")
    print(f"games:    {completed}/{requested}")
    print(f"win:      {stats['eval/win_rate']:.4f}")
    print(f"loss:     {stats['eval/loss_rate']:.4f}")
    print(f"draw:     {stats['eval/draw_rate']:.4f}")
    print(f"ongoing:  {stats['eval/ongoing_rate']:.4f}")
    print(f"avg_len:  {stats['eval/avg_game_len']:.1f}")
    print(f"Wilson95: [{low:.4f}, {high:.4f}]")
    if low >= 0.90:
        print("=> 90% bar: CI lower bound >= 0.90 (passed)")
    elif high >= 0.90:
        print("=> 90% bar: CI includes 0.90 (more games required)")
    else:
        print("=> 90% bar: not yet established at 95% confidence")


if __name__ == "__main__":
    main()
