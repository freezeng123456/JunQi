#!/usr/bin/env python3
"""Dense vs-random evaluation with Wilson 95% confidence intervals.

Use this script (not a single 128-game eval) to judge whether the policy
has crossed the 90% win-rate bar against a uniform-random opponent.

Example::

    python scripts/eval_random_dense.py \\
        --ckpt exps/v42_planB_T_vs_random/ckpt_best.pt \\
        --config configs/v42_planB_T_vs_random.yaml \\
        --num-games 2048
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
    ap.add_argument("--trained-team", type=int, default=0, choices=[0, 1])
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for eval_random_dense")

    from junqi_rl.analysis.random_eval import evaluate_vs_random_gpu
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    with open(args.config, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    net_cfg = cfg_yaml.get("net") or cfg_yaml.get("ppo", {}).get("net")
    if net_cfg is None:
        raise SystemExit(f"No net section in {args.config}")

    policy = JunqiNet(JunqiNetConfig(**net_cfg)).to(args.device)
    sd = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    if isinstance(sd, dict) and "ema" in sd:
        ema_sd = sd["ema"]
        # Training checkpoints store EMAPolicy.state_dict() → {"decay", "shadow"}.
        if isinstance(ema_sd, dict) and "shadow" in ema_sd:
            policy.load_state_dict(ema_sd["shadow"])
        else:
            policy.load_state_dict(ema_sd)
    elif isinstance(sd, dict) and "net" in sd:
        policy.load_state_dict(sd["net"])
    elif isinstance(sd, dict) and "policy" in sd:
        policy.load_state_dict(sd["policy"])
    else:
        policy.load_state_dict(sd)
    policy.eval()

    stats = evaluate_vs_random_gpu(
        policy,
        num_envs=min(64, args.num_games),
        num_games=args.num_games,
        trained_team=args.trained_team,
        max_moves=args.max_steps,
        device=args.device,
        seed=args.seed_base,
        greedy=args.greedy,
    )

    requested = int(stats["eval/requested_games"])
    completed = int(stats["eval/num_games"])
    low = stats["eval/win_rate_ci95_low"]
    high = stats["eval/win_rate_ci95_high"]

    print(f"ckpt:     {args.ckpt}")
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
