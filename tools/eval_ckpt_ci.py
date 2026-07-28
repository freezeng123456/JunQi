"""Re-evaluate a ckpt over many seeds to get a tight confidence interval."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import numpy as np
import torch

from junqi_rl.analysis import eval_vs_random
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def main():
    ckpt_path = Path(sys.argv[1] if len(sys.argv) > 1
                     else "exps/toy_shaped/ckpt_best.pt")
    cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=2, depth=2,
                         embed_dim=64, n_head=4, ff_factor=2, dropout=0.0)
    policy = JunqiNet(cfg).to("cuda").eval()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    policy.load_state_dict(ckpt["ema"])
    print(f"# Loaded {ckpt_path} (rollout={ckpt.get('rollout')})")

    # Accumulate over 8 seeds, team=0 AND team=1
    all_wins = []
    total_games = 0
    for team in (0, 1):
        for seed in range(8):
            stats = eval_vs_random(
                policy, num_games=128, trained_team=team,
                max_steps=1500, device="cuda", seed_base=seed * 1000,
                greedy=False,
            )
            all_wins.append(stats["trained_win_rate"])
            total_games += 128
            print(f"  team={team} seed={seed}  win={stats['trained_win_rate']:.3f}")
    mean = float(np.mean(all_wins))
    std = float(np.std(all_wins))
    # 95% CI assuming ~binomial: std_err ≈ sqrt(p(1-p)/N_total)
    p_hat = mean
    se = (p_hat * (1 - p_hat) / total_games) ** 0.5
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    print(f"\n==> win_rate = {mean:.4f}  [95%CI: {ci_lo:.4f}, {ci_hi:.4f}]  "
          f"(std across runs = {std:.4f}, total_games = {total_games})")


if __name__ == "__main__":
    main()
