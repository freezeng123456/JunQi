"""Fresh (untrained) policy baseline — 16 seeds × 128 games."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from junqi_rl.analysis import eval_vs_random
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def main():
    cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=2, depth=2,
                         embed_dim=64, n_head=4, ff_factor=2, dropout=0.0)
    torch.manual_seed(0)
    policy = JunqiNet(cfg).to("cuda").eval()
    print("# Untrained policy baseline — 16 seeds × 128 games")

    all_wins = []
    total = 0
    for team in (0, 1):
        for seed in range(8):
            stats = eval_vs_random(
                policy, num_games=128, trained_team=team,
                max_steps=1500, device="cuda", seed_base=seed * 1000,
                greedy=False,
            )
            all_wins.append(stats["trained_win_rate"])
            total += 128
    mean = float(np.mean(all_wins))
    p_hat = mean
    se = (p_hat * (1 - p_hat) / total) ** 0.5
    print(f"\n==> baseline win_rate = {mean:.4f}  [95%CI: {mean - 1.96*se:.4f}, {mean + 1.96*se:.4f}]")


if __name__ == "__main__":
    main()
