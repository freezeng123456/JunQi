"""Quick sanity: run eval_vs_random with untrained policy to see baseline."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from junqi_rl.analysis import eval_vs_random
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def main():
    cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=2, depth=2,
                         embed_dim=64, n_head=4, ff_factor=2, dropout=0.0)
    policy = JunqiNet(cfg).to("cuda").eval()
    print("# Untrained vs random baseline")
    for max_steps in (400, 1000, 2000):
        stats = eval_vs_random(policy, num_games=64, trained_team=0,
                               max_steps=max_steps, device="cuda", seed_base=0)
        print(f"  max_steps={max_steps:4d}   "
              f"win={stats['trained_win_rate']:.3f}  "
              f"loss={stats['trained_loss_rate']:.3f}  "
              f"draw={stats['draw_rate']:.3f}  "
              f"ongoing={stats['ongoing_rate']:.3f}  "
              f"mean_len={stats['mean_length']:.1f}")


if __name__ == "__main__":
    main()
