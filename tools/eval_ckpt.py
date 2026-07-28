"""Load the last saved best ckpt and re-evaluate with bigger sample size."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import torch

from junqi_rl.analysis import eval_vs_random
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def main():
    import sys
    ckpt_path = Path(sys.argv[1] if len(sys.argv) > 1
                     else "exps/toy_t4_beat_random/ckpt_best.pt")
    if not ckpt_path.exists():
        print("no ckpt at", ckpt_path); sys.exit(1)

    cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=2, depth=2,
                         embed_dim=64, n_head=4, ff_factor=2, dropout=0.0)
    policy = JunqiNet(cfg).to("cuda").eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    policy.load_state_dict(ckpt["ema"])
    print(f"loaded ckpt @ rollout {ckpt.get('rollout', '?')}, "
          f"saved eval: {ckpt.get('eval_stats')}")

    # Re-evaluate with bigger sample
    for (greedy, team) in [(False, 0), (True, 0), (False, 1)]:
        stats = eval_vs_random(policy, num_games=128, trained_team=team,
                               max_steps=1500, device="cuda", seed_base=999,
                               greedy=greedy)
        print(f"  team={team} greedy={greedy}  "
              f"win={stats['trained_win_rate']:.3f}  "
              f"loss={stats['trained_loss_rate']:.3f}  "
              f"draw={stats['draw_rate']:.3f}  "
              f"ongoing={stats['ongoing_rate']:.3f}  "
              f"mean_len={stats['mean_length']:.1f}")


if __name__ == "__main__":
    main()
