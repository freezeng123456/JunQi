#!/usr/bin/env python3
"""Paired GPU head-to-head between two JunqiNet checkpoints.

Loads each checkpoint with the stem it was trained with (GraphStem or
legacy CNN).  Current env observations are 317-channel; CNN weights that
expect the old 412-channel layout get ``piece_slot`` expanded back to
``piece_id`` (seat × 30 + slot) so the rest of the planes line up.

Example::

    python scripts/eval_h2h.py \\
        --first /path/to/graph_stem/ckpt_000100.pt \\
        --second /path/to/cnn/ckpt_000482.pt \\
        --num-games 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
_JUNQI_RL = ROOT / "junqi_rl"
for _p in (str(ROOT), str(_JUNQI_RL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from junqi_core.observation import CHANNEL_LAYOUT
from junqi_core.rules import SLOTS_PER_SEAT
from junqi_rl.analysis.random_eval import evaluate_paired_head_to_head
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


_LEGACY_PIECE_ID = 120


def _channel_to_slot() -> list[int]:
    from junqi_core.observation import _SLOT_TO_CHANNEL

    mapping = [-1] * 25
    for slot in range(SLOTS_PER_SEAT):
        ch = int(_SLOT_TO_CHANNEL[slot])
        if ch >= 0:
            mapping[ch] = slot
    if any(s < 0 for s in mapping):
        raise RuntimeError("piece_slot → setup-slot map is incomplete")
    return mapping


_CHANNEL_TO_SLOT = _channel_to_slot()


def expand_obs_317_to_412(obs: Tensor) -> Tensor:
    """Rebuild the 412-channel layout from a current 317-channel observation."""
    if obs.size(1) == _LEGACY_PIECE_ID + 317 - 25:
        return obs
    if obs.size(1) != 317:
        raise ValueError(f"expected 317 spatial channels, got {obs.size(1)}")

    slot = CHANNEL_LAYOUT["piece_slot"]
    prefix = obs[:, : slot.start]
    slot_ch = obs[:, slot]
    suffix = obs[:, slot.stop :]

    own = (obs[:, CHANNEL_LAYOUT["piece_own"]].abs() > 0).any(dim=1)
    team = (
        (obs[:, CHANNEL_LAYOUT["prob_teammate"]].abs() > 0).any(dim=1)
        | (obs[:, CHANNEL_LAYOUT["dark_teammate"]].abs() > 0).any(dim=1)
    )
    left = (obs[:, CHANNEL_LAYOUT["piece_left_side_enemy"]].abs() > 0).any(dim=1)
    right = (obs[:, CHANNEL_LAYOUT["piece_right_side_enemy"]].abs() > 0).any(dim=1)

    seat = torch.zeros(own.shape, dtype=torch.long, device=obs.device)
    seat = torch.where(right, torch.full_like(seat, 3), seat)
    seat = torch.where(left, torch.full_like(seat, 1), seat)
    seat = torch.where(team, torch.full_like(seat, 2), seat)
    seat = torch.where(own, torch.zeros_like(seat), seat)

    occupied = slot_ch.abs().sum(dim=1) > 0
    ch = slot_ch.abs().argmax(dim=1)
    slot_lut = torch.tensor(_CHANNEL_TO_SLOT, device=obs.device, dtype=torch.long)
    setup_slot = slot_lut[ch]
    pid = seat * SLOTS_PER_SEAT + setup_slot

    piece_id = obs.new_zeros(obs.size(0), _LEGACY_PIECE_ID, obs.size(2), obs.size(3))
    b, y, x = torch.where(occupied)
    if b.numel():
        vals = slot_ch[b, ch[b, y, x], y, x]
        piece_id[b, pid[b, y, x], y, x] = vals

    out = torch.cat([prefix, piece_id, suffix], dim=1)
    if out.size(1) != 412:
        raise RuntimeError(f"expanded obs has {out.size(1)} channels, want 412")
    return out


class Expand317To412(nn.Module):
    """Forwards 317-channel env obs into a 412-channel CNN policy."""

    def __init__(self, net: JunqiNet) -> None:
        super().__init__()
        self.net = net

    def eval(self):  # type: ignore[override]
        self.net.eval()
        return super().eval()

    def act(self, obs_spatial: Tensor, obs_global: Tensor, legal_mask: Tensor):
        return self.net.act(expand_obs_317_to_412(obs_spatial), obs_global, legal_mask)

    def act_greedy(self, obs_spatial: Tensor, obs_global: Tensor, legal_mask: Tensor):
        return self.net.act_greedy(
            expand_obs_317_to_412(obs_spatial), obs_global, legal_mask
        )


def _cfg_from_ckpt(state: dict, policy: dict) -> JunqiNetConfig:
    raw = state.get("train_cfg")
    net = raw.get("net") if isinstance(raw, dict) else None
    if isinstance(net, dict) and net.get("embed_dim"):
        kwargs = {
            k: net[k]
            for k in (
                "cnn_channels",
                "cnn_layers",
                "depth",
                "embed_dim",
                "n_head",
                "ff_factor",
                "dropout",
                "pos_emb_std",
                "use_cat_vf",
                "action_key_dim",
            )
            if k in net
        }
        return JunqiNetConfig(**kwargs)
    raise SystemExit("checkpoint is missing train_cfg['net']")


def load_policy(path: str, kind: str, device: torch.device) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if kind not in state:
        raise SystemExit(f"{path} has no '{kind}' weights")
    weights = state[kind]
    is_cnn = any(k.startswith("cnn.") for k in weights)
    in_ch = 412 if is_cnn else 317
    if is_cnn:
        in_ch = int(weights["cnn.net.0.weight"].shape[1])
    elif "stem.embed.weight" in weights:
        in_ch = int(weights["stem.embed.weight"].shape[1])

    cfg = _cfg_from_ckpt(state, weights)
    cfg.use_graph_stem = not is_cnn
    cfg.in_channels = in_ch
    net = JunqiNet(cfg)
    missing, unexpected = net.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"state_dict mismatch for {path}\n"
            f"  missing={missing}\n  unexpected={unexpected}"
        )
    net.to(device).eval()
    if is_cnn and in_ch == 412:
        return Expand317To412(net).to(device).eval()
    return net


def _print_metrics(tag: str, metrics: dict[str, float]) -> None:
    print(
        f"[{tag}]  win={metrics.get('h2h/win_rate', 0.0):.3f}  "
        f"loss={metrics.get('h2h/loss_rate', 0.0):.3f}  "
        f"draw={metrics.get('h2h/draw_rate', 0.0):.3f}  "
        f"ongoing={metrics.get('h2h/ongoing_rate', 0.0):.3f}  "
        f"avg_len={metrics.get('h2h/avg_game_len', 0.0):.0f}  "
        f"done={int(metrics.get('h2h/num_games', 0.0))}/"
        f"{int(metrics.get('h2h/requested_games', 0.0))}  "
        f"ci95=[{metrics.get('h2h/win_rate_ci95_low', 0.0):.3f}, "
        f"{metrics.get('h2h/win_rate_ci95_high', 1.0):.3f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", required=True, help="First policy (metrics POV)")
    ap.add_argument("--second", required=True, help="Opponent policy")
    ap.add_argument("--num-games", type=int, default=256)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--max-moves", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--checkpoint-kind", choices=["policy", "ema"], default="policy")
    ap.add_argument("--greedy", action="store_true", help="Argmax instead of sampling")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for eval_h2h")

    device = torch.device(args.device)
    first = load_policy(args.first, args.checkpoint_kind, device)
    second = load_policy(args.second, args.checkpoint_kind, device)
    print(f"[h2h] first  = {args.first}")
    print(f"[h2h] second = {args.second}")
    print(
        f"[h2h] games={args.num_games} envs={args.num_envs} "
        f"greedy={args.greedy} kind={args.checkpoint_kind}"
    )

    metrics = evaluate_paired_head_to_head(
        first,
        second,
        num_games=args.num_games,
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        max_moves=args.max_moves,
        autocast_dtype=torch.bfloat16,
        greedy=args.greedy,
    )
    _print_metrics("h2h first vs second", metrics)


if __name__ == "__main__":
    main()
