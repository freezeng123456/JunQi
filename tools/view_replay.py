"""tools/view_replay.py — Pretty-print a TrajectoryWithPolicy file.

Shows per-step:
  * acting seat
  * chosen move (world coords)
  * value estimate V(s)
  * top-K legal actions with probabilities
  * if beliefs were recorded: the most likely enemy piece type per cell

Usage:
    python3 tools/view_replay.py path/to/recording.npz [--step N] [--topk 8]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from junqi_core.replay_with_policy import TrajectoryWithPolicy
from junqi_core.rules import Seat


_SEAT_NAMES = ["SOUTH", "WEST", "NORTH", "EAST"]


def _decode(action_id: int) -> tuple[tuple[int, int], tuple[int, int]]:
    src_flat = action_id // 289
    dst_flat = action_id %  289
    sy, sx = divmod(src_flat, 17)
    dy, dx = divmod(dst_flat, 17)
    return (sx, sy), (dx, dy)


def _print_step(traj: TrajectoryWithPolicy, t: int, topk: int) -> None:
    row = traj.actions[t]
    seat_v = int(row[0])
    seat_name = _SEAT_NAMES[seat_v]
    src = (int(row[1]), int(row[2]))
    dst = (int(row[3]), int(row[4]))
    value = float(traj.values[t])

    print(f"\n━━━━━ step {t:3d}  seat={seat_name}  chose ({src[0]},{src[1]})→({dst[0]},{dst[1]})  V(s)={value:+.3f}")

    k_stored = traj.top_k
    k = min(topk, k_stored)
    ids = traj.top_action_ids[t]
    probs = traj.top_probs[t]
    print(f"  top-{k} candidate moves (world frame):")
    for i in range(k):
        aid = int(ids[i])
        p   = float(probs[i])
        if p == 0.0 and i > 0:
            break
        s, d = _decode(aid)
        marker = "►" if (s == src and d == dst) else " "
        print(f"    {marker}  ({s[0]:2d},{s[1]:2d})→({d[0]:2d},{d[1]:2d})  "
              f"p={p:6.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to TrajectoryWithPolicy .npz file")
    p.add_argument("--step", type=int, default=None, help="Only print this step")
    p.add_argument("--topk", type=int, default=8, help="Top-K moves to show")
    p.add_argument("--range", default=None,
                   help="Range 'a-b' (inclusive) to print, overrides --step")
    args = p.parse_args()

    traj = TrajectoryWithPolicy.load(args.path)
    T = traj.num_steps

    print(f"# {args.path}")
    print(f"  steps={T}  top_k_stored={traj.top_k}  "
          f"first_seat={traj.first_seat.name}  show_mode={traj.show_mode.name}")
    print(f"  rng_seed={traj.rng_seed}  final_state_hash=0x{traj.final_state_hash & 0xFFFFFFFFFFFFFFFF:016x}")

    if args.step is not None:
        _print_step(traj, args.step, args.topk)
    elif args.range is not None:
        a, b = args.range.split("-")
        a, b = int(a), int(b)
        for t in range(max(0, a), min(T, b + 1)):
            _print_step(traj, t, args.topk)
    else:
        # Print summary: first 3, middle, last 3
        samples = sorted(set([0, 1, 2, T // 2, T - 3, T - 2, T - 1]))
        samples = [t for t in samples if 0 <= t < T]
        for t in samples:
            _print_step(traj, t, args.topk)


if __name__ == "__main__":
    main()
