#!/usr/bin/env python3
"""Inspect a JunQi ``.npz`` replay without a GUI.

Examples::

    python tools/replay_viewer.py game.npz --step 120
    python tools/replay_viewer.py rl_game.npz --all --json > frames.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from junqi_core.replay_viewer import ReplayViewer, frame_summary, load_replay


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("replay", type=Path, help="Trajectory .npz file")
    p.add_argument("--step", type=int, default=None, help="step to inspect (0 = initial)")
    p.add_argument("--all", action="store_true", help="emit every frame")
    p.add_argument("--json", action="store_true", help="emit JSON/JSONL instead of text")
    p.add_argument("--validate", action="store_true", help="replay and verify final hash")
    return p


def main() -> int:
    args = _parser().parse_args()
    source = load_replay(str(args.replay))
    if args.validate:
        source.validate()

    viewer = ReplayViewer(source)
    if args.all:
        frames = [viewer.reset()]
        for _ in range(viewer.length):
            frames.append(viewer.next())
    else:
        frames = [viewer.seek(viewer.length if args.step is None else args.step)]

    for frame in frames:
        data = frame_summary(frame)
        if args.json:
            print(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
        else:
            action = data.get("action", {})
            result = data.get("result", {})
            policy = data.get("policy", {})
            line = (
                f"step={data['step']:>4} turn={data['turn']:<5} "
                f"event={result.get('event', 'START'):<8} "
                f"action={action.get('src', '-')}>{action.get('dst', '')} "
                f"alive={data['alive_pieces']}"
            )
            if policy:
                line += f" value={policy['value']:.4f} chosen={policy['chosen_action_id']}"
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
