#!/usr/bin/env python3
"""Summarize actual first actions recorded in training_games.jsonl.

The training collector records the first action applied to every completed
game. This script intentionally does not run a policy on a synthetic board;
it measures the distribution that the self-play process actually generated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(path: Path) -> dict[str, object]:
    from junqi_core.board import COMPACT_TO_FLAT, flat_to_xy

    counts: Counter[int] = Counter()
    total = 0
    missing = 0
    # A long run can disable per-game JSON records. In that mode use the
    # cumulative first_action_counts written into game_progress.jsonl.
    with path.open(encoding="utf-8") as probe:
        first_line = next((line for line in probe if line.strip()), "")
    first_row = json.loads(first_line) if first_line else {}
    if "first_action_counts" in first_row:
        last = first_row
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = json.loads(line)
        counts.update({int(k): int(v) for k, v in last["first_action_counts"].items()})
        total = sum(counts.values())
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                action = int(row.get("first_action", -1))
                if action < 0:
                    missing += 1
                    continue
                counts[action] += 1
                total += 1

    source_counts = Counter()
    for action, n in counts.items():
        source_counts[action // 129] += n

    entropy = 0.0
    if total:
        entropy = -sum(
            (n / total) * math.log(n / total)
            for n in counts.values()
            if n
        )
    top = []
    for action, n in counts.most_common(10):
        source, destination = divmod(action, 129)
        flat = int(COMPACT_TO_FLAT[source])
        x, y = flat_to_xy(flat)
        top.append({
            "action": action,
            "source": source,
            "destination": destination,
            "canonical_xy": [x, y],
            "count": n,
            "mass": n / total if total else 0.0,
        })
    return {
        "games_with_first_move": total,
        "games_missing_first_move": missing,
        "unique_actions": len(counts),
        "unique_sources": len(source_counts),
        "entropy_nats": entropy,
        "top_actions": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=Path, help="training_games.jsonl")
    args = parser.parse_args()
    print(json.dumps(summarize(args.games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
