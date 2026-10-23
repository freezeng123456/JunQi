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
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = int(row.get("first_source", -1))
            if source < 0:
                missing += 1
                continue
            counts[source] += 1
            total += 1

    entropy = 0.0
    if total:
        entropy = -sum(
            (n / total) * math.log(n / total)
            for n in counts.values()
            if n
        )
    top = []
    for source, n in counts.most_common(10):
        flat = int(COMPACT_TO_FLAT[source])
        x, y = flat_to_xy(flat)
        top.append({
            "source": source,
            "canonical_xy": [x, y],
            "count": n,
            "mass": n / total if total else 0.0,
        })
    return {
        "games_with_first_move": total,
        "games_missing_first_move": missing,
        "unique_sources": len(counts),
        "entropy_nats": entropy,
        "top_sources": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=Path, help="training_games.jsonl")
    args = parser.parse_args()
    print(json.dumps(summarize(args.games), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
