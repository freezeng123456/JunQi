"""Diagnostic: compare v17 and v19 TensorBoard metrics around the loss=0 transition."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(path: Path) -> EventAccumulator:
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea: EventAccumulator, key: str) -> list[tuple[int, float]]:
    if key not in ea.Tags()["scalars"]:
        return []
    return [(e.step, e.value) for e in ea.Scalars(key)]


def at_step(ea: EventAccumulator, key: str, target: int, tol: int = 5) -> float | None:
    pts = series(ea, key)
    if not pts:
        return None
    closest = min(pts, key=lambda p: abs(p[0] - target))
    if abs(closest[0] - target) > tol:
        return None
    return closest[1]


def find_event(run_dir: str) -> Path | None:
    cands = glob.glob(str(Path(run_dir) / "logs" / "events.out.tfevents.*"))
    if not cands:
        return None
    return Path(cands[0])


def main() -> None:
    runs = {
        "v17": "/data/home/freezeng/data/workspace/JunQi/exps/beat_random_v17_ataraxos_aligned",
        "v18": "/data/home/freezeng/data/workspace/JunQi/exps/beat_random_v18_buffer_fix",
        "v19": "/data/home/freezeng/data/workspace/JunQi/exps/beat_random_v19_buffer_right",
    }
    eas: dict[str, EventAccumulator] = {}
    for k, v in runs.items():
        p = find_event(v)
        if p is None:
            print(f"[{k}] no event file found under {v}")
            continue
        eas[k] = load(p)
        print(f"[{k}] loaded {p.name}")

    print()

    keys_of_interest = [
        "eval/win_rate",
        "eval/avg_game_len",
        "rollout/mean_return",
        "rollout/mean_advantage",
        "rollout/std_advantage",
        "train/nan_skip_total",
        "train/num_updates",
        "train/policy_loss",
        "train/value_loss",
        "arr_train/policy_loss",
        "arr_train/value_loss",
        "arr_train/entropy",
        "arr_train/clip_fraction",
        "arr_train/reg_temp",
        "arr/pool_size",
        "arr/fallback_rate",
    ]

    # Compare each key across runs at anchor rollouts.
    anchors = [50, 100, 150, 200, 240]
    for key in keys_of_interest:
        print(f"\n=== {key} ===")
        header = f"{'run':4s}  " + "  ".join(f"rollout={a:3d}" for a in anchors)
        print(header)
        for name, ea in eas.items():
            vals = []
            for a in anchors:
                v = at_step(ea, key, a)
                vals.append(f"{v:11.4g}" if v is not None else "         NA")
            print(f"{name:4s}  " + "  ".join(vals))


if __name__ == "__main__":
    main()
