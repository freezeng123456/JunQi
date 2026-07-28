"""Diagnostic: compare v23 (belief enabled but no refresh fired post-warmup
until rollout 20→never actually refreshed past rollout 20→320 crash) vs v24
(chunked refresh fires every 5 rollouts after warmup=100).

Key questions:
1. Is the win_rate collapse in v24 after rollout 100 caused by the belief
   refresh (new) or by a pre-existing trend (also in v23)?
2. Are belief-train metrics (ce/accuracy) sane or garbage?
3. Are belief-infer metrics (mean_entropy/max_prob_mean) sane?
"""
from __future__ import annotations

from pathlib import Path
import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load(path: Path) -> EventAccumulator:
    ea = EventAccumulator(str(path), size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea, key):
    if key not in ea.Tags()["scalars"]:
        return []
    return [(e.step, e.value) for e in ea.Scalars(key)]


def latest_tb(run_dir: Path) -> Path | None:
    matches = sorted(glob.glob(str(run_dir / "logs" / "events.out.tfevents*")))
    return Path(matches[-1]) if matches else None


def print_curve(name: str, pts, stride: int = 10):
    if not pts:
        print(f"  {name}: (no data)")
        return
    print(f"  {name}:")
    for s, v in pts[::stride]:
        print(f"    step={s:4d}  {v:+.4f}")
    s, v = pts[-1]
    print(f"    step={s:4d}  {v:+.4f}  [LAST]")


def dump(run_dir: Path, label: str):
    print(f"\n=== {label}: {run_dir.name} ===")
    tb = latest_tb(run_dir)
    if not tb:
        print(f"  (no tb file)"); return
    ea = load(tb)
    tags = ea.Tags()["scalars"]
    print(f"  all tags ({len(tags)}):")
    for t in sorted(tags):
        print(f"    - {t}")
    print()

    for key in [
        "eval/win_rate",
        "eval/avg_game_len",
        "train/policy_loss",
        "train/value_loss",
        "train/entropy",
        "rollout/mean_return",
        # belief-specific
        "belief_train/ce_loss",
        "belief_train/uniform_ce",
        "belief_train/accuracy",
        "belief_train/n_revealed",
        "belief_infer/mean_entropy",
        "belief_infer/max_prob_mean",
    ]:
        pts = series(ea, key)
        if pts:
            print_curve(key, pts, stride=max(1, len(pts) // 15))


if __name__ == "__main__":
    root = Path("/data/home/freezeng/data/workspace/JunQi/exps")
    dump(root / "beat_random_v20_v17replay", label="v20 (no belief, baseline)")
    dump(root / "beat_random_v22_belief", label="v22 (belief+refresh=1, crashed R54)")
    dump(root / "beat_random_v23_belief_fixed", label="v23 (belief+refresh=5, warmup=100, crashed R320)")
    dump(root / "beat_random_v24_belief_chunked", label="v24 (current)")
