"""Diagnostic: root-cause the loss_p=0.0000 / loss_v=0.0000 plateau.

Hypothesis: with num_epochs_per_rollout=1, PPO ratio is identically 1.0 on
every update, so loss reduces to REINFORCE: `-adv_norm * log_prob`. When
the value net has learned the (negative, near-constant) return, raw
advantages collapse to ~0, their std collapses, and `adv_norm = (adv -
mean) / (std + 1e-8)` becomes numerically unstable. After the 0.75
quantile filter, the *kept* samples might carry |adv_norm| far below what
would produce a visible loss — or the fp16 autocast underflows tiny
products of ratio*adv_norm.

This script quantifies that across v20/v22/v23/v24/v25 by plotting:
  - rollout/mean_advantage      (signed)
  - rollout/std_advantage       (scale)
  - rollout/num_valid           (samples past the filter)
  - rollout/mean_return         (what value net is predicting)
  - train/entropy_loss          (policy collapse proxy)
  - train/kl_loss               (new-vs-old-policy divergence)
  - train/policy_loss           (the 0.0000 we're explaining)
  - train/value_loss
  - train/nan_skip_total        (fp16 blow-ups)

Output: aligned time-series so we can eyeball the correlation between
loss=0 onset and advantage collapse.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

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


def fmt_val(v: float | None) -> str:
    if v is None:
        return "     .  "
    if abs(v) < 1e-4:
        return f"{v:+.2e}"
    return f"{v:+.4f}"


def find_loss_zero_onset(pts, tol: float = 1e-5) -> int | None:
    """First step where |policy_loss| < tol for 3 consecutive points."""
    streak = 0
    for s, v in pts:
        if abs(v) < tol:
            streak += 1
            if streak >= 3:
                return s - 2 * (pts[1][0] - pts[0][0] if len(pts) > 1 else 5)
        else:
            streak = 0
    return None


def aligned_dump(run_dir: Path, label: str, keys: list[str], stride_rollouts: int = 20):
    print(f"\n{'='*100}")
    print(f"  {label} — {run_dir.name}")
    print(f"{'='*100}")
    tb = latest_tb(run_dir)
    if not tb:
        print(f"  (no tb file)")
        return

    ea = load(tb)

    # Report the loss=0 onset
    pl = series(ea, "train/policy_loss")
    onset = find_loss_zero_onset(pl, tol=1e-5) if pl else None
    if onset is not None:
        print(f"  policy_loss→0 onset (abs < 1e-5, 3 consec): rollout ≈ {onset}")
    elif pl:
        print(f"  policy_loss never hit 0 (last = {pl[-1][1]:.6f})")
    else:
        print("  (no policy_loss series)")

    # Build an index per-step per-key.
    data: dict[str, dict[int, float]] = {}
    all_steps: set[int] = set()
    for k in keys:
        s = series(ea, k)
        data[k] = {step: val for step, val in s}
        all_steps.update(data[k].keys())

    if not all_steps:
        print("  (no data for any tag)")
        return

    # Stride rollouts and find nearest sample per tag
    min_step = min(all_steps)
    max_step = max(all_steps)
    check_steps = list(range(min_step, max_step + 1, stride_rollouts))
    if max_step not in check_steps:
        check_steps.append(max_step)

    # Compact key names for the header (last component after /)
    short_keys = [k.split("/")[-1][:13] for k in keys]
    header = f"  {'step':>5}  " + "  ".join(f"{sk:>9}" for sk in short_keys)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for step in check_steps:
        row = [f"{step:>5}"]
        for k in keys:
            d = data[k]
            if not d:
                row.append("     .  ")
                continue
            # Find closest step within stride/2
            closest = min(d.keys(), key=lambda s: abs(s - step))
            if abs(closest - step) <= stride_rollouts // 2:
                row.append(fmt_val(d[closest]))
            else:
                row.append("     .  ")
        print("  " + "  ".join(f"{r:>9}" for r in row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stride", type=int, default=20,
        help="rollouts between rows in the printed table",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="override the default run list (space-separated exp dir basenames)",
    )
    args = parser.parse_args()

    # Order: baseline v20 (no belief), v22/v23 (crashed but got past R80
    # loss=0 threshold), v24 (collapsed), v25 (current).
    default_runs = [
        ("v17 (Ataraxos-aligned, storage=2048)", "beat_random_v17_ataraxos_aligned"),
        ("v20 (v17 replay, no-OOM baseline)",    "beat_random_v20_v17_noOOM"),
        ("v22 (belief refresh=1, R54 OOM)",      "beat_random_v22_belief"),
        ("v23 (belief refresh=5, R320 OOM)",     "beat_random_v23_belief_fixed"),
        ("v24 (chunked, collapsed R89)",         "beat_random_v24_belief_chunked"),
        ("v25 (seed=123, current)",              "beat_random_v25_belief_seed123"),
    ]
    if args.runs:
        default_runs = [(name, name) for name in args.runs]

    root = Path("/data/home/freezeng/data/workspace/JunQi/exps")

    # Scalars we care about: flow from advantage/return → filter → loss
    keys = [
        # Rollout-side signals
        "rollout/mean_return",
        "rollout/mean_advantage",
        "rollout/std_advantage",
        "rollout/num_valid",
        # Training dynamics
        "train/policy_loss",
        "train/value_loss",
        "train/entropy_loss",
        "train/kl_loss",
        # NaN counter (fp16 poison)
        "train/nan_skip_total",
    ]

    for label, rel in default_runs:
        aligned_dump(root / rel, label, keys, stride_rollouts=args.stride)

    print()
    print("="*100)
    print("  Interpretation cheat-sheet:")
    print("="*100)
    print("  - If policy_loss→0 coincides with std_advantage→tiny: value net")
    print("    is overfit, advantage signal collapsed. Fix: higher adv_filt_thresh")
    print("    (so we only train on informative samples) OR vf_coef lower (slow")
    print("    down value net) OR num_epochs_per_rollout=4 (give ratio room to move).")
    print("  - If kl_loss spikes before loss→0: policy updated aggressively,")
    print("    then collapsed to a peaky distribution. Fix: kl_coef up, or lr down.")
    print("  - If nan_skip_total rising: fp16 blow-ups. Fix: dtype=float32 or")
    print("    max_grad_norm smaller.")
    print("  - If num_valid stays flat but loss→0: filter is fine, problem is")
    print("    in the loss itself (likely advantage underflow).")


if __name__ == "__main__":
    main()
