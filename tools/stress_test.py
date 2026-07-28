
"""Offline stress test: run many random self-play games and write a CSV report.

Usage:
  python -m tools.stress_test --games 1000 --max-steps 4000 --out report.csv
  python -m tools.stress_test --games 100 --track-beliefs

Writes:
  report.csv   per-game rows (seed, steps, reason, winner, steps_per_sec, violations)
  summary.json aggregated metrics

Exit code:
  0  all games finished with zero invariant violations
  1  at least one violation detected (see report.csv for details)

See docs/DECISIONS.md ADR-017 for validation strategy rationale.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

from junqi_core.simulator import (
    StressSummary,
    simulate_random_game,
    validate_trace_invariants,
)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="junqi_core random self-play stress test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games", type=int, default=1000, help="Number of games to run")
    p.add_argument("--max-steps", type=int, default=4000, help="Per-game step cap")
    p.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed; seeds used are [start, start+games)",
    )
    p.add_argument(
        "--track-beliefs", action="store_true",
        help="Maintain BeliefTensor per observer (3× slower but checks I1/I2)",
    )
    p.add_argument(
        "--skip-reproducibility", action="store_true",
        help="Skip V1 reproducibility check (2× faster; enable for smoke tests)",
    )
    p.add_argument(
        "--out", type=Path, default=Path(".junqi_tmp") / "stress_report.csv",
        help="Per-game CSV report output",
    )
    p.add_argument(
        "--summary", type=Path, default=Path(".junqi_tmp") / "stress_summary.json",
        help="Aggregated summary JSON output",
    )
    p.add_argument(
        "--progress-every", type=int, default=50,
        help="Print progress line every N games (0 = silent)",
    )
    return p.parse_args(argv)


# ===========================================================================
# Main driver
# ===========================================================================


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    summary = StressSummary()
    rows: list[dict[str, object]] = []
    total_violations = 0
    t_wall0 = perf_counter()

    for i in range(args.games):
        seed = args.seed_start + i
        trace = simulate_random_game(
            seed=seed,
            max_steps=args.max_steps,
            track_beliefs=args.track_beliefs,
        )
        violations = validate_trace_invariants(
            trace, recheck_hashes=not args.skip_reproducibility,
        )
        total_violations += len(violations)
        summary.register(trace, violations)

        rows.append({
            "seed": seed,
            "num_steps": trace.num_steps,
            "reason": trace.termination_reason,
            "winner_team": (
                "draw" if trace.draw
                else ("0" if trace.winner_team == 0
                      else "1" if trace.winner_team == 1
                      else "none")
            ),
            "terminal_hash": trace.terminal_hash,
            "steps_per_sec": f"{trace.steps_per_sec():.1f}",
            "time_step_sec": f"{trace.time_step_total:.4f}",
            "time_belief_sec": f"{trace.time_belief_total:.4f}",
            "violations": len(violations),
            "violation_detail": "; ".join(
                f"{v.kind}:{v.detail}" for v in violations
            ) if violations else "",
        })

        if args.progress_every and (i + 1) % args.progress_every == 0:
            elapsed = perf_counter() - t_wall0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"[{i + 1}/{args.games}] "
                f"violations={total_violations} "
                f"mean_steps={summary.mean_steps:.1f} "
                f"steps/s={summary.steps_per_sec():.0f} "
                f"games/s={rate:.2f}",
                flush=True,
            )

    # --- Write CSV -------------------------------------------------------
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["seed"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # --- Write JSON summary ---------------------------------------------
    summary_dict = summary.as_dict()
    summary_dict["wall_time_sec"] = round(perf_counter() - t_wall0, 3)
    summary_dict["total_violations"] = total_violations
    summary_dict["track_beliefs"] = args.track_beliefs
    summary_dict["reproducibility_checked"] = not args.skip_reproducibility
    summary_dict["seed_range"] = [args.seed_start, args.seed_start + args.games]

    with args.summary.open("w") as f:
        json.dump(summary_dict, f, indent=2, sort_keys=True)

    # --- Print final report ---------------------------------------------
    print()
    print("=" * 60)
    print("Stress test summary")
    print("=" * 60)
    for k, v in sorted(summary_dict.items()):
        print(f"  {k}: {v}")
    print("=" * 60)
    print(f"CSV:     {args.out}")
    print(f"Summary: {args.summary}")

    return 0 if total_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
