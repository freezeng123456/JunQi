"""Phase 0.3 T7 M6 - Performance benchmark for step() and build_observation().

NOT A TEST. This script is intentionally kept out of CI because its
outputs (steps/sec, observation us) are timing-sensitive and depend on
the host machine, NumPy build, BLAS, etc. Run it manually to certify
M6's two hard numerical targets:

  * step() throughput >= 3000 plays/s single-thread Python.
  * build_observation() mean latency < 2 ms on a mid-game state.

Usage:
  python -m tools.benchmark_t7
  python -m tools.benchmark_t7 --games 20 --steps-per-game 200
  python -m tools.benchmark_t7 --obs-iters 2000 --obs-warmup 200
  python -m tools.benchmark_t7 --observer SOUTH --show-mode DARK

Exit code:
  0  both targets met.
  1  one or both targets missed (useful for bisect scripts).
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from time import perf_counter

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import build_observation
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState


# ---------------------------------------------------------------------------
# step() throughput
# ---------------------------------------------------------------------------


def bench_step(
    *,
    games: int,
    steps_per_game: int,
    show_mode: ShowMode,
    seed_start: int = 0,
) -> dict[str, float]:
    """Benchmark step() in two complementary modes.

    * step_only_per_sec: time spent INSIDE ``GameState.step(action)``
      only (exclusive of legal-action enumeration and RNG sampling).
      This is the narrow metric compared against ADR-017's 3000/s
      Phase 0.2 T5 baseline.
    * loop_per_sec: end-to-end random-policy rollout speed including
      ``state.legal_actions(seat=...)`` and ``rng.choice(...)``. Useful
      as a rollout-throughput indicator for PPO self-play planning.
    """
    total_steps = 0
    step_only_elapsed = 0.0
    loop_elapsed = 0.0
    terminated_early = 0

    for i in range(games):
        rng = random.Random(seed_start + i)
        state = GameState.new_game(
            generate_random_setup(rng), show_mode=show_mode
        )
        loop_t0 = perf_counter()
        for _ in range(steps_per_game):
            if state.terminated:
                terminated_early += 1
                break
            seat = state.turn
            legal = state.legal_actions(seat=seat)
            if not legal:
                break
            action = rng.choice(legal)
            step_t0 = perf_counter()
            state, _ = state.step(action)
            step_only_elapsed += perf_counter() - step_t0
            total_steps += 1
        loop_elapsed += perf_counter() - loop_t0

    step_rate = total_steps / step_only_elapsed if step_only_elapsed > 0 else 0.0
    loop_rate = total_steps / loop_elapsed if loop_elapsed > 0 else 0.0
    return {
        "total_steps": total_steps,
        "step_only_elapsed_s": step_only_elapsed,
        "step_only_per_sec": step_rate,
        "loop_elapsed_s": loop_elapsed,
        "loop_per_sec": loop_rate,
        "games": games,
        "terminated_early": terminated_early,
    }


# ---------------------------------------------------------------------------
# build_observation() latency
# ---------------------------------------------------------------------------


def bench_observation(
    *,
    n_iter: int,
    warmup: int,
    seed: int,
    observer: Seat,
    show_mode: ShowMode,
    advance_steps: int,
) -> dict[str, float]:
    # Build a non-trivial mid-game state.
    rng = random.Random(seed)
    state = GameState.new_game(
        generate_random_setup(rng), show_mode=show_mode
    )
    for _ in range(advance_steps):
        if state.terminated:
            break
        seat = state.turn
        legal = state.legal_actions(seat=seat)
        if not legal:
            break
        action = rng.choice(legal)
        state, _ = state.step(action)

    belief = BeliefTensor.initial(state, observer)

    # Warm-up (static-board cache, allocator pools, CPU caches).
    for _ in range(warmup):
        build_observation(state, belief, observer)

    # Per-iteration timings so we can report distribution, not just mean.
    samples = []
    for _ in range(n_iter):
        t0 = perf_counter()
        build_observation(state, belief, observer)
        samples.append(perf_counter() - t0)

    samples.sort()
    mean_us = statistics.mean(samples) * 1e6
    median_us = statistics.median(samples) * 1e6
    p95_us = samples[int(0.95 * len(samples))] * 1e6 if samples else 0.0
    p99_us = samples[int(0.99 * len(samples))] * 1e6 if samples else 0.0
    min_us = samples[0] * 1e6 if samples else 0.0
    max_us = samples[-1] * 1e6 if samples else 0.0

    return {
        "n_iter": n_iter,
        "warmup": warmup,
        "mean_us": mean_us,
        "median_us": median_us,
        "p95_us": p95_us,
        "p99_us": p99_us,
        "min_us": min_us,
        "max_us": max_us,
        "observer": observer.name,
        "show_mode": show_mode.name,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="T7 M6 performance benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--steps-per-game", type=int, default=200)
    p.add_argument("--obs-iters", type=int, default=1000)
    p.add_argument("--obs-warmup", type=int, default=50)
    p.add_argument("--obs-advance", type=int, default=40)
    p.add_argument(
        "--observer", choices=[s.name for s in Seat], default="SOUTH"
    )
    p.add_argument(
        "--show-mode",
        choices=[m.name for m in ShowMode],
        default="HALF_DARK",
    )
    p.add_argument(
        "--step-target-per-sec", type=float, default=3000.0,
        help="M6 pass/fail threshold for step() throughput",
    )
    p.add_argument(
        "--obs-target-ms", type=float, default=2.0,
        help="M6 pass/fail threshold for build_observation() mean latency",
    )
    return p.parse_args(argv)


def _fmt(n: float, suffix: str = "") -> str:
    return f"{n:,.2f}{suffix}"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    show_mode = ShowMode[args.show_mode]
    observer = Seat[args.observer]

    print("=" * 66)
    print("T7 M6 benchmark  (Phase 0.3)")
    print("=" * 66)
    print(
        f"  show_mode={show_mode.name}  observer={observer.name}  "
        f"games={args.games} x steps_per_game={args.steps_per_game}"
    )
    print()

    # --- step() throughput -----------------------------------------------
    print("Section 1 / 2  -  step() throughput")
    print("-" * 66)
    s = bench_step(
        games=args.games,
        steps_per_game=args.steps_per_game,
        show_mode=show_mode,
    )
    print(f"  total_steps        : {int(s['total_steps']):,}")
    print(f"  games              : {int(s['games'])}")
    print(f"  terminated_early   : {int(s['terminated_early'])}")
    print(f"  step_only_elapsed  : {_fmt(s['step_only_elapsed_s'], 's')}")
    print(f"  step_only_per_sec  : {_fmt(s['step_only_per_sec'])}   <-- T5 baseline metric")
    print(f"  loop_elapsed       : {_fmt(s['loop_elapsed_s'], 's')}")
    print(f"  loop_per_sec       : {_fmt(s['loop_per_sec'])}   (incl. legal_actions + RNG)")
    step_pass = s["step_only_per_sec"] >= args.step_target_per_sec
    print(
        f"  TARGET (step_only_per_sec >= {args.step_target_per_sec:,.0f}/s): "
        f"{'PASS' if step_pass else 'FAIL'}"
    )
    print()

    # --- build_observation() latency -------------------------------------
    print("Section 2 / 2  -  build_observation() latency")
    print("-" * 66)
    o = bench_observation(
        n_iter=args.obs_iters,
        warmup=args.obs_warmup,
        seed=10007,
        observer=observer,
        show_mode=show_mode,
        advance_steps=args.obs_advance,
    )
    print(f"  n_iter           : {int(o['n_iter']):,}")
    print(f"  warmup           : {int(o['warmup']):,}")
    print(f"  mean             : {_fmt(o['mean_us'], ' us')}")
    print(f"  median           : {_fmt(o['median_us'], ' us')}")
    print(f"  p95              : {_fmt(o['p95_us'], ' us')}")
    print(f"  p99              : {_fmt(o['p99_us'], ' us')}")
    print(f"  min              : {_fmt(o['min_us'], ' us')}")
    print(f"  max              : {_fmt(o['max_us'], ' us')}")
    target_us = args.obs_target_ms * 1000.0
    obs_pass = o["mean_us"] < target_us
    print(
        f"  TARGET (mean < {args.obs_target_ms:g} ms): "
        f"{'PASS' if obs_pass else 'FAIL'}"
    )
    print()

    # --- Summary ---------------------------------------------------------
    print("=" * 66)
    print(
        f"Overall: step()={'PASS' if step_pass else 'FAIL'}  "
        f"build_observation()={'PASS' if obs_pass else 'FAIL'}"
    )
    print("=" * 66)

    return 0 if (step_pass and obs_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
