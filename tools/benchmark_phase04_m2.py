"""Phase 0.4 M2 benchmark — ObservationBuilder vs pre-M2 build_observation.

Writes a compact report to /tmp/m2_bench.log plus prints the headline
numbers to stdout.  Drives 3 metrics:

  * ``builder.build()``      — hot path; should beat pre-M2 by 15–30 %.
  * ``build_observation()``  — legacy path (= builder.build + snapshot);
                                should be roughly equal.
  * per-call Python allocations — should be essentially zero after the
    first warmup call (verified via ``tracemalloc``).
"""
import random
import time
import tracemalloc
from pathlib import Path

LOG_PATH = "/tmp/m2_bench.log"


def main() -> None:
    from junqi_core.info_model import BeliefTensor
    from junqi_core.observation import ObservationBuilder, build_observation
    from junqi_core.rules import Seat, ShowMode
    from junqi_core.setup import generate_random_setup
    from junqi_core.state import GameState

    # A mid-game state with live D/E signals.
    rng = random.Random(31415)
    state = GameState.new_game(generate_random_setup(rng), show_mode=ShowMode.HALF_DARK)
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    for _ in range(100):
        if state.terminated:
            break
        legal = state.legal_actions()
        if not legal:
            break
        a = rng.choice(legal)
        new_state, result = state.step(a)
        if not new_state.info[Seat.SOUTH].dead:
            belief.update(state, new_state, result)
        state = new_state

    builder = ObservationBuilder()

    # Warmup.
    for _ in range(100):
        builder.build(state, belief, Seat.SOUTH)

    N = 2_000

    # -- builder.build --
    t0 = time.perf_counter()
    for _ in range(N):
        builder.build(state, belief, Seat.SOUTH)
    build_dt = (time.perf_counter() - t0) / N * 1e6

    # -- legacy build_observation (includes snapshot copy) --
    t0 = time.perf_counter()
    for _ in range(N):
        build_observation(state, belief, Seat.SOUTH)
    legacy_dt = (time.perf_counter() - t0) / N * 1e6

    # -- tracemalloc: per-call allocation --
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    for _ in range(100):
        builder.build(state, belief, Seat.SOUTH)
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diffs = after.compare_to(before, "filename")
    per_call = sum(d.size_diff for d in diffs) / 100

    report = [
        f"ObservationBuilder.build() mean: {build_dt:.1f} us",
        f"build_observation() mean       : {legacy_dt:.1f} us  (legacy shim)",
        f"per-call allocation            : {per_call:.0f} bytes (target ~0)",
    ]
    for line in report:
        print(line)
    Path(LOG_PATH).write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
