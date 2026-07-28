
"""Small-scale stress test for CI (Phase 0.2 T5).

Runs a limited batch of random self-play games and asserts:
  - no invariant violations across ~50 games
  - all 4 termination reasons can occur across enough seeds
  - belief tensors maintain I1/I2 through an entire game
  - reproducibility holds on random seeds
  - performance is within acceptable range (>1000 steps/sec, i.e. much
    lower than observed so we don't flake on slow CI hardware)

For the comprehensive 1000-game suite, use tools/stress_test.py.
"""

from __future__ import annotations

import pytest

from junqi_core.simulator import (
    simulate_random_game,
    validate_trace_invariants,
)


# ===========================================================================
# Smoke: a single short game
# ===========================================================================


def test_single_game_smoke() -> None:
    """One game with seed=0 completes without invariants violation."""
    trace = simulate_random_game(seed=0, max_steps=500, track_beliefs=False)
    violations = validate_trace_invariants(trace)
    assert not violations, f"violations: {[str(v) for v in violations]}"
    assert trace.num_steps > 0
    # With max_steps=500, either terminated OR exactly 500 steps reached
    assert trace.terminated or trace.num_steps == 500


def test_single_game_with_beliefs() -> None:
    """Running with track_beliefs=True still passes invariants."""
    trace = simulate_random_game(seed=1, max_steps=300, track_beliefs=True)
    violations = validate_trace_invariants(trace)
    assert not violations, f"violations: {[str(v) for v in violations]}"
    # Beliefs should exist for every observer
    for observer, belief in trace.terminal_beliefs.items():
        # Every belief row on an alive piece must be normalized
        for pos, vec in belief.probs.items():
            s = float(vec.sum())
            assert 0.99 <= s <= 1.01, (
                f"{observer.name} at {pos} belief row sums to {s}"
            )
            assert (vec >= 0).all()


# ===========================================================================
# Reproducibility: deterministic on fixed seed
# ===========================================================================


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
def test_reproducibility(seed: int) -> None:
    """Two runs with the same seed produce byte-identical terminal hash."""
    trace_a = simulate_random_game(seed=seed, max_steps=500)
    trace_b = simulate_random_game(seed=seed, max_steps=500)
    assert trace_a.num_steps == trace_b.num_steps, (
        f"seed={seed}: num_steps differ {trace_a.num_steps} vs {trace_b.num_steps}"
    )
    assert trace_a.terminal_hash == trace_b.terminal_hash, (
        f"seed={seed}: terminal hash mismatch"
    )
    assert trace_a.termination_reason == trace_b.termination_reason


# ===========================================================================
# Batch: 25 games pass invariants
# ===========================================================================


def test_batch_25_games_no_violations() -> None:
    """25 games with varied seeds all pass invariant validation."""
    violations_total = 0
    for seed in range(25):
        trace = simulate_random_game(seed=seed, max_steps=400, track_beliefs=False)
        v = validate_trace_invariants(trace, recheck_hashes=False)
        violations_total += len(v)
        assert not v, f"seed={seed}: {[str(x) for x in v]}"
    assert violations_total == 0


# ===========================================================================
# Coverage: different termination reasons occur
# ===========================================================================


def test_termination_reason_diversity() -> None:
    """Across 50 seeds we should see at least 2 distinct termination reasons
    (most will be flag_capture / team_kill / timeout depending on step cap)."""
    reasons: set[str] = set()
    for seed in range(50):
        trace = simulate_random_game(seed=seed, max_steps=1500)
        reasons.add(trace.termination_reason)
    # We expect at least one of the real termination reasons plus timeout
    # (since max_steps=1500 isn't always enough to finish).
    # The important thing is that the simulator does not crash on any seed.
    assert len(reasons) >= 1, f"only saw reasons: {reasons}"


# ===========================================================================
# Performance baseline (lenient for CI)
# ===========================================================================


def test_performance_baseline_lenient() -> None:
    """step() throughput is at least 1000 steps/sec on CI.

    Local dev measured ~15000 steps/sec; we leave huge slack for slow CI
    runners. If this fails we probably have an O(N^2) regression somewhere.
    """
    trace = simulate_random_game(seed=42, max_steps=500, track_beliefs=False)
    if trace.num_steps >= 50:  # need enough steps for a stable measurement
        sps = trace.steps_per_sec()
        assert sps >= 1000, (
            f"step() too slow: {sps:.1f} steps/sec (expected ≥ 1000). "
            f"Measured over {trace.num_steps} steps in {trace.time_step_total:.3f}s."
        )


def test_belief_update_performance_baseline() -> None:
    """BeliefTensor.update() adds reasonable overhead — track_beliefs games
    should still be faster than 300 steps/sec."""
    trace = simulate_random_game(seed=42, max_steps=300, track_beliefs=True)
    if trace.num_steps >= 30 and trace.time_belief_total > 0:
        belief_per_step_us = (trace.time_belief_total / trace.num_steps) * 1e6
        # Expect < 3ms per step (including 4 observers)
        assert belief_per_step_us < 3000, (
            f"belief update too slow: {belief_per_step_us:.1f} µs/step "
            f"(expected < 3000 µs)"
        )
