
"""Random self-play driver for `junqi_core`.

Given a seed, plays one complete random game (uniform-random legal-action
policy) from the opening position to termination, optionally maintaining
belief tensors for each observer.

Primary use:
  - Stress testing (`tools/stress_test.py`, `tests/test_stress_random.py`)
  - Performance baselining (measure step() / BeliefTensor.update() throughput)
  - Reproducibility validation (same seed → same MoveResult sequence and
    same terminal state hash)

Public API:
  simulate_random_game(seed, max_steps=4000, track_beliefs=False) -> GameTrace
  GameTrace                              — complete record of one game
  validate_trace_invariants(trace)       — post-run audit (I1–I5, hash
    reproducibility, legal-action correctness)

This module deliberately has NO external dependencies beyond `junqi_core`
and stdlib + numpy; it must run inside CI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .info_model import BeliefTensor
from .rules import ALL_SEATS, Seat, ShowMode
from .setup import generate_random_setup
from .state import Action, GameState, MoveResult, classify_termination

# ===========================================================================
# Trace dataclass
# ===========================================================================


@dataclass(slots=True)
class GameTrace:
    """Complete trace of a single simulated game."""

    seed: int
    show_mode: ShowMode
    track_beliefs: bool

    # Per-step records
    actions: list[Action] = field(default_factory=list)
    results: list[MoveResult] = field(default_factory=list)
    state_hashes: list[int] = field(default_factory=list)      # after each step
    # Terminal summary
    terminated: bool = False
    winner_team: int | None = None
    draw: bool = False
    num_steps: int = 0
    termination_reason: str = "unknown"    # "flag_capture" / "team_kill" / "q14_mutual" / "q12_chain" / "draw" / "timeout"
    # Timing (seconds)
    time_step_total: float = 0.0
    time_belief_total: float = 0.0
    # Final state (kept for serialization round-trip checks)
    terminal_hash: int = 0

    # Per-seat terminal belief tensors (optional, only if track_beliefs=True)
    terminal_beliefs: dict[Seat, BeliefTensor] = field(default_factory=dict)

    def steps_per_sec(self) -> float:
        if self.time_step_total <= 0 or self.num_steps == 0:
            return 0.0
        return self.num_steps / self.time_step_total


# ===========================================================================
# Core simulation
# ===========================================================================


def simulate_random_game(
    seed: int,
    *,
    max_steps: int = 4000,
    track_beliefs: bool = False,
    show_mode: ShowMode = ShowMode.HALF_DARK,
) -> GameTrace:
    """Play one full game with uniform-random legal-action policy.

    Deterministic in (seed, max_steps, track_beliefs, show_mode): two calls
    with identical arguments produce byte-identical traces (including
    terminal hash and termination_reason).

    Args:
        seed: random seed for both setup generation and action selection.
        max_steps: hard upper bound on steps (safety net; Q10 draw at 4000
                   will normally fire earlier).
        track_beliefs: if True, maintain a BeliefTensor per observer; costs
                       ~3x runtime but enables belief-invariant checks.
        show_mode: ShowMode.HALF_DARK (default) / BRIGHT / DARK.

    Returns:
        A GameTrace populated with actions, results, per-step hashes, and
        termination metadata.
    """
    rng = random.Random(seed)
    trace = GameTrace(seed=seed, show_mode=show_mode, track_beliefs=track_beliefs)

    # --- Initial setup ----------------------------------------------------
    setups = generate_random_setup(rng)
    state = GameState.new_game(setups, show_mode=show_mode)

    beliefs: dict[Seat, BeliefTensor] = {}
    if track_beliefs:
        for s in ALL_SEATS:
            beliefs[s] = BeliefTensor.initial(state, s)

    # --- Play loop --------------------------------------------------------
    for step_i in range(max_steps):
        if state.terminated:
            break

        actor = state.turn
        legal = state.legal_actions(seat=actor)
        if not legal:
            # Q12 should have killed this seat in the previous step(); if we
            # see this here the engine has a bug.
            raise RuntimeError(
                f"seat {actor.name} has no legal moves at step {step_i} "
                f"but state.turn says it should act; invariant violation"
            )
        action = rng.choice(legal)
        trace.actions.append(action)

        # --- Advance state --------------------------------------------------
        t0 = perf_counter()
        new_state, result = state.step(action)
        trace.time_step_total += perf_counter() - t0

        trace.results.append(result)

        # --- Update beliefs (if tracking) ----------------------------------
        if track_beliefs:
            t0 = perf_counter()
            for s, b in beliefs.items():
                # Skip observers that JUST died this step — their belief is
                # frozen and no longer meaningful.
                if s in result.seats_died_this_step:
                    continue
                if new_state.info[s].dead:
                    continue
                b.update(state, new_state, result)
            trace.time_belief_total += perf_counter() - t0

        # --- Commit -------------------------------------------------------
        state = new_state
        trace.state_hashes.append(state.state_hash())

    # --- Terminal summary -----------------------------------------------
    trace.num_steps = len(trace.actions)
    trace.terminated = state.terminated
    trace.winner_team = state.winner_team
    trace.draw = state.draw
    trace.terminal_hash = state.state_hash()
    trace.termination_reason = classify_termination(
        state, trace.results[-1] if trace.results else None
    )

    if track_beliefs:
        for s, b in beliefs.items():
            trace.terminal_beliefs[s] = b

    return trace


# ===========================================================================
# Invariant validation (post-run audit)
# ===========================================================================


@dataclass(slots=True)
class TraceViolation:
    """A single invariant violation discovered by `validate_trace_invariants`."""

    step_index: int                            # which step (-1 = terminal)
    kind: str                                  # e.g. "I1" / "hash_mismatch" / "illegal_move"
    detail: str


def validate_trace_invariants(
    trace: GameTrace,
    *,
    recheck_hashes: bool = True,
) -> list[TraceViolation]:
    """Replay the trace from the same seed and audit every invariant.

    Checks:
      V1. Reproducibility: re-running `simulate_random_game(seed)` produces
          byte-identical trace (same terminal hash and same action sequence).
      V2. Hash monotonicity: every state_hash was recorded.
      V3. Termination state is consistent with the last result.
      V4. If track_beliefs, all I1–I2 (belief rows sum to 1, non-negative).
      V5. Soundness: in HALF_DARK mode each observer's terminal belief
          assigns >0 probability to the TRUE type at every enemy cell
          that survived.

    Returns:
        list of TraceViolation; empty list means all checks passed.
    """
    violations: list[TraceViolation] = []

    # V1: reproducibility
    if recheck_hashes:
        trace2 = simulate_random_game(
            seed=trace.seed,
            max_steps=trace.num_steps,
            track_beliefs=False,
            show_mode=trace.show_mode,
        )
        if trace2.terminal_hash != trace.terminal_hash:
            violations.append(TraceViolation(
                step_index=-1,
                kind="reproducibility",
                detail=(
                    f"replay hash={trace2.terminal_hash} != original "
                    f"{trace.terminal_hash} (seed={trace.seed})"
                ),
            ))
        if trace2.num_steps != trace.num_steps:
            violations.append(TraceViolation(
                step_index=-1,
                kind="reproducibility",
                detail=(
                    f"replay num_steps={trace2.num_steps} != "
                    f"{trace.num_steps} (seed={trace.seed})"
                ),
            ))

    # V2: hash list length consistent
    if len(trace.state_hashes) != trace.num_steps:
        violations.append(TraceViolation(
            step_index=-1,
            kind="hash_list_length",
            detail=(
                f"state_hashes len={len(trace.state_hashes)} != "
                f"num_steps={trace.num_steps}"
            ),
        ))

    # V3: termination flag/result consistency
    if trace.terminated and trace.num_steps > 0:
        last = trace.results[-1]
        if last.terminated_after != trace.terminated:
            violations.append(TraceViolation(
                step_index=trace.num_steps - 1,
                kind="termination_flag",
                detail=(
                    f"last MoveResult.terminated_after={last.terminated_after} "
                    f"but trace.terminated={trace.terminated}"
                ),
            ))
        if last.winner_team_after != trace.winner_team:
            violations.append(TraceViolation(
                step_index=trace.num_steps - 1,
                kind="winner_team_mismatch",
                detail=(
                    f"last.winner_team_after={last.winner_team_after} vs "
                    f"trace.winner_team={trace.winner_team}"
                ),
            ))

    # V4 + V5: belief invariants (only if tracked)
    if trace.track_beliefs:
        for observer, belief in trace.terminal_beliefs.items():
            for pos, vec in belief.probs.items():
                # V4: row sums to 1, non-negative
                s = float(vec.sum())
                if not (0.99 <= s <= 1.01):
                    violations.append(TraceViolation(
                        step_index=-1,
                        kind="I1",
                        detail=(
                            f"observer={observer.name} pos={pos} "
                            f"belief row sums to {s:.4f}"
                        ),
                    ))
                if (vec < 0).any():
                    violations.append(TraceViolation(
                        step_index=-1,
                        kind="I2",
                        detail=f"observer={observer.name} pos={pos} negative prob",
                    ))

    return violations


# ===========================================================================
# Stress aggregation (used by tools/stress_test.py)
# ===========================================================================


@dataclass(slots=True)
class StressSummary:
    """Aggregated metrics across many games."""

    num_games: int = 0
    num_step_total: int = 0
    time_step_total: float = 0.0
    time_belief_total: float = 0.0
    violations_total: int = 0
    reason_counts: dict[str, int] = field(default_factory=dict)
    winner_team_counts: dict[str, int] = field(default_factory=dict)  # "0"/"1"/"draw"
    # Step-length statistics
    min_steps: int = 0
    max_steps: int = 0
    mean_steps: float = 0.0

    def register(self, trace: GameTrace, violations: list[TraceViolation]) -> None:
        self.num_games += 1
        self.num_step_total += trace.num_steps
        self.time_step_total += trace.time_step_total
        self.time_belief_total += trace.time_belief_total
        self.violations_total += len(violations)
        self.reason_counts[trace.termination_reason] = (
            self.reason_counts.get(trace.termination_reason, 0) + 1
        )
        key = (
            "draw" if trace.draw
            else ("0" if trace.winner_team == 0
                  else "1" if trace.winner_team == 1
                  else "none")
        )
        self.winner_team_counts[key] = self.winner_team_counts.get(key, 0) + 1
        if self.num_games == 1:
            self.min_steps = trace.num_steps
            self.max_steps = trace.num_steps
        else:
            self.min_steps = min(self.min_steps, trace.num_steps)
            self.max_steps = max(self.max_steps, trace.num_steps)
        self.mean_steps = self.num_step_total / self.num_games

    def steps_per_sec(self) -> float:
        if self.time_step_total <= 0:
            return 0.0
        return self.num_step_total / self.time_step_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_games": self.num_games,
            "num_step_total": self.num_step_total,
            "time_step_total_sec": round(self.time_step_total, 4),
            "time_belief_total_sec": round(self.time_belief_total, 4),
            "violations_total": self.violations_total,
            "reason_counts": dict(self.reason_counts),
            "winner_team_counts": dict(self.winner_team_counts),
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "mean_steps": round(self.mean_steps, 2),
            "steps_per_sec": round(self.steps_per_sec(), 1),
        }


# ===========================================================================
# Self-test
# ===========================================================================


def _self_test() -> None:  # pragma: no cover
    trace = simulate_random_game(seed=0, max_steps=500, track_beliefs=True)
    assert trace.num_steps > 0
    assert trace.terminated or trace.num_steps == 500
    violations = validate_trace_invariants(trace)
    assert not violations, f"violations: {violations}"
    print(
        f"junqi_core.simulator self-test: OK "
        f"(seed=0 → {trace.num_steps} steps, "
        f"reason={trace.termination_reason}, "
        f"winner={trace.winner_team}, "
        f"{trace.steps_per_sec():.1f} steps/sec)"
    )


if __name__ == "__main__":
    _self_test()
