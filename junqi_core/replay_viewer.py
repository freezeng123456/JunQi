"""Small, dependency-free replay adapter for clients and RL dashboards.

The GTK client has a legacy slider, while RL recordings live in ``.npz``
files.  This module provides one stable, seekable API for both: a caller can
scrub to a step, render ``frame.state`` and inspect the exact public
``MoveResult`` without re-implementing replay logic in a UI.

It deliberately does not depend on GTK, FastAPI or Torch, so it is safe to
use in training workers and in headless CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from junqi_core.replay import ReplayCursor, ReplayFrame, Trajectory
from junqi_core.replay_with_policy import StepPolicyRecord, TrajectoryWithPolicy
from junqi_core.rules import ALL_SEATS
from junqi_core.state import Action, GameState, MoveResult


@dataclass(frozen=True, slots=True)
class ViewerFrame:
    """A core replay frame plus optional per-step policy artefacts."""

    core: ReplayFrame
    policy: StepPolicyRecord | None = None

    @property
    def step(self) -> int:
        return self.core.step

    @property
    def state(self) -> GameState:
        return self.core.state

    @property
    def action(self) -> Action | None:
        return self.core.action

    @property
    def result(self) -> MoveResult | None:
        return self.core.result


class ReplayViewer:
    """Seekable viewer for a bare or policy-augmented trajectory."""

    def __init__(self, source: Trajectory | TrajectoryWithPolicy) -> None:
        self.source = source
        if isinstance(source, TrajectoryWithPolicy):
            self._policy_source: TrajectoryWithPolicy | None = source
            base = source.as_trajectory()
        else:
            self._policy_source = None
            base = source
        self._cursor = ReplayCursor(base)

    @property
    def position(self) -> int:
        return self._cursor.position

    @property
    def length(self) -> int:
        return self._cursor.length

    def frame(self) -> ViewerFrame:
        core = self._cursor.frame
        policy = None
        if self._policy_source is not None and core.step > 0:
            policy = self._policy_source.step_record(core.step - 1)
        return ViewerFrame(core=core, policy=policy)

    def seek(self, step: int) -> ViewerFrame:
        self._cursor.seek(step)
        return self.frame()

    def next(self) -> ViewerFrame:
        self._cursor.step_forward()
        return self.frame()

    def previous(self) -> ViewerFrame:
        self._cursor.step_backward()
        return self.frame()

    def reset(self) -> ViewerFrame:
        self._cursor.reset()
        return self.frame()


def frame_summary(frame: ViewerFrame) -> dict[str, Any]:
    """Return JSON-friendly state/action/policy data for dashboards."""

    state = frame.state
    summary: dict[str, Any] = {
        "step": frame.step,
        "move_counter": int(state.move_counter),
        "turn": state.turn.name,
        "terminated": bool(state.terminated),
        "draw": bool(state.draw),
        "winner_team": state.winner_team,
        "alive_pieces": {
            seat.name: sum(1 for piece in state.pieces.values() if piece.seat is seat)
            for seat in ALL_SEATS
        },
    }
    if frame.action is not None:
        summary["action"] = {
            "seat": frame.action.seat.name,
            "src": list(frame.action.src),
            "dst": list(frame.action.dst),
        }
    if frame.result is not None:
        summary["result"] = {
            "event": frame.result.event.name,
            "flag_reveal_src": bool(frame.result.flag_reveal_src),
            "flag_reveal_dst": bool(frame.result.flag_reveal_dst),
            "flag_captured": bool(frame.result.flag_captured),
            "seats_died": [seat.name for seat in frame.result.seats_died_this_step],
            "terminated_after": bool(frame.result.terminated_after),
            "draw_after": bool(frame.result.draw_after),
        }
    if frame.policy is not None:
        summary["policy"] = {
            "acting_seat": ALL_SEATS[frame.policy.acting_seat].name,
            "action_source": frame.policy.action_source_name,
            "value": float(frame.policy.value),
            "chosen_action_id": int(frame.policy.chosen_action_id),
            "top_action_ids": frame.policy.top_action_ids.astype(np.int64).tolist(),
            "top_probs": frame.policy.top_probs.astype(np.float64).tolist(),
        }
    return summary


def load_replay(path: str) -> Trajectory | TrajectoryWithPolicy:
    """Load either supported ``.npz`` replay format by its header."""

    with np.load(path, allow_pickle=True) as data:
        kind = str(data["kind"]) if "kind" in data.files else ""
    if kind.startswith("with_policy_v"):
        return TrajectoryWithPolicy.load(path)
    return Trajectory.load(path)


__all__ = ["ReplayViewer", "ViewerFrame", "frame_summary", "load_replay"]
