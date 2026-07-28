"""junqi_core.replay_with_policy — Replay trajectories augmented with
per-step policy logits + value estimates for analysis / visualisation.

This module sits on top of :class:`junqi_core.replay.Trajectory`.  Where
a bare Trajectory records only the action sequence (enough to reproduce
game state), :class:`TrajectoryWithPolicy` *additionally* stores, per step:

  * the full policy probability distribution over legal actions
    (top-K for compactness; default K=16)
  * the value estimate V(s) (scalar or categorical)
  * the acting seat's observation snapshot (optional — off by default
    because it's 116 KB per step)
  * the per-seat belief distribution (optional — 12 x 289 per seat,
    4 x 12 x 289 x 4 bytes = 55 KB per step if stored)

This is the data object consumed by the replay viewer (CLI / notebook)
to surface "why did the agent play this move?" and "what does the agent
think the opponent's flag is?".

File format: `.npz` with a header field ``kind = "with_policy_v2"``.
Version 2 adds an ``action_sources`` column while remaining able to load
version-1 recordings.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from junqi_core.replay import Trajectory, _row_to_action
from junqi_core.rules import RULES_VERSION, Seat, ShowMode
from junqi_core.setup import setup_from_names, setup_to_names
from junqi_core.state import GameState, MoveResult

# 12 tracked piece types used by the belief representation (see
# _movegen_tables.NUM_TRACKED_TYPES).  Explicit here to avoid a dependency
# loop.
NUM_TRACKED_TYPES = 12
NUM_CELLS = 289
NUM_SEATS = 4

ACTION_SOURCE_POLICY_SAMPLE = 0
ACTION_SOURCE_POLICY_GREEDY = 1
ACTION_SOURCE_RANDOM_OPPONENT = 2
ACTION_SOURCE_NAMES = (
    "policy_sample",
    "policy_greedy",
    "random_opponent",
)


@dataclass
class StepPolicyRecord:
    """Policy artefacts for a single step.  Indexed into the parent
    :class:`TrajectoryWithPolicy` arrays."""

    step: int                 # 0-based step index
    acting_seat: int          # 0..3
    top_action_ids: np.ndarray  # (K,) int32  world-frame flat action ids
    top_probs: np.ndarray       # (K,) float32
    value: float                # V(s) scalar estimate
    chosen_action_id: int       # world-frame; == top_action_ids[0] if greedy
    action_source: int = ACTION_SOURCE_POLICY_SAMPLE

    @property
    def action_source_name(self) -> str:
        if 0 <= self.action_source < len(ACTION_SOURCE_NAMES):
            return ACTION_SOURCE_NAMES[self.action_source]
        return "unknown"


@dataclass
class TrajectoryWithPolicy:
    """Trajectory + per-step policy data.

    Mirrors :class:`Trajectory` for reproducibility, plus three dense
    per-step arrays:

      * ``top_action_ids`` : (T, K) int32   — world-frame flat ids
      * ``top_probs``      : (T, K) float32 — sorted descending
      * ``values``         : (T,)   float32 — scalar value estimates

    Optional per-step tensors (may be all-zero / empty if not recorded):

      * ``beliefs``        : (T, NUM_SEATS, NUM_TRACKED_TYPES, NUM_CELLS)
                             float16 — belief-net output per seat,
                             sparse-ish (most cells empty), 55 KB/step.

    Use :meth:`save` / :meth:`load` for on-disk persistence.
    """

    setups: Any             # SetupArray
    actions: np.ndarray     # (T, 5) int16
    rng_seed: int
    final_state_hash: int
    first_seat: Seat = Seat.SOUTH
    show_mode: ShowMode = ShowMode.HALF_DARK

    # Policy data
    top_action_ids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=np.int32))
    top_probs: np.ndarray      = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float32))
    values: np.ndarray         = field(default_factory=lambda: np.zeros((0,),   dtype=np.float32))
    acting_seats: np.ndarray   = field(default_factory=lambda: np.zeros((0,),   dtype=np.int8))
    action_sources: np.ndarray | None = None

    # Optional belief snapshot per step (one entry per acting step).
    beliefs: np.ndarray | None = None   # (T, 4, 12, 289) float16, optional

    # Version tags
    rules_version: str = RULES_VERSION
    state_version: str = "2.0"
    policy_version: str = "with_policy_v2"

    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.int16)
        if self.actions.ndim != 2 or self.actions.shape[1] != 5:
            raise ValueError(
                f"actions must have shape (T, 5), got {self.actions.shape}"
            )
        t = int(self.actions.shape[0])
        self.top_action_ids = np.asarray(self.top_action_ids, dtype=np.int32)
        self.top_probs = np.asarray(self.top_probs, dtype=np.float32)
        self.values = np.asarray(self.values, dtype=np.float32)
        self.acting_seats = np.asarray(self.acting_seats, dtype=np.int8)
        if self.action_sources is None:
            self.action_sources = np.full(
                (t,), ACTION_SOURCE_POLICY_SAMPLE, dtype=np.int8
            )
        else:
            self.action_sources = np.asarray(self.action_sources, dtype=np.int8)
        if self.top_action_ids.ndim != 2 or self.top_action_ids.shape[0] != t:
            raise ValueError("top_action_ids must have shape (T, K)")
        if self.top_probs.shape != self.top_action_ids.shape:
            raise ValueError("top_probs must have the same shape as top_action_ids")
        if (
            self.values.shape != (t,)
            or self.acting_seats.shape != (t,)
            or self.action_sources.shape != (t,)
        ):
            raise ValueError(
                "values, acting_seats and action_sources must have shape (T,)"
            )
        if np.any((self.acting_seats < 0) | (self.acting_seats > 3)):
            raise ValueError("acting_seats must be in [0, 3]")
        if np.any(
            (self.action_sources < 0)
            | (self.action_sources >= len(ACTION_SOURCE_NAMES))
        ):
            raise ValueError("action_sources contains an unsupported value")
        if self.beliefs is not None:
            self.beliefs = np.asarray(self.beliefs, dtype=np.float16)
            expected = (t, NUM_SEATS, NUM_TRACKED_TYPES, NUM_CELLS)
            if self.beliefs.shape != expected:
                raise ValueError(f"beliefs must have shape {expected}, got {self.beliefs.shape}")

    # ------------------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def top_k(self) -> int:
        return int(self.top_action_ids.shape[1]) if self.top_action_ids.ndim == 2 else 0

    # ------------------------------------------------------------------
    # View as bare Trajectory (to reuse the deterministic replay engine)
    # ------------------------------------------------------------------
    def as_trajectory(self) -> Trajectory:
        return Trajectory(
            setups=self.setups,
            actions=self.actions,
            rng_seed=self.rng_seed,
            final_state_hash=self.final_state_hash,
            first_seat=self.first_seat,
            show_mode=self.show_mode,
            rules_version=self.rules_version,
            state_version=self.state_version,
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save(self, path: str | os.PathLike[str]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        assert self.action_sources is not None
        setup_names = np.asarray(setup_to_names(self.setups), dtype=object)
        extras: dict[str, Any] = {}
        if self.beliefs is not None:
            extras["beliefs"] = self.beliefs.astype(np.float16, copy=False)
        np.savez(
            str(p),
            kind=np.array(self.policy_version),
            setups=setup_names,
            action_log=self.actions.astype(np.int16, copy=False),
            rng_seed=np.int64(self.rng_seed),
            rules_version=np.array(self.rules_version),
            state_version=np.array(self.state_version),
            final_state_hash=np.int64(self.final_state_hash),
            first_seat=np.int8(int(self.first_seat.value)),
            show_mode=np.int8(int(self.show_mode.value)),
            top_action_ids=self.top_action_ids.astype(np.int32, copy=False),
            top_probs=self.top_probs.astype(np.float32, copy=False),
            values=self.values.astype(np.float32, copy=False),
            acting_seats=self.acting_seats.astype(np.int8, copy=False),
            action_sources=self.action_sources.astype(np.int8, copy=False),
            meta_json=np.array(json.dumps(self.meta, ensure_ascii=False, default=str)),
            **extras,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> TrajectoryWithPolicy:
        with np.load(str(path), allow_pickle=True) as data:
            kind = str(data.get("kind", "")) if "kind" in data.files else ""
            if kind and not kind.startswith("with_policy_v"):
                raise ValueError(
                    f"Not a TrajectoryWithPolicy file (kind={kind!r})"
                )
            setup_names = data["setups"].tolist()
            beliefs = data["beliefs"] if "beliefs" in data.files else None
            action_sources = (
                np.asarray(data["action_sources"], dtype=np.int8)
                if "action_sources" in data.files
                else None
            )
            meta: dict[str, Any] = {}
            if "meta_json" in data.files:
                try:
                    loaded_meta = json.loads(str(data["meta_json"]))
                    if isinstance(loaded_meta, dict):
                        meta = loaded_meta
                except (TypeError, ValueError, json.JSONDecodeError):
                    raise ValueError("invalid replay meta_json") from None
            return cls(
                setups=setup_from_names(setup_names),
                actions=np.asarray(data["action_log"], dtype=np.int16),
                rng_seed=int(data["rng_seed"]),
                final_state_hash=int(data["final_state_hash"]),
                first_seat=Seat(int(data["first_seat"])),
                show_mode=ShowMode(int(data["show_mode"])),
                top_action_ids=np.asarray(data["top_action_ids"], dtype=np.int32),
                top_probs=np.asarray(data["top_probs"], dtype=np.float32),
                values=np.asarray(data["values"], dtype=np.float32),
                acting_seats=np.asarray(data["acting_seats"], dtype=np.int8),
                action_sources=action_sources,
                rules_version=str(data["rules_version"]),
                state_version=str(data["state_version"]),
                beliefs=beliefs,
                meta=meta,
            )

    # ------------------------------------------------------------------
    # Frame-level access
    # ------------------------------------------------------------------
    def step_record(self, t: int) -> StepPolicyRecord:
        """Return the :class:`StepPolicyRecord` for the ``t``-th step."""
        assert self.action_sources is not None
        return StepPolicyRecord(
            step=t,
            acting_seat=int(self.acting_seats[t]),
            top_action_ids=self.top_action_ids[t],
            top_probs=self.top_probs[t],
            value=float(self.values[t]),
            chosen_action_id=_chosen_world_action_id(self.actions[t]),
            action_source=int(self.action_sources[t]),
        )

    def replay_with_records(
        self,
    ) -> Iterator[tuple[GameState, MoveResult, StepPolicyRecord]]:
        """Yield ``(post_state, move_result, step_record)`` per step.

        ``post_state`` is the mutated state after applying the action.
        The :class:`StepPolicyRecord` is drawn from the stored policy
        arrays — so users can visualise, for each frame, what the agent
        thought was likely and how it valued the position.
        """
        state = GameState.new_game(
            self.setups, show_mode=self.show_mode, first_seat=self.first_seat,
        )
        T = self.num_steps
        for t in range(T):
            action = _row_to_action(self.actions[t])
            result = state.step_inplace(action)
            yield state, result, self.step_record(t)

    def validate(self) -> None:
        """Validate both deterministic game state and policy-array lengths."""
        self.as_trajectory().validate()

    def viewer(self) -> Any:
        """Return the policy-aware headless viewer adapter."""
        from junqi_core.replay_viewer import ReplayViewer

        return ReplayViewer(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chosen_world_action_id(action_row: np.ndarray) -> int:
    """Convert a (5,) row [seat, sx, sy, dx, dy] back to a world flat id.

    We rebuild the flat id on the fly so the replay file doesn't need a
    redundant action-id column.
    """
    _, sx, sy, dx, dy = (int(v) for v in action_row)
    src_flat = sy * 17 + sx
    dst_flat = dy * 17 + dx
    return int(src_flat * 289 + dst_flat)


def probs_to_top_k(
    probs: np.ndarray,  # (FLAT_ACTION_DIM,) float32
    legal_mask: np.ndarray,  # (FLAT_ACTION_DIM,) bool
    k: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Utility: pick top-K (id, prob) pairs from a masked policy.

    Illegal actions get prob = -inf (never selected).  Returns two arrays
    of length ``k`` (padded with zeros when fewer legal actions exist).
    """
    probs = np.asarray(probs, dtype=np.float32)
    legal_mask = np.asarray(legal_mask, dtype=bool)
    if probs.ndim != 1 or legal_mask.shape != probs.shape:
        raise ValueError("probs and legal_mask must be one-dimensional and equal-sized")
    if k < 0:
        raise ValueError("k must be non-negative")
    # NaNs can appear briefly during a failed mixed-precision forward.  They
    # must never become the top-ranked explanation in a replay viewer.
    finite_probs = np.nan_to_num(probs, nan=-np.inf, posinf=1.0, neginf=-np.inf)
    masked = np.where(legal_mask, finite_probs, -np.inf)
    n_legal = int(legal_mask.sum())
    if n_legal == 0 or k == 0:
        return (
            np.zeros((k,), dtype=np.int32),
            np.zeros((k,), dtype=np.float32),
        )
    take = min(k, n_legal)
    candidates = np.flatnonzero(legal_mask)
    # Lexicographic ordering makes equal-probability top-K rows stable across
    # NumPy versions and therefore diff-friendly in RL progress reports.
    top_sorted = candidates[np.lexsort((candidates, -masked[candidates]))][:take]
    out_ids = np.zeros((k,), dtype=np.int32)
    out_probs = np.zeros((k,), dtype=np.float32)
    out_ids[:take] = top_sorted.astype(np.int32)
    out_probs[:take] = masked[top_sorted].astype(np.float32)
    return out_ids, out_probs


__all__ = [
    "ACTION_SOURCE_NAMES",
    "ACTION_SOURCE_POLICY_GREEDY",
    "ACTION_SOURCE_POLICY_SAMPLE",
    "ACTION_SOURCE_RANDOM_OPPONENT",
    "NUM_CELLS",
    "NUM_SEATS",
    "NUM_TRACKED_TYPES",
    "StepPolicyRecord",
    "TrajectoryWithPolicy",
    "probs_to_top_k",
]
