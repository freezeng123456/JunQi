"""junqi_core.replay — bit-exact game replay on top of SoA GameState.

Phase 0.4 M7 / ADR-122 follow-on. Implements the replay protocol of
``docs/RULES.md`` §9 on top of the M1 ``GameState`` (SoA + incremental
Zobrist).

File format (.npz) — keys:
    setups              : unicode ndarray, shape (4, 30), dtype=np.str_ of
                          piece-name strings (see setup.setup_to_names).
    action_log          : int16 ndarray of shape (T, 5) with columns
                          (seat, src_x, src_y, dst_x, dst_y).
    rng_seed            : int64 scalar; -1 if unseeded.
    rules_version       : unicode ndarray[0] (single string).
    state_version       : unicode ndarray[0] (single string = "2.0").
    final_state_hash    : int64 scalar (state_hash after last step; used
                          for replay-determinism self-check).
    first_seat          : int8 (0..3) — Seat.value of the first seat.
    show_mode           : int8 — ShowMode.value.

Public API:
    Trajectory(setups, actions, rng_seed, rules_version, state_version,
               final_state_hash, first_seat, show_mode)
        .save(path) -> None
        .load(path) -> Trajectory                       [classmethod]
        .replay(*, return_move_results=False) -> tuple  (final_state[, results])
        .replay_iter() -> Iterator[tuple[GameState, Action, MoveResult]]
        .validate() -> None — raises AssertionError on any determinism
                              mismatch (used by tests).

    record_trajectory(env, policy, *, max_steps=1000, seed=None) -> Trajectory
        Run ``env`` under ``policy(state) -> action_id`` until done or
        ``max_steps`` reached; return the resulting :class:`Trajectory`.

Design notes:

* The on-disk representation deliberately stores **actions only**, not
  the broadcast ``MoveResult`` objects.  The latter are reproducible
  from ``(setups, actions)`` via the game engine; storing them would
  bloat .npz files (each has ~15 fields) and introduce a second source
  of truth.  The `replay_iter` method emits live ``MoveResult`` objects
  from the engine so downstream code (e.g., T8 experience rebuild) can
  consume them without touching disk twice.

* ``final_state_hash`` is the ADR-117 incremental Zobrist.  After load,
  the replay walks the ``action_log`` and asserts
  ``final.state_hash() == final_state_hash``.  Any engine regression
  that breaks determinism (reseeded RNG, rule drift, etc.) surfaces as
  a hash mismatch here.

* ``setups`` round-trips via :func:`setup.setup_to_names` /
  :func:`setup.setup_from_names` — those are the stable string form
  used by golden JSONs, so a replay created today remains loadable
  even if PieceType enum slots are renumbered.

This module is new; no legacy callers to preserve.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from junqi_core.board import is_on_board
from junqi_core.rules import RULES_VERSION, Seat, ShowMode
from junqi_core.setup import setup_from_names, setup_to_names
from junqi_core.state import Action, GameState, MoveResult

if TYPE_CHECKING:  # pragma: no cover
    from junqi_core.setup import SetupArray


# ---------------------------------------------------------------------------
# Trajectory dataclass
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """A recorded game: setups + action sequence + determinism checksum.

    Fields
    ------
    setups
        The :class:`SetupArray` (4 x 30 PieceType) used to seed the
        game.  Stored as piece-name strings on disk via
        :func:`setup_to_names`.
    actions
        ``np.ndarray[int16]`` of shape ``(T, 5)``; each row is
        ``(seat, src_x, src_y, dst_x, dst_y)``.  Seat values match
        :class:`Seat` integer codes.
    rng_seed
        The RNG seed used to generate ``setups`` if any; ``-1`` means
        "setups were produced out-of-band" (manual / golden).
    rules_version, state_version
        Pinned at creation time.
    final_state_hash
        ADR-117 Zobrist hash of the final :class:`GameState`.  Used by
        :meth:`validate` to self-check determinism on load.
    first_seat, show_mode
        Construction arguments for :meth:`GameState.new_game` — stored
        so a replay can reproduce the exact state trajectory without
        additional metadata.
    """

    setups: SetupArray
    actions: np.ndarray                # (T, 5) int16
    rng_seed: int
    final_state_hash: int
    first_seat: Seat = Seat.SOUTH
    show_mode: ShowMode = ShowMode.HALF_DARK
    rules_version: str = RULES_VERSION
    state_version: str = "2.0"
    # Optional column: total steps (derived from actions.shape[0]).
    # Exposed for ergonomic access so callers don't re-shape actions.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize and validate the action log at the file boundary.

        Previously malformed rows were accepted until a viewer attempted to
        replay them, which made corrupt/truncated files look like engine bugs.
        Failing early also keeps the C/NPZ replay formats consistent.
        """
        self.actions = np.asarray(self.actions, dtype=np.int16)
        if self.actions.ndim != 2 or self.actions.shape[1] != 5:
            raise ValueError(
                f"actions must have shape (T, 5), got {self.actions.shape}"
            )
        if self.actions.size:
            rows = self.actions.astype(np.int64, copy=False)
            if np.any((rows[:, 0] < 0) | (rows[:, 0] > 3)):
                raise ValueError("action seat must be in [0, 3]")
            if np.any((rows[:, 1:] < 0) | (rows[:, 1:] >= 17)):
                raise ValueError("action coordinates must be in [0, 16]")
            if np.any((rows[:, 1] == rows[:, 3]) & (rows[:, 2] == rows[:, 4])):
                raise ValueError("action src and dst must differ")
            # 17x17 includes non-playable geometry cells; reject those here
            # so a saved replay can never fail halfway through a viewer.
            for row in rows:
                if not is_on_board(int(row[1]), int(row[2])) or not is_on_board(
                    int(row[3]), int(row[4])
                ):
                    raise ValueError(f"action points off playable board: {row.tolist()}")
        if self.first_seat not in (Seat.SOUTH, Seat.WEST, Seat.NORTH, Seat.EAST):
            raise ValueError(f"invalid first_seat: {self.first_seat!r}")
        if self.show_mode not in (ShowMode.BRIGHT, ShowMode.DARK, ShowMode.HALF_DARK):
            raise ValueError(f"invalid show_mode: {self.show_mode!r}")

    # ------------------------------------------------------------------
    # Derived
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        return int(self.actions.shape[0])

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write this trajectory to ``path`` as an uncompressed .npz."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        setup_names = np.asarray(
            setup_to_names(self.setups), dtype=np.str_,
        )
        np.savez(
            str(p),
            setups=setup_names,
            action_log=self.actions.astype(np.int16, copy=False),
            rng_seed=np.int64(self.rng_seed),
            rules_version=np.array(self.rules_version),
            state_version=np.array(self.state_version),
            final_state_hash=np.int64(self.final_state_hash),
            first_seat=np.int8(int(self.first_seat.value)),
            show_mode=np.int8(int(self.show_mode.value)),
            meta_json=np.array(json.dumps(self.meta, ensure_ascii=False, default=str)),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Trajectory:
        """Load a trajectory previously saved with :meth:`save`."""
        with np.load(str(path), allow_pickle=False) as data:
            setup_names = data["setups"].tolist()
            setups = setup_from_names(setup_names)
            actions = np.asarray(data["action_log"], dtype=np.int16)
            rv = str(data["rules_version"])
            if not rv.startswith("1."):
                raise ValueError(
                    f"incompatible replay rules_version {rv!r} (expect 1.x)"
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
                setups=setups,
                actions=actions,
                rng_seed=int(data["rng_seed"]),
                rules_version=rv,
                state_version=str(data["state_version"]),
                final_state_hash=int(data["final_state_hash"]),
                first_seat=Seat(int(data["first_seat"])),
                show_mode=ShowMode(int(data["show_mode"])),
                meta=meta,
            )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self, *, return_move_results: bool = False,
    ) -> (
        GameState
        | tuple[GameState, list[MoveResult]]
    ):
        """Replay the trajectory on a fresh engine and return the final state.

        If ``return_move_results`` is True, also returns the full list of
        :class:`MoveResult` produced step by step.
        """
        state = GameState.new_game(
            self.setups, show_mode=self.show_mode, first_seat=self.first_seat,
        )
        results: list[MoveResult] = []
        for row in self.actions:
            action = _row_to_action(row)
            result = state.step_inplace(action)
            if return_move_results:
                results.append(result)
        if return_move_results:
            return state, results
        return state

    def replay_iter(self) -> Iterator[tuple[GameState, Action, MoveResult]]:
        """Yield ``(post_state, action, move_result)`` per step.

        ``post_state`` is the mutated :class:`GameState` right after the
        action is applied; callers must not keep references across
        iterations without calling ``state.clone()``.
        """
        state = GameState.new_game(
            self.setups, show_mode=self.show_mode, first_seat=self.first_seat,
        )
        for row in self.actions:
            action = _row_to_action(row)
            result = state.step_inplace(action)
            yield state, action, result

    # ------------------------------------------------------------------
    # Self-check
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Replay from scratch and assert determinism invariants.

        Raises ``AssertionError`` if the final Zobrist hash or step
        count does not match what was recorded.
        """
        state = self.replay()
        assert isinstance(state, GameState)
        got = state.state_hash()
        if got != self.final_state_hash:
            raise AssertionError(
                f"replay determinism broken: expected final state_hash "
                f"{self.final_state_hash}, got {got}"
            )

    def cursor(self) -> ReplayCursor:
        """Return a seekable cursor for manual/GUI replay.

        The cursor keeps cloned state checkpoints, so stepping backward or
        scrubbing a slider never mutates the trajectory or shares a mutable
        ``GameState`` with the caller.
        """
        return ReplayCursor(self)


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


def record_trajectory(
    *,
    setups: SetupArray | None = None,
    rng_seed: int | None = None,
    policy: Callable[[GameState], int] | None = None,
    max_steps: int = 1000,
    first_seat: Seat = Seat.SOUTH,
    show_mode: ShowMode = ShowMode.HALF_DARK,
) -> Trajectory:
    """Run a single game under ``policy`` and capture it as a Trajectory.

    Parameters
    ----------
    setups
        Pre-generated :class:`SetupArray`; if ``None`` a random one is
        drawn via ``generate_random_setup(random.Random(rng_seed))`` and
        ``rng_seed`` is recorded on the trajectory.
    rng_seed
        Seed for random-setup generation AND default uniform-random
        policy when ``policy is None``.  ``None`` means
        "non-deterministic" (system entropy) and is recorded as ``-1``.
    policy
        ``policy(state) -> world_frame_flat_action_id``.  If ``None``,
        a uniform-random legal action is sampled each step.
    max_steps
        Hard cap on the number of recorded actions.  The trajectory
        terminates earlier if the game ends.
    first_seat, show_mode
        Forwarded to :meth:`GameState.new_game`.
    """
    import random as _random

    from junqi_core.setup import generate_random_setup

    rng = _random.Random(rng_seed) if rng_seed is not None else _random.Random()
    recorded_seed = -1 if rng_seed is None else int(rng_seed)

    if setups is None:
        setups = generate_random_setup(rng)

    state = GameState.new_game(
        setups, show_mode=show_mode, first_seat=first_seat,
    )

    def _default_policy(s: GameState) -> int:
        ids = s.legal_action_ids()
        if ids.size == 0:
            return -1
        return int(ids[rng.randrange(int(ids.size))])

    pol = policy if policy is not None else _default_policy

    rows: list[tuple[int, int, int, int, int]] = []
    for _ in range(max_steps):
        if state.terminated:
            break
        aid = pol(state)
        if aid < 0:
            break
        src_flat = aid // 289
        dst_flat = aid %  289
        src = (src_flat %  17, src_flat // 17)
        dst = (dst_flat %  17, dst_flat // 17)
        action = Action(seat=state.turn, src=src, dst=dst)
        rows.append((int(state.turn.value), src[0], src[1], dst[0], dst[1]))
        state.step_inplace(action)

    actions = (
        np.asarray(rows, dtype=np.int16).reshape(-1, 5)
        if rows else np.zeros((0, 5), dtype=np.int16)
    )
    return Trajectory(
        setups=setups,
        actions=actions,
        rng_seed=recorded_seed,
        final_state_hash=int(state.state_hash()),
        first_seat=first_seat,
        show_mode=show_mode,
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _row_to_action(row: np.ndarray) -> Action:
    seat = Seat(int(row[0]))
    return Action(
        seat=seat,
        src=(int(row[1]), int(row[2])),
        dst=(int(row[3]), int(row[4])),
    )


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """A stable state snapshot returned by :class:`ReplayCursor`.

    ``step`` is the number of actions already applied.  Step zero is the
    initial position and therefore has no action/result.
    """

    step: int
    state: GameState
    action: Action | None = None
    result: MoveResult | None = None


class ReplayCursor:
    """Seekable, deterministic cursor over a :class:`Trajectory`.

    ``seek(n)`` is intentionally simple and robust: it reuses cached cloned
    snapshots and only applies the delta from the nearest cached position.
    This is fast enough for human replay and avoids the old GTK slider's
    global static ``preStep`` state leaking between files.
    """

    def __init__(self, trajectory: Trajectory, *, checkpoint_interval: int = 32) -> None:
        self.trajectory = trajectory
        self.checkpoint_interval = max(1, int(checkpoint_interval))
        initial = GameState.new_game(
            trajectory.setups,
            show_mode=trajectory.show_mode,
            first_seat=trajectory.first_seat,
        )
        # A 1,000-step RL game does not need 1,000 full GameState copies.
        # Keep sparse checkpoints and the small MoveResult list instead.
        self._checkpoints: dict[int, GameState] = {0: initial.clone()}
        self._current_state = initial
        self._results: list[MoveResult | None] = [None] * (trajectory.num_steps + 1)
        self._position = 0

    @property
    def position(self) -> int:
        return self._position

    @property
    def length(self) -> int:
        return self.trajectory.num_steps

    @property
    def frame(self) -> ReplayFrame:
        """Return an immutable wrapper containing a cloned current state."""
        return ReplayFrame(
            step=self._position,
            state=self._current_state.clone(),
            action=(
                None
                if self._position == 0
                else _row_to_action(self.trajectory.actions[self._position - 1])
            ),
            result=self._results[self._position],
        )

    def reset(self) -> ReplayFrame:
        self._current_state = self._checkpoints[0].clone()
        self._position = 0
        return self.frame

    def step_forward(self, count: int = 1) -> ReplayFrame:
        if count < 0:
            return self.step_backward(-count)
        target = min(self.length, self._position + int(count))
        return self.seek(target)

    def step_backward(self, count: int = 1) -> ReplayFrame:
        if count < 0:
            return self.step_forward(-count)
        self._position = max(0, self._position - int(count))
        return self.frame

    def seek(self, step: int) -> ReplayFrame:
        target = int(step)
        if not 0 <= target <= self.length:
            raise IndexError(f"replay step {target} outside [0, {self.length}]")
        if target < self._position:
            checkpoint = max(k for k in self._checkpoints if k <= target)
            self._current_state = self._checkpoints[checkpoint].clone()
            self._position = checkpoint
        while self._position < target:
            index = self._position
            state = self._current_state
            action = _row_to_action(self.trajectory.actions[index])
            result = state.step_inplace(action)
            self._position += 1
            self._current_state = state
            self._results[self._position] = result
            if self._position % self.checkpoint_interval == 0:
                self._checkpoints[self._position] = state.clone()
        if target == self.length:
            self._checkpoints[target] = self._current_state.clone()
        return self.frame


__all__ = ["ReplayCursor", "ReplayFrame", "Trajectory", "record_trajectory"]
