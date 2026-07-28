"""junqi_rl.env — RL adapter: ``JunqiEnv`` / ``VectorJunqiEnv`` (ADR-122).

A single stable entry point for the RL training loop.  The env wraps the
full junqi-core stack (GameState, 4× BeliefTensor, ObservationBuilder)
and presents a 4-headed self-play interface:

    env = JunqiEnv()
    obs = env.reset(seed=42)              # obs: dict[Seat -> ObservationTensor]
    while not done:
        seat = env.current_seat()
        ids  = env.legal_action_ids(seat)
        aid  = int(policy(obs[seat], ids))  # world-frame flat id
        obs, reward, done, info = env.step(aid)

Design contract (ADR-122):

* ``action_id`` is **world-frame**: ``action_id = src_flat * 289 +
  dst_flat`` with ``src_flat = sy * 17 + sx``.  The policy network
  predicts actions in the *canonical* frame and the caller must
  un-rotate before calling ``step``.  Utility helpers
  :func:`action_id_to_src_dst` and :func:`src_dst_to_action_id` are
  provided; :func:`unrotate_action_id` converts canonical-frame flat
  ids to world frame in one call.
* Observation dict values are always **canonical-frame** (each
  ObservationTensor is anchored at the corresponding observer's seat).
* Reward is a per-seat 4-tuple sourced from
  :meth:`GameState.team_rewards`; it is ``(0, 0, 0, 0)`` on every
  non-terminal step.
* ``info`` carries the last :class:`MoveResult` plus terminal flags
  (``draw``, ``winner_team``, ``termination_reason``).
* The env is single-threaded and not re-entrant.  ``VectorJunqiEnv``
  stacks N of these with a shared :class:`ObservationBuilder` and a
  slab-allocated observation tensor.

See ``docs/DECISIONS.md`` ADR-122 for the full decision record.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from junqi_core.board import BOARD_SIZE, NUM_CELLS
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import (
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationBuilder,
    ObservationTensor,
)
from junqi_core.rotation import canonical_to_world, world_to_canonical
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState, MoveResult
from junqi_core.board import (
    NUM_ON_BOARD_CELLS,
    COMPACT_TO_FLAT,
    FLAT_TO_COMPACT,
)


# ---------------------------------------------------------------------------
# Flat action-id helpers (ADR-119 encoding: src_flat * 289 + dst_flat)
# ---------------------------------------------------------------------------


def action_id_to_src_dst(
    action_id: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Decode a world-frame flat action id to ``(src_xy, dst_xy)``."""
    if not 0 <= action_id < NUM_CELLS * NUM_CELLS:
        raise ValueError(
            f"action_id {action_id} outside [0, {NUM_CELLS * NUM_CELLS})"
        )
    src_flat = action_id // NUM_CELLS
    dst_flat = action_id %  NUM_CELLS
    return (
        (src_flat %  BOARD_SIZE, src_flat // BOARD_SIZE),
        (dst_flat %  BOARD_SIZE, dst_flat // BOARD_SIZE),
    )


def src_dst_to_action_id(
    src: tuple[int, int], dst: tuple[int, int],
) -> int:
    """Encode a world-frame ``(src, dst)`` pair to a flat action id."""
    sx, sy = src
    dx, dy = dst
    return (sy * BOARD_SIZE + sx) * NUM_CELLS + (dy * BOARD_SIZE + dx)


def unrotate_action_id(canonical_action_id: int, acting_seat: Seat) -> int:
    """Convert a canonical-frame flat action id to a world-frame one.

    Policy networks emit canonical-frame actions (the input observation
    is canonical).  This helper composes :func:`action_id_to_src_dst`,
    :func:`canonical_to_world`, and :func:`src_dst_to_action_id` so
    callers don't have to juggle coordinate-frame boilerplate.
    """
    src_c, dst_c = action_id_to_src_dst(canonical_action_id)
    src_w = canonical_to_world(*src_c, acting_seat)
    dst_w = canonical_to_world(*dst_c, acting_seat)
    return src_dst_to_action_id(src_w, dst_w)


def unrotate_compact_action_id(compact_action_id: int, acting_seat: Seat) -> int:
    """Convert a compact canonical-frame action id to a world-frame full action id.

    Compact action = compact_src * 129 + compact_dst, where compact indices
    refer to the 129 on-board cells only.

    Returns a world-frame full action id (src_flat_289 * 289 + dst_flat_289).
    """
    compact_src = compact_action_id // NUM_ON_BOARD_CELLS
    compact_dst = compact_action_id % NUM_ON_BOARD_CELLS
    # Map compact → canonical flat (17×17)
    src_can_flat = int(COMPACT_TO_FLAT[compact_src])
    dst_can_flat = int(COMPACT_TO_FLAT[compact_dst])
    # canonical → world
    src_c = (src_can_flat % BOARD_SIZE, src_can_flat // BOARD_SIZE)
    dst_c = (dst_can_flat % BOARD_SIZE, dst_can_flat // BOARD_SIZE)
    src_w = canonical_to_world(*src_c, acting_seat)
    dst_w = canonical_to_world(*dst_c, acting_seat)
    return src_dst_to_action_id(src_w, dst_w)


def rotate_to_compact_action_id(world_action_id: int, acting_seat: Seat) -> int:
    """Convert a world-frame full action id to compact canonical-frame.

    Inverse of :func:`unrotate_compact_action_id`.
    """
    src_w, dst_w = action_id_to_src_dst(world_action_id)
    src_c = world_to_canonical(*src_w, acting_seat)
    dst_c = world_to_canonical(*dst_w, acting_seat)
    src_flat = src_c[1] * BOARD_SIZE + src_c[0]
    dst_flat = dst_c[1] * BOARD_SIZE + dst_c[0]
    return int(FLAT_TO_COMPACT[src_flat]) * NUM_ON_BOARD_CELLS + int(FLAT_TO_COMPACT[dst_flat])


def rotate_action_id(world_action_id: int, acting_seat: Seat) -> int:
    """Inverse of :func:`unrotate_action_id`."""
    src_w, dst_w = action_id_to_src_dst(world_action_id)
    src_c = world_to_canonical(*src_w, acting_seat)
    dst_c = world_to_canonical(*dst_w, acting_seat)
    return src_dst_to_action_id(src_c, dst_c)


# ---------------------------------------------------------------------------
# Step-info record (value of the 4th ``info`` element)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JunqiStepInfo:
    """Per-step metadata returned by :meth:`JunqiEnv.step`.

    Fields
    ------
    acting_seat
        The seat that just moved (the env's current_seat *before* step).
    move_result
        The :class:`MoveResult` produced by ``state.step``.  Useful for
        experience replay, reward shaping, and UI.
    terminated
        Whether the game ended this step.
    draw
        True iff game ended in a draw (both teams lose / 50-move rule).
    winner_team
        0, 1, or None.  Defined only when ``terminated and not draw``.
    termination_reason
        Terminal reason string (``state.termination_reason``) or None.
    """

    acting_seat: Seat
    move_result: MoveResult
    terminated: bool
    draw: bool
    winner_team: int | None
    termination_reason: str | None


# ---------------------------------------------------------------------------
# JunqiEnv — single-env, 4-headed self-play
# ---------------------------------------------------------------------------


class JunqiEnv:
    """Single-game Gym-ish environment.

    The env owns:
      * one :class:`GameState` (the authoritative game);
      * four :class:`BeliefTensor` instances (one per seat);
      * one :class:`ObservationBuilder` (shared across seats).

    Observations returned by :meth:`reset` / :meth:`step` are a
    ``dict[Seat, ObservationTensor]`` with one entry per seat.  Each
    entry is a freshly-allocated :class:`ObservationTensor` (snapshot)
    so callers can retain references across steps without aliasing.

    Parameters
    ----------
    show_mode
        Visibility level for the game (default ``HALF_DARK`` for RL).
    builder
        Optional shared :class:`ObservationBuilder`; one is created if
        not provided.
    max_num_moves
        Override the maximum number of moves before a forced draw.  The
        global ``junqi_core`` default is ``MAX_NUM_MOVES = 4000``.  When
        ``max_num_moves < 4000``, episodes are cut short early; the draw
        reward ``(0, 0, 0, 0)`` is returned.  Setting this to ``None``
        uses the global constant.
    """

    def __init__(
        self,
        *,
        show_mode: ShowMode = ShowMode.HALF_DARK,
        builder: ObservationBuilder | None = None,
        max_num_moves: int | None = None,
    ) -> None:
        from junqi_core.rules import MAX_NUM_MOVES as _DEFAULT_MAX
        self.show_mode: ShowMode = show_mode
        self.max_num_moves: int = max_num_moves if max_num_moves is not None else _DEFAULT_MAX
        self._state: GameState | None = None
        self._beliefs: dict[Seat, BeliefTensor] = {}
        self._builder: ObservationBuilder = builder or ObservationBuilder()

    # ------------------------------------------------------------------
    # Gym-ish API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[Seat, ObservationTensor]:
        """Start a fresh game and return observations for all 4 seats."""
        rng = random.Random(seed)
        setup = generate_random_setup(rng)
        self._state = GameState.new_game(setup, show_mode=self.show_mode)
        self._beliefs = {
            s: BeliefTensor.initial(self._state, s) for s in ALL_SEATS
        }
        return self._build_all_observations()

    def step(
        self, action_id: int,
    ) -> tuple[
        dict[Seat, ObservationTensor],
        tuple[int, int, int, int],
        bool,
        JunqiStepInfo,
    ]:
        """Apply a world-frame flat action id.

        Returns ``(obs_dict, reward_tuple, done, info)``.

        * ``obs_dict`` — one :class:`ObservationTensor` per seat (freshly
          snapshotted; safe to retain).
        * ``reward_tuple`` — 4-tuple ``(r_south, r_west, r_north, r_east)``
          of ints (+1 win, -1 loss, 0 draw, 0 non-terminal).
        * ``done`` — True iff the game just terminated (includes draws).
        * ``info`` — :class:`JunqiStepInfo` carrying acting seat, the
          :class:`MoveResult`, and terminal flags.
        """
        state = self._assert_reset()
        if state.terminated:
            raise RuntimeError(
                "JunqiEnv.step called after termination; call reset() first"
            )

        acting_seat = state.turn
        src, dst = action_id_to_src_dst(action_id)
        action = Action(seat=acting_seat, src=src, dst=dst)

        new_state, result = state.step(action)
        # Update each seat's belief with the pre/post-state pair.  The
        # BeliefTensor contract is independent across seats (each seat
        # only incorporates its own visibility rules).
        for seat, belief in self._beliefs.items():
            belief.update(state, new_state, result)
        self._state = new_state

        obs = self._build_all_observations()
        done = new_state.terminated

        # Enforce custom max_num_moves limit (may be shorter than the
        # global junqi_core constant).
        forced_draw = (
            not done
            and new_state.move_counter >= self.max_num_moves
        )
        if forced_draw:
            done = True

        reward = (new_state.team_rewards() or (0, 0, 0, 0)) if (done and not forced_draw) else (0, 0, 0, 0)
        is_draw = bool(getattr(new_state, "draw", False)) or forced_draw
        info = JunqiStepInfo(
            acting_seat=acting_seat,
            move_result=result,
            terminated=done,
            draw=is_draw,
            winner_team=(
                new_state.winner_team
                if done and not is_draw
                else None
            ),
            termination_reason=(
                "max_num_moves" if forced_draw
                else getattr(new_state, "termination_reason", None)
            ),
        )
        return obs, reward, done, info

    def _step_game_only(
        self, action_id: int,
    ) -> tuple[
        tuple[int, int, int, int],
        bool,
        JunqiStepInfo,
    ]:
        """Apply a world-frame action without rebuilding observations.

        This is the hot-path method used by :class:`VectorJunqiEnv` which
        will rebuild observations in bulk via :meth:`_fill_all_obs` after
        all envs have been stepped.  Skipping the per-env obs build avoids
        O(N×4) redundant :class:`ObservationBuilder` calls.

        Returns ``(reward_tuple, done, info)`` — same semantics as
        :meth:`step` except the obs return value is omitted.
        """
        state = self._assert_reset()
        if state.terminated:
            raise RuntimeError(
                "JunqiEnv.step called after termination; call reset() first"
            )

        acting_seat = state.turn
        src, dst = action_id_to_src_dst(action_id)
        action = Action(seat=acting_seat, src=src, dst=dst)

        new_state, result = state.step(action)
        for seat, belief in self._beliefs.items():
            belief.update(state, new_state, result)
        self._state = new_state

        done = new_state.terminated
        forced_draw = (
            not done
            and new_state.move_counter >= self.max_num_moves
        )
        if forced_draw:
            done = True

        reward = (new_state.team_rewards() or (0, 0, 0, 0)) if (done and not forced_draw) else (0, 0, 0, 0)
        is_draw = bool(getattr(new_state, "draw", False)) or forced_draw
        info = JunqiStepInfo(
            acting_seat=acting_seat,
            move_result=result,
            terminated=done,
            draw=is_draw,
            winner_team=(
                new_state.winner_team
                if done and not is_draw
                else None
            ),
            termination_reason=(
                "max_num_moves" if forced_draw
                else getattr(new_state, "termination_reason", None)
            ),
        )
        return reward, done, info

    # ------------------------------------------------------------------
    # Pass-through / introspection
    # ------------------------------------------------------------------

    def legal_action_ids(self, seat: Seat | None = None) -> np.ndarray:
        """Flat world-frame legal action ids for ``seat`` (default: current)."""
        return self._assert_reset().legal_action_ids(seat)

    def current_seat(self) -> Seat:
        """Seat whose turn it currently is."""
        return self._assert_reset().turn

    @property
    def state(self) -> GameState:
        """Read-only access to the underlying :class:`GameState`."""
        return self._assert_reset()

    @property
    def beliefs(self) -> dict[Seat, BeliefTensor]:
        """Read-only access to the four :class:`BeliefTensor`s."""
        return self._beliefs

    @property
    def builder(self) -> ObservationBuilder:
        """The :class:`ObservationBuilder` used for obs assembly."""
        return self._builder

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_reset(self) -> GameState:
        if self._state is None:
            raise RuntimeError("JunqiEnv.reset() must be called before use")
        return self._state

    def _build_all_observations(self) -> dict[Seat, ObservationTensor]:
        """Build obs for all 4 seats using the shared builder.

        Each seat's obs is ``.snapshot()``-ed so subsequent ``build()``
        calls don't alias previous dict entries.
        """
        state = self._assert_reset()
        out: dict[Seat, ObservationTensor] = {}
        for seat in ALL_SEATS:
            obs = self._builder.build(state, self._beliefs[seat], seat)
            out[seat] = obs.snapshot()
        return out


# ---------------------------------------------------------------------------
# VectorJunqiEnv — N parallel envs sharing one slab
# ---------------------------------------------------------------------------


class VectorJunqiEnv:
    """N parallel single-process :class:`JunqiEnv` instances.

    Shares one :class:`ObservationBuilder` across all envs and writes
    observations directly into a preallocated slab
    ``(N, 4, OBS_CHANNELS, 17, 17) float32`` plus
    ``(N, 4, OBS_GLOBAL_DIMS) float32``.

    Unlike :class:`JunqiEnv`, the vector env does NOT return per-seat
    observation dicts — instead it exposes the slabs as attributes
    :attr:`obs_spatial` and :attr:`obs_global`; the trainer reads rows
    directly.  This is the zero-copy surface for a PPO collector.

    Performance notes
    -----------------
    * Game steps use :meth:`JunqiEnv._step_game_only` which skips the
      redundant per-env obs build (observations are rebuilt in bulk by
      :meth:`_fill_all_obs` on the main thread after all steps complete).
    * An optional :class:`~concurrent.futures.ThreadPoolExecutor` can be
      enabled via ``num_workers > 1``.  Whether it helps depends on the
      CPU and how much of the game logic actually releases the GIL.  On
      Python-heavy workloads threading may add overhead; profile with
      ``scripts/profile_env.py --workers_sweep`` before enabling.
    * With ``num_workers=1`` (the default) no pool is created and the
      code path is a tight sequential loop.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        show_mode: ShowMode = ShowMode.HALF_DARK,
        max_num_moves: int | None = None,
        num_workers: int = 1,
    ) -> None:
        """Initialise the vector env.

        Parameters
        ----------
        num_envs
            Number of parallel game environments.
        show_mode
            Visibility mode for every environment.
        max_num_moves
            Optional per-episode step limit (overrides the junqi_core
            global ``MAX_NUM_MOVES = 4000``).
        num_workers
            Thread-pool size for parallel game stepping.  Defaults to
            ``1`` (sequential, no threading overhead).  Set to a value
            greater than 1 only when profiling confirms a speedup (the
            Python GIL limits gains on GIL-bound game logic).
        """
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        self.num_envs = int(num_envs)
        self.show_mode = show_mode
        self._builder = ObservationBuilder()
        self._envs: list[JunqiEnv] = [
            JunqiEnv(show_mode=show_mode, builder=self._builder, max_num_moves=max_num_moves)
            for _ in range(self.num_envs)
        ]
        # Slabs: shape (N, 4 seats, C, H, W) and (N, 4, G).
        self.obs_spatial = np.zeros(
            (self.num_envs, 4, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        self.obs_global = np.zeros(
            (self.num_envs, 4, OBS_GLOBAL_DIMS), dtype=np.float32,
        )
        # Per-env bookkeeping.
        self._done = np.zeros(self.num_envs, dtype=bool)

        # Thread pool for parallel game stepping.
        _workers = num_workers if num_workers is not None else min(self.num_envs, 8)
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=_workers) if _workers > 1 else None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self, seed_base: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset all N envs and fill the observation slab.

        Returns
        -------
        (obs_spatial, obs_global)
            The internal slabs, for convenience.  They are the same
            arrays as :attr:`obs_spatial` / :attr:`obs_global`.
        """
        for i, env in enumerate(self._envs):
            sd = (seed_base + i) if seed_base is not None else None
            env.reset(seed=sd)
        self._done.fill(False)
        self._fill_all_obs()
        return self.obs_spatial, self.obs_global

    def step(
        self, action_ids: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray,      # obs slabs (spatial, global)
        np.ndarray,                  # reward (N, 4) int32
        np.ndarray,                  # done   (N,)  bool
        list[JunqiStepInfo],         # info   length N
    ]:
        """Apply one action per env and refresh the slab.

        Parameters
        ----------
        action_ids
            ndarray of shape (N,) int; world-frame flat action ids.
            Entries where the corresponding env is already ``done`` are
            ignored (the env's slab rows are not overwritten).
        """
        if action_ids.shape != (self.num_envs,):
            raise ValueError(
                f"action_ids shape mismatch: expected ({self.num_envs},), "
                f"got {action_ids.shape}"
            )
        rewards = np.zeros((self.num_envs, 4), dtype=np.int32)
        infos: list[JunqiStepInfo | None] = [None] * self.num_envs

        if self._pool is not None:
            # ---- Parallel game stepping via thread pool ----
            # Submit all active envs; collect futures in order.
            futures = {}
            for i, env in enumerate(self._envs):
                if not self._done[i]:
                    futures[i] = self._pool.submit(
                        env._step_game_only, int(action_ids[i])
                    )
            for i, fut in futures.items():
                rwd, done, info = fut.result()
                rewards[i] = rwd
                self._done[i] = done
                infos[i] = info
        else:
            # ---- Sequential fallback (num_workers=1) ----
            for i, env in enumerate(self._envs):
                if self._done[i]:
                    continue
                rwd, done, info = env._step_game_only(int(action_ids[i]))
                rewards[i] = rwd
                self._done[i] = done
                infos[i] = info

        self._fill_all_obs()
        # Type narrowing: by the end of the loop, every non-"was already
        # done" entry has a JunqiStepInfo.  For already-done entries we
        # keep a sentinel None; trainers treat those as "skip".
        return (
            self.obs_spatial, self.obs_global, rewards,
            self._done.copy(), infos,  # type: ignore[return-value]
        )

    # ------------------------------------------------------------------
    # Slab filling
    # ------------------------------------------------------------------

    def _fill_all_obs(self) -> None:
        """Re-fill the ``(N, 4, ...)`` slabs for every non-done env.

        Done envs keep whatever last obs they had (the trainer should
        treat those slots as terminal and drop them).  Uses
        ``ObservationBuilder.build_observations_batch`` writing directly
        through a reshaped ``(N*4, ...)`` view of the slab — no scatter
        step, no intermediate allocation.
        """
        # Reshape the (N, 4, ...) slab to (N*4, ...) — this is a view,
        # not a copy, because (N, 4) axes are contiguous by construction.
        sp_flat = self.obs_spatial.reshape(
            self.num_envs * 4, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE,
        )
        gl_flat = self.obs_global.reshape(self.num_envs * 4, OBS_GLOBAL_DIMS)

        flat_states: list[GameState] = []
        flat_beliefs: list[BeliefTensor] = []
        flat_observers: list[Seat] = []
        active_indices: list[int] = []
        for i, env in enumerate(self._envs):
            if self._done[i]:
                continue
            state = env.state
            for s_idx, seat in enumerate(ALL_SEATS):
                flat_states.append(state)
                flat_beliefs.append(env._beliefs[seat])
                flat_observers.append(seat)
                active_indices.append(i * 4 + s_idx)
        if not active_indices:
            return

        if len(active_indices) == sp_flat.shape[0]:
            # Happy path: no done env yet, write the whole slab at once.
            self._builder.build_observations_batch(
                flat_states, flat_beliefs, flat_observers, sp_flat, gl_flat,
            )
            return

        # Mixed: some envs are done (rows must stay untouched).  Build
        # into a compact per-active scratch slab and then a single
        # fancy-index assignment copies into the full slab.  This is
        # one memcpy of (active_count, C, H, W) + one of (active_count, G)
        # vs. the per-k loop in the previous implementation.
        k = len(active_indices)
        sp_scratch = np.empty(
            (k, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32,
        )
        gl_scratch = np.empty((k, OBS_GLOBAL_DIMS), dtype=np.float32)
        self._builder.build_observations_batch(
            flat_states, flat_beliefs, flat_observers, sp_scratch, gl_scratch,
        )
        idx = np.asarray(active_indices, dtype=np.intp)
        sp_flat[idx] = sp_scratch
        gl_flat[idx] = gl_scratch

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def envs(self) -> list[JunqiEnv]:
        """Read-only list of underlying :class:`JunqiEnv` instances."""
        return list(self._envs)

    @property
    def done(self) -> np.ndarray:
        """Per-env done flag, shape ``(num_envs,)`` bool."""
        return self._done

    def current_seats(self) -> list[Seat]:
        """Per-env current seat, length N."""
        return [e.current_seat() if not d else ALL_SEATS[0]
                for e, d in zip(self._envs, self._done)]

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the internal thread pool (if any).

        Call this when the env is no longer needed to release OS threads.
        After ``close()``, further calls to :meth:`step` will fail.
        """
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
