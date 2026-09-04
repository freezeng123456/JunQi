"""Observation tensor builder for 4-player Junqi RL networks.

Phase 0.4 M2 (ADR-118): the observation pipeline is now built around
:class:`ObservationBuilder`, which owns 3 pre-allocated NumPy buffers
(one world-frame spatial, one canonical-frame spatial, one global) and
recycles them on every ``build()`` call.  Channel writers accept a
pre-allocated output slice and fill it in-place using vectorized SoA
reads on :class:`junqi_core.state.GameState` (ADR-117 M1).

Design contract (see `docs/ARCHITECTURE.md` §4 and ADR-106, ADR-118):

  * Output tensor is ALWAYS in the observer's canonical frame: the observer
    sits at SOUTH (bottom, canonical y in [11, 16]); teammate is at NORTH
    (top); `left_side_enemy` occupies canonical x in [0, 5]; `right_side_enemy`
    occupies canonical x in [11, 16].

  * `OBS_CHANNELS` spatial channels (412 as of the CombatMemory v6 layout)
    and `OBS_GLOBAL_DIMS` global scalars (28).  Exact layout is pinned in the
    `CHANNEL_LAYOUT` / `GLOBAL_LAYOUT` module-level constants below; changes
    require an ADR.

  * Seat-indexed channels (Dead / Flag-revealed) are stored in observer-sorted
    order ``[me, teammate, left_side, right_side]``, matching the canonical-
    rotation philosophy (network is frame-invariant; channel semantics do not
    leak observer identity).

  * Teammate piece types live in the `prob_teammate` channel group: a single
    12-channel probability distribution sourced from
    ``BeliefTensor.get(teammate_pos)``.  Under BRIGHT / HALF_DARK this
    degenerates to a one-hot vector (teammate is fully observed); under DARK
    it is a non-degenerate posterior.  Channel layout is show-mode invariant.

  * **Buffer-reuse contract (ADR-118)**: :meth:`ObservationBuilder.build`
    returns an :class:`ObservationTensor` whose ``.spatial`` / ``.global_``
    fields are *views* into the builder's internal buffers.  The next
    ``.build()`` call WILL overwrite them.  Callers that need a persistent
    copy must call ``.snapshot()`` (returns an ``ObservationTensor`` backed
    by fresh ndarrays) — RL rollouts don't need this because they forward
    the obs immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .board import (
    BOARD_SIZE,
    cell_info,
    is_camp,
    is_nine_grid,
    is_railway,
    is_stronghold,
)
from .info_model import NUM_TRACKED_TYPES, TRACKED_TYPES, BeliefTensor
from .rotation import rotate_planes
from .rules import (
    MAX_NUM_MOVES,
    MAX_NUM_MOVES_BETWEEN_ATTACKS,
    DeathReason,
    PieceType,
    Seat,
    ShowMode,
)
from .state import GameState

# ===========================================================================
# Channel layout (frozen, Phase 0.3 T7)
# ===========================================================================

_CH_PIECE_OWN_SIZE: Final[int] = NUM_TRACKED_TYPES
_CH_PROB_TEAMMATE_SIZE: Final[int] = NUM_TRACKED_TYPES
_CH_DARK_TEAMMATE_SIZE: Final[int] = 1
_CH_PIECE_LEFT_SIDE_ENEMY_SIZE: Final[int] = 1
_CH_PIECE_RIGHT_SIDE_ENEMY_SIZE: Final[int] = 1
_CH_BELIEF_LEFT_SIDE_SIZE: Final[int] = NUM_TRACKED_TYPES
_CH_BELIEF_RIGHT_SIDE_SIZE: Final[int] = NUM_TRACKED_TYPES
_CH_DEAD_FLAGS_SIZE: Final[int] = 3  # teammate, left_enemy, right_enemy (me removed: always 0)
_CH_FLAG_REVEALED_SIZE: Final[int] = 4
_CH_BOARD_STATIC_SIZE: Final[int] = 6
_CH_TURN_HISTORY_SIZE: Final[int] = 2  # draw_progress + move_progress (is_my_turn/phase removed)

# T7 / ADR-115 — 32 channels (5 groups, each split into ours/theirs).
_CH_MOVE_BUCKET_SIZE: Final[int] = 4 * 2
_CH_ACTIVE_EAT_BUCKET_SIZE: Final[int] = 4 * 2
_CH_PASSIVE_SURVIVE_BUCKET_SIZE: Final[int] = 4 * 2
_CH_DEATH_REASON_SIZE: Final[int] = 3 * 4   # 3 me + 3 teammate + 3 left_enemy + 3 right_enemy
_CH_DEAD_AT_ZERO_SIZE: Final[int] = 1 * 2
_CH_PIECE_ID_SIZE: Final[int] = 120  # one-hot per piece_id (4 seats × 30 slots)
_CH_MOVE_HISTORY_SIZE: Final[int] = 32  # src_dst_planes: 32-step history

# --- CombatMemory v4 channel group sizes ---
# Two layers, total 50 + 46 = 96 channels:
#   Layer 1 (45 ch) projects state[observer][enemy_pid] to enemy alive cells.
#   Layer 2 ( 5 ch) projects {state[opp_left] AND state[opp_right]} to my own
#     alive cells (theory-of-mind; AND yields path-revealed-only knowledge).
#   Layer 3 (46 ch, ADR-129 v5 PR 2026-Q2) — finer per-pid identity tracking
#     to let the network distinguish "killer K ate observer's slot-i piece"
#     from "killer K ate slot-j piece" (the same multi-hot type alone hides
#     this).  All 46 channels are derived from the SAME state arrays as
#     Layer 1, so no new SoA fields are needed:
#       cm_kill_mine_count[12]   per-type COUNT (popcount of direct ∩ type mask)
#       cm_kill_mine_slot[30]    slot-i bit lit at enemy cell if direct
#                                eat-bitmap covers observer's slot-i piece
#       cm_recency[4]            sigmoid(elapsed/τ) on direct/chain/floor steps
#   Layer 4 (60 ch, v6 PR 2026-Q2 — reverse projection) — for each of
#     observer's own 30 pids, project onto its current cell (alive) or
#     zero_pos (dead) the bitmap of which of the 60 ENEMY pids
#     (left + right opponents) have directly OR chain-eaten it.  This
#     pairs symmetrically with cm_kill_mine_slot (which lights up the
#     KILLER cell): cm_eaten_by_pid lights up the VICTIM's home cell
#     (or current cell if still alive — though by definition victims
#     are dead, so live cells get 0; but observer's surviving pids may
#     still report 0, signalling "not yet eaten").  DARK-safe:
#     cm.eaten_by_pid is only written for observer's own mpid (see
#     combat_memory.py reverse projection).
#       cm_eaten_by_pid[60]  channels 0..29 = (observer+1)%4 (left enemy)
#                            channels 30..59 = (observer+3)%4 (right enemy)
# OBS_CHANNELS = 256 + 50 + 46 + 60 = 412.
_CH_CM_KILL_MINE_TYPE_SIZE:    Final[int] = NUM_TRACKED_TYPES  # 12, multi-hot
_CH_CM_KILL_MINE_GE_SIZE:      Final[int] = 3                  # ≥1 / ≥2 / ≥3
_CH_CM_KILL_OTHER_GE_SIZE:     Final[int] = 3                  # ≥1 / ≥2 / ≥3
_CH_CM_CHAIN_TYPE_SIZE:        Final[int] = NUM_TRACKED_TYPES  # 12, multi-hot
_CH_CM_CHAIN_GE_SIZE:          Final[int] = 3                  # ≥1 / ≥2 / ≥3
_CH_CM_FLOOR_GE_SIZE:          Final[int] = 9                  # ≥GONGB ... ≥SILING cumulative
_CH_CM_IS_GONGB_SIZE:          Final[int] = 1
_CH_CM_NOT_GONGB_SIZE:         Final[int] = 1
_CH_CM_DILEI_CANDIDATE_SIZE:   Final[int] = 1                  # runtime computed
# Layer 2 (theory-of-mind on my own alive pieces).
_CH_CM_MY_KILL_COUNT_GE_SIZE:    Final[int] = 3
_CH_CM_MY_IS_GONGB_SIZE:         Final[int] = 1
_CH_CM_MY_DILEI_CANDIDATE_SIZE:  Final[int] = 1
# Layer 3 (per-pid identity tail; ADR-129 v5).
_CH_CM_KILL_MINE_COUNT_SIZE:     Final[int] = NUM_TRACKED_TYPES   # 12, normalized count
_CH_CM_KILL_MINE_SLOT_SIZE:      Final[int] = 30                  # observer-local slot bit
_CH_CM_RECENCY_SIZE:             Final[int] = 4                   # 4 fresh-signal scalars
# Layer 4 (reverse projection; v6).
_CH_CM_EATEN_BY_PID_SIZE:        Final[int] = 60                  # 30 left + 30 right enemy pids

MOVE_BUCKET_COUNT: Final[int] = 4
ACTIVE_EAT_BUCKET_COUNT: Final[int] = 4
PASSIVE_SURVIVE_BUCKET_COUNT: Final[int] = 4
DEATH_REASON_COUNT: Final[int] = 3


def _layout_slices() -> dict[str, slice]:
    order = [
        ("piece_own", _CH_PIECE_OWN_SIZE),
        ("prob_teammate", _CH_PROB_TEAMMATE_SIZE),
        ("dark_teammate", _CH_DARK_TEAMMATE_SIZE),
        ("piece_left_side_enemy", _CH_PIECE_LEFT_SIDE_ENEMY_SIZE),
        ("piece_right_side_enemy", _CH_PIECE_RIGHT_SIDE_ENEMY_SIZE),
        ("belief_left_side", _CH_BELIEF_LEFT_SIDE_SIZE),
        ("belief_right_side", _CH_BELIEF_RIGHT_SIDE_SIZE),
        ("dead_flags", _CH_DEAD_FLAGS_SIZE),
        ("flag_revealed", _CH_FLAG_REVEALED_SIZE),
        ("board_static", _CH_BOARD_STATIC_SIZE),
        ("turn_history", _CH_TURN_HISTORY_SIZE),
        ("move_bucket", _CH_MOVE_BUCKET_SIZE),
        ("active_eat_bucket", _CH_ACTIVE_EAT_BUCKET_SIZE),
        ("passive_survive_bucket", _CH_PASSIVE_SURVIVE_BUCKET_SIZE),
        ("death_reason", _CH_DEATH_REASON_SIZE),
        ("dead_at_zero", _CH_DEAD_AT_ZERO_SIZE),
        ("piece_id", _CH_PIECE_ID_SIZE),
        ("move_history", _CH_MOVE_HISTORY_SIZE),
        # ---- CombatMemory v4 tail (50 ch); pre-CM indices [0, 256) preserved ----
        # Layer 1 (45 ch) — projected to enemy alive pieces.
        ("cm_kill_mine_type", _CH_CM_KILL_MINE_TYPE_SIZE),
        ("cm_kill_mine_ge", _CH_CM_KILL_MINE_GE_SIZE),
        ("cm_kill_other_ge", _CH_CM_KILL_OTHER_GE_SIZE),
        ("cm_chain_type", _CH_CM_CHAIN_TYPE_SIZE),
        ("cm_chain_ge", _CH_CM_CHAIN_GE_SIZE),
        ("cm_floor_ge", _CH_CM_FLOOR_GE_SIZE),
        ("cm_is_gongb", _CH_CM_IS_GONGB_SIZE),
        ("cm_not_gongb", _CH_CM_NOT_GONGB_SIZE),
        ("cm_dilei_candidate", _CH_CM_DILEI_CANDIDATE_SIZE),
        # Layer 2 (5 ch) — projected to my own alive pieces (theory-of-mind).
        ("cm_my_kill_count_ge", _CH_CM_MY_KILL_COUNT_GE_SIZE),
        ("cm_my_is_gongb", _CH_CM_MY_IS_GONGB_SIZE),
        ("cm_my_dilei_candidate", _CH_CM_MY_DILEI_CANDIDATE_SIZE),
        # Layer 3 (46 ch, ADR-129 v5) — per-pid identity tail.
        ("cm_kill_mine_count", _CH_CM_KILL_MINE_COUNT_SIZE),
        ("cm_kill_mine_slot", _CH_CM_KILL_MINE_SLOT_SIZE),
        ("cm_recency", _CH_CM_RECENCY_SIZE),
        # Layer 4 (60 ch, v6) — reverse projection on observer's own pids.
        ("cm_eaten_by_pid", _CH_CM_EATEN_BY_PID_SIZE),
    ]
    out: dict[str, slice] = {}
    start = 0
    for name, size in order:
        out[name] = slice(start, start + size)
        start += size
    return out


CHANNEL_LAYOUT: Final[dict[str, slice]] = _layout_slices()
OBS_CHANNELS: Final[int] = sum(s.stop - s.start for s in CHANNEL_LAYOUT.values())
assert OBS_CHANNELS == 412, (
    f"Channel layout must sum to 412, got {OBS_CHANNELS}"
)

_GLOB_REMAINING_LEFT_SIZE: Final[int] = NUM_TRACKED_TYPES
_GLOB_REMAINING_RIGHT_SIZE: Final[int] = NUM_TRACKED_TYPES
_GLOB_FLAG_REVEALED_SIZE: Final[int] = 4


def _global_slices() -> dict[str, slice]:
    order = [
        ("remaining_left_side", _GLOB_REMAINING_LEFT_SIZE),
        ("remaining_right_side", _GLOB_REMAINING_RIGHT_SIZE),
        ("flag_revealed", _GLOB_FLAG_REVEALED_SIZE),
    ]
    out: dict[str, slice] = {}
    start = 0
    for name, size in order:
        out[name] = slice(start, start + size)
        start += size
    return out


GLOBAL_LAYOUT: Final[dict[str, slice]] = _global_slices()
OBS_GLOBAL_DIMS: Final[int] = sum(s.stop - s.start for s in GLOBAL_LAYOUT.values())
assert OBS_GLOBAL_DIMS == 28, f"Global layout must sum to 28, got {OBS_GLOBAL_DIMS}"


# ===========================================================================
# Static lookup tables (built once at import)
# ===========================================================================

# PieceType.value -> index in the 12-wide TRACKED_TYPES channel group;
# -1 for NONE/DARK (skipped by writers).  Indexed by piece_type_arr directly.
_TYPE_TO_IDX_MAP: Final[dict[PieceType, int]] = {
    t: i for i, t in enumerate(TRACKED_TYPES)
}

_PIECETYPE_TO_TRACKED_IDX: Final[np.ndarray] = (
    _build_piecetype_idx := lambda: (
        arr := np.full(max(pt.value for pt in PieceType) + 1, -1, dtype=np.int8),
        [arr.__setitem__(pt.value, _TYPE_TO_IDX_MAP[pt]) for pt in _TYPE_TO_IDX_MAP],
        arr,
    )[2]
)()

# DeathReason.value -> 0/1/2 index; -1 for absent entries.
_REASON_TO_IDX_MAP: Final[dict[DeathReason, int]] = {
    DeathReason.KILLED_BY_ENEMY: 0,
    DeathReason.HIT_MINE_OR_BOMB: 1,
    DeathReason.MUTUAL: 2,
}

_DEATH_REASON_TO_IDX: Final[np.ndarray] = (
    _build_reason_idx := lambda: (
        arr := np.full(max(r.value for r in DeathReason) + 1, -1, dtype=np.int8),
        [arr.__setitem__(r.value, _REASON_TO_IDX_MAP[r]) for r in _REASON_TO_IDX_MAP],
        arr,
    )[2]
)()

# Seat-value -> team (0/1); used for ours-vs-theirs bit masks.
_SEAT_TEAM: Final[np.ndarray] = np.array(
    [s.team for s in (Seat.SOUTH, Seat.WEST, Seat.NORTH, Seat.EAST)],
    dtype=np.int8,
)


# ===========================================================================
# CombatMemory v4 channel-writer helper tables (built at import)
# ===========================================================================


def _popcount_pid_pair(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Popcount of (lo, hi) uint64 pair → number of pid bits set.
    Used for the ``ge_*`` count thresholds.  Python loop version; the
    only build-time cost scales with the enemy-pid count (≤ 60), so
    the overhead is < 50 µs per build.
    """
    flat_lo = lo.reshape(-1).astype(np.uint64, copy=False)
    flat_hi = hi.reshape(-1).astype(np.uint64, copy=False)
    out = np.empty(flat_lo.shape, dtype=np.int16)
    for i in range(flat_lo.shape[0]):
        out[i] = int(flat_lo[i]).bit_count() + int(flat_hi[i]).bit_count()
    return out.reshape(lo.shape)



def _build_board_static_world() -> np.ndarray:
    """Compute the (6, 17, 17) board-topology plane stack once at import.

    Every plane here must be invariant under 90-degree rotation, because the
    stack is built in the world frame and then rotated into each observer's
    canonical frame.  A plane that is not rotation-invariant would encode the
    observer's seat identity, which the canonical frame exists to remove.

    That rules out a one-hot-per-curve encoding: ``rot90`` permutes the four
    corner curves in a 4-cycle (1 -> 4 -> 3 -> 2), so "curve 1" would light up
    a different corner for each observer.  Plane 4 therefore marks the union
    of all four curves, which is rotation-invariant and is what movement
    generation actually keys on (a rail cell where an engineer may turn).
    Plane 5 is reserved; distinguishing individual curves needs one plane per
    curve, which would change ``OBS_CHANNELS``.
    """
    planes = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            if is_camp(x, y):
                planes[0, y, x] = 1.0
            if is_stronghold(x, y):
                planes[1, y, x] = 1.0
            if is_railway(x, y):
                planes[2, y, x] = 1.0
            if is_nine_grid(x, y):
                planes[3, y, x] = 1.0
            if cell_info(x, y).curve_rail != 0:
                planes[4, y, x] = 1.0
    return planes


_BOARD_STATIC_WORLD: Final[np.ndarray] = _build_board_static_world()


# ===========================================================================
# ObservationTensor
# ===========================================================================


@dataclass(frozen=True, slots=True)
class ObservationTensor:
    """Canonical-frame observation for a single acting seat.

    Attributes:
        spatial: float32 array of shape (OBS_CHANNELS, 17, 17). Indexing
            convention is `[channel, y, x]`.
        global_: float32 array of shape (OBS_GLOBAL_DIMS,).
        observer: Seat whose canonical frame this tensor is expressed in.

    Buffer-reuse note (ADR-118)
    ---------------------------
    When produced by :meth:`ObservationBuilder.build`, ``spatial`` and
    ``global_`` are *views* into the builder's internal buffers and will
    be overwritten by the next ``build()``.  Use :meth:`snapshot` to
    detach into freshly allocated ndarrays.
    """

    spatial: np.ndarray
    global_: np.ndarray
    observer: Seat

    def __post_init__(self) -> None:
        if self.spatial.shape != (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"spatial must be ({OBS_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}); "
                f"got {self.spatial.shape}"
            )
        if self.spatial.dtype != np.float32:
            raise TypeError(f"spatial must be float32; got {self.spatial.dtype}")
        if self.global_.shape != (OBS_GLOBAL_DIMS,):
            raise ValueError(
                f"global_ must be ({OBS_GLOBAL_DIMS},); got {self.global_.shape}"
            )
        if self.global_.dtype != np.float32:
            raise TypeError(f"global_ must be float32; got {self.global_.dtype}")

    def channel(self, name: str) -> np.ndarray:
        """Return the named channel group (view, not copy)."""
        return self.spatial[CHANNEL_LAYOUT[name]]

    def global_slice(self, name: str) -> np.ndarray:
        """Return the named global feature range (view, not copy)."""
        return self.global_[GLOBAL_LAYOUT[name]]

    def snapshot(self) -> ObservationTensor:
        """Return a detached copy whose buffers are NOT shared with any builder.

        Required if the caller wants to hold an ObservationTensor across
        subsequent ``ObservationBuilder.build()`` calls.
        """
        return ObservationTensor(
            spatial=self.spatial.copy(),
            global_=self.global_.copy(),
            observer=self.observer,
        )


# ===========================================================================
# ObservationBuilder (Phase 0.4 M2, ADR-118)
# ===========================================================================


class ObservationBuilder:
    """Reusable, allocation-free builder for :class:`ObservationTensor`.

    Owns 3 pre-allocated NumPy buffers:

      * ``_world``     (OBS_CHANNELS, 17, 17) float32 — world-frame scratch.
      * ``_canonical`` (OBS_CHANNELS, 17, 17) float32 — canonical-frame output.
      * ``_global``    (28,)         float32 — global feature vector.

    :meth:`build` zeros the buffers, runs the 16 vectorized writers, runs a
    central rotation into ``_canonical``, and returns an
    :class:`ObservationTensor` whose ``.spatial`` / ``.global_`` are views
    into the owned buffers.  **The views are invalidated on the next
    ``build()``.**  Callers that need persistence call ``.snapshot()``.

    Thread-safety: each thread needs its own instance.
    """

    __slots__ = ("_canonical", "_global", "_world")

    def __init__(self) -> None:
        self._world = np.zeros(
            (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self._canonical = np.zeros_like(self._world)
        self._global = np.zeros(OBS_GLOBAL_DIMS, dtype=np.float32)

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    def build(
        self,
        state: GameState,
        belief: BeliefTensor,
        observer: Seat,
    ) -> ObservationTensor:
        """Build ``observer``'s canonical-frame observation.

        Preconditions:
          * ``belief.observer is observer``.
          * ``state.show_mode`` matches ``belief.show_mode``.

        The returned :class:`ObservationTensor` shares its buffers with
        this builder.  The views are invalidated on the next ``build()``
        or ``build_into()`` call (see ADR-118).
        """
        self._fill_into(
            state, belief, observer, self._world, self._canonical, self._global,
        )
        return ObservationTensor(
            spatial=self._canonical,
            global_=self._global,
            observer=observer,
        )

    def build_into(
        self,
        state: GameState,
        belief: BeliefTensor,
        observer: Seat,
        out_spatial: np.ndarray,
        out_global: np.ndarray,
    ) -> None:
        """Build directly into caller-provided output buffers.

        This is the zero-copy entry point used by
        :meth:`build_observations_batch` and (transitively) by the torch
        memory bridge.  Exactly equivalent to :meth:`build` modulo where
        the output lands.

        Parameters
        ----------
        state, belief, observer : see :meth:`build`.
        out_spatial : ndarray[OBS_CHANNELS, 17, 17] float32
            Canonical-frame spatial output.  Must be C-contiguous.
        out_global  : ndarray[OBS_GLOBAL_DIMS] float32
            Global feature output.  Must be C-contiguous.

        Notes
        -----
        * ``out_spatial`` / ``out_global`` may be views into a larger
          tensor (for batch assembly) or be numpy wrappers around torch
          CPU tensors (for zero-copy staging).  The builder writes
          through them but never retains references.
        * The builder's own scratch ``_world`` is still used for the
          world-frame assembly (rotation requires a separate source
          buffer); the rotated output goes directly into
          ``out_spatial``.
        """
        if out_spatial.shape != (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"out_spatial shape mismatch: expected "
                f"({OBS_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}), "
                f"got {out_spatial.shape}"
            )
        if out_spatial.dtype != np.float32:
            raise TypeError(
                f"out_spatial must be float32, got {out_spatial.dtype}"
            )
        if out_global.shape != (OBS_GLOBAL_DIMS,):
            raise ValueError(
                f"out_global shape mismatch: expected ({OBS_GLOBAL_DIMS},), "
                f"got {out_global.shape}"
            )
        if out_global.dtype != np.float32:
            raise TypeError(
                f"out_global must be float32, got {out_global.dtype}"
            )
        self._fill_into(state, belief, observer, self._world, out_spatial, out_global)

    def build_observations_batch(
        self,
        states: list[GameState] | tuple[GameState, ...],
        beliefs: list[BeliefTensor] | tuple[BeliefTensor, ...],
        observers: list[Seat] | tuple[Seat, ...],
        out_spatial: np.ndarray,
        out_global: np.ndarray,
    ) -> None:
        """Fill ``(N, 412, 17, 17)`` + ``(N, 28)`` buffers in place.

        Each index ``i`` is equivalent to::

            build_into(states[i], beliefs[i], observers[i],
                       out_spatial[i], out_global[i])

        Implementation is a simple Python-level for-loop: the per-state
        cost is dominated by the 16 vectorized writers.  Phase 1 will
        introduce a proper batched SoA kernel that collapses the outer
        loop into one numpy pass (ADR-126); this function is the
        API seam that does not change when that happens.

        Shape contract
        --------------
        * ``out_spatial``: shape ``(N, OBS_CHANNELS, 17, 17)`` float32, C-contig.
        * ``out_global`` : shape ``(N, OBS_GLOBAL_DIMS)`` float32, C-contig.
        * ``len(states) == len(beliefs) == len(observers) == N``.
        """
        n = len(states)
        if len(beliefs) != n or len(observers) != n:
            raise ValueError(
                f"batch length mismatch: states={n}, beliefs={len(beliefs)}, "
                f"observers={len(observers)}"
            )
        if out_spatial.shape != (n, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"out_spatial shape mismatch: expected "
                f"({n}, {OBS_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}), "
                f"got {out_spatial.shape}"
            )
        if out_global.shape != (n, OBS_GLOBAL_DIMS):
            raise ValueError(
                f"out_global shape mismatch: expected ({n}, {OBS_GLOBAL_DIMS}), "
                f"got {out_global.shape}"
            )
        if out_spatial.dtype != np.float32 or out_global.dtype != np.float32:
            raise TypeError("out_spatial and out_global must be float32")
        for i in range(n):
            self._fill_into(
                states[i], beliefs[i], observers[i],
                self._world, out_spatial[i], out_global[i],
            )

    # -----------------------------------------------------------------
    # Internal assembly (shared by build, build_into, batch)
    # -----------------------------------------------------------------

    def _fill_into(
        self,
        state: GameState,
        belief: BeliefTensor,
        observer: Seat,
        world_buf: np.ndarray,        # scratch (OBS_CHANNELS, 17, 17) float32
        canonical_out: np.ndarray,    # final   (OBS_CHANNELS, 17, 17) float32
        global_out: np.ndarray,       # final   (28,)          float32
    ) -> None:
        """Core assembly logic used by every public entry point."""
        if belief.observer is not observer:
            raise ValueError(
                f"belief.observer={belief.observer.name} does not match "
                f"observer={observer.name}; rebuild belief with the correct "
                f"observer"
            )

        # Phase 0.4 M6 — lazy belief tensor sync.  ``BeliefTensor.update``
        # now only marks the tensor mirror dirty; the refresh happens
        # here, right before the writers start reading it.  No-op when
        # the mirror is already fresh (e.g. after ``BeliefTensor.initial``).
        belief.ensure_synced(state)

        # Zero world + global scratch in-place (one C-level memset each).
        # ``canonical_out`` is always fully overwritten by the rotation
        # below, so no zeroing is required.
        world_buf.fill(0.0)
        global_out.fill(0.0)

        # Pre-extract SoA columns we will re-use across writers.
        alive = state.alive
        pos_x = state.pos_x
        pos_y = state.pos_y
        piece_seat_arr = state.piece_seat_arr
        piece_type_arr = state.piece_type_arr
        live_mask = alive                                    # (120,) bool
        live_pids = np.nonzero(live_mask)[0]                 # (N,)
        if live_pids.size:
            lx = pos_x[live_pids].astype(np.intp, copy=False)
            ly = pos_y[live_pids].astype(np.intp, copy=False)
            lseat = piece_seat_arr[live_pids]
            ltype = piece_type_arr[live_pids]
            ltype_idx = _PIECETYPE_TO_TRACKED_IDX[ltype]     # (N,) int8; -1 if untracked
        else:
            lx = ly = np.empty(0, dtype=np.intp)
            lseat = ltype = ltype_idx = np.empty(0, dtype=np.int8)

        # Seat masks (live-piece index -> bool).
        obs_val = observer.value
        teammate_val = observer.teammate.value
        left_val = observer.left_side_enemy.value
        right_val = observer.right_side_enemy.value
        obs_team = int(_SEAT_TEAM[obs_val])
        order_vals = (obs_val, teammate_val, left_val, right_val)

        # --- 1. piece_own --------------------------------------------------
        _write_piece_own(
            out=world_buf[CHANNEL_LAYOUT["piece_own"]],
            lx=lx, ly=ly, lseat=lseat, ltype_idx=ltype_idx,
            owner_val=obs_val,
        )

        # --- 2. prob_teammate (12 planes from BeliefTensor) ---------------
        _write_prob_teammate(
            out=world_buf[CHANNEL_LAYOUT["prob_teammate"]],
            lx=lx, ly=ly, lseat=lseat,
            live_pids=live_pids,
            teammate_val=teammate_val,
            belief=belief,
        )

        # --- 3. dark_teammate (mask; only non-zero under DARK) ------------
        if state.show_mode is ShowMode.DARK:
            _write_enemy_mask(
                out=world_buf[CHANNEL_LAYOUT["dark_teammate"]][0],
                lx=lx, ly=ly, lseat=lseat, target_seat_val=teammate_val,
            )

        # --- 4. piece_left_side_enemy ------------------------------------
        _write_enemy_mask(
            out=world_buf[CHANNEL_LAYOUT["piece_left_side_enemy"]][0],
            lx=lx, ly=ly, lseat=lseat, target_seat_val=left_val,
        )

        # --- 5. piece_right_side_enemy -----------------------------------
        _write_enemy_mask(
            out=world_buf[CHANNEL_LAYOUT["piece_right_side_enemy"]][0],
            lx=lx, ly=ly, lseat=lseat, target_seat_val=right_val,
        )

        # --- 6. belief_left_side (12 planes per enemy) -------------------
        _write_belief_side(
            out=world_buf[CHANNEL_LAYOUT["belief_left_side"]],
            lx=lx, ly=ly, lseat=lseat,
            live_pids=live_pids,
            target_seat_val=left_val,
            belief=belief,
        )

        # --- 7. belief_right_side ----------------------------------------
        _write_belief_side(
            out=world_buf[CHANNEL_LAYOUT["belief_right_side"]],
            lx=lx, ly=ly, lseat=lseat,
            live_pids=live_pids,
            target_seat_val=right_val,
            belief=belief,
        )

        # --- 8. dead_flags (3 constant planes: teammate, left, right) -----
        #     me is always alive (can't observe while dead), so skipped.
        _write_constant_per_seat(
            out=world_buf[CHANNEL_LAYOUT["dead_flags"]],
            order_vals=(teammate_val, left_val, right_val),
            per_seat_values=state.seat_dead_arr.astype(np.float32, copy=False),
        )

        # --- 9. flag_revealed (4 constant planes, observer-sorted) -------
        _write_constant_per_seat(
            out=world_buf[CHANNEL_LAYOUT["flag_revealed"]],
            order_vals=order_vals,
            per_seat_values=state.seat_flag_revealed_arr.astype(np.float32, copy=False),
        )

        # --- 10. board_static (6 planes, precomputed) --------------------
        world_buf[CHANNEL_LAYOUT["board_static"]] = _BOARD_STATIC_WORLD

        # --- 11. turn_history (2 constant scalars) -----------------------
        _write_turn_history(
            out=world_buf[CHANNEL_LAYOUT["turn_history"]],
            state=state,
            observer=observer,
        )

        # --- 12. move_bucket (A group: 4 ours + 4 theirs) ----------------
        ours_mask_b, theirs_mask_b = _build_bucket_masks(
            live_pids, lseat, belief, obs_team
        )
        _write_bucket_group(
            out=world_buf[CHANNEL_LAYOUT["move_bucket"]],
            mode="exact",
            live_pids=live_pids,
            lx=lx, ly=ly,
            ours_mask=ours_mask_b, theirs_mask=theirs_mask_b,
            counters=state.move_count_arr,
            bucket_count=MOVE_BUCKET_COUNT,
        )

        # --- 13. active_eat_bucket (B group: cumulative) -----------------
        _write_bucket_group(
            out=world_buf[CHANNEL_LAYOUT["active_eat_bucket"]],
            mode="cumulative",
            live_pids=live_pids,
            lx=lx, ly=ly,
            ours_mask=ours_mask_b, theirs_mask=theirs_mask_b,
            counters=state.active_eat_arr,
            bucket_count=ACTIVE_EAT_BUCKET_COUNT,
        )

        # --- 14. passive_survive_bucket (C group: cumulative) ------------
        _write_bucket_group(
            out=world_buf[CHANNEL_LAYOUT["passive_survive_bucket"]],
            mode="cumulative",
            live_pids=live_pids,
            lx=lx, ly=ly,
            ours_mask=ours_mask_b, theirs_mask=theirs_mask_b,
            counters=state.passive_surv_arr,
            bucket_count=PASSIVE_SURVIVE_BUCKET_COUNT,
        )

        # --- 15. death_reason (D group: 3 me + 3 teammate + 3 left + 3 right)
        _write_death_reason(
            out=world_buf[CHANNEL_LAYOUT["death_reason"]],
            state=state,
            me_seat_val=obs_val,
            teammate_seat_val=teammate_val,
            left_seat_val=left_val,
            right_seat_val=right_val,
        )

        # --- 16. dead_at_zero (E group: 1 ours + 1 theirs) ---------------
        _write_dead_at_zero(
            out=world_buf[CHANNEL_LAYOUT["dead_at_zero"]],
            state=state,
            obs_team=obs_team,
        )

        # --- 17. piece_id (120 one-hot planes, one per piece_id) ----------
        _write_piece_id(
            out=world_buf[CHANNEL_LAYOUT["piece_id"]],
            live_pids=live_pids,
            lx=lx,
            ly=ly,
        )

        # --- 18. move_history (32 src_dst_planes) -------------------------
        _write_move_history(
            out=world_buf[CHANNEL_LAYOUT["move_history"]],
            state=state,
        )

        # --- 19. CombatMemory v4 + v5 tail (96 channels) ------------------
        _write_combat_memory(
            out_kill_mine_type=world_buf[CHANNEL_LAYOUT["cm_kill_mine_type"]],
            out_kill_mine_ge=world_buf[CHANNEL_LAYOUT["cm_kill_mine_ge"]],
            out_kill_other_ge=world_buf[CHANNEL_LAYOUT["cm_kill_other_ge"]],
            out_chain_type=world_buf[CHANNEL_LAYOUT["cm_chain_type"]],
            out_chain_ge=world_buf[CHANNEL_LAYOUT["cm_chain_ge"]],
            out_floor_ge=world_buf[CHANNEL_LAYOUT["cm_floor_ge"]],
            out_is_gongb=world_buf[CHANNEL_LAYOUT["cm_is_gongb"]],
            out_not_gongb=world_buf[CHANNEL_LAYOUT["cm_not_gongb"]],
            out_dilei_candidate=world_buf[CHANNEL_LAYOUT["cm_dilei_candidate"]],
            out_my_kill_count_ge=world_buf[CHANNEL_LAYOUT["cm_my_kill_count_ge"]],
            out_my_is_gongb=world_buf[CHANNEL_LAYOUT["cm_my_is_gongb"]],
            out_my_dilei_candidate=world_buf[CHANNEL_LAYOUT["cm_my_dilei_candidate"]],
            # Layer 3 (v5): per-pid identity tail.
            out_kill_mine_count=world_buf[CHANNEL_LAYOUT["cm_kill_mine_count"]],
            out_kill_mine_slot=world_buf[CHANNEL_LAYOUT["cm_kill_mine_slot"]],
            out_recency=world_buf[CHANNEL_LAYOUT["cm_recency"]],
            out_eaten_by_pid=world_buf[CHANNEL_LAYOUT["cm_eaten_by_pid"]],
            state=state,
            observer_val=obs_val,
            live_pids=live_pids,
            lx=lx,
            ly=ly,
        )

        # ---- global features ----
        _write_global_features(
            out=global_out, state=state, belief=belief, order_vals=order_vals,
        )

        # ---- Rotate world_buf -> canonical_out via a single np.rot90 + copy ----
        #
        # ``rotate_planes`` returns a strided view; np.copyto does the
        # ~117 KiB transfer C-side in one shot into the caller-provided
        # destination buffer.
        np.copyto(canonical_out, rotate_planes(world_buf, observer))


# ===========================================================================
# Module-level convenience wrapper (maintained for legacy test entry points)
# ===========================================================================
#
# ADR-118 mandates deleting ``build_observation``; in M2 we keep a thin
# wrapper around a per-call :class:`ObservationBuilder` so existing tests
# and tools continue to work.  The wrapper calls :meth:`snapshot` so the
# returned tensor owns its own buffers (no shared state between calls).
# M3 is the scheduled removal point once all call sites migrate.

_DEFAULT_BUILDER_SLOT: ObservationBuilder | None = None


def build_observation(
    state: GameState,
    belief: BeliefTensor,
    observer: Seat,
) -> ObservationTensor:
    """Legacy convenience wrapper around :class:`ObservationBuilder`.

    Each call constructs an ObservationTensor whose buffers are *detached*
    from any shared builder — so multiple calls can be held simultaneously
    (this preserves the pre-M2 API semantics).  Prefer the builder API in
    new code; see ADR-118.
    """
    global _DEFAULT_BUILDER_SLOT
    if _DEFAULT_BUILDER_SLOT is None:
        _DEFAULT_BUILDER_SLOT = ObservationBuilder()
    obs = _DEFAULT_BUILDER_SLOT.build(state, belief, observer)
    return obs.snapshot()  # detach buffers for legacy callers


# ===========================================================================
# Vectorized channel writers (all in-place on a pre-allocated `out` slice).
# ===========================================================================
#
# Conventions
# -----------
# * ``out`` is a view into ``ObservationBuilder._world`` of the right
#   shape for the channel group (``(C, 17, 17)`` or ``(17, 17)``).
# * ``lx`` / ``ly`` are intp arrays of length ``N = sum(alive)`` carrying
#   the (x, y) of every live piece, extracted once at the top of
#   ``ObservationBuilder.build``.  ``lseat`` / ``ltype_idx`` are the
#   matching seat-value / tracked-type-index arrays.
# * Every writer fills ONLY its output slice (already zeroed by the
#   caller) — nothing is cleared here.


def _write_piece_own(
    out: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lseat: np.ndarray,
    ltype_idx: np.ndarray,
    owner_val: int,
) -> None:
    """(12, H, W): 1.0 at each own-piece cell in its tracked-type channel."""
    sel = (lseat == owner_val) & (ltype_idx >= 0)
    if not sel.any():
        return
    out[ltype_idx[sel], ly[sel], lx[sel]] = 1.0


def _write_enemy_mask(
    out: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lseat: np.ndarray,
    target_seat_val: int,
) -> None:
    """(H, W): 1.0 at every cell occupied by ``target_seat_val``."""
    sel = (lseat == target_seat_val)
    if not sel.any():
        return
    out[ly[sel], lx[sel]] = 1.0


def _write_prob_teammate(
    out: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lseat: np.ndarray,
    live_pids: np.ndarray,
    teammate_val: int,
    belief: BeliefTensor,
) -> None:
    """(12, H, W): per-cell P(type) at teammate cells.

    Phase 0.4 M3 (ADR-120): reads ``belief.probs_arr`` (shape
    ``(num_pids, 12)``) in a single fancy-index + transpose, no Python
    per-piece loop.
    """
    sel = (lseat == teammate_val)
    if not sel.any():
        return
    ys = ly[sel]
    xs = lx[sel]
    # shape (N_sel, 12) -> (12, N_sel) -> scatter into (12, H, W).
    vecs = belief.probs_arr[live_pids[sel]]           # (N_sel, 12)
    out[:, ys, xs] = vecs.T


def _write_belief_side(
    out: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    lseat: np.ndarray,
    live_pids: np.ndarray,
    target_seat_val: int,
    belief: BeliefTensor,
) -> None:
    """(12, H, W): per-cell P(type) at enemy cells for a given side.

    Phase 0.4 M3 (ADR-120): vectorized via ``belief.probs_arr``.
    """
    sel = (lseat == target_seat_val)
    if not sel.any():
        return
    ys = ly[sel]
    xs = lx[sel]
    vecs = belief.probs_arr[live_pids[sel]]           # (N_sel, 12)
    out[:, ys, xs] = vecs.T


def _write_constant_per_seat(
    out: np.ndarray,
    order_vals: tuple[int, ...],
    per_seat_values: np.ndarray,
) -> None:
    """(K, H, W): constant planes in observer-sorted order.

    ``per_seat_values`` is indexed by the raw Seat.value (0..3); the
    output channels are laid out according to ``order_vals``.
    """
    for ch, sv in enumerate(order_vals):
        v = float(per_seat_values[sv])
        if v != 0.0:
            out[ch].fill(v)


def _write_turn_history(
    out: np.ndarray,
    state: GameState,
    observer: Seat,
) -> None:
    """(2, H, W): draw_progress, move_progress.

    draw_progress = moves_since_last_combat / 200 (clamp [0,1]).
    move_progress = move_counter / 4000 (clamp [0,1]).

    Removed channels (redundant):
      - is_my_turn: always 1.0 in the single-seat hot path.
      - game_phase: deterministic discretisation of move_progress.
    """
    draw_progress = min(
        float(state.moves_since_last_combat) / MAX_NUM_MOVES_BETWEEN_ATTACKS,
        1.0,
    )
    move_progress = min(float(state.move_counter) / MAX_NUM_MOVES, 1.0)

    out[0].fill(draw_progress)
    out[1].fill(move_progress)


_ONEHOT_EPS: Final[float] = 1e-6


def _exact_bucket(counter: int) -> int:
    if counter <= 0:
        return 0
    if counter == 1:
        return 1
    if counter == 2:
        return 2
    return 3


def _cumulative_top_bucket(counter: int) -> int:
    if counter <= 0:
        return 0
    if counter >= 3:
        return 3
    return counter  # 1 or 2


def _write_bucket_group(
    *,
    out: np.ndarray,
    mode: str,                     # "exact" or "cumulative"
    live_pids: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
    ours_mask: np.ndarray,
    theirs_mask: np.ndarray,
    counters: np.ndarray,          # full (num_pids,) counter array from state
    bucket_count: int,
) -> None:
    """(2*bucket_count, H, W): bucketed per-piece counters.

    ``mode``:
      * ``"exact"``      — light only plane ``bucket(c)``.
      * ``"cumulative"`` — light planes 0..top_bucket(c) inclusive.

    ``out`` is split into halves [ours | theirs] of size ``bucket_count``.
    ``ours_mask`` / ``theirs_mask`` are pre-computed boolean arrays of
    shape ``(len(live_pids),)`` — ``theirs_mask`` already has the D-3
    visibility filter applied.  Both masks are computed once per
    ``ObservationBuilder.build`` call and reused across the 3 bucket
    groups (see ``_build_bucket_masks``).

    Phase 0.4 M3: no per-mask ``.any()`` probes — empty fancy-index
    assignments are no-ops, so we just issue them.
    """
    if live_pids.size == 0:
        return

    half = bucket_count

    # Bucket index per live piece: 0/1/2/3 (saturate at 3).
    cnt = counters[live_pids]
    buckets = np.minimum(cnt.astype(np.intp, copy=False), 3)

    if mode == "exact":
        # Two fancy-index assignments.
        for base, mask in ((0, ours_mask), (half, theirs_mask)):
            sub_buckets = buckets[mask]
            sub_ys = ly[mask]
            sub_xs = lx[mask]
            out[base + sub_buckets, sub_ys, sub_xs] = 1.0
    else:
        # cumulative: for each bucket index k, pick pieces whose
        # top >= k and light plane (base + k) at their cells.
        for base, mask in ((0, ours_mask), (half, theirs_mask)):
            tops = buckets[mask]
            sub_ys = ly[mask]
            sub_xs = lx[mask]
            for k in range(bucket_count):
                sub = tops >= k
                out[base + k, sub_ys[sub], sub_xs[sub]] = 1.0


def _build_bucket_masks(
    live_pids: np.ndarray,
    lseat: np.ndarray,
    belief: BeliefTensor,
    obs_team: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (ours_mask, theirs_mask) once for all 3 bucket groups.

    Enemy bucket data (move_count, active_eat, passive_surv) is derived
    from publicly broadcast MoveResult events — every player can count
    these.  No D-3 visibility filter is applied.
    """
    if live_pids.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty
    ours_mask = (_SEAT_TEAM[lseat] == obs_team)
    theirs_mask = ~ours_mask
    return ours_mask, theirs_mask


def _write_death_reason(
    out: np.ndarray,
    state: GameState,
    me_seat_val: int,
    teammate_seat_val: int,
    left_seat_val: int,
    right_seat_val: int,
) -> None:
    """(12, H, W): 3×4 death-reason planes, one group per seat perspective.

    Planes  0- 2: me        (observer's own pieces)
    Planes  3- 5: teammate
    Planes  6- 8: left-side enemy
    Planes  9-11: right-side enemy

    Each group has 3 planes for DeathReason: KILLED_BY_ENEMY / HIT_MINE_OR_BOMB / MUTUAL.
    Anchored at the world-frame death location.
    """
    if not state.deaths:
        return

    R = DEATH_REASON_COUNT     # = 3

    pids = np.fromiter(state.deaths.keys(), dtype=np.int64, count=len(state.deaths))
    if pids.size == 0:
        return
    reason_vals = state.death_reason_arr[pids]
    step_vals = state.death_step_arr[pids]
    seat_vals = state.piece_seat_arr[pids]
    flat = state.death_loc_flat_arr[pids]
    ys = (flat // BOARD_SIZE).astype(np.intp, copy=False)
    xs = (flat %  BOARD_SIZE).astype(np.intp, copy=False)

    reason_idx = _DEATH_REASON_TO_IDX[reason_vals.astype(np.int64, copy=False)]
    valid = (reason_idx >= 0) & (step_vals > 0)
    if not valid.all():
        reason_idx = reason_idx[valid]
        seat_vals = seat_vals[valid]
        xs = xs[valid]
        ys = ys[valid]

    # Map each dead piece to its seat group offset
    # me → 0, teammate → R, left → 2R, right → 3R
    base_offset = np.full_like(reason_idx, -1)
    base_offset[seat_vals == me_seat_val]       = 0
    base_offset[seat_vals == teammate_seat_val]  = R
    base_offset[seat_vals == left_seat_val]      = R * 2
    base_offset[seat_vals == right_seat_val]     = R * 3

    mask = base_offset >= 0
    ch = (base_offset[mask] + reason_idx[mask]).astype(np.intp)
    out[ch, ys[mask], xs[mask]] = 1.0


def _write_dead_at_zero(
    out: np.ndarray,
    state: GameState,
    obs_team: int,
) -> None:
    """(2, H, W): dead-at-zero mask (1 ours, 1 theirs), anchored at zero cell."""
    if not state.deaths:
        return

    pids = np.fromiter(state.deaths.keys(), dtype=np.int64, count=len(state.deaths))
    if pids.size == 0:
        return
    seat_vals = state.piece_seat_arr[pids]
    zx = state.zero_x[pids]
    zy = state.zero_y[pids]
    valid = (zx >= 0) & (zy >= 0)
    if not valid.all():
        seat_vals = seat_vals[valid]
        zx = zx[valid]
        zy = zy[valid]
    ours = (_SEAT_TEAM[seat_vals] == obs_team).astype(np.intp, copy=False)
    # ours -> plane 0, theirs -> plane 1.
    ch = (1 - ours)
    out[ch, zy.astype(np.intp, copy=False), zx.astype(np.intp, copy=False)] = 1.0


def _write_piece_id(
    out: np.ndarray,
    live_pids: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
) -> None:
    """(120, H, W): one-hot per live piece_id.

    Each live piece ``p`` writes 1.0 at ``out[p, ly, lx]`` where ``p`` is
    the piece_id (0..119).  Dead pieces contribute nothing (their channels
    remain zero).

    piece_id is public information: all players can observe *which cell* a
    piece occupies and track its identity across moves even though the
    piece TYPE is hidden.
    """
    if live_pids.size == 0:
        return
    out[live_pids, ly, lx] = 1.0


def _write_move_history(
    out: np.ndarray,
    state: GameState,
) -> None:
    """(32, H, W): src_dst_planes — last 32 moves.

    Channel delta=0 is the most recent move; delta=31 is the oldest.
    Source cell gets -1.0, destination cell gets +1.0 (world frame;
    the canonical rotation is applied by the caller afterwards).

    ``state.move_history`` is a list of ``(src_flat, dst_flat)`` tuples,
    most-recent-last, capped at 32 entries.
    """
    hist = state.move_history
    n = len(hist)
    if n == 0:
        return
    for delta in range(min(n, 32)):
        src_flat, dst_flat = hist[-(delta + 1)]  # most recent first
        src_x = src_flat % BOARD_SIZE
        src_y = src_flat // BOARD_SIZE
        dst_x = dst_flat % BOARD_SIZE
        dst_y = dst_flat // BOARD_SIZE
        out[delta, src_y, src_x] = -1.0
        out[delta, dst_y, dst_x] = 1.0


def _write_combat_memory(
    *,
    out_kill_mine_type: np.ndarray,        # (12, 17, 17)
    out_kill_mine_ge: np.ndarray,          # (3, 17, 17)
    out_kill_other_ge: np.ndarray,         # (3, 17, 17)
    out_chain_type: np.ndarray,            # (12, 17, 17)
    out_chain_ge: np.ndarray,              # (3, 17, 17)
    out_floor_ge: np.ndarray,              # (9, 17, 17)
    out_is_gongb: np.ndarray,              # (1, 17, 17)
    out_not_gongb: np.ndarray,             # (1, 17, 17)
    out_dilei_candidate: np.ndarray,       # (1, 17, 17)
    out_my_kill_count_ge: np.ndarray,      # (3, 17, 17)
    out_my_is_gongb: np.ndarray,           # (1, 17, 17)
    out_my_dilei_candidate: np.ndarray,    # (1, 17, 17)
    # Layer 3 (v5) — per-pid identity tail.
    out_kill_mine_count: np.ndarray,       # (12, 17, 17)
    out_kill_mine_slot: np.ndarray,        # (30, 17, 17)
    out_recency: np.ndarray,               # (4, 17, 17)
    # Layer 4 (v6) — reverse projection on observer's own pieces.
    out_eaten_by_pid: np.ndarray,          # (60, 17, 17)
    state: GameState,
    observer_val: int,
    live_pids: np.ndarray,
    lx: np.ndarray,
    ly: np.ndarray,
) -> None:
    """Project the v4 CombatMemoryState onto two layers of channels.

    Layer 1 (45 ch): ``state.combat_memory[observer]`` viewed at every
    enemy alive piece's current cell.

    Layer 2 (5 ch): theory-of-mind — what *opponents* know about my own
    alive pieces.  Computed by AND-aggregating the two opponent
    observers' state — this filters to public-only information (DARK
    rule: any single-opponent fact may be from their own-seat
    visibility, but AND of two opponents is path-revealed and chain
    knowledge that crossed both, which is necessarily public).

    All channels are binary (0/1).

    `dilei_candidate` is computed at *runtime*:
        candidate = alive AND
                    move_count == 0 AND
                    zero_pos in seat-back-two-rows AND
                    NOT attacked_by_known_gongb[observer]
    """
    cm = state.combat_memory
    if live_pids.size == 0:
        return

    obs_team = int(_SEAT_TEAM[observer_val])
    seats = state.piece_seat_arr[live_pids]
    enemy_mask = _SEAT_TEAM[seats] != obs_team
    mine_mask = seats == observer_val

    # ===========================================================================
    # Layer 1 — projected to enemy alive pieces
    # ===========================================================================
    if enemy_mask.any():
        epids = live_pids[enemy_mask]
        ex = lx[enemy_mask]
        ey = ly[enemy_mask]

        # ---- Slice CombatMemoryState rows for these enemy pids ----
        direct_lo = cm.direct_ate_my_pid_lo[observer_val, epids]
        direct_hi = cm.direct_ate_my_pid_hi[observer_val, epids]
        direct_tm = cm.direct_ate_my_type_mask[observer_val, epids]
        chain_lo  = cm.chain_pid_lo[observer_val, epids]
        chain_hi  = cm.chain_pid_hi[observer_val, epids]
        chain_tm  = cm.chain_ate_my_type_mask[observer_val, epids]
        other_cnt = cm.direct_other_count[observer_val, epids].astype(np.int32, copy=False)
        rfloor    = cm.rank_floor[observer_val, epids]
        is_gongb  = cm.is_gongb[observer_val, epids]
        not_gongb = cm.not_gongb[observer_val, epids]
        attacked_g = cm.attacked_by_known_gongb[observer_val, epids]

        # ---- kill_mine: count from pid bitmap, type from mask ----
        mine_count = _popcount_pid_pair(direct_lo, direct_hi).astype(np.int32)
        for k, thresh in enumerate((1, 2, 3)):
            sel = mine_count >= thresh
            if sel.any():
                ix = np.where(sel)[0]
                out_kill_mine_ge[k, ey[ix], ex[ix]] = 1.0
        for t in range(NUM_TRACKED_TYPES):
            bit = np.uint16(1 << t)
            sel = (direct_tm & bit).astype(bool)
            if sel.any():
                ix = np.where(sel)[0]
                out_kill_mine_type[t, ey[ix], ex[ix]] = 1.0

        # ---- kill_other: count threshold from direct_other_count ----
        for k, thresh in enumerate((1, 2, 3)):
            sel = other_cnt >= thresh
            if sel.any():
                ix = np.where(sel)[0]
                out_kill_other_ge[k, ey[ix], ex[ix]] = 1.0

        # ---- chain: count from full pid bitmap, type from mask ----
        chain_count = _popcount_pid_pair(chain_lo, chain_hi).astype(np.int32)
        for k, thresh in enumerate((1, 2, 3)):
            sel = chain_count >= thresh
            if sel.any():
                ix = np.where(sel)[0]
                out_chain_ge[k, ey[ix], ex[ix]] = 1.0
        for t in range(NUM_TRACKED_TYPES):
            bit = np.uint16(1 << t)
            sel = (chain_tm & bit).astype(bool)
            if sel.any():
                ix = np.where(sel)[0]
                out_chain_type[t, ey[ix], ex[ix]] = 1.0

        # ---- floor_ge: cumulative one-hot ----
        for g in range(9):
            thresh = g + 1
            sel = rfloor >= thresh
            if sel.any():
                ix = np.where(sel)[0]
                out_floor_ge[g, ey[ix], ex[ix]] = 1.0

        # ---- is_gongb / not_gongb (raw bool) ----
        if is_gongb.any():
            ix = np.where(is_gongb)[0]
            out_is_gongb[0, ey[ix], ex[ix]] = 1.0
        if not_gongb.any():
            ix = np.where(not_gongb)[0]
            out_not_gongb[0, ey[ix], ex[ix]] = 1.0

        # ---- dilei_candidate (runtime computed, binary) ----
        # Conditions:
        #   1) zero_pos is in this enemy's own back-two-rows  (initial pos)
        #   2) move_count_arr[pid] == 0  (mine never moved)
        #   3) NOT attacked_by_known_gongb[observer][pid]
        #
        # If a known-GONGB has visibly attacked it and it survived, it
        # is NOT a mine (D2 in design).  We rely on attacked_by_known_gongb
        # being set in apply_combat_event.
        zero_x = state.zero_x[epids]
        zero_y = state.zero_y[epids]
        mc = state.move_count_arr[epids]
        e_seats = seats[enemy_mask]
        # vectorized seat-back-two-rows test
        in_back = np.zeros(epids.shape, dtype=bool)
        for s_val, pos_axis, lo, hi in (
            (Seat.SOUTH.value, zero_y, 15, 16),
            (Seat.NORTH.value, zero_y, 0, 1),
            (Seat.WEST.value, zero_x, 0, 1),
            (Seat.EAST.value, zero_x, 15, 16),
        ):
            sel = (e_seats == s_val) & (pos_axis >= lo) & (pos_axis <= hi)
            in_back |= sel
        cand = in_back & (mc == 0) & (~attacked_g)
        if cand.any():
            ix = np.where(cand)[0]
            out_dilei_candidate[0, ey[ix], ex[ix]] = 1.0

    # ===========================================================================
    # Layer 2 — theory-of-mind, projected to my own alive pieces
    # ===========================================================================
    if mine_mask.any():
        mpids = live_pids[mine_mask]
        mx = lx[mine_mask]
        my = ly[mine_mask]

        # Two opponents (seat ± 1).  AND-aggregate to keep only public info.
        left_opp  = (observer_val + 1) % 4
        right_opp = (observer_val + 3) % 4

        l_direct_lo = cm.direct_ate_my_pid_lo[left_opp, mpids]
        l_direct_hi = cm.direct_ate_my_pid_hi[left_opp, mpids]
        r_direct_lo = cm.direct_ate_my_pid_lo[right_opp, mpids]
        r_direct_hi = cm.direct_ate_my_pid_hi[right_opp, mpids]
        # Counts per opponent: kill_mine_count + direct_other_count.
        l_mine_count = _popcount_pid_pair(l_direct_lo, l_direct_hi).astype(np.int32)
        r_mine_count = _popcount_pid_pair(r_direct_lo, r_direct_hi).astype(np.int32)
        l_other_cnt  = cm.direct_other_count[left_opp,  mpids].astype(np.int32, copy=False)
        r_other_cnt  = cm.direct_other_count[right_opp, mpids].astype(np.int32, copy=False)
        # AND aggregation = element-wise minimum on counts.
        public_kill_count = np.minimum(
            l_mine_count + l_other_cnt, r_mine_count + r_other_cnt
        )
        for k, thresh in enumerate((1, 2, 3)):
            sel = public_kill_count >= thresh
            if sel.any():
                ix = np.where(sel)[0]
                out_my_kill_count_ge[k, my[ix], mx[ix]] = 1.0

        # is_gongb (path-revealed iff BOTH opponents see it).
        l_isg = cm.is_gongb[left_opp,  mpids]
        r_isg = cm.is_gongb[right_opp, mpids]
        public_isg = l_isg & r_isg
        if public_isg.any():
            ix = np.where(public_isg)[0]
            out_my_is_gongb[0, my[ix], mx[ix]] = 1.0

        # dilei_candidate from opponents' AND.  Uses ``attacked_by_known_gongb``
        # — if either opponent has seen a known-GONGB attack on me, the AND
        # includes that fact (i.e. NOT both → not_attacked = OR).  But for
        # candidate we want "neither opponent knows of a GONGB attack".
        l_atk = cm.attacked_by_known_gongb[left_opp,  mpids]
        r_atk = cm.attacked_by_known_gongb[right_opp, mpids]
        public_attacked_by_gongb = l_atk | r_atk  # if any opponent saw it, info is public
        zero_x = state.zero_x[mpids]
        zero_y = state.zero_y[mpids]
        mc = state.move_count_arr[mpids]
        m_seats = seats[mine_mask]
        in_back = np.zeros(mpids.shape, dtype=bool)
        for s_val, lo, hi, axis in (
            (Seat.SOUTH.value, 15, 16, zero_y),
            (Seat.NORTH.value, 0, 1, zero_y),
            (Seat.WEST.value,  0, 1, zero_x),
            (Seat.EAST.value, 15, 16, zero_x),
        ):
            sel = (m_seats == s_val) & (axis >= lo) & (axis <= hi)
            in_back |= sel
        cand = in_back & (mc == 0) & (~public_attacked_by_gongb)
        if cand.any():
            ix = np.where(cand)[0]
            out_my_dilei_candidate[0, my[ix], mx[ix]] = 1.0

    # ===========================================================================
    # Layer 3 (ADR-129 v5) — per-pid identity tail.
    #
    # Goal: let the network distinguish "killer K ate observer's slot 7
    # piece" from "killer K ate observer's slot 12 piece".  The 12-bit
    # type multi-hot in Layer 1 alone hides this.
    #
    # All 46 channels are derived from the SAME state arrays as Layer 1
    # (no new SoA fields), so they cost nothing in memory and stay
    # byte-identical with the GPU mirror.
    #
    # Channels:
    #   cm_kill_mine_count[12]   per-type COUNT (popcount of direct-pid
    #                            bitmap restricted to victims of that
    #                            type), normalized to [0, 1] by /3
    #   cm_kill_mine_slot[30]    slot-i bit lit at enemy cell if the
    #                            observer's pid (obs_seat*30 + i) is set
    #                            in direct_ate_my_pid_lo/hi
    #   cm_recency[4]            sigmoid-fresh signals on
    #                            last_direct_step / last_chain_step /
    #                            rank_floor_step (rank-3 + chain-2 + dir-1)
    # ===========================================================================
    if enemy_mask.any():
        epids = live_pids[enemy_mask]
        ex = lx[enemy_mask]
        ey = ly[enemy_mask]

        # Re-fetch direct bitmaps for these enemy pids (Layer 1 already
        # read them; we reread because the local variables there are
        # gated by the same enemy_mask block but Python doesn't keep
        # them across the Layer 2 block scoping cleanly).
        direct_lo = cm.direct_ate_my_pid_lo[observer_val, epids]
        direct_hi = cm.direct_ate_my_pid_hi[observer_val, epids]

        # ---- Layer 3.1: cm_kill_mine_count[12] ----
        # For each tracked-type t, compute popcount of (direct_lo, direct_hi)
        # restricted to observer's pids of type t.  We build the per-type
        # mask from state.piece_type_arr restricted to observer's 30 pids.
        obs_pid_lo = observer_val * 30
        obs_pid_hi_excl = obs_pid_lo + 30  # exclusive upper bound
        my_types = state.piece_type_arr[obs_pid_lo:obs_pid_hi_excl]      # (30,)
        # Tracked-type indices for each of the 30 observer pids.
        my_type_idx = _PIECETYPE_TO_TRACKED_IDX[my_types]                 # (30,) int8
        for t in range(NUM_TRACKED_TYPES):
            # Build pid mask: 1 << (pid_global - 0) for pid_global in
            # observer's pid range whose type == t.  All observer pids
            # are < 64 except observer_val=2 (NORTH: pids 60..89) and
            # observer_val=3 (EAST: pids 90..119) which spill into HI.
            slot_indices = np.where(my_type_idx == t)[0]                  # local 0..29 indices
            if slot_indices.size == 0:
                continue
            global_pids = (obs_pid_lo + slot_indices).astype(np.int64)
            t_mask_lo = np.uint64(0)
            t_mask_hi = np.uint64(0)
            for gp in global_pids.tolist():
                if gp < 64:
                    t_mask_lo |= np.uint64(1) << np.uint64(gp)
                else:
                    t_mask_hi |= np.uint64(1) << np.uint64(gp - 64)
            cnt_per_pid = _popcount_pid_pair(
                direct_lo & t_mask_lo, direct_hi & t_mask_hi
            ).astype(np.int32)                                            # (Nenemy,)
            sel = cnt_per_pid > 0
            if sel.any():
                ix = np.where(sel)[0]
                # Normalize to [0, 1] by /3 (saturating).
                vals = np.minimum(cnt_per_pid[ix].astype(np.float32) / 3.0, 1.0)
                out_kill_mine_count[t, ey[ix], ex[ix]] = vals

        # ---- Layer 3.2: cm_kill_mine_slot[30] ----
        # Slot bit s: lit iff direct-pid bitmap covers observer's
        # global pid (obs_pid_lo + s).
        for s in range(30):
            gp = obs_pid_lo + s
            if gp < 64:
                bit = np.uint64(1) << np.uint64(gp)
                sel = (direct_lo & bit) != np.uint64(0)
            else:
                bit = np.uint64(1) << np.uint64(gp - 64)
                sel = (direct_hi & bit) != np.uint64(0)
            if sel.any():
                ix = np.where(sel)[0]
                out_kill_mine_slot[s, ey[ix], ex[ix]] = 1.0

        # ---- Layer 3.3: cm_recency[4] ----
        # Tau values (steps).  Larger tau decays slower.  We use
        # exponential: fresh = exp(-elapsed/tau) clamped to [0, 1].
        # Plane order:
        #   0: direct event, tau=32   ("did this killer eat me recently?")
        #   1: direct event, tau=256  ("ever seen direct activity?")
        #   2: chain event,  tau=64
        #   3: rank-floor lift, tau=128
        cur_step = int(state.move_counter)
        last_direct = cm.last_direct_step[observer_val, epids].astype(np.int32, copy=False)
        last_chain  = cm.last_chain_step[observer_val, epids].astype(np.int32, copy=False)
        last_floor  = cm.rank_floor_step[observer_val, epids].astype(np.int32, copy=False)
        for plane_idx, (last_step, tau) in enumerate((
            (last_direct, 32.0),
            (last_direct, 256.0),
            (last_chain,  64.0),
            (last_floor,  128.0),
        )):
            valid = last_step >= 0
            if not valid.any():
                continue
            elapsed = (cur_step - last_step).astype(np.float32)
            fresh = np.exp(-np.maximum(elapsed, 0.0) / tau, dtype=np.float32)
            fresh = np.where(valid, fresh, 0.0).astype(np.float32)
            sel = fresh > 0.0
            if sel.any():
                ix = np.where(sel)[0]
                out_recency[plane_idx, ey[ix], ex[ix]] = fresh[ix]

    # ===========================================================================
    # Layer 4 (v6) — reverse projection on observer's own pieces.
    #
    # For each of observer's 30 own pids (alive or dead), look up
    # ``cm.eaten_by_pid[observer, mpid]`` (a 120-bit map of which pids
    # have directly OR chain-eaten mpid).  Restrict to the 60 enemy
    # pids (left_opp + right_opp).  Project onto:
    #   * mpid's current cell if alive (alive ⇒ never been eaten ⇒ all
    #     bits are 0; signal is informative on freshly-revived edge
    #     cases — currently always 0 by construction);
    #   * mpid's zero_pos cell if dead (the home square).
    #
    # Channels:
    #   0..29  = (observer+1)%4 left enemy's slot 0..29
    #   30..59 = (observer+3)%4 right enemy's slot 0..29
    #
    # DARK boundary: ``cm.eaten_by_pid`` is only ever populated by
    # combat_memory.apply_combat_event for observer's own mpids; for
    # other seats the entire row is zero, so even if we read it we
    # would get 0 — but we explicitly restrict mpids to observer's
    # range below for clarity.
    # ===========================================================================
    obs_pid_lo_g = observer_val * 30
    my_pids_global = np.arange(obs_pid_lo_g, obs_pid_lo_g + 30, dtype=np.int64)
    # Compute projection cells: alive → current pos; dead → zero_pos.
    alive_mask = state.alive[my_pids_global]
    px = np.where(alive_mask, state.pos_x[my_pids_global], state.zero_x[my_pids_global])
    py = np.where(alive_mask, state.pos_y[my_pids_global], state.zero_y[my_pids_global])
    # Filter out pids whose cell is invalid (e.g. unassigned slot in
    # partially-filled boards or sentinel -1).
    valid = (px >= 0) & (py >= 0)
    if valid.any():
        v_idx = np.where(valid)[0]
        e_lo = cm.eaten_by_pid_lo[observer_val, my_pids_global[v_idx]]
        e_hi = cm.eaten_by_pid_hi[observer_val, my_pids_global[v_idx]]
        vx = px[v_idx].astype(np.intp, copy=False)
        vy = py[v_idx].astype(np.intp, copy=False)
        left_opp  = (observer_val + 1) % 4
        right_opp = (observer_val + 3) % 4
        for ch_offset, opp_seat in ((0, left_opp), (30, right_opp)):
            opp_pid_lo = opp_seat * 30
            for s in range(30):
                gp = opp_pid_lo + s
                if gp < 64:
                    bit = np.uint64(1) << np.uint64(gp)
                    sel = (e_lo & bit) != np.uint64(0)
                else:
                    bit = np.uint64(1) << np.uint64(gp - 64)
                    sel = (e_hi & bit) != np.uint64(0)
                if sel.any():
                    ix = np.where(sel)[0]
                    out_eaten_by_pid[ch_offset + s, vy[ix], vx[ix]] = 1.0


def _write_global_features(
    out: np.ndarray,
    state: GameState,
    belief: BeliefTensor,
    order_vals: tuple[int, int, int, int],
) -> None:
    """(28,): enemy inventories + flag-reveal scalars.

    ``order_vals`` = ``(me, teammate, left_side_enemy, right_side_enemy)``
    pre-computed by :meth:`ObservationBuilder.build`.

    Phase 0.4 M3 (ADR-120): vectorized.  Reads
    ``belief.remaining_arr[(4, 12)]`` directly (one slice per side) and
    ``state.seat_flag_revealed_arr`` for the 4 flag-reveal scalars.
    """
    left_val = order_vals[2]
    right_val = order_vals[3]

    left_base = GLOBAL_LAYOUT["remaining_left_side"].start
    right_base = GLOBAL_LAYOUT["remaining_right_side"].start
    out[left_base:left_base + NUM_TRACKED_TYPES] = belief.remaining_arr[left_val]
    out[right_base:right_base + NUM_TRACKED_TYPES] = belief.remaining_arr[right_val]

    flag_base = GLOBAL_LAYOUT["flag_revealed"].start
    fr = state.seat_flag_revealed_arr
    out[flag_base + 0] = float(fr[order_vals[0]])
    out[flag_base + 1] = float(fr[order_vals[1]])
    out[flag_base + 2] = float(fr[left_val])
    out[flag_base + 3] = float(fr[right_val])


# ===========================================================================
# Utilities
# ===========================================================================


def _type_to_idx(pt: PieceType) -> int | None:
    """Map PieceType to 12-wide channel index, or None for NONE/DARK.

    Kept for backward-compat (test modules import it).
    """
    return _TYPE_TO_IDX_MAP.get(pt)


def _observer_sorted_seats(observer: Seat) -> tuple[Seat, Seat, Seat, Seat]:
    """(me, teammate, left_side_enemy, right_side_enemy)."""
    return (
        observer,
        observer.teammate,
        observer.left_side_enemy,
        observer.right_side_enemy,
    )


def channel_name(channel_idx: int) -> str:
    """Return a human-readable name for a spatial-channel index."""
    if not 0 <= channel_idx < OBS_CHANNELS:
        raise IndexError(f"channel_idx {channel_idx} out of [0, {OBS_CHANNELS})")
    for name, sl in CHANNEL_LAYOUT.items():
        if sl.start <= channel_idx < sl.stop:
            return f"{name}[{channel_idx - sl.start}]"
    raise AssertionError(f"unreachable: channel {channel_idx} not in layout")


# ===========================================================================
# Torch bridge (ADR-124, Phase 0.4 M5)
# ===========================================================================
#
# A PyTorch CPU tensor and its ``numpy()`` view share the same underlying
# storage: writing into the numpy array writes into the tensor's memory.
# We exploit that to let the vectorized SoA writers above land data
# directly in a tensor that can then be shipped to GPU with a single
# ``.to(device, non_blocking=True)`` (a DMA transfer).
#
# Why not just pass a torch tensor to ``build_observations_batch``?  The
# writers use numpy fancy-indexing (``out[idx, ys, xs] = ...``) and
# boolean masks; torch's index rules diverge in subtle ways.  The clean
# separation is: the tensor provides the memory, we wrap it in a numpy
# array (zero-copy) for the duration of the write, and torch sees the
# final values when we return.
#
# This is ADR-124's exact contract.  On CUDA we would need a
# ``__cuda_array_interface__``-based bridge; that is ADR-F3 and comes
# after Phase 1's batched CPU env lands.


def numpy_view_of_torch_cpu(tensor: Any) -> np.ndarray:
    """Return a numpy view sharing storage with a CPU torch tensor.

    Kept import-light: we do NOT unconditionally import torch at module
    load time (torch is a heavy optional dep).  Callers that opt into the
    bridge have torch installed; callers that only use numpy paths never
    pay the import cost.
    """
    import torch  # type: ignore[import-not-found]  # optional runtime dependency
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(tensor).__name__}")
    if tensor.device.type != "cpu":
        raise ValueError(
            f"torch bridge requires CPU tensors; got device={tensor.device}. "
            "Stage on CPU and use tensor.to('cuda', non_blocking=True) to "
            "transfer."
        )
    if tensor.dtype != torch.float32:
        raise TypeError(
            f"torch bridge requires float32 tensors; got {tensor.dtype}"
        )
    if not tensor.is_contiguous():
        raise ValueError(
            "torch bridge requires a contiguous tensor; call .contiguous() "
            "before binding"
        )
    return np.asarray(tensor.numpy())  # zero-copy; shares storage


# Monkey-patch builder methods onto ObservationBuilder below so we do
# not clutter the main class body.  Kept here to keep the torch-specific
# code isolated from the numpy hot path above.
def _builder_build_observations_batch_torch(
    self: ObservationBuilder,
    states: list[GameState] | tuple[GameState, ...],
    beliefs: list[BeliefTensor] | tuple[BeliefTensor, ...],
    observers: list[Seat] | tuple[Seat, ...],
    spatial_tensor: Any,  # torch.Tensor (N, OBS_CHANNELS, 17, 17) float32 CPU
    global_tensor: Any,   # torch.Tensor (N, OBS_GLOBAL_DIMS)       float32 CPU
) -> None:
    """Write a batch of observations directly into CPU torch tensors.

    The two tensors must already be sized and CPU-resident; this call
    does not allocate anything.  On return, the caller typically does
    ``spatial_tensor.to(device, non_blocking=True)`` to ship to GPU
    without an additional host-side copy.

    See :meth:`build_observations_batch` for the non-torch signature.
    """
    sp_np = numpy_view_of_torch_cpu(spatial_tensor)
    gl_np = numpy_view_of_torch_cpu(global_tensor)
    self.build_observations_batch(states, beliefs, observers, sp_np, gl_np)


# Attach as method.
ObservationBuilder.build_observations_batch_torch = (  # type: ignore[attr-defined]
    _builder_build_observations_batch_torch
)


# ===========================================================================
# Module self-check: runs at import to catch layout drift early.
# ===========================================================================


def _self_check() -> None:
    """Verify layout invariants at import time."""
    assert OBS_CHANNELS == 412
    assert OBS_GLOBAL_DIMS == 28
    assert CHANNEL_LAYOUT["move_bucket"].stop - CHANNEL_LAYOUT["move_bucket"].start == 8
    assert CHANNEL_LAYOUT["active_eat_bucket"].stop - CHANNEL_LAYOUT["active_eat_bucket"].start == 8
    assert CHANNEL_LAYOUT["passive_survive_bucket"].stop - CHANNEL_LAYOUT["passive_survive_bucket"].start == 8
    assert CHANNEL_LAYOUT["death_reason"].stop - CHANNEL_LAYOUT["death_reason"].start == 12
    assert CHANNEL_LAYOUT["dead_at_zero"].stop - CHANNEL_LAYOUT["dead_at_zero"].start == 2
    assert CHANNEL_LAYOUT["piece_id"].stop - CHANNEL_LAYOUT["piece_id"].start == 120
    assert CHANNEL_LAYOUT["move_history"].stop - CHANNEL_LAYOUT["move_history"].start == 32
    # ---- CombatMemory v4 tail (50 channels) ----
    assert CHANNEL_LAYOUT["cm_kill_mine_type"].stop - CHANNEL_LAYOUT["cm_kill_mine_type"].start == 12
    assert CHANNEL_LAYOUT["cm_kill_mine_ge"].stop - CHANNEL_LAYOUT["cm_kill_mine_ge"].start == 3
    assert CHANNEL_LAYOUT["cm_kill_other_ge"].stop - CHANNEL_LAYOUT["cm_kill_other_ge"].start == 3
    assert CHANNEL_LAYOUT["cm_chain_type"].stop - CHANNEL_LAYOUT["cm_chain_type"].start == 12
    assert CHANNEL_LAYOUT["cm_chain_ge"].stop - CHANNEL_LAYOUT["cm_chain_ge"].start == 3
    assert CHANNEL_LAYOUT["cm_floor_ge"].stop - CHANNEL_LAYOUT["cm_floor_ge"].start == 9
    assert CHANNEL_LAYOUT["cm_is_gongb"].stop - CHANNEL_LAYOUT["cm_is_gongb"].start == 1
    assert CHANNEL_LAYOUT["cm_not_gongb"].stop - CHANNEL_LAYOUT["cm_not_gongb"].start == 1
    assert CHANNEL_LAYOUT["cm_dilei_candidate"].stop - CHANNEL_LAYOUT["cm_dilei_candidate"].start == 1
    assert CHANNEL_LAYOUT["cm_my_kill_count_ge"].stop - CHANNEL_LAYOUT["cm_my_kill_count_ge"].start == 3
    assert CHANNEL_LAYOUT["cm_my_is_gongb"].stop - CHANNEL_LAYOUT["cm_my_is_gongb"].start == 1
    assert CHANNEL_LAYOUT["cm_my_dilei_candidate"].stop - CHANNEL_LAYOUT["cm_my_dilei_candidate"].start == 1
    # ---- CombatMemory v5 layer-3 (46 channels) ----
    assert CHANNEL_LAYOUT["cm_kill_mine_count"].stop - CHANNEL_LAYOUT["cm_kill_mine_count"].start == 12
    assert CHANNEL_LAYOUT["cm_kill_mine_slot"].stop - CHANNEL_LAYOUT["cm_kill_mine_slot"].start == 30
    assert CHANNEL_LAYOUT["cm_recency"].stop - CHANNEL_LAYOUT["cm_recency"].start == 4
    # ---- CombatMemory v6 layer-4 (60 channels) ----
    assert CHANNEL_LAYOUT["cm_eaten_by_pid"].stop - CHANNEL_LAYOUT["cm_eaten_by_pid"].start == 60
    # Sanity: pre-CombatMemory channels [0, 256) preserved bit-for-bit.
    assert CHANNEL_LAYOUT["move_history"].stop == 256
    running = 0
    for _name, sl in CHANNEL_LAYOUT.items():
        assert sl.start == running
        running = sl.stop
    assert running == OBS_CHANNELS
    running = 0
    for _name, sl in GLOBAL_LAYOUT.items():
        assert sl.start == running
        running = sl.stop
    assert running == OBS_GLOBAL_DIMS
    assert _BOARD_STATIC_WORLD.shape == (6, BOARD_SIZE, BOARD_SIZE)


_self_check()


if __name__ == "__main__":
    _self_check()
    print(
        f"junqi_core.observation self-check: OK "
        f"(channels={OBS_CHANNELS}, global={OBS_GLOBAL_DIMS})"
    )
    print("Channel layout:")
    for name, sl in CHANNEL_LAYOUT.items():
        print(f"  [{sl.start:3d}..{sl.stop:3d})  {name}")
