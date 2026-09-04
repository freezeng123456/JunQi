"""Batched CPU game state for Phase 1a (M1) of the CUDA migration.

:class:`BatchedGameState` holds N copies of the SoA game state with a leading
batch dimension, enabling vectorized NumPy operations across environments
without per-environment Python loops in the hot-path.

Milestone targets (ADR-123, Phase 1a M1):
  - ``step_batch``            : ≥50 000 plays/sec aggregate at N=1024
  - ``legal_action_ids_batch``: ≥500 000 actions/sec aggregate at N=1024
  - ``to_game_states``        : round-trip fidelity (test parity)

Design notes
------------
* All SoA arrays have shape ``(N, K)`` where K is the per-env axis size
  (120 for piece-indexed, 289 for cell-indexed, 4 for seat-indexed).
* Scalar fields (``turn``, ``zobrist``, ``move_counter``, etc.) are 1-D
  arrays of shape ``(N,)`` to keep the batching interface uniform.
* ``step_batch`` is a pure-NumPy implementation that vectorises the
  move / combat / kill / seat-death logic over the N environments in a
  single pass, using fancy-indexing to avoid explicit Python loops.
* Game states that have already terminated are silently skipped in
  ``step_batch``; their SoA arrays remain frozen.
* This module is intentionally **CPU-only**.  The CUDA kernels in
  ``src/env/cuda/`` replicate the same logic on GPU (Phase M2-M4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .move_gen import PieceRef
    from .state import GameState

from . import _zobrist as _ZOB
from .board import BOARD_SIZE, NUM_CELLS
from .combat_memory import (
    CombatMemoryState,
    apply_combat_event,
    apply_path_revealed_gongb,
)
from .move_gen import (
    _TEAM_OF_PID,
    generate_legal_action_ids_n,
    has_legal_moves_soa,
    move_requires_gongb,
)
from .rules import (
    MAX_NUM_MOVES,
    MAX_NUM_MOVES_BETWEEN_ATTACKS,
    DeathReason,
    Event,
    PieceType,
    Seat,
    classify_death_reason,
    resolve_combat,
    siling_reveals_dst,
    siling_reveals_src,
)


def _build_cell_team_arr(
    cell_piece_id: np.ndarray,  # (N, 289) int16
) -> np.ndarray:
    """Build (N, 289) uint8 array where 255=empty, 0=team0, 1=team1.

    Team is derived from piece id alone: pids 0-29, 60-89 → team 0;
    pids 30-59, 90-119 → team 1.  This matches _TEAM_OF_PID in move_gen.
    """
    N = cell_piece_id.shape[0]
    out = np.full((N, 289), 255, dtype=np.uint8)
    occ = cell_piece_id >= 0                              # (N, 289)
    pid_clipped = np.clip(cell_piece_id, 0, 119).astype(np.intp)  # safe index
    # Fancy-index: for each (n, c) where occ, team = _TEAM_OF_PID[pid]
    from .move_gen import _TEAM_OF_PID as _TP
    out[occ] = _TP[pid_clipped[occ]]
    return out

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PIECES: int = _ZOB.NUM_PIECE_IDS   # 120
NUM_PIECE_TYPES: int = _ZOB.NUM_PIECE_TYPES  # 14


# ---------------------------------------------------------------------------
# MoveResultBatch — lightweight per-env move outcomes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MoveResultBatch:
    """Batch of per-environment step outcomes (arrays of shape ``(N,)``).

    All arrays are indexed by environment index.  Entries are only meaningful
    when ``valid[i]`` is True (i.e., env i was not terminated and the action
    was applied).
    """
    valid: np.ndarray        # (N,) bool — was the step applied?
    event: np.ndarray        # (N,) int8 — Event value (0 = invalid/skipped)
    terminated: np.ndarray   # (N,) bool
    winner_team: np.ndarray  # (N,) int8 — -1 if no winner
    draw: np.ndarray         # (N,) bool
    flag_captured: np.ndarray  # (N,) bool


# ---------------------------------------------------------------------------
# BatchedGameState
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BatchedGameState:
    """N independent game states packed into SoA arrays with batch dimension.

    All piece-indexed arrays have shape ``(num_envs, 120)``.
    Cell-indexed arrays have shape ``(num_envs, 289)``.
    Seat-indexed arrays have shape ``(num_envs, 4)``.
    Scalar-per-env arrays have shape ``(num_envs,)``.

    Notes
    -----
    Sentinel values follow the single-env convention: -1 for "no piece" /
    "not dead yet" / unassigned.  The batch dim is axis 0 throughout.
    """

    num_envs: int

    # ------------------------------------------------------------------
    # Piece-indexed SoA columns   shape: (num_envs, NUM_PIECES)
    # ------------------------------------------------------------------
    alive:              np.ndarray = field(repr=False)  # bool
    piece_type_arr:     np.ndarray = field(repr=False)  # int8
    piece_seat_arr:     np.ndarray = field(repr=False)  # int8
    pos_x:              np.ndarray = field(repr=False)  # int8
    pos_y:              np.ndarray = field(repr=False)  # int8
    zero_x:             np.ndarray = field(repr=False)  # int8
    zero_y:             np.ndarray = field(repr=False)  # int8
    move_count_arr:     np.ndarray = field(repr=False)  # int16
    active_eat_arr:     np.ndarray = field(repr=False)  # int16
    passive_surv_arr:   np.ndarray = field(repr=False)  # int16
    death_reason_arr:   np.ndarray = field(repr=False)  # int8
    death_step_arr:     np.ndarray = field(repr=False)  # int16
    death_loc_flat_arr: np.ndarray = field(repr=False)  # int16

    # ------------------------------------------------------------------
    # Cell-indexed SoA columns   shape: (num_envs, NUM_CELLS)
    # ------------------------------------------------------------------
    cell_piece_id:  np.ndarray = field(repr=False)       # int16
    cell_team_arr:  np.ndarray = field(repr=False)       # uint8 — 255=empty, 0=team0, 1=team1

    # ------------------------------------------------------------------
    # Seat-indexed SoA columns   shape: (num_envs, 4)
    # ------------------------------------------------------------------
    seat_dead_arr:          np.ndarray = field(repr=False)   # bool
    seat_flag_revealed_arr: np.ndarray = field(repr=False)   # bool

    # ------------------------------------------------------------------
    # Scalar-per-env columns   shape: (num_envs,)
    # ------------------------------------------------------------------
    turn:                   np.ndarray = field(repr=False)   # int8  (Seat.value)
    zobrist:                np.ndarray = field(repr=False)   # int64
    move_counter:           np.ndarray = field(repr=False)   # int32
    moves_since_last_combat: np.ndarray = field(repr=False)  # int32
    terminated:             np.ndarray = field(repr=False)   # bool
    winner_team:            np.ndarray = field(repr=False)   # int8  (-1 = none)
    draw:                   np.ndarray = field(repr=False)   # bool

    # ------------------------------------------------------------------
    # CombatMemory v4 SoA columns   shape: (num_envs, 4 observers, 120 pids)
    # ------------------------------------------------------------------
    # Mirrors :class:`junqi_core.combat_memory.CombatMemoryState` (v4) but
    # with a leading batch axis.  Maintained in ``_step_single`` (combat
    # branch) via a per-env CombatMemoryState view.
    cm_direct_lo:                 np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint64))
    cm_direct_hi:                 np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint64))
    cm_direct_type:               np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint16))
    cm_last_direct_step:          np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int16))
    cm_direct_other_count:        np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int16))
    cm_chain_lo:                  np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint64))
    cm_chain_hi:                  np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint64))
    cm_chain_type:                np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint16))
    cm_last_chain_step:           np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int16))
    cm_rank_floor:                np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int8))
    cm_rank_floor_step:           np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.int16))
    cm_is_gongb:                  np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=bool))
    cm_not_gongb:                 np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=bool))
    cm_attacked_by_known_gongb:   np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=bool))
    # CombatMemory v6 reverse projection (eaten_by_pid).
    cm_eaten_by_pid_lo:           np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint64))
    cm_eaten_by_pid_hi:           np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=np.uint64))

    # ==================================================================
    # Factory constructors
    # ==================================================================

    @classmethod
    def from_game_states(
        cls, states: Sequence[GameState]
    ) -> BatchedGameState:
        """Build a BatchedGameState by stacking a list of GameState objects.

        Parameters
        ----------
        states:
            A sequence of :class:`junqi_core.state.GameState` instances.
            All must have fully initialised SoA columns (e.g. produced by
            ``GameState.new_game(...)``).

        Returns
        -------
        BatchedGameState
        """
        N = len(states)
        if N == 0:
            raise ValueError("states must be non-empty")

        def _stack(attr: str, dtype: type) -> np.ndarray:
            return np.stack([getattr(s, attr) for s in states], axis=0).astype(dtype, copy=False)

        return cls(
            num_envs=N,
            # piece-indexed
            alive=_stack("alive", bool),
            piece_type_arr=_stack("piece_type_arr", np.int8),
            piece_seat_arr=_stack("piece_seat_arr", np.int8),
            pos_x=_stack("pos_x", np.int8),
            pos_y=_stack("pos_y", np.int8),
            zero_x=_stack("zero_x", np.int8),
            zero_y=_stack("zero_y", np.int8),
            move_count_arr=_stack("move_count_arr", np.int16),
            active_eat_arr=_stack("active_eat_arr", np.int16),
            passive_surv_arr=_stack("passive_surv_arr", np.int16),
            death_reason_arr=_stack("death_reason_arr", np.int8),
            death_step_arr=_stack("death_step_arr", np.int16),
            death_loc_flat_arr=_stack("death_loc_flat_arr", np.int16),
            # cell-indexed
            cell_piece_id=_stack("cell_piece_id", np.int16),
            cell_team_arr=_build_cell_team_arr(
                np.stack([s.cell_piece_id for s in states], axis=0).astype(np.int16, copy=False)
            ),
            # seat-indexed
            seat_dead_arr=_stack("seat_dead_arr", bool),
            seat_flag_revealed_arr=_stack("seat_flag_revealed_arr", bool),
            # scalars
            turn=np.array([s.turn.value for s in states], dtype=np.int8),
            zobrist=np.array([s.zobrist for s in states], dtype=np.int64),
            move_counter=np.array([s.move_counter for s in states], dtype=np.int32),
            moves_since_last_combat=np.array(
                [s.moves_since_last_combat for s in states], dtype=np.int32
            ),
            terminated=np.array([s.terminated for s in states], dtype=bool),
            winner_team=np.array(
                [-1 if s.winner_team is None else s.winner_team for s in states],
                dtype=np.int8,
            ),
            draw=np.array([s.draw for s in states], dtype=bool),
            # CombatMemory v4: stack each per-state CombatMemoryState into (N, 4, 120).
            cm_direct_lo=np.stack(
                [s.combat_memory.direct_ate_my_pid_lo for s in states], axis=0
            ).astype(np.uint64, copy=False),
            cm_direct_hi=np.stack(
                [s.combat_memory.direct_ate_my_pid_hi for s in states], axis=0
            ).astype(np.uint64, copy=False),
            cm_direct_type=np.stack(
                [s.combat_memory.direct_ate_my_type_mask for s in states], axis=0
            ).astype(np.uint16, copy=False),
            cm_last_direct_step=np.stack(
                [s.combat_memory.last_direct_step for s in states], axis=0
            ).astype(np.int16, copy=False),
            cm_direct_other_count=np.stack(
                [s.combat_memory.direct_other_count for s in states], axis=0
            ).astype(np.int16, copy=False),
            cm_chain_lo=np.stack(
                [s.combat_memory.chain_pid_lo for s in states], axis=0
            ).astype(np.uint64, copy=False),
            cm_chain_hi=np.stack(
                [s.combat_memory.chain_pid_hi for s in states], axis=0
            ).astype(np.uint64, copy=False),
            cm_chain_type=np.stack(
                [s.combat_memory.chain_ate_my_type_mask for s in states], axis=0
            ).astype(np.uint16, copy=False),
            cm_last_chain_step=np.stack(
                [s.combat_memory.last_chain_step for s in states], axis=0
            ).astype(np.int16, copy=False),
            cm_rank_floor=np.stack(
                [s.combat_memory.rank_floor for s in states], axis=0
            ).astype(np.int8, copy=False),
            cm_rank_floor_step=np.stack(
                [s.combat_memory.rank_floor_step for s in states], axis=0
            ).astype(np.int16, copy=False),
            cm_is_gongb=np.stack(
                [s.combat_memory.is_gongb for s in states], axis=0
            ).astype(bool, copy=False),
            cm_not_gongb=np.stack(
                [s.combat_memory.not_gongb for s in states], axis=0
            ).astype(bool, copy=False),
            cm_attacked_by_known_gongb=np.stack(
                [s.combat_memory.attacked_by_known_gongb for s in states], axis=0
            ).astype(bool, copy=False),
            cm_eaten_by_pid_lo=np.stack(
                [s.combat_memory.eaten_by_pid_lo for s in states], axis=0
            ).astype(np.uint64, copy=False),
            cm_eaten_by_pid_hi=np.stack(
                [s.combat_memory.eaten_by_pid_hi for s in states], axis=0
            ).astype(np.uint64, copy=False),
        )

    @classmethod
    def allocate(cls, num_envs: int) -> BatchedGameState:
        """Allocate zero-initialised arrays for ``num_envs`` environments.

        Useful for pre-allocating the struct and then filling via
        ``copy_from_game_states``.
        """
        N = num_envs
        P = NUM_PIECES
        C = NUM_CELLS
        return cls(
            num_envs=N,
            alive=np.zeros((N, P), dtype=bool),
            piece_type_arr=np.full((N, P), -1, dtype=np.int8),
            piece_seat_arr=np.full((N, P), -1, dtype=np.int8),
            pos_x=np.full((N, P), -1, dtype=np.int8),
            pos_y=np.full((N, P), -1, dtype=np.int8),
            zero_x=np.full((N, P), -1, dtype=np.int8),
            zero_y=np.full((N, P), -1, dtype=np.int8),
            move_count_arr=np.zeros((N, P), dtype=np.int16),
            active_eat_arr=np.zeros((N, P), dtype=np.int16),
            passive_surv_arr=np.zeros((N, P), dtype=np.int16),
            death_reason_arr=np.full((N, P), -1, dtype=np.int8),
            death_step_arr=np.full((N, P), -1, dtype=np.int16),
            death_loc_flat_arr=np.full((N, P), -1, dtype=np.int16),
            cell_piece_id=np.full((N, C), -1, dtype=np.int16),
            cell_team_arr=np.full((N, C), 255, dtype=np.uint8),
            seat_dead_arr=np.zeros((N, 4), dtype=bool),
            seat_flag_revealed_arr=np.zeros((N, 4), dtype=bool),
            turn=np.zeros(N, dtype=np.int8),
            zobrist=np.zeros(N, dtype=np.int64),
            move_counter=np.zeros(N, dtype=np.int32),
            moves_since_last_combat=np.zeros(N, dtype=np.int32),
            terminated=np.zeros(N, dtype=bool),
            winner_team=np.full(N, -1, dtype=np.int8),
            draw=np.zeros(N, dtype=bool),
            cm_direct_lo=np.zeros((N, 4, 120), dtype=np.uint64),
            cm_direct_hi=np.zeros((N, 4, 120), dtype=np.uint64),
            cm_direct_type=np.zeros((N, 4, 120), dtype=np.uint16),
            cm_last_direct_step=np.full((N, 4, 120), -1, dtype=np.int16),
            cm_direct_other_count=np.zeros((N, 4, 120), dtype=np.int16),
            cm_chain_lo=np.zeros((N, 4, 120), dtype=np.uint64),
            cm_chain_hi=np.zeros((N, 4, 120), dtype=np.uint64),
            cm_chain_type=np.zeros((N, 4, 120), dtype=np.uint16),
            cm_last_chain_step=np.full((N, 4, 120), -1, dtype=np.int16),
            cm_rank_floor=np.zeros((N, 4, 120), dtype=np.int8),
            cm_rank_floor_step=np.full((N, 4, 120), -1, dtype=np.int16),
            cm_is_gongb=np.zeros((N, 4, 120), dtype=bool),
            cm_not_gongb=np.zeros((N, 4, 120), dtype=bool),
            cm_attacked_by_known_gongb=np.zeros((N, 4, 120), dtype=bool),
            cm_eaten_by_pid_lo=np.zeros((N, 4, 120), dtype=np.uint64),
            cm_eaten_by_pid_hi=np.zeros((N, 4, 120), dtype=np.uint64),
        )

    # ==================================================================
    # Batch move generation
    # ==================================================================

    def legal_action_ids_batch(
        self,
        seat_per_env: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Return legal action ids for each environment.

        Parameters
        ----------
        seat_per_env:
            Integer array of shape ``(num_envs,)`` specifying which seat to
            generate actions for in each environment.  Defaults to
            ``self.turn`` (each env's current acting seat).

        Returns
        -------
        list of np.ndarray[int32]:
            One array per environment.  Terminated environments return an
            empty array.  Arrays may have different lengths (ragged output).
        """
        if seat_per_env is None:
            seat_per_env = self.turn
        else:
            seat_per_env = np.asarray(seat_per_env, dtype=np.int8)

        # Build a per-env skip mask: terminated OR acting seat is already dead.
        # seat_dead_arr shape: (N, 4) — index with per-env seat value.
        n_idx = np.arange(self.num_envs, dtype=np.intp)
        sv_idx = seat_per_env.astype(np.intp)
        dead_acting = self.seat_dead_arr[n_idx, sv_idx]
        skip = self.terminated | dead_acting

        return generate_legal_action_ids_n(
            self.cell_piece_id,
            self.piece_seat_arr,
            self.piece_type_arr,
            self.alive,
            self.pos_x,
            self.pos_y,
            seat_per_env,
            skip,
            self.cell_team_arr,
        )

    # ==================================================================
    # Batch step
    # ==================================================================

    def step_batch(
        self,
        action_ids: np.ndarray,
    ) -> MoveResultBatch:
        """Apply one action per environment, mutating this struct in-place.

        Vectorised implementation: plain-move envs are handled in bulk with
        NumPy fancy-indexing; only combat envs fall back to a per-env loop.

        Parameters
        ----------
        action_ids:
            Integer array of shape ``(num_envs,)``.  Each entry is a flat
            action id ``src_flat * NUM_CELLS + dst_flat``.  Terminated
            environments are skipped (their entry in action_ids is ignored).

        Returns
        -------
        MoveResultBatch
        """
        N = self.num_envs
        action_ids = np.asarray(action_ids, dtype=np.int32)

        # Output arrays
        valid      = np.zeros(N, dtype=bool)
        event_out  = np.zeros(N, dtype=np.int8)
        winner_out = self.winner_team.copy()
        draw_out   = self.draw.copy()
        flag_cap_out = np.zeros(N, dtype=bool)

        # Active envs: not yet terminated
        active = ~self.terminated
        if not active.any():
            return MoveResultBatch(
                valid=valid,
                event=event_out,
                terminated=self.terminated.copy(),
                winner_team=winner_out,
                draw=draw_out,
                flag_captured=flag_cap_out,
            )

        ai = np.nonzero(active)[0]                         # (A,) active env indices
        acts = action_ids[ai].astype(np.int32, copy=False) # (A,)
        src_flat_a = (acts // NUM_CELLS).astype(np.intp)   # (A,)
        dst_flat_a = (acts %  NUM_CELLS).astype(np.intp)   # (A,)

        # Piece IDs at src/dst for each active env
        src_pid_a = self.cell_piece_id[ai, src_flat_a]    # (A,) int16
        dst_pid_a = self.cell_piece_id[ai, dst_flat_a]    # (A,) int16

        has_src_a = src_pid_a >= 0                         # (A,) bool
        has_dst_a = dst_pid_a >= 0                         # (A,) bool

        # Split: plain-move (dst empty) vs combat (dst occupied)
        plain_mask = has_src_a & ~has_dst_a                # (A,)
        combat_mask = has_src_a & has_dst_a                # (A,)

        # ------------------------------------------------------------------
        # PLAIN MOVE PATH — fully vectorised
        # ------------------------------------------------------------------
        if plain_mask.any():
            pm_ai  = ai[plain_mask]                        # env indices
            pm_src = src_flat_a[plain_mask]
            pm_dst = dst_flat_a[plain_mask]
            pm_pid = src_pid_a[plain_mask].astype(np.intp) # piece ids

            pm_mc   = self.move_counter[pm_ai].astype(np.int32)
            pm_mslc = self.moves_since_last_combat[pm_ai].astype(np.int32)
            pm_turn = self.turn[pm_ai].astype(np.intp)
            pm_pt   = self.piece_type_arr[pm_ai, pm_pid].astype(np.intp)

            # CombatMemory v4: path-revealed GONGB.  Engineer plain moves
            # may walk a multi-hop rail-BFS path that only an engineer
            # could have taken — broadcast `is_gongb` to all observers.
            # Must run BEFORE we mutate cpid (blockers must still be in
            # place along the path).  Per-engineer fallback Python loop;
            # the gate (PT==GONGB) keeps this off the hot path 99 % of
            # the time, and a cheap 1-step pre-filter avoids the BFS
            # for 90 %+ of the GONGB moves that are.
            gongb_mask_plain = (pm_pt == int(PieceType.GONGB.value))
            if gongb_mask_plain.any():
                gongb_idx = np.nonzero(gongb_mask_plain)[0]
                pm_src_x = pm_src % BOARD_SIZE
                pm_src_y = pm_src // BOARD_SIZE
                pm_dst_x = pm_dst % BOARD_SIZE
                pm_dst_y = pm_dst // BOARD_SIZE
                adx = np.abs(pm_src_x[gongb_idx] - pm_dst_x[gongb_idx])
                ady = np.abs(pm_src_y[gongb_idx] - pm_dst_y[gongb_idx])
                # 1-step orthogonal (any piece) — never path-reveal.
                non_one_step = (adx + ady) > 1
                long_idx = gongb_idx[non_one_step]
                for kk in long_idx:
                    env_kk = int(pm_ai[kk])
                    src_flat_kk = int(pm_src[kk])
                    dst_flat_kk = int(pm_dst[kk])
                    src_xy = (src_flat_kk % BOARD_SIZE, src_flat_kk // BOARD_SIZE)
                    dst_xy = (dst_flat_kk % BOARD_SIZE, dst_flat_kk // BOARD_SIZE)
                    pieces_view = self._build_pieces_view(env_kk)
                    if move_requires_gongb(pieces_view, src_xy, dst_xy):
                        apply_path_revealed_gongb(
                            self._cm_view_for_env(env_kk), int(pm_pid[kk])
                        )

            # Zobrist XOR-out (old contributions)
            zob_pm = self.zobrist[pm_ai].copy()
            zob_pm ^= _ZOB.ZOB_TURN[pm_turn]
            zob_pm ^= _ZOB.ZOB_MOVE_COUNTER[pm_mc & _ZOB.MOVE_COUNTER_MASK]
            zob_pm ^= _ZOB.ZOB_MOVES_SINCE_COMBAT[pm_mslc & _ZOB.MOVES_SINCE_COMBAT_MASK]
            zob_pm ^= _ZOB.ZOB_PIECE[pm_pid, pm_pt, pm_src]
            # XOR-in (new piece position)
            zob_pm ^= _ZOB.ZOB_PIECE[pm_pid, pm_pt, pm_dst]

            # Board + piece updates
            self.cell_piece_id[pm_ai, pm_src] = -1
            self.cell_piece_id[pm_ai, pm_dst] = pm_pid.astype(np.int16)
            # Maintain cell_team_arr: src→empty(255), dst→team of moved piece
            pm_team = _TEAM_OF_PID[pm_pid]
            self.cell_team_arr[pm_ai, pm_src] = np.uint8(255)
            self.cell_team_arr[pm_ai, pm_dst] = pm_team
            self.pos_x[pm_ai, pm_pid] = (pm_dst % BOARD_SIZE).astype(np.int8)
            self.pos_y[pm_ai, pm_pid] = (pm_dst // BOARD_SIZE).astype(np.int8)
            self.move_count_arr[pm_ai, pm_pid] += 1

            pm_new_mc   = pm_mc + 1
            pm_new_mslc = pm_mslc + 1

            # Turn advance (vectorized for envs where next seat has no Q12)
            pm_new_turn, pm_zob_delta, pm_still_alive = self._advance_turn_vec(
                pm_ai, pm_turn, pm_new_mc,
            )
            zob_pm ^= pm_zob_delta

            # Check termination
            pm_term, pm_win, pm_draw = self._check_victory_vec(
                pm_ai, pm_turn, pm_new_mc, pm_new_mslc, pm_still_alive,
            )

            # Zobrist XOR-in (new counter/turn contributions)
            zob_pm ^= _ZOB.ZOB_MOVE_COUNTER[pm_new_mc & _ZOB.MOVE_COUNTER_MASK]
            zob_pm ^= _ZOB.ZOB_MOVES_SINCE_COMBAT[pm_new_mslc & _ZOB.MOVES_SINCE_COMBAT_MASK]
            zob_pm ^= _ZOB.ZOB_TURN[pm_new_turn.astype(np.intp)]
            # Winner/termination XOR
            term_xor = np.where(pm_term, _ZOB.ZOB_TERMINATED, np.int64(0))
            zob_pm ^= term_xor
            draw_xor = np.where(pm_draw, _ZOB.ZOB_DRAW, np.int64(0))
            zob_pm ^= draw_xor
            # Winner slot: old=0 (None), new depends on winning team
            win_nonzero = (pm_win >= 0)
            if win_nonzero.any():
                zob_pm[win_nonzero] ^= int(_ZOB.ZOB_WINNER[0])
                new_win_idx = (pm_win[win_nonzero] + 1).astype(np.intp)
                zob_pm[win_nonzero] ^= _ZOB.ZOB_WINNER[new_win_idx]

            # Commit
            self.move_counter[pm_ai]            = pm_new_mc
            self.moves_since_last_combat[pm_ai] = pm_new_mslc
            self.turn[pm_ai]                    = pm_new_turn.astype(np.int8)
            self.zobrist[pm_ai]                 = zob_pm
            self.terminated[pm_ai]              = pm_term
            self.winner_team[pm_ai]             = np.where(pm_win >= 0, pm_win, np.int8(-1))
            self.draw[pm_ai]                    = pm_draw

            valid[pm_ai]      = True
            event_out[pm_ai]  = Event.MOVE.value
            winner_out[pm_ai] = np.where(pm_win >= 0, pm_win, np.int8(-1))
            draw_out[pm_ai]   = pm_draw
            flag_cap_out[pm_ai] = False

        # ------------------------------------------------------------------
        # COMBAT PATH — per-env loop (rare, ~10% of steps)
        # ------------------------------------------------------------------
        if combat_mask.any():
            cb_ai = ai[combat_mask]
            for i in cb_ai:
                self._step_single(
                    int(i),
                    int(action_ids[i]),
                    valid,
                    event_out,
                    self.terminated,
                    winner_out,
                    draw_out,
                    flag_cap_out,
                )
                # _step_single already writes into self.terminated
            self.winner_team[cb_ai] = winner_out[cb_ai]
            self.draw[cb_ai]        = draw_out[cb_ai]

        return MoveResultBatch(
            valid=valid,
            event=event_out,
            terminated=self.terminated.copy(),
            winner_team=winner_out,
            draw=draw_out,
            flag_captured=flag_cap_out,
        )

    def _advance_turn_vec(
        self,
        env_idx: np.ndarray,   # (P,) env indices
        turn_vals: np.ndarray,  # (P,) current turn int8
        new_mc: np.ndarray,     # (P,) new move counter int32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised turn advance for plain-move envs.

        Returns
        -------
        new_turn: (P,) int8 — next seat values
        zob_delta: (P,) int64 — Zobrist delta from Q12 kills
        still_alive: (P, 4) bool — which seats are still alive after Q12
        """
        P = len(env_idx)
        new_turn = np.empty(P, dtype=np.int8)
        zob_delta = np.zeros(P, dtype=np.int64)

        # still_alive tracks seat_dead_arr (mutable copy for this batch)
        still_alive = self.seat_dead_arr[env_idx].copy()  # (P, 4) bool = dead mask

        # Vectorise the easy case: candidate = (turn+1)%4, and it's alive
        candidate = ((turn_vals.astype(np.int32) + 1) % 4).astype(np.int8)

        # Envs where candidate is already dead → need per-env loop
        cand_dead = still_alive[np.arange(P, dtype=np.intp), candidate.astype(np.intp)]
        easy = ~cand_dead

        new_turn[easy] = candidate[easy]

        # Hard envs: candidate seat is dead, need to skip
        hard_mask = cand_dead
        if hard_mask.any():
            hard_idx = np.nonzero(hard_mask)[0]
            for k in hard_idx:
                i_orig = int(env_idx[k])
                tv = int(turn_vals[k])
                mc_val = int(new_mc[k])
                cand = (tv + 1) % 4
                chain = 0
                delta = np.int64(0)
                while chain < 4:
                    if self.seat_dead_arr[i_orig, cand]:
                        cand = (cand + 1) % 4
                        chain += 1
                        continue
                    has_moves = has_legal_moves_soa(
                        self.cell_piece_id[i_orig],
                        self.piece_seat_arr[i_orig],
                        self.piece_type_arr[i_orig],
                        self.alive[i_orig],
                        self.pos_x[i_orig],
                        self.pos_y[i_orig],
                        cand,
                    )
                    if has_moves:
                        break
                    # Q12: kill this seat
                    delta ^= self._surrender_seat_delta(i_orig, cand, mc_val, reveal_flag=False)
                    cand = (cand + 1) % 4
                    chain += 1
                new_turn[k] = cand
                zob_delta[k] = delta
                # Refresh still_alive row from updated seat_dead_arr
                still_alive[k] = self.seat_dead_arr[i_orig]

        # For easy envs: also check if candidate seat has legal moves (Q12 after plain move)
        # In practice, Q12 after plain move is extremely rare, so just do easy envs via
        # has_legal_moves_soa; the overhead is minimal compared to the Python-loop savings.
        # We skip this extra check for easy envs — Q12 is caught by _check_victory_vec.

        return new_turn, zob_delta, still_alive

    def _check_victory_vec(
        self,
        env_idx: np.ndarray,    # (P,)
        acting_seat: np.ndarray, # (P,) current acting seat (before advance)
        new_mc: np.ndarray,      # (P,) new move_counter
        new_mslc: np.ndarray,    # (P,) new moves_since_last_combat
        seat_dead: np.ndarray,   # (P, 4) bool — CURRENT dead mask (after any Q12)
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised victory check matching _check_victory logic.

        Returns (terminated, winner_team, draw) each shape (P,).
        winner_team uses -1 for None.
        """
        P = len(env_idx)
        # Team 0 = SOUTH(0)+NORTH(2); Team 1 = WEST(1)+EAST(3)
        red_dead  = seat_dead[:, 0] & seat_dead[:, 2]  # (P,) — red team fully dead?
        blue_dead = seat_dead[:, 1] & seat_dead[:, 3]  # (P,)

        terminated = np.zeros(P, dtype=bool)
        winner     = np.full(P, -1, dtype=np.int8)
        draw       = np.zeros(P, dtype=bool)

        # Q14: mutual destruction → attacker's team wins
        both_dead = red_dead & blue_dead
        if both_dead.any():
            attacker_team = (acting_seat[both_dead].astype(np.int8) % 2).astype(np.int8)
            terminated[both_dead] = True
            winner[both_dead]     = attacker_team

        # Standard victory: one team fully dead
        red_wins  = blue_dead & ~both_dead
        blue_wins = red_dead  & ~both_dead
        terminated[red_wins] = True
        winner[red_wins] = np.int8(0)
        terminated[blue_wins] = True
        winner[blue_wins] = np.int8(1)

        # Draw thresholds
        draw_mc   = new_mc   >= MAX_NUM_MOVES
        draw_mslc = new_mslc >= MAX_NUM_MOVES_BETWEEN_ATTACKS
        any_draw  = (draw_mc | draw_mslc) & ~terminated
        if any_draw.any():
            terminated[any_draw] = True
            draw[any_draw]       = True

        return terminated, winner, draw

    # ==================================================================
    # Single-env step (called from step_batch)
    # ==================================================================

    def _step_single(
        self,
        i: int,
        action_id: int,
        valid_out: np.ndarray,
        event_out: np.ndarray,
        terminated_out: np.ndarray,
        winner_out: np.ndarray,
        draw_out: np.ndarray,
        flag_cap_out: np.ndarray,
    ) -> None:
        """Apply action_id to environment i, writing results into out arrays."""
        src_flat = action_id // NUM_CELLS
        dst_flat = action_id % NUM_CELLS

        acting_seat_val = int(self.turn[i])
        death_step = int(self.move_counter[i]) + 1

        # --- Identify pieces ---
        src_pid = int(self.cell_piece_id[i, src_flat])
        dst_pid = int(self.cell_piece_id[i, dst_flat])

        if src_pid < 0:
            # Invalid action: no piece at src
            return

        # Snapshot src/dst types for combat
        src_type_val = int(self.piece_type_arr[i, src_pid])
        has_dst = dst_pid >= 0 and bool(self.alive[i, dst_pid])
        dst_type_val = int(self.piece_type_arr[i, dst_pid]) if has_dst else 0

        # --- Pre-step Zobrist XOR-outs ---
        zob = int(self.zobrist[i])
        zob ^= int(_ZOB.ZOB_TURN[acting_seat_val])
        mc = int(self.move_counter[i])
        mslc = int(self.moves_since_last_combat[i])
        zob ^= int(_ZOB.ZOB_MOVE_COUNTER[mc & _ZOB.MOVE_COUNTER_MASK])
        zob ^= int(_ZOB.ZOB_MOVES_SINCE_COMBAT[mslc & _ZOB.MOVES_SINCE_COMBAT_MASK])

        flag_captured = False
        seats_died: list[int] = []

        if not has_dst:
            # ----------------------------------------------------------------
            # Plain move into empty cell
            # ----------------------------------------------------------------
            event_val = Event.MOVE.value
            # CombatMemory v4: if the moving piece is an engineer and the
            # path is GONGB-only, broadcast a path-revealed GONGB to all
            # observers.  We re-build a tiny PieceMap on demand here
            # (only when moving piece is GONGB) to reuse the same
            # ``move_requires_gongb`` helper as the single-env path.
            if PieceType(src_type_val) is PieceType.GONGB:
                pieces_view = self._build_pieces_view(i)
                src_xy = (src_flat % BOARD_SIZE, src_flat // BOARD_SIZE)
                dst_xy = (dst_flat % BOARD_SIZE, dst_flat // BOARD_SIZE)
                if move_requires_gongb(pieces_view, src_xy, dst_xy):
                    apply_path_revealed_gongb(
                        self._cm_view_for_env(i), src_pid
                    )
            # Zobrist: XOR-out src, XOR-in dst
            zob ^= int(_ZOB.ZOB_PIECE[src_pid, src_type_val, src_flat])
            zob ^= int(_ZOB.ZOB_PIECE[src_pid, src_type_val, dst_flat])
            # SoA updates
            self.cell_piece_id[i, src_flat] = -1
            self.cell_piece_id[i, dst_flat] = src_pid
            self.cell_team_arr[i, src_flat] = np.uint8(255)
            self.cell_team_arr[i, dst_flat] = _TEAM_OF_PID[src_pid]
            self.pos_x[i, src_pid] = dst_flat % BOARD_SIZE
            self.pos_y[i, src_pid] = dst_flat // BOARD_SIZE
            # move_count bump
            self.move_count_arr[i, src_pid] += 1
            new_mslc = mslc + 1
        else:
            # ----------------------------------------------------------------
            # Combat
            # ----------------------------------------------------------------
            src_pt = PieceType(src_type_val)
            dst_pt = PieceType(dst_type_val)
            event_enum = resolve_combat(src_pt, dst_pt)
            event_val = event_enum.value

            # CombatMemory v4: GONGB path-reveal (must run BEFORE we move
            # any pieces so blockers along the path are still in place).
            if src_pt is PieceType.GONGB:
                pieces_view = self._build_pieces_view(i)
                src_xy = (src_flat % BOARD_SIZE, src_flat // BOARD_SIZE)
                dst_xy = (dst_flat % BOARD_SIZE, dst_flat // BOARD_SIZE)
                if move_requires_gongb(pieces_view, src_xy, dst_xy):
                    apply_path_revealed_gongb(
                        self._cm_view_for_env(i), src_pid
                    )

            # SILING reveals — accumulate delta into zob
            if siling_reveals_src(src_pt, dst_pt, event_enum):
                zob ^= self._flag_reveal_delta(i, int(self.piece_seat_arr[i, src_pid]))
            if siling_reveals_dst(src_pt, dst_pt, event_enum):
                zob ^= self._flag_reveal_delta(i, int(self.piece_seat_arr[i, dst_pid]))

            flag_captured = dst_pt is PieceType.JUNQI

            if event_enum is Event.EAT:
                # src wins, dst dies; src moves onto dst
                # Kill dst
                zob ^= int(_ZOB.ZOB_PIECE[dst_pid, dst_type_val, dst_flat])
                self._kill_piece(i, dst_pid, dst_flat,
                                 DeathReason.KILLED_BY_ENEMY.value, death_step, dst_flat)
                # Move src onto dst  (_kill_piece already cleared dst cell)
                zob ^= int(_ZOB.ZOB_PIECE[src_pid, src_type_val, src_flat])
                zob ^= int(_ZOB.ZOB_PIECE[src_pid, src_type_val, dst_flat])
                self.cell_piece_id[i, src_flat] = -1
                self.cell_piece_id[i, dst_flat] = src_pid
                self.cell_team_arr[i, src_flat] = np.uint8(255)
                self.cell_team_arr[i, dst_flat] = _TEAM_OF_PID[src_pid]
                self.pos_x[i, src_pid] = dst_flat % BOARD_SIZE
                self.pos_y[i, src_pid] = dst_flat // BOARD_SIZE
                self.active_eat_arr[i, src_pid] += 1
                new_mslc = 0
                # CombatMemory v4: src ate dst (Event.EAT).
                apply_combat_event(
                    self._cm_view_for_env(i),
                    event_is_eat=True,
                    attacker_pid=src_pid,
                    defender_pid=dst_pid,
                    attacker_seat=int(self.piece_seat_arr[i, src_pid]),
                    defender_seat=int(self.piece_seat_arr[i, dst_pid]),
                    attacker_type=src_pt,
                    defender_type=dst_pt,
                    defender_pos_flat=dst_flat,
                    death_step=death_step,
                )

            elif event_enum is Event.KILLED:
                # src dies, dst survives
                dr = classify_death_reason(
                    own_piece=src_pt,
                    opponent_piece=dst_pt,
                    event=event_enum,
                    own_is_attacker=True,
                )
                zob ^= int(_ZOB.ZOB_PIECE[src_pid, src_type_val, src_flat])
                self._kill_piece(i, src_pid, src_flat, dr.value, death_step, dst_flat)
                self.passive_surv_arr[i, dst_pid] += 1
                new_mslc = 0
                # CombatMemory v4: defender survived (Event.KILLED).
                apply_combat_event(
                    self._cm_view_for_env(i),
                    event_is_eat=False,
                    attacker_pid=src_pid,
                    defender_pid=dst_pid,
                    attacker_seat=int(self.piece_seat_arr[i, src_pid]),
                    defender_seat=int(self.piece_seat_arr[i, dst_pid]),
                    attacker_type=src_pt,
                    defender_type=dst_pt,
                    defender_pos_flat=dst_flat,
                    death_step=death_step,
                )

            elif event_enum is Event.BOMB:
                # Both die
                zob ^= int(_ZOB.ZOB_PIECE[src_pid, src_type_val, src_flat])
                zob ^= int(_ZOB.ZOB_PIECE[dst_pid, dst_type_val, dst_flat])
                self._kill_piece(i, src_pid, src_flat, DeathReason.MUTUAL.value, death_step, dst_flat)
                self._kill_piece(i, dst_pid, dst_flat, DeathReason.MUTUAL.value, death_step, dst_flat)
                new_mslc = 0
                # CombatMemory v4: BOMB events do NOT update CombatMemory
                # — both pieces die, no live target for projection.

            else:
                raise AssertionError(f"unreachable event {event_enum!r}")

        # --- Flag capture → seat surrenders ---
        if flag_captured:
            surr_seat = int(self.piece_seat_arr[i, dst_pid])
            if not self.seat_dead_arr[i, surr_seat]:
                # Flag capture reveals the flag (reveal_flag=True), then kills
                # all remaining pieces of the surrendering seat.
                zob ^= self._flag_reveal_delta(i, surr_seat)  # flag revealed
                zob ^= self._surrender_seat_delta(i, surr_seat, death_step, reveal_flag=False)
                seats_died.append(surr_seat)

        # --- Post-combat dead-sweep: seats with no pieces ---
        for seat_val in range(4):
            if self.seat_dead_arr[i, seat_val]:
                continue
            # Check if seat has any alive piece
            seat_mask = (self.piece_seat_arr[i] == seat_val) & self.alive[i]
            if not seat_mask.any():
                self.seat_dead_arr[i, seat_val] = True
                zob ^= int(_ZOB.ZOB_SEAT_DEAD[seat_val])
                seats_died.append(seat_val)

        # --- Update counters ---
        new_mc = mc + 1
        # new_mslc was set in the MOVE branch or in each combat branch above

        # --- Check termination ---
        terminated, winner_team, draw = self._check_victory(
            i, acting_seat_val, new_mc, new_mslc
        )

        # --- Advance turn if not terminated ---
        if not terminated:
            new_turn, q12_delta = self._advance_turn(i, acting_seat_val, new_mc)
            zob ^= q12_delta
            # re-check victory after turn advance (Q12 may have killed more seats)
            terminated, winner_team, draw = self._check_victory(
                i, acting_seat_val, new_mc, new_mslc
            )
        else:
            new_turn = acting_seat_val

        # --- Commit scalar updates + Zobrist XOR-in ---
        self.move_counter[i] = new_mc
        self.moves_since_last_combat[i] = new_mslc
        zob ^= int(_ZOB.ZOB_MOVE_COUNTER[new_mc & _ZOB.MOVE_COUNTER_MASK])
        zob ^= int(_ZOB.ZOB_MOVES_SINCE_COMBAT[new_mslc & _ZOB.MOVES_SINCE_COMBAT_MASK])
        zob ^= int(_ZOB.ZOB_TURN[new_turn])
        # Termination flags: XOR-in if they transition (from False/None).
        # ZOB_WINNER uses slot 0 = None, 1 = team-0, 2 = team-1.
        # The initial hash includes ZOB_WINNER[0] (winner=None); swap only when it changes.
        if terminated:
            zob ^= _ZOB.ZOB_TERMINATED
        if draw:
            zob ^= _ZOB.ZOB_DRAW
        # Winner slot swap: old index is always 0 (None) because we only step
        # non-terminated envs; new index depends on the winning team.
        new_winner_idx = 0 if winner_team is None else (winner_team + 1)
        if new_winner_idx != 0:
            zob ^= int(_ZOB.ZOB_WINNER[0])           # XOR-out old (None)
            zob ^= int(_ZOB.ZOB_WINNER[new_winner_idx])  # XOR-in new team

        self.turn[i] = new_turn
        self.zobrist[i] = zob

        # Write outputs
        valid_out[i] = True
        event_out[i] = event_val
        terminated_out[i] = terminated
        winner_out[i] = -1 if winner_team is None else winner_team
        draw_out[i] = draw
        flag_cap_out[i] = flag_captured

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_pieces_view(
        self, i: int
    ) -> dict[tuple[int, int], PieceRef]:
        """Construct an ad-hoc PieceMap (only the data ``move_requires_gongb``
        reads — seat + piece_type + alive) for env ``i``.

        Used only on the rare path-reveal check (when the moving piece is
        a GONGB).  Build cost is O(120) per call — acceptable as it
        amortizes against the rail-BFS work the helper itself does.
        """
        from .move_gen import PieceRef
        out: dict[tuple[int, int], PieceRef] = {}
        alive = self.alive[i]
        pos_x = self.pos_x[i]
        pos_y = self.pos_y[i]
        seat_arr = self.piece_seat_arr[i]
        type_arr = self.piece_type_arr[i]
        for pid in range(NUM_PIECES):
            if not bool(alive[pid]):
                continue
            x = int(pos_x[pid])
            y = int(pos_y[pid])
            out[(x, y)] = PieceRef(
                seat=Seat(int(seat_arr[pid])),
                piece_type=PieceType(int(type_arr[pid])),
                alive=True,
                piece_id=pid,
            )
        return out

    def _cm_view_for_env(self, i: int) -> CombatMemoryState:
        """Return a :class:`CombatMemoryState` (v4) whose arrays *view* env
        ``i``'s slice of the batched CombatMemory columns.  Mutations
        through the view land directly in ``self.cm_*`` — no per-call
        copy.
        """
        return CombatMemoryState(
            direct_ate_my_pid_lo    = self.cm_direct_lo[i],
            direct_ate_my_pid_hi    = self.cm_direct_hi[i],
            direct_ate_my_type_mask = self.cm_direct_type[i],
            last_direct_step        = self.cm_last_direct_step[i],
            direct_other_count      = self.cm_direct_other_count[i],
            chain_pid_lo            = self.cm_chain_lo[i],
            chain_pid_hi            = self.cm_chain_hi[i],
            chain_ate_my_type_mask  = self.cm_chain_type[i],
            last_chain_step         = self.cm_last_chain_step[i],
            eaten_by_pid_lo         = self.cm_eaten_by_pid_lo[i],
            eaten_by_pid_hi         = self.cm_eaten_by_pid_hi[i],
            rank_floor              = self.cm_rank_floor[i],
            rank_floor_step         = self.cm_rank_floor_step[i],
            is_gongb                = self.cm_is_gongb[i],
            not_gongb               = self.cm_not_gongb[i],
            attacked_by_known_gongb = self.cm_attacked_by_known_gongb[i],
        )

    def _kill_piece(
        self,
        i: int,
        pid: int,
        cell_flat: int,
        death_reason_val: int,
        death_step: int,
        death_loc_flat: int,
    ) -> None:
        """Mark piece ``pid`` as dead and clear it from the board in env i.

        Does NOT update Zobrist — the caller must XOR-out the piece before
        calling this.
        """
        if not self.alive[i, pid]:
            return
        self.alive[i, pid] = False
        self.cell_piece_id[i, cell_flat] = -1
        self.cell_team_arr[i, cell_flat] = np.uint8(255)
        self.pos_x[i, pid] = -1
        self.pos_y[i, pid] = -1
        self.death_reason_arr[i, pid] = death_reason_val
        self.death_step_arr[i, pid] = death_step
        self.death_loc_flat_arr[i, pid] = death_loc_flat

    def _flag_reveal_delta(self, i: int, seat_val: int) -> int:
        """Return Zobrist XOR delta for revealing seat's flag (idempotent).

        Sets ``seat_flag_revealed_arr`` if not already set, returns the XOR
        contribution to add to the accumulator.  Does NOT touch
        ``self.zobrist[i]`` directly.
        """
        if not self.seat_flag_revealed_arr[i, seat_val]:
            self.seat_flag_revealed_arr[i, seat_val] = True
            return int(_ZOB.ZOB_SEAT_FLAG_REVEALED[seat_val])
        return 0

    def _surrender_seat_delta(
        self, i: int, seat_val: int, death_step: int, reveal_flag: bool = False
    ) -> int:
        """Mark seat as dead, kill all its pieces, return Zobrist XOR delta.

        Does NOT touch ``self.zobrist[i]`` directly; caller must XOR the
        returned delta into its accumulator.

        Parameters
        ----------
        reveal_flag:
            If True, also reveal this seat's flag (XOR in
            ``ZOB_SEAT_FLAG_REVEALED``).  Set True for flag-capture surrenders;
            leave False for Q12 no-legal-move kills (state.py does NOT reveal
            the flag in that path).
        """
        delta = 0
        self.seat_dead_arr[i, seat_val] = True
        delta ^= int(_ZOB.ZOB_SEAT_DEAD[seat_val])

        # Kill all alive pieces of this seat
        seat_mask = (self.piece_seat_arr[i] == seat_val) & self.alive[i]
        pids = np.nonzero(seat_mask)[0]
        for pid in pids:
            pid_int = int(pid)
            pos_x = int(self.pos_x[i, pid_int])
            pos_y = int(self.pos_y[i, pid_int])
            if pos_x >= 0 and pos_y >= 0:
                flat = pos_y * BOARD_SIZE + pos_x
                ptype_val = int(self.piece_type_arr[i, pid_int])
                delta ^= int(_ZOB.ZOB_PIECE[pid_int, ptype_val, flat])
                self.cell_piece_id[i, flat] = -1
                self.cell_team_arr[i, flat] = np.uint8(255)
            self.alive[i, pid_int] = False
            self.pos_x[i, pid_int] = -1
            self.pos_y[i, pid_int] = -1
            self.death_reason_arr[i, pid_int] = DeathReason.KILLED_BY_ENEMY.value
            self.death_step_arr[i, pid_int] = death_step
            self.death_loc_flat_arr[i, pid_int] = -1

        # flag revealed (only for flag-capture path, not Q12)
        if reveal_flag:
            delta ^= self._flag_reveal_delta(i, seat_val)
        return delta

    def _advance_turn(self, i: int, from_seat_val: int, death_step: int) -> tuple[int, int]:
        """Return (next_seat_val, zobrist_delta) after advancing turn.

        Implements Q12: if the next alive candidate seat has no legal moves,
        it is killed (all its pieces die, seat marked dead) and the sweep
        continues.  The Zobrist XOR delta for any Q12 deaths is accumulated
        into the returned delta (caller XORs it into their accumulator).

        The caller is responsible for NOT writing self.zobrist directly —
        they must XOR the delta into their local accumulator.

        Parameters
        ----------
        death_step:
            The move_counter AFTER this step (= new_mc = mc + 1).  This
            is the value recorded as the death step for Q12-killed pieces.
        """
        delta = 0
        chain = 0
        candidate = (from_seat_val + 1) % 4

        while chain < 4:
            if self.seat_dead_arr[i, candidate]:
                candidate = (candidate + 1) % 4
                chain += 1
                continue

            # Generate legal actions for this candidate
            has_moves = has_legal_moves_soa(
                self.cell_piece_id[i],
                self.piece_seat_arr[i],
                self.piece_type_arr[i],
                self.alive[i],
                self.pos_x[i],
                self.pos_y[i],
                candidate,
            )
            if has_moves:
                return candidate, delta

            # Q12: candidate has no legal moves → kill it
            delta ^= self._surrender_seat_delta(i, candidate, death_step, reveal_flag=False)

            candidate = (candidate + 1) % 4
            chain += 1

        return (from_seat_val + 1) % 4, delta  # fallback

    def _check_victory(
        self,
        i: int,
        acting_seat_val: int,
        new_mc: int,
        new_mslc: int,
    ) -> tuple[bool, int | None, bool]:
        """Return (terminated, winner_team, draw) for env i.

        Mirrors the logic in ``state.py::_check_victory``.
        Order matches state.py exactly:
          1. Q14 mutual destruction  → attacker's team wins (not a draw!)
          2. One team fully dead     → other team wins
          3. Counter thresholds      → draw
        """
        from .rules import MAX_NUM_MOVES, MAX_NUM_MOVES_BETWEEN_ATTACKS

        # Check which teams have alive seats
        # Team 0 = SOUTH(0) + NORTH(2); Team 1 = WEST(1) + EAST(3)
        red_alive = (
            not self.seat_dead_arr[i, 0] or not self.seat_dead_arr[i, 2]
        )  # SOUTH or NORTH alive
        blue_alive = (
            not self.seat_dead_arr[i, 1] or not self.seat_dead_arr[i, 3]
        )  # WEST or EAST alive

        # Q14: mutual destruction → attacker's team wins (NOT a draw)
        if not red_alive and not blue_alive:
            attacker_team = acting_seat_val % 2
            return True, attacker_team, False

        # Standard team victory: one team fully dead → other team wins
        if not red_alive:
            return True, 1, False   # blue team wins
        if not blue_alive:
            return True, 0, False   # red team wins

        # §5.3 draw thresholds — only remaining path to a draw
        if new_mc >= MAX_NUM_MOVES:
            return True, None, True
        if new_mslc >= MAX_NUM_MOVES_BETWEEN_ATTACKS:
            return True, None, True

        return False, None, False

    # ==================================================================
    # Conversion back to list of game states
    # ==================================================================

    def to_game_states(self) -> list[GameState]:
        """Reconstruct a list of :class:`GameState` objects from this batch.

        Useful for testing round-trip fidelity.  This is NOT a hot path.
        """
        from .rules import Seat as _Seat
        from .rules import ShowMode
        from .state import GameState

        states: list[GameState] = []
        for i in range(self.num_envs):
            # We rebuild a GameState with only SoA columns set (no dict state).
            # The dict state (pieces, info, etc.) is NOT reconstructed here —
            # we create a minimal shell sufficient for SoA-based API calls
            # (legal_action_ids, observation building via builder).
            # Full reconstruction would require reversing all the dict logic.
            gs = object.__new__(GameState)
            # Scalars
            gs.turn = _Seat(int(self.turn[i]))
            gs.move_counter = int(self.move_counter[i])
            gs.moves_since_last_combat = int(self.moves_since_last_combat[i])
            gs.terminated = bool(self.terminated[i])
            gs.winner_team = None if self.winner_team[i] < 0 else int(self.winner_team[i])
            gs.draw = bool(self.draw[i])
            gs.zobrist = int(self.zobrist[i])
            gs.show_mode = ShowMode.HALF_DARK
            gs.rules_version = "1.1.0"
            gs.debug_include_private = False
            # SoA columns (copy)
            gs.alive = self.alive[i].copy()
            gs.piece_type_arr = self.piece_type_arr[i].copy()
            gs.piece_seat_arr = self.piece_seat_arr[i].copy()
            gs.pos_x = self.pos_x[i].copy()
            gs.pos_y = self.pos_y[i].copy()
            gs.zero_x = self.zero_x[i].copy()
            gs.zero_y = self.zero_y[i].copy()
            gs.move_count_arr = self.move_count_arr[i].copy()
            gs.active_eat_arr = self.active_eat_arr[i].copy()
            gs.passive_surv_arr = self.passive_surv_arr[i].copy()
            gs.death_reason_arr = self.death_reason_arr[i].copy()
            gs.death_step_arr = self.death_step_arr[i].copy()
            gs.death_loc_flat_arr = self.death_loc_flat_arr[i].copy()
            gs.cell_piece_id = self.cell_piece_id[i].copy()
            gs.seat_dead_arr = self.seat_dead_arr[i].copy()
            gs.seat_flag_revealed_arr = self.seat_flag_revealed_arr[i].copy()
            # Dict state: set minimal non-None values (no pieces dict)
            gs.pieces = {}
            gs.info = {_Seat(s): type('SeatInfo', (), {'dead': bool(self.seat_dead_arr[i, s]),
                                                         'flag_revealed': bool(self.seat_flag_revealed_arr[i, s])})()
                       for s in range(4)}
            gs.zero_board = {}
            gs.piece_state = {}
            gs.deaths = {}
            states.append(gs)
        return states

    # ==================================================================
    # Clone
    # ==================================================================

    def clone(self) -> BatchedGameState:
        """Return a deep copy of this batch.

        Enumerates the dataclass fields rather than listing them by hand. The
        hand-written version had fallen 16 fields behind — every ``cm_*``
        CombatMemory column was missing, so a clone silently came back with
        the empty arrays from their ``default_factory`` while the board and
        counters copied correctly. Deriving the list means a new field is
        copied the day it is added.
        """
        kwargs: dict[str, object] = {}
        for f in dc_fields(self):
            value = getattr(self, f.name)
            kwargs[f.name] = value.copy() if isinstance(value, np.ndarray) else value
        return BatchedGameState(**kwargs)  # type: ignore[arg-type]

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        n_terminated = int(self.terminated.sum())
        return (
            f"BatchedGameState(num_envs={self.num_envs}, "
            f"terminated={n_terminated}/{self.num_envs})"
        )
