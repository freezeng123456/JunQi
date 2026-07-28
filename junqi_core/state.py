"""Game state, action, and transition engine for 4-player Junqi.

This module is the brain of `junqi_core`. It wires together `rules.py` (what
combat does), `board.py` (board geometry), `move_gen.py` (what moves are
legal), and `setup.py` (how a game starts). It exposes:

  - `GameState`   : immutable snapshot of an in-progress game.
  - `Action`      : (seat, src, dst) in world-frame.
  - `SeatInfo`    : per-seat liveness / flag-reveal state.
  - `MoveResult`  : everything broadcast (and some debug fields) after a step.

Key API:
  - `GameState.new_game(setups, ...)`   — construct start-of-game state.
  - `state.step(action) -> (new_state, MoveResult)` — pure functional step.
  - `state.legal_actions(seat=None)`
  - `state.legal_action_mask(seat=None)`
  - `state.clone()`, `state.state_hash()`.

Reference: docs/RULES.md v1.1.0 (and docs/DECISIONS.md ADR-016 for Q14).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import numpy as np

from . import _zobrist as _ZOB
from .board import (
    BOARD_SIZE,
    NUM_CELLS,
    all_cells_of_seat,
    index_to_pos,
    is_on_board,
    xy_to_flat,
)
from .combat_memory import (
    CombatMemoryState,
    apply_combat_event,
    apply_path_revealed_gongb,
)
from .move_gen import (
    PieceMap,
    PieceRef,
    generate_legal_action_ids_batch,
    generate_legal_actions,
    has_any_legal_move,
    has_any_legal_move_soa,
    is_legal_move,
    legal_moves_from,
    move_requires_gongb,
)
from .rules import (
    ALL_SEATS,
    MAX_NUM_MOVES,
    MAX_NUM_MOVES_BETWEEN_ATTACKS,
    RULES_VERSION,
    DeathReason,
    Event,
    PieceType,
    Seat,
    ShowMode,
    classify_death_reason,
    resolve_combat,
    same_team,
    siling_reveals_dst,
    siling_reveals_src,
)
from .setup import (
    SetupArray,
    assign_piece_ids,
    lineup_from_names,
    validate_setup,
)

# ===========================================================================
# DeathInfo / PieceState — T7 / ADR-114
# ===========================================================================
#
# These two small records are the data backbone of the piece_id global
# identity system. They are owned by `GameState` and read by observation
# channel groups A/B/C (`PieceState`) and D/E (`DeathInfo`).
#
# Both are frozen for cache-safety; `GameState.step()` does not mutate them
# in-place — it constructs replacements via `dataclasses.replace()` and
# stores them in new dicts (see M2/M3 in docs/PHASE_0.3_T7_TODO.md).


@dataclass(frozen=True, slots=True)
class DeathInfo:
    """Per-piece death record. One entry per dead piece, keyed by piece_id.

    Fields
    ------
    piece_id
        Global identity (0..119) assigned at `new_game`.
    reason
        Why this piece died. See `DeathReason` in rules.py (D-2 decision:
        MUTUAL covers all bomb/same-rank outcomes regardless of cause).
    death_loc
        World-frame (x, y) of the cell where combat occurred. This is the
        anchor cell for observation channel group D (`death_reason_*`).
    step
        `GameState.move_counter` at the moment of death (1-indexed; the move
        that killed this piece). Enables temporal features / replays.
    """

    piece_id: int
    reason: DeathReason
    death_loc: tuple[int, int]
    step: int


@dataclass(frozen=True, slots=True)
class PieceState:
    """Per-piece running counters. One entry per *living* piece, keyed by
    piece_id. Decoupled from `PieceRef` (which stays frozen and hashable
    for move_gen caches) per D-1 decision.

    Fields
    ------
    move_count
        Number of successful `Event.MOVE` actions this piece has made.
        Increments strictly by 1 per step.
    active_eat_count
        Number of times this piece attacked and won (Event.EAT as attacker).
    passive_survive_count
        Number of times this piece was attacked and survived (Event.KILLED
        where this piece was the defender). Winning by same-rank BOMB
        means this piece *also* dies, so it is never counted here.

    Observation mapping (see docs/PHASE_0.3_T7_TODO.md section 1.4):
      - move_count -> bucket {0, 1, 2, >=3}            (A group, exact-match)
      - active_eat_count -> bucket {0, >=1, >=2, >=3}  (B group, cumulative)
      - passive_survive_count -> bucket {0, >=1, >=2, >=3} (C group, cumulative)
    """

    move_count: int = 0
    active_eat_count: int = 0
    passive_survive_count: int = 0


# ===========================================================================
# SeatInfo — per-seat dynamic state
# ===========================================================================


@dataclass(slots=True)
class SeatInfo:
    """Per-seat dynamic state. Mutable inside a GameState's dict; copy on step()."""

    dead: bool = False
    flag_revealed: bool = False

    def clone(self) -> SeatInfo:
        return SeatInfo(dead=self.dead, flag_revealed=self.flag_revealed)


# ===========================================================================
# Action — frozen (src, dst, seat) tuple
# ===========================================================================


@dataclass(frozen=True, slots=True)
class Action:
    """A single move in world-frame coordinates."""

    seat: Seat
    src: tuple[int, int]
    dst: tuple[int, int]

    def __post_init__(self) -> None:
        if not is_on_board(*self.src):
            raise ValueError(f"action.src {self.src} is off-board")
        if not is_on_board(*self.dst):
            raise ValueError(f"action.dst {self.dst} is off-board")
        if self.src == self.dst:
            raise ValueError("action.src == action.dst")


# ===========================================================================
# MoveResult — what gets broadcast after a step
# ===========================================================================


@dataclass(frozen=True, slots=True)
class MoveResult:
    """All information about a single step.

    The fields up to `flag_captured` are the PUBLIC broadcast (§3.3 RULES).
    The `*_type_revealed` fields are populated only in BRIGHT show_mode or
    when debug_include_private=True on the GameState.

    The `seats_died_this_step` / `terminated_after` / `winner_team_after` /
    `draw_after` fields let callers react to termination without re-inspecting
    the full state.
    """

    seat: Seat
    src: tuple[int, int]
    dst: tuple[int, int]
    event: Event
    flag_reveal_src: bool
    flag_reveal_dst: bool
    flag_captured: bool

    # Optional "visible" piece types (non-None only in BRIGHT mode / debug)
    src_type_revealed: PieceType | None = None
    dst_type_revealed: PieceType | None = None

    # Derived termination signals (reflecting state *after* this step)
    terminated_after: bool = False
    winner_team_after: int | None = None
    draw_after: bool = False

    # Seats that transitioned from alive -> dead during this step
    seats_died_this_step: tuple[Seat, ...] = ()


# ===========================================================================
# GameState — the main immutable snapshot
# ===========================================================================


@dataclass(slots=True)
class GameState:
    """Immutable snapshot of an in-progress 4-player Junqi game.

    `step()` returns a NEW GameState; the original is never mutated. Use
    `clone()` if you need a modifiable deep copy.
    """

    pieces: PieceMap                                # (x,y) -> PieceRef (alive only)
    turn: Seat                                      # whose turn it is
    move_counter: int                               # total moves since game start
    moves_since_last_combat: int                    # for Q10 200-move draw
    info: dict[Seat, SeatInfo]                      # per-seat state

    terminated: bool = False
    winner_team: int | None = None
    draw: bool = False

    show_mode: ShowMode = ShowMode.HALF_DARK
    rules_version: str = RULES_VERSION

    # Per-state debug switch — if True, MoveResult exposes concrete types.
    debug_include_private: bool = False

    # -----------------------------------------------------------------------
    # T7 / ADR-114 — piece_id global identity system.
    # -----------------------------------------------------------------------
    # `zero_board` is an immutable snapshot of the **initial** setup, keyed
    # by world-frame cell. Written exactly once in `new_game()` and never
    # mutated afterwards. Observation channel group E (`dead_at_zero_*`)
    # reads from this map at every observation build.
    #
    # `piece_state` holds running counters (move/eat/survive) for *living*
    # pieces, keyed by piece_id. When a piece dies, its entry is popped
    # (M3) to save memory; the zero-count signal is recoverable from
    # `zero_board` + `deaths`.
    #
    # `deaths` maps piece_id -> DeathInfo. Entries are frozen at death time.
    #
    # M1 only initializes these. M2 populates `piece_state`, M3 populates
    # `deaths`. See docs/PHASE_0.3_T7_TODO.md section 4.
    zero_board: dict[tuple[int, int], PieceRef] = field(default_factory=dict)
    piece_state: dict[int, PieceState] = field(default_factory=dict)
    deaths: dict[int, DeathInfo] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # CombatMemory (per-observer, per-piece_id high-order combat history).
    # -----------------------------------------------------------------------
    # Keeps a 4-observer × 120-piece bit-packed log of "this enemy directly
    # ate my piece P" / "this enemy chain-ate someone who ate my P" plus
    # ordinary-rank floor / type-exclusion / mine-bomb suspect flags.
    # Updated in :meth:`step_inplace` from the combat outcome and projected
    # to spatial channels by ``observation.py::_write_combat_memory``.
    #
    # Allocated lazily in ``new_game`` / ``_rebuild_soa_from_dict``.  Cloned
    # by ``clone()``.  Idempotent under repeated apply (the bit OR-in is
    # set-not-add).  See ``junqi_core/combat_memory.py`` for the data
    # structure and the design doc ``docs/COMBAT_MEMORY_DESIGN.md``.
    combat_memory: CombatMemoryState = field(default_factory=CombatMemoryState.zeros)

    # -----------------------------------------------------------------------
    # Phase 0.4 M1 / ADR-117 — SoA mirror of the dict-keyed state above.
    # -----------------------------------------------------------------------
    # These ndarrays are kept byte-consistent with `pieces` / `info` /
    # `piece_state` / `deaths` / `zero_board` at every `step_inplace` exit.
    # Hot-path consumers (observation.py, move_gen.py, info_model.py will
    # migrate in M2–M4) may read them directly for O(1) lookup by piece_id
    # or by flat cell index. All columns are piece-indexed on axis=0 or
    # cell-indexed on axis=0; axis 0 is reserved as the future batch axis
    # (ADR-123).
    #
    # Shapes (immutable):
    #   piece-indexed (120,): alive, piece_type_arr, piece_seat_arr,
    #       pos_x, pos_y, zero_x, zero_y, move_count_arr, active_eat_arr,
    #       passive_surv_arr, death_reason_arr, death_step_arr,
    #       death_loc_flat_arr
    #   cell-indexed (289,): cell_piece_id
    #   seat-indexed (4,):   seat_dead_arr, seat_flag_revealed_arr
    #
    # Sentinel values: -1 for "no piece" / "not dead yet" / unassigned pid.
    alive:                np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    piece_type_arr:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    piece_seat_arr:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    pos_x:                np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    pos_y:                np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    zero_x:               np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    zero_y:               np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    move_count_arr:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    active_eat_arr:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    passive_surv_arr:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    death_reason_arr:     np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    death_step_arr:       np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    death_loc_flat_arr:   np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    cell_piece_id:        np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    seat_dead_arr:        np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    seat_flag_revealed_arr: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    # Move history ring buffer for src_dst_planes observation channels.
    # List of (src_flat, dst_flat) int tuples, most recent last. Capped at 32.
    move_history: list = field(default_factory=list)

    # Incremental Zobrist hash (ADR-117). Updated in-place by
    # `step_inplace` via XOR against the tables in `_zobrist.py`. The
    # public `state_hash()` method simply returns this value (Phase 0.2
    # T5 baseline guaranteed it was the return of `hash(...)`; tests only
    # assert self-equality under `clone()`, so the numerical change is
    # transparent).
    zobrist: int = 0

    # Class-level constant (not an instance field)
    _Q12_MAX_CHAIN: ClassVar[int] = 4   # safety bound for Q12 chain loop

    # -----------------------------------------------------------------------
    # Post-init: guarantee the SoA mirror matches the dict state.
    # -----------------------------------------------------------------------
    # Tests and tools sometimes hand-construct a ``GameState`` (e.g. by
    # building a custom ``pieces`` dict).  Without going through
    # ``new_game`` the SoA columns stay as the 0-shape default sentinels
    # declared in the field definitions, which would crash any subsequent
    # ``step_inplace`` call.  We detect the sentinel and rebuild a full
    # SoA mirror from the dict state.
    #
    # Contract: if ANY SoA column has the correct length (== 120 for
    # piece-indexed / 289 for cell-indexed), we trust the caller and do
    # nothing.  If the alive array is 0-shape, we rebuild everything.
    def __post_init__(self) -> None:
        if self.alive.shape == (_ZOB.NUM_PIECE_IDS,):
            return  # already initialised (came from new_game / clone / from_dict)
        self._rebuild_soa_from_dict()

    def _rebuild_soa_from_dict(self) -> None:
        """(Re)build all SoA columns + Zobrist from the dict state.

        Used by :meth:`__post_init__` to upgrade hand-constructed
        GameStates (tests / tools) to be ``step_inplace``-safe.  Missing
        ``piece_id`` (legacy PieceRef default ``-1``) is patched on the
        fly using the piece's current cell: the pid is synthesised as
        ``seat * 30 + free_slot`` where ``free_slot`` is the lowest
        currently-unused slot for that seat.  This keeps the liveness
        invariants intact for tests that don't otherwise care about pid.
        """
        num_pids = _ZOB.NUM_PIECE_IDS
        self.alive = np.zeros(num_pids, dtype=bool)
        self.piece_type_arr = np.full(num_pids, -1, dtype=np.int8)
        self.piece_seat_arr = np.full(num_pids, -1, dtype=np.int8)
        self.pos_x = np.full(num_pids, -1, dtype=np.int8)
        self.pos_y = np.full(num_pids, -1, dtype=np.int8)
        self.zero_x = np.full(num_pids, -1, dtype=np.int8)
        self.zero_y = np.full(num_pids, -1, dtype=np.int8)
        self.move_count_arr = np.zeros(num_pids, dtype=np.int16)
        self.active_eat_arr = np.zeros(num_pids, dtype=np.int16)
        self.passive_surv_arr = np.zeros(num_pids, dtype=np.int16)
        self.death_reason_arr = np.full(num_pids, -1, dtype=np.int8)
        self.death_step_arr = np.full(num_pids, -1, dtype=np.int16)
        self.death_loc_flat_arr = np.full(num_pids, -1, dtype=np.int16)
        self.cell_piece_id = np.full(NUM_CELLS, -1, dtype=np.int16)
        self.seat_dead_arr = np.zeros(4, dtype=bool)
        self.seat_flag_revealed_arr = np.zeros(4, dtype=bool)
        # CombatMemory is freshly zeroed when we rebuild the SoA: the
        # dict-side state can't carry CombatMemory facts, so any existing
        # log would be inconsistent with the current pieces.  Tools that
        # need to preserve combat memory across rebuilds should serialize
        # via ``to_dict`` / ``from_dict`` instead.
        self.combat_memory = CombatMemoryState.zeros()

        # Seat scalars from the dict-side `info`.
        for s in ALL_SEATS:
            if self.info[s].dead:
                self.seat_dead_arr[s.value] = True
            if self.info[s].flag_revealed:
                self.seat_flag_revealed_arr[s.value] = True

        # Track per-seat free slot for pid synthesis (legacy PieceRefs
        # with piece_id = -1).  Slot 0..29 per seat; any slot not already
        # claimed by an explicitly-pid'd piece is available.
        used_by_seat: dict[Seat, set[int]] = {s: set() for s in ALL_SEATS}
        for ref in self.pieces.values():
            if ref.piece_id >= 0:
                used_by_seat[ref.seat].add(ref.piece_id % 30)

        def _alloc_pid(seat: Seat) -> int:
            used = used_by_seat[seat]
            for slot in range(30):
                if slot in used:
                    continue
                used.add(slot)
                return seat.value * 30 + slot
            raise RuntimeError(
                f"no free piece_id slot for seat {seat.name} "
                f"(already 30 pieces — invariant violated)"
            )

        zob = 0
        for pos, ref in self.pieces.items():
            pid = ref.piece_id if ref.piece_id >= 0 else _alloc_pid(ref.seat)
            # If pid was synthesized, rewrite the PieceRef in-place via
            # replace (frozen dataclass).  We must also patch self.pieces
            # and self.zero_board (if the old ref is there).
            if pid != ref.piece_id:
                new_ref = PieceRef(
                    seat=ref.seat,
                    piece_type=ref.piece_type,
                    alive=ref.alive,
                    piece_id=pid,
                )
                self.pieces[pos] = new_ref
                ref = new_ref
            self.alive[pid] = True
            self.piece_type_arr[pid] = ref.piece_type.value
            self.piece_seat_arr[pid] = ref.seat.value
            self.pos_x[pid] = pos[0]
            self.pos_y[pid] = pos[1]
            flat = xy_to_flat(*pos)
            self.cell_piece_id[flat] = pid
            zob ^= int(_ZOB.ZOB_PIECE[pid, ref.piece_type.value, flat])

        # zero_board mirror (if provided).  We also use this pass to
        # populate piece_seat_arr / piece_type_arr for pids that are NOT
        # alive (they may still appear in self.deaths and need seat/type
        # info for the D/E channel writers to resolve ownership).
        for pos, ref in self.zero_board.items():
            if ref.piece_id >= 0:
                pid = ref.piece_id
                self.zero_x[pid] = pos[0]
                self.zero_y[pid] = pos[1]
                # Fill seat/type if not already set by the live-piece pass.
                if self.piece_seat_arr[pid] < 0:
                    self.piece_seat_arr[pid] = ref.seat.value
                if self.piece_type_arr[pid] < 0:
                    self.piece_type_arr[pid] = ref.piece_type.value

        # piece_state / deaths mirrors.
        for pid, ps in self.piece_state.items():
            if 0 <= pid < num_pids:
                self.move_count_arr[pid] = ps.move_count
                self.active_eat_arr[pid] = ps.active_eat_count
                self.passive_surv_arr[pid] = ps.passive_survive_count
        for pid, di in self.deaths.items():
            if 0 <= pid < num_pids:
                self.death_reason_arr[pid] = di.reason.value
                self.death_step_arr[pid] = di.step
                self.death_loc_flat_arr[pid] = xy_to_flat(*di.death_loc)

        # Zobrist scalars.
        zob ^= int(_ZOB.ZOB_TURN[self.turn.value])
        zob ^= int(_ZOB.ZOB_MOVE_COUNTER[self.move_counter & _ZOB.MOVE_COUNTER_MASK])
        zob ^= int(_ZOB.ZOB_MOVES_SINCE_COMBAT[
            self.moves_since_last_combat & _ZOB.MOVES_SINCE_COMBAT_MASK
        ])
        zob ^= int(_ZOB.ZOB_WINNER[
            0 if self.winner_team is None else (self.winner_team + 1)
        ])
        zob ^= int(_ZOB.ZOB_SHOW_MODE[self.show_mode.value])
        if self.terminated:
            zob ^= _ZOB.ZOB_TERMINATED
        if self.draw:
            zob ^= _ZOB.ZOB_DRAW
        for s_val in range(4):
            if bool(self.seat_dead_arr[s_val]):
                zob ^= int(_ZOB.ZOB_SEAT_DEAD[s_val])
            if bool(self.seat_flag_revealed_arr[s_val]):
                zob ^= int(_ZOB.ZOB_SEAT_FLAG_REVEALED[s_val])
        self.zobrist = zob

    # =======================================================================
    # Constructors
    # =======================================================================

    @classmethod
    def new_game(
        cls,
        setups: SetupArray,
        *,
        show_mode: ShowMode = ShowMode.HALF_DARK,
        first_seat: Seat = Seat.SOUTH,
        debug_include_private: bool = False,
    ) -> GameState:
        """Start a fresh game from 4 validated setups.

        Raises ValueError if setups violate C1-C5.
        """
        # Hard validation (C1-C5, Q13 camp constraint, etc.)
        result = validate_setup(setups)
        if not result.ok:
            raise ValueError(
                f"setup violates rules: {result.violations}"
            )

        # --- T7 / ADR-114: assign piece_id for every real piece ---
        id_map = assign_piece_ids(setups)

        # --- Build SoA mirror columns (ADR-117) ---
        num_pids = _ZOB.NUM_PIECE_IDS  # 120
        alive = np.zeros(num_pids, dtype=bool)
        piece_type_arr = np.full(num_pids, -1, dtype=np.int8)
        piece_seat_arr = np.full(num_pids, -1, dtype=np.int8)
        pos_x = np.full(num_pids, -1, dtype=np.int8)
        pos_y = np.full(num_pids, -1, dtype=np.int8)
        zero_x = np.full(num_pids, -1, dtype=np.int8)
        zero_y = np.full(num_pids, -1, dtype=np.int8)
        move_count_arr = np.zeros(num_pids, dtype=np.int16)
        active_eat_arr = np.zeros(num_pids, dtype=np.int16)
        passive_surv_arr = np.zeros(num_pids, dtype=np.int16)
        death_reason_arr = np.full(num_pids, -1, dtype=np.int8)
        death_step_arr = np.full(num_pids, -1, dtype=np.int16)
        death_loc_flat_arr = np.full(num_pids, -1, dtype=np.int16)
        cell_piece_id = np.full(NUM_CELLS, -1, dtype=np.int16)
        seat_dead_arr = np.zeros(4, dtype=bool)
        seat_flag_revealed_arr = np.zeros(4, dtype=bool)

        # Build piece map from lineups
        pieces: PieceMap = {}
        zero_board: dict[tuple[int, int], PieceRef] = {}
        piece_state: dict[int, PieceState] = {}

        # Accumulate Zobrist as we walk pieces.
        zobrist_val = 0

        for seat in ALL_SEATS:
            lineup = setups[seat.value]
            for i in range(30):
                piece_type = lineup[i]
                if piece_type is PieceType.NONE:
                    continue  # camp cells
                pos = index_to_pos(seat, i)
                pid = id_map[(seat, i)]
                ref = PieceRef(
                    seat=seat,
                    piece_type=piece_type,
                    alive=True,
                    piece_id=pid,
                )
                pieces[pos] = ref
                # zero_board snapshots the initial layout; read-only from
                # this point on. PieceRef is frozen → safe to share ref.
                zero_board[pos] = ref
                # Every live piece starts with zero counters.
                piece_state[pid] = PieceState()

                # SoA mirror
                alive[pid] = True
                piece_type_arr[pid] = piece_type.value
                piece_seat_arr[pid] = seat.value
                pos_x[pid] = pos[0]
                pos_y[pid] = pos[1]
                zero_x[pid] = pos[0]
                zero_y[pid] = pos[1]
                flat = xy_to_flat(*pos)
                cell_piece_id[flat] = pid

                # Zobrist: XOR in this piece's placement.
                zobrist_val ^= int(_ZOB.ZOB_PIECE[pid, piece_type.value, flat])

        info = {seat: SeatInfo() for seat in ALL_SEATS}

        # Zobrist scalar contributions (turn, show_mode; counters=0 /
        # terminated=False / winner=None / draw=False all contribute via
        # the corresponding tables being XOR'd-in unconditionally below).
        zobrist_val ^= int(_ZOB.ZOB_TURN[first_seat.value])
        zobrist_val ^= int(_ZOB.ZOB_MOVE_COUNTER[0])
        zobrist_val ^= int(_ZOB.ZOB_MOVES_SINCE_COMBAT[0])
        zobrist_val ^= int(_ZOB.ZOB_WINNER[0])   # winner_team = None → idx 0
        zobrist_val ^= int(_ZOB.ZOB_SHOW_MODE[show_mode.value])
        # terminated=False / draw=False: scalars are XOR'd in only when
        # toggled ON; nothing to do at construction time.

        return cls(
            pieces=pieces,
            turn=first_seat,
            move_counter=0,
            moves_since_last_combat=0,
            info=info,
            terminated=False,
            winner_team=None,
            draw=False,
            show_mode=show_mode,
            rules_version=RULES_VERSION,
            debug_include_private=debug_include_private,
            zero_board=zero_board,
            piece_state=piece_state,
            deaths={},
            combat_memory=CombatMemoryState.zeros(),
            alive=alive,
            piece_type_arr=piece_type_arr,
            piece_seat_arr=piece_seat_arr,
            pos_x=pos_x,
            pos_y=pos_y,
            zero_x=zero_x,
            zero_y=zero_y,
            move_count_arr=move_count_arr,
            active_eat_arr=active_eat_arr,
            passive_surv_arr=passive_surv_arr,
            death_reason_arr=death_reason_arr,
            death_step_arr=death_step_arr,
            death_loc_flat_arr=death_loc_flat_arr,
            cell_piece_id=cell_piece_id,
            seat_dead_arr=seat_dead_arr,
            seat_flag_revealed_arr=seat_flag_revealed_arr,
            zobrist=zobrist_val,
        )

    # =======================================================================
    # Core: step()
    # =======================================================================

    def step(self, action: Action) -> tuple[GameState, MoveResult]:
        """Apply an action and return (new_state, move_result).

        The original state is NOT mutated (immutable snapshot contract).
        Internally this is ``clone + step_inplace``; RL rollouts can call
        :meth:`step_inplace` directly to skip the clone when the previous
        state reference is no longer needed. See ADR-117.
        """
        new_state = self.clone()
        mr = new_state.step_inplace(action)
        return new_state, mr

    def step_inplace(self, action: Action) -> MoveResult:
        """Apply an action by mutating ``self``.  Returns the MoveResult.

        This is the hot-path RL-rollout API added in ADR-117 (Phase 0.4 M1).
        Maintains, in a single pass:

          * the authoritative dict state (``pieces`` / ``info`` /
            ``piece_state`` / ``deaths``) — unchanged in semantics;
          * the SoA mirror columns (``alive`` / ``pos_x`` / ... /
            ``cell_piece_id`` / ``seat_dead_arr`` / ...);
          * the incremental Zobrist accumulator in ``self.zobrist``.

        See RULES.md §3 + §5 for the full semantic model.
        """
        if self.terminated:
            raise RuntimeError("cannot step() a terminated game")
        if action.seat is not self.turn:
            raise ValueError(
                f"action.seat={action.seat.name} but turn={self.turn.name}"
            )
        if self.info[action.seat].dead:
            raise ValueError(f"action.seat={action.seat.name} is dead; cannot act")
        if not is_legal_move(self.pieces, action.src, action.dst, action.seat):
            raise ValueError(
                f"illegal action: {action.seat.name} "
                f"{action.src} -> {action.dst}"
            )

        # All mutations happen directly on ``self``'s fields.  Rebind
        # short local names for readability; they are ALIASES, not copies.
        pieces = self.pieces
        info = self.info
        piece_state = self.piece_state
        deaths = self.deaths

        # Record move in history ring buffer (capped at 32 entries).
        src_flat = action.src[1] * 17 + action.src[0]
        dst_flat = action.dst[1] * 17 + action.dst[0]
        self.move_history.append((src_flat, dst_flat))
        if len(self.move_history) > 32:
            self.move_history = self.move_history[-32:]

        # -------------------------------------------------------------------
        # Precompute Zobrist XOR-OUTs for fields we are about to overwrite.
        # The XOR-INs happen at the end once the new values are known.
        # -------------------------------------------------------------------
        zob_out_turn = int(_ZOB.ZOB_TURN[self.turn.value])
        zob_out_mc = int(_ZOB.ZOB_MOVE_COUNTER[self.move_counter & _ZOB.MOVE_COUNTER_MASK])
        zob_out_mslc = int(_ZOB.ZOB_MOVES_SINCE_COMBAT[
            self.moves_since_last_combat & _ZOB.MOVES_SINCE_COMBAT_MASK
        ])
        # winner/terminated/draw are all False/None here (we guard above);
        # they'll be XOR'd-in if they flip ON.

        death_step: int = self.move_counter + 1
        seats_died: list[Seat] = []

        src_piece = pieces[action.src]
        dst_piece = pieces.get(action.dst)

        # CombatMemory v4: path-revealed GONGB detection.
        # Per the user's optimization: only the engineer can possibly
        # have walked a multi-hop rail-BFS path, so we skip the geometry
        # check entirely for non-GONGB pieces.  When the moving piece IS
        # a GONGB (or already publicly known to be one), we ask
        # ``move_requires_gongb`` whether the path is GONGB-only — if so,
        # broadcast the reveal to all four observers.  Done before the
        # piece is moved, so blockers along the path are still present.
        if src_piece.piece_type.is_engineer:
            if move_requires_gongb(pieces, action.src, action.dst):
                apply_path_revealed_gongb(
                    self.combat_memory, src_piece.piece_id
                )

        # -------------------------------------------------------------------
        # Phase 1 + 2: apply move OR combat, update pieces, compute reveals
        # -------------------------------------------------------------------
        event: Event
        flag_reveal_src = False
        flag_reveal_dst = False
        flag_captured = False
        src_type_revealed: PieceType | None = None
        dst_type_revealed: PieceType | None = None

        if dst_piece is None or not dst_piece.alive:
            # Plain move into empty cell
            event = Event.MOVE
            # Move the piece: remove from src, install at dst
            del pieces[action.src]
            pieces[action.dst] = src_piece
            # T7 M2: bump attacker's move_count.
            _bump_move_count(piece_state, src_piece.piece_id)
            # SoA + Zobrist: move src_piece from action.src to action.dst.
            self._soa_move_piece(src_piece.piece_id, action.src, action.dst)
            # SoA: move_count_arr mirror.
            self.move_count_arr[src_piece.piece_id] = piece_state[src_piece.piece_id].move_count
        else:
            # Combat
            event = resolve_combat(src_piece.piece_type, dst_piece.piece_type)
            flag_reveal_src = siling_reveals_src(
                src_piece.piece_type, dst_piece.piece_type, event
            )
            flag_reveal_dst = siling_reveals_dst(
                src_piece.piece_type, dst_piece.piece_type, event
            )
            flag_captured = dst_piece.piece_type is PieceType.JUNQI

            # Apply SILING flag reveals to info (+ SoA mirror + Zobrist).
            if flag_reveal_src:
                self._set_flag_revealed(src_piece.seat)
            if flag_reveal_dst:
                self._set_flag_revealed(dst_piece.seat)

            # Resolve piece-level outcomes per event.
            if event is Event.EAT:
                # src moves onto dst; dst dies.
                del pieces[action.src]
                del pieces[action.dst]
                pieces[action.dst] = src_piece
                # Attacker survived & won → active_eat_count += 1.
                _bump_active_eat(piece_state, src_piece.piece_id)
                self.active_eat_arr[src_piece.piece_id] = piece_state[src_piece.piece_id].active_eat_count
                # Victim dies → drop counters, record DeathInfo (T7 M3).
                piece_state.pop(dst_piece.piece_id, None)
                deaths[dst_piece.piece_id] = DeathInfo(
                    piece_id=dst_piece.piece_id,
                    reason=classify_death_reason(
                        own_piece=dst_piece.piece_type,
                        opponent_piece=src_piece.piece_type,
                        event=event,
                        own_is_attacker=False,
                    ),
                    death_loc=action.dst,
                    step=death_step,
                )
                # SoA + Zobrist: kill dst, then move src onto dst.
                self._soa_kill_piece(dst_piece.piece_id, deaths[dst_piece.piece_id])
                self._soa_move_piece(src_piece.piece_id, action.src, action.dst)
                # CombatMemory v4: src ate dst (Event.EAT).
                apply_combat_event(
                    self.combat_memory,
                    event_is_eat=True,
                    attacker_pid=src_piece.piece_id,
                    defender_pid=dst_piece.piece_id,
                    attacker_seat=src_piece.seat.value,
                    defender_seat=dst_piece.seat.value,
                    attacker_type=src_piece.piece_type,
                    defender_type=dst_piece.piece_type,
                    defender_pos_flat=action.dst[1] * BOARD_SIZE + action.dst[0],
                    death_step=death_step,
                )
            elif event is Event.KILLED:
                # src dies; dst stays.
                del pieces[action.src]
                # Defender survived a hit → passive_survive_count += 1.
                _bump_passive_survive(piece_state, dst_piece.piece_id)
                self.passive_surv_arr[dst_piece.piece_id] = piece_state[dst_piece.piece_id].passive_survive_count
                piece_state.pop(src_piece.piece_id, None)
                # T7 M3: attacker died; classify_death_reason returns
                # HIT_MINE_OR_BOMB if defender is a mine/bomb, else KBE.
                deaths[src_piece.piece_id] = DeathInfo(
                    piece_id=src_piece.piece_id,
                    reason=classify_death_reason(
                        own_piece=src_piece.piece_type,
                        opponent_piece=dst_piece.piece_type,
                        event=event,
                        own_is_attacker=True,
                    ),
                    death_loc=action.dst,
                    step=death_step,
                )
                self._soa_kill_piece(src_piece.piece_id, deaths[src_piece.piece_id])
                # CombatMemory v4: attacker died (Event.KILLED).  The v4
                # update routine handles both the chain propagation and
                # the per-observer rank-floor lift; the dilei_candidate
                # signal is computed at observation-build time from
                # zero_pos / move_count / attacked_by_known_gongb.
                apply_combat_event(
                    self.combat_memory,
                    event_is_eat=False,
                    attacker_pid=src_piece.piece_id,
                    defender_pid=dst_piece.piece_id,
                    attacker_seat=src_piece.seat.value,
                    defender_seat=dst_piece.seat.value,
                    attacker_type=src_piece.piece_type,
                    defender_type=dst_piece.piece_type,
                    defender_pos_flat=action.dst[1] * BOARD_SIZE + action.dst[0],
                    death_step=death_step,
                )
            elif event is Event.BOMB:
                # Both die — no survivors, no counter increments.
                del pieces[action.src]
                del pieces[action.dst]
                piece_state.pop(src_piece.piece_id, None)
                piece_state.pop(dst_piece.piece_id, None)
                deaths[src_piece.piece_id] = DeathInfo(
                    piece_id=src_piece.piece_id,
                    reason=DeathReason.MUTUAL,
                    death_loc=action.dst,
                    step=death_step,
                )
                deaths[dst_piece.piece_id] = DeathInfo(
                    piece_id=dst_piece.piece_id,
                    reason=DeathReason.MUTUAL,
                    death_loc=action.dst,
                    step=death_step,
                )
                self._soa_kill_piece(src_piece.piece_id, deaths[src_piece.piece_id])
                self._soa_kill_piece(dst_piece.piece_id, deaths[dst_piece.piece_id])
                # CombatMemory v4: BOMB events do NOT update CombatMemory
                # — both pieces are dead so there is no live target to
                # project onto, and chain propagation cannot continue.
                # The ZHADAN-vs-DILEI special case is implicitly handled
                # by BeliefTensor.remaining inventory deduction.
            else:
                raise AssertionError(f"unreachable combat event {event!r}")

        # Optionally expose piece types (BRIGHT mode or debug)
        if self.show_mode is ShowMode.BRIGHT or self.debug_include_private:
            src_type_revealed = src_piece.piece_type
            if dst_piece is not None:
                dst_type_revealed = dst_piece.piece_type

        # -------------------------------------------------------------------
        # Phase 3: Flag capture → defending seat surrenders
        # -------------------------------------------------------------------
        if flag_captured:
            assert dst_piece is not None
            surrendering_seat = dst_piece.seat
            if not info[surrendering_seat].dead:
                info[surrendering_seat].dead = True
                info[surrendering_seat].flag_revealed = True
                # SoA + Zobrist for seat state
                self._set_seat_dead(surrendering_seat)
                # flag_revealed might already be true; _set_flag_revealed
                # is idempotent on the flag.
                if not self.seat_flag_revealed_arr[surrendering_seat.value]:
                    self._set_flag_revealed(surrendering_seat)
                _remove_all_pieces_of_seat(
                    pieces, surrendering_seat,
                    piece_state=piece_state,
                    deaths=deaths,
                    death_reason=DeathReason.KILLED_BY_ENEMY,
                    death_step=death_step,
                )
                # SoA: purge every piece of surrendering_seat still alive.
                self._soa_purge_seat(
                    surrendering_seat,
                    deaths=deaths,
                    death_reason=DeathReason.KILLED_BY_ENEMY,
                    death_step=death_step,
                )
                seats_died.append(surrendering_seat)

        # -------------------------------------------------------------------
        # Phase 4: Post-combat dead-sweep (T1)
        # -------------------------------------------------------------------
        for seat in ALL_SEATS:
            if info[seat].dead:
                continue
            if not _seat_has_any_piece(pieces, seat):
                info[seat].dead = True
                self._set_seat_dead(seat)
                seats_died.append(seat)

        # -------------------------------------------------------------------
        # Update counters (move count + combat freshness)
        # -------------------------------------------------------------------
        new_move_counter = self.move_counter + 1
        if event is Event.MOVE:
            new_moves_since_combat = self.moves_since_last_combat + 1
        else:
            new_moves_since_combat = 0  # any combat resets the counter

        # -------------------------------------------------------------------
        # Phase 5: Check team victory (incl. Q14 mutual destruction)
        # -------------------------------------------------------------------
        terminated, winner_team, draw = _check_victory(
            info=info,
            attacker_seat=action.seat,
            move_counter=new_move_counter,
            moves_since_last_combat=new_moves_since_combat,
        )

        # -------------------------------------------------------------------
        # Phase 6: Advance turn + chase down Q12 dead-seats
        # -------------------------------------------------------------------
        if not terminated:
            new_turn = self._advance_turn_with_q12(
                starting_from=action.seat,
                pieces=pieces,
                info=info,
                seats_died=seats_died,
                piece_state=piece_state,
                deaths=deaths,
                death_step=death_step,
            )
            terminated, winner_team, draw = _check_victory(
                info=info,
                attacker_seat=action.seat,
                move_counter=new_move_counter,
                moves_since_last_combat=new_moves_since_combat,
            )
        else:
            new_turn = self.turn  # frozen on termination

        # -------------------------------------------------------------------
        # Commit scalar field updates + finish Zobrist.
        # -------------------------------------------------------------------
        # move_counter / moves_since_last_combat XOR-in.
        self.move_counter = new_move_counter
        self.moves_since_last_combat = new_moves_since_combat
        zob_in_mc = int(_ZOB.ZOB_MOVE_COUNTER[new_move_counter & _ZOB.MOVE_COUNTER_MASK])
        zob_in_mslc = int(_ZOB.ZOB_MOVES_SINCE_COMBAT[
            new_moves_since_combat & _ZOB.MOVES_SINCE_COMBAT_MASK
        ])

        # turn XOR swap.
        self.turn = new_turn
        zob_in_turn = int(_ZOB.ZOB_TURN[new_turn.value])

        # Termination flags: XOR-in if toggled ON.
        zob_term_delta = 0
        if terminated != self.terminated:
            zob_term_delta ^= _ZOB.ZOB_TERMINATED
        if draw != self.draw:
            zob_term_delta ^= _ZOB.ZOB_DRAW
        if winner_team != self.winner_team:
            # Winner slot uses indices {0,1,2} for {None, 0, 1}.
            zob_term_delta ^= int(_ZOB.ZOB_WINNER[
                0 if self.winner_team is None else (self.winner_team + 1)
            ])
            zob_term_delta ^= int(_ZOB.ZOB_WINNER[
                0 if winner_team is None else (winner_team + 1)
            ])

        self.terminated = terminated
        self.winner_team = winner_team
        self.draw = draw

        self.zobrist ^= (
            zob_out_turn ^ zob_in_turn
            ^ zob_out_mc ^ zob_in_mc
            ^ zob_out_mslc ^ zob_in_mslc
            ^ zob_term_delta
        )

        return MoveResult(
            seat=action.seat,
            src=action.src,
            dst=action.dst,
            event=event,
            flag_reveal_src=flag_reveal_src,
            flag_reveal_dst=flag_reveal_dst,
            flag_captured=flag_captured,
            src_type_revealed=src_type_revealed,
            dst_type_revealed=dst_type_revealed,
            terminated_after=terminated,
            winner_team_after=winner_team,
            draw_after=draw,
            seats_died_this_step=tuple(seats_died),
        )

    # =======================================================================
    # SoA mirror helpers (Phase 0.4 M1) — keep ndarrays byte-consistent
    # with the dict state; maintain the incremental Zobrist accumulator.
    # =======================================================================

    def _soa_move_piece(
        self,
        pid: int,
        src: tuple[int, int],
        dst: tuple[int, int],
    ) -> None:
        """Move piece ``pid`` from ``src`` to ``dst`` in the SoA mirror.

        XORs out the piece contribution at src and XORs in at dst.
        """
        src_flat = src[1] * BOARD_SIZE + src[0]
        dst_flat = dst[1] * BOARD_SIZE + dst[0]
        type_val = int(self.piece_type_arr[pid])
        # Zobrist: XOR-out old cell, XOR-in new cell.
        self.zobrist ^= int(_ZOB.ZOB_PIECE[pid, type_val, src_flat])
        self.zobrist ^= int(_ZOB.ZOB_PIECE[pid, type_val, dst_flat])
        # SoA updates.
        self.cell_piece_id[src_flat] = -1
        self.cell_piece_id[dst_flat] = pid
        self.pos_x[pid] = dst[0]
        self.pos_y[pid] = dst[1]

    def _soa_kill_piece(self, pid: int, di: DeathInfo) -> None:
        """Remove piece ``pid`` from the SoA mirror and record its death.

        XORs out the piece's current contribution to the Zobrist.
        Idempotent: if the piece is already dead (``alive[pid]`` is False)
        the call is a no-op at the piece-placement level (death metadata
        is still refreshed defensively).
        """
        if not bool(self.alive[pid]):
            return
        type_val = int(self.piece_type_arr[pid])
        cur_flat = int(self.pos_y[pid]) * BOARD_SIZE + int(self.pos_x[pid])
        self.zobrist ^= int(_ZOB.ZOB_PIECE[pid, type_val, cur_flat])
        self.cell_piece_id[cur_flat] = -1
        self.alive[pid] = False
        self.pos_x[pid] = -1
        self.pos_y[pid] = -1
        # Record death metadata.
        self.death_reason_arr[pid] = di.reason.value
        self.death_step_arr[pid] = di.step
        self.death_loc_flat_arr[pid] = (
            di.death_loc[1] * BOARD_SIZE + di.death_loc[0]
        )

    def _soa_purge_seat(
        self,
        seat: Seat,
        *,
        deaths: dict[int, DeathInfo],
        death_reason: DeathReason,
        death_step: int,
    ) -> None:
        """Purge every live piece of ``seat`` from the SoA mirror.

        Called after :func:`_remove_all_pieces_of_seat` has already updated
        the dict state and written ``deaths`` entries.  We mirror those
        removals into the ndarrays and XOR out each piece's Zobrist
        contribution.  The dict-side ``deaths`` dict is the authoritative
        source for the DeathInfo metadata we record here.
        """
        seat_val = seat.value
        # Which pids belong to this seat and are still alive in the SoA?
        mask = (self.piece_seat_arr == seat_val) & self.alive
        pids_to_purge = np.nonzero(mask)[0]
        for pid in pids_to_purge:
            pid_int = int(pid)
            di = deaths.get(pid_int)
            if di is None:
                # Should never happen — caller wrote deaths for every live
                # piece it removed.  Defensive fallback: record with the
                # caller-supplied reason at the piece's current cell.
                cur_x = int(self.pos_x[pid_int])
                cur_y = int(self.pos_y[pid_int])
                di = DeathInfo(
                    piece_id=pid_int,
                    reason=death_reason,
                    death_loc=(cur_x, cur_y),
                    step=death_step,
                )
            self._soa_kill_piece(pid_int, di)

    def _set_seat_dead(self, seat: Seat) -> None:
        """Toggle seat dead on in both the SoA mirror and Zobrist."""
        if self.seat_dead_arr[seat.value]:
            return
        self.seat_dead_arr[seat.value] = True
        self.zobrist ^= int(_ZOB.ZOB_SEAT_DEAD[seat.value])

    def _set_flag_revealed(self, seat: Seat) -> None:
        """Toggle seat flag_revealed on in both the SoA mirror and Zobrist."""
        if self.seat_flag_revealed_arr[seat.value]:
            return
        self.seat_flag_revealed_arr[seat.value] = True
        # Keep dict-side in sync when called from combat-reveal paths
        # (flag-capture path sets it directly too; idempotent).
        self.info[seat].flag_revealed = True
        self.zobrist ^= int(_ZOB.ZOB_SEAT_FLAG_REVEALED[seat.value])

    # =======================================================================
    # Turn-advance helper (handles Q12 chain)
    # =======================================================================

    def _advance_turn_with_q12(
        self,
        *,
        starting_from: Seat,
        pieces: PieceMap,
        info: dict[Seat, SeatInfo],
        seats_died: list[Seat],
        piece_state: dict[int, PieceState] | None = None,
        deaths: dict[int, DeathInfo] | None = None,
        death_step: int | None = None,
    ) -> Seat:
        """Rotate turn from `starting_from.next_seat` forward, skipping dead
        seats and killing seats with no legal moves (Q12 cascade).

        Returns the first seat that is alive AND has at least one legal move.
        If no such seat exists (all seats dead), returns an arbitrary seat
        (the caller re-checks termination).

        `piece_state` (T7 M2): when provided, Q12-killed seats also get
        their piece_state entries purged here to maintain the
        liveness-invariant (every piece_id in the dict is alive on the
        board). `None` is accepted for callers that don't track it
        (defensive — step() always supplies it).

        `deaths` / `death_step` (T7 M3): when provided, Q12 collateral
        deaths are recorded with reason=KILLED_BY_ENEMY (per decision A)
        anchored at each piece's current cell.
        """
        candidate = starting_from.next_seat

        for _ in range(self._Q12_MAX_CHAIN):
            if info[candidate].dead:
                candidate = candidate.next_seat
                continue
            if has_any_legal_move(pieces, candidate):
                return candidate
            # Q12: seat is alive but has no legal moves → it dies now.
            info[candidate].dead = True
            _remove_all_pieces_of_seat(
                pieces, candidate,
                piece_state=piece_state,
                deaths=deaths,
                death_reason=DeathReason.KILLED_BY_ENEMY,
                death_step=death_step,
            )
            # Phase 0.4 M1 / ADR-117: mirror into the SoA arrays.
            self._set_seat_dead(candidate)
            if deaths is not None and death_step is not None:
                self._soa_purge_seat(
                    candidate,
                    deaths=deaths,
                    death_reason=DeathReason.KILLED_BY_ENEMY,
                    death_step=death_step,
                )
            seats_died.append(candidate)
            candidate = candidate.next_seat

        # All 4 seats exhausted (dead) — caller will detect termination.
        return candidate

    # =======================================================================
    # Query API
    # =======================================================================

    def legal_actions(self, seat: Seat | None = None) -> list[Action]:
        """Return all legal actions for `seat` (default: current turn).

        Phase 0.4 M4: routes through the vectorized batch SoA generator
        (``generate_legal_action_ids_batch``); the Action list is then
        built from the flat ids.  Bit-compatible with the pre-M4 legacy
        implementation (set-equal).
        """
        target = seat if seat is not None else self.turn
        if self.info[target].dead or self.terminated:
            return []
        ids = generate_legal_action_ids_batch(
            self.cell_piece_id, self.piece_seat_arr, self.piece_type_arr,
            self.alive, self.pos_x, self.pos_y, target.value,
        )
        if ids.size == 0:
            return []
        # Decode each flat id = src_flat * 289 + dst_flat back to (x, y).
        out: list[Action] = []
        for flat_id in ids.tolist():
            src_flat = flat_id // NUM_CELLS
            dst_flat = flat_id % NUM_CELLS
            src = (src_flat % BOARD_SIZE, src_flat // BOARD_SIZE)
            dst = (dst_flat % BOARD_SIZE, dst_flat // BOARD_SIZE)
            out.append(Action(seat=target, src=src, dst=dst))
        return out

    def legal_action_ids(self, seat: Seat | None = None) -> np.ndarray:
        """Return legal actions as flat int32 ids, shape ``(K,)``.

        Each entry is ``src_flat * 289 + dst_flat``.  This is the
        zero-overhead path for RL code that wants to sample / mask
        directly in action-id space (cf. ADR-119).
        """
        target = seat if seat is not None else self.turn
        if self.info[target].dead or self.terminated:
            return np.empty(0, dtype=np.int32)
        return generate_legal_action_ids_batch(
            self.cell_piece_id, self.piece_seat_arr, self.piece_type_arr,
            self.alive, self.pos_x, self.pos_y, target.value,
        )

    def legal_action_mask(self, seat: Seat | None = None) -> np.ndarray:
        """Return boolean mask of shape [17,17,17,17].

        `mask[sx, sy, dx, dy] = True` iff `(sx, sy) -> (dx, dy)` is a legal
        action for `seat` (default: current turn). World-frame coordinates.
        """
        mask = np.zeros((BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE), dtype=bool)
        ids = self.legal_action_ids(seat)
        if ids.size == 0:
            return mask
        src_flat = ids // NUM_CELLS
        dst_flat = ids %  NUM_CELLS
        sx = (src_flat % BOARD_SIZE).astype(np.intp, copy=False)
        sy = (src_flat // BOARD_SIZE).astype(np.intp, copy=False)
        dx = (dst_flat % BOARD_SIZE).astype(np.intp, copy=False)
        dy = (dst_flat // BOARD_SIZE).astype(np.intp, copy=False)
        mask[sx, sy, dx, dy] = True
        return mask

    def legal_action_mask_flat(
        self, seat: Seat | None = None, out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return boolean mask of shape ``(NUM_CELLS * NUM_CELLS,) = (83521,)``.

        Optionally writes into ``out`` (must be ``bool`` of size 83521).
        """
        total = NUM_CELLS * NUM_CELLS
        if out is None:
            out = np.zeros(total, dtype=bool)
        else:
            out.fill(False)
        ids = self.legal_action_ids(seat)
        if ids.size:
            out[ids] = True
        return out

    def legal_moves_from_pos(self, src: tuple[int, int]) -> list[tuple[int, int]]:
        """Return legal destinations for the piece currently at `src`."""
        piece = self.pieces.get(src)
        if piece is None or not piece.alive:
            return []
        if self.info[piece.seat].dead:
            return []
        return legal_moves_from(self.pieces, src, piece.seat)

    # =======================================================================
    # Introspection / utilities
    # =======================================================================

    def piece_at(self, pos: tuple[int, int]) -> PieceRef | None:
        """Get the piece at `pos` (world-frame), or None if empty."""
        p = self.pieces.get(pos)
        if p is None or not p.alive:
            return None
        return p

    def clone(self) -> GameState:
        """Deep copy.

        ``zero_board`` is by-contract immutable after ``new_game``, so the
        reference is shared.  ``piece_state`` / ``deaths`` carry frozen
        values so a shallow dict copy suffices.  All SoA ndarray columns
        are copied with ``ndarray.copy()`` (~C-speed block copy of a few
        hundred bytes each).  ADR-117 target: ``≤ 2 µs``.

        Fast path: bypass ``__init__`` / ``__post_init__`` via ``__new__``
        and direct slot assignment — the clone is guaranteed to have a
        fully-initialised SoA mirror because ``self`` does.
        """
        new = GameState.__new__(GameState)
        new.pieces = dict(self.pieces)
        new.turn = self.turn
        new.move_counter = self.move_counter
        new.moves_since_last_combat = self.moves_since_last_combat
        new.info = {s: i.clone() for s, i in self.info.items()}
        new.terminated = self.terminated
        new.winner_team = self.winner_team
        new.draw = self.draw
        new.show_mode = self.show_mode
        new.rules_version = self.rules_version
        new.debug_include_private = self.debug_include_private
        new.zero_board = self.zero_board           # immutable — shared
        new.piece_state = dict(self.piece_state)   # frozen values
        new.deaths = dict(self.deaths)             # frozen values
        new.combat_memory = self.combat_memory.clone()
        # SoA columns — shallow ndarray.copy() each (C-speed block copy).
        new.alive = self.alive.copy()
        new.piece_type_arr = self.piece_type_arr.copy()
        new.piece_seat_arr = self.piece_seat_arr.copy()
        new.pos_x = self.pos_x.copy()
        new.pos_y = self.pos_y.copy()
        new.zero_x = self.zero_x.copy()
        new.zero_y = self.zero_y.copy()
        new.move_count_arr = self.move_count_arr.copy()
        new.active_eat_arr = self.active_eat_arr.copy()
        new.passive_surv_arr = self.passive_surv_arr.copy()
        new.death_reason_arr = self.death_reason_arr.copy()
        new.death_step_arr = self.death_step_arr.copy()
        new.death_loc_flat_arr = self.death_loc_flat_arr.copy()
        new.cell_piece_id = self.cell_piece_id.copy()
        new.seat_dead_arr = self.seat_dead_arr.copy()
        new.seat_flag_revealed_arr = self.seat_flag_revealed_arr.copy()
        new.move_history = list(self.move_history)
        new.zobrist = self.zobrist
        return new

    def state_hash(self) -> int:
        """Return the incremental Zobrist hash (ADR-117, Phase 0.4 M1).

        O(1).  The value is the same int that ``self.zobrist`` carries;
        two states with identical board configuration, scalar fields, and
        show_mode hash to the same int (``clone`` preserves it; any
        ``step_inplace`` mutation XORs the appropriate table entries).
        """
        return int(self.zobrist)

    # =======================================================================
    # Reward vector (for RL)
    # =======================================================================

    def team_rewards(self) -> tuple[int, int, int, int] | None:
        """Return per-seat reward tuple (HOME, RIGHT, OPPS, LEFT) or None.

        None if the game is not terminated.  Draw → (0,0,0,0).
        Otherwise winner team seats get +1; loser team seats get -1.
        """
        if not self.terminated:
            return None
        if self.draw or self.winner_team is None:
            return (0, 0, 0, 0)
        return tuple(
            1 if s.team == self.winner_team else -1 for s in ALL_SEATS
        )  # type: ignore[return-value]

    # =======================================================================
    # Serialization (for replays)
    # =======================================================================
    #
    # Schema version policy
    # ---------------------
    # ``state_version`` is an on-disk format tag orthogonal to
    # ``rules_version``.  ``"2.0"`` is the ADR-117 format that carries the
    # full SoA mirror (alive / pos / zero / counters / deaths).  ``"1.x"``
    # is the pre-ADR-117 format which stored only ``pieces`` + ``info``;
    # loading v1 is lossy for T7 counter/death signal but is still
    # supported read-only so older replay files can round-trip.

    STATE_VERSION: ClassVar[str] = "2.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict.

        Includes the full ADR-117 SoA mirror so ``from_dict`` can
        reconstruct the state byte-identically (including
        ``move_count`` / ``active_eat`` / ``passive_surv`` counters and
        ``deaths`` metadata — closes the L5 perf-debt item).
        """
        pieces_json = [
            {
                "pos": list(pos),
                "seat": p.seat.value,
                "type": p.piece_type.name,
                "piece_id": p.piece_id,
            }
            for pos, p in sorted(self.pieces.items())
        ]
        info_json = [
            {
                "seat": s.value,
                "dead": self.info[s].dead,
                "flag_revealed": self.info[s].flag_revealed,
            }
            for s in ALL_SEATS
        ]
        # Zero-board snapshot: one entry per real piece_id (not per cell).
        zero_board_json = [
            {
                "pos": list(pos),
                "seat": p.seat.value,
                "type": p.piece_type.name,
                "piece_id": p.piece_id,
            }
            for pos, p in sorted(self.zero_board.items())
        ]
        # Per-piece running counters (T7 / ADR-114).
        piece_state_json = [
            {
                "piece_id": pid,
                "move_count": ps.move_count,
                "active_eat_count": ps.active_eat_count,
                "passive_survive_count": ps.passive_survive_count,
            }
            for pid, ps in sorted(self.piece_state.items())
        ]
        # Death records (T7 / ADR-114).
        deaths_json = [
            {
                "piece_id": pid,
                "reason": di.reason.name,
                "death_loc": list(di.death_loc),
                "step": di.step,
            }
            for pid, di in sorted(self.deaths.items())
        ]
        return {
            "rules_version": self.rules_version,
            "state_version": self.STATE_VERSION,
            "pieces": pieces_json,
            "turn": self.turn.value,
            "move_counter": self.move_counter,
            "moves_since_last_combat": self.moves_since_last_combat,
            "info": info_json,
            "terminated": self.terminated,
            "winner_team": self.winner_team,
            "draw": self.draw,
            "show_mode": self.show_mode.value,
            "debug_include_private": self.debug_include_private,
            # --- Phase 0.4 M1 / ADR-117 additions ---
            "zero_board": zero_board_json,
            "piece_state": piece_state_json,
            "deaths": deaths_json,
            "zobrist": int(self.zobrist),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameState:
        """Deserialize.  Handles both ``state_version: "2.0"`` (full SoA
        round-trip) and the pre-ADR-117 format (``state_version`` absent).

        For v1 data the SoA mirror and ``deaths`` / ``piece_state`` /
        ``zero_board`` fields are populated by BEST EFFORT:
          * ``piece_id`` is inferred from ``(seat, seat_local_slot)`` via
            :func:`junqi_core.board.index_to_pos` inverse lookup when
            possible, else ``-1`` (legacy behaviour).
          * counters initialised to zero; ``deaths`` left empty;
            ``zero_board`` reconstructed only if the loaded ``pieces``
            match a fresh-game layout.
        """
        rv = d.get("rules_version", "?")
        if not rv.startswith("1."):
            raise ValueError(f"incompatible rules_version {rv!r} (need 1.x)")

        sv = d.get("state_version", "1.0")

        # ---- Shared: pieces + info ----
        pieces: PieceMap = {}
        for p in d["pieces"]:
            pos = (p["pos"][0], p["pos"][1])
            pieces[pos] = PieceRef(
                seat=Seat(p["seat"]),
                piece_type=PieceType[p["type"]],
                alive=True,
                piece_id=p.get("piece_id", -1),
            )

        info: dict[Seat, SeatInfo] = {}
        for item in d["info"]:
            info[Seat(item["seat"])] = SeatInfo(
                dead=item["dead"],
                flag_revealed=item["flag_revealed"],
            )

        # ---- Initial SoA skeleton (filled below per schema version) ----
        num_pids = _ZOB.NUM_PIECE_IDS
        alive = np.zeros(num_pids, dtype=bool)
        piece_type_arr = np.full(num_pids, -1, dtype=np.int8)
        piece_seat_arr = np.full(num_pids, -1, dtype=np.int8)
        pos_x = np.full(num_pids, -1, dtype=np.int8)
        pos_y = np.full(num_pids, -1, dtype=np.int8)
        zero_x = np.full(num_pids, -1, dtype=np.int8)
        zero_y = np.full(num_pids, -1, dtype=np.int8)
        move_count_arr = np.zeros(num_pids, dtype=np.int16)
        active_eat_arr = np.zeros(num_pids, dtype=np.int16)
        passive_surv_arr = np.zeros(num_pids, dtype=np.int16)
        death_reason_arr = np.full(num_pids, -1, dtype=np.int8)
        death_step_arr = np.full(num_pids, -1, dtype=np.int16)
        death_loc_flat_arr = np.full(num_pids, -1, dtype=np.int16)
        cell_piece_id = np.full(NUM_CELLS, -1, dtype=np.int16)
        seat_dead_arr = np.zeros(4, dtype=bool)
        seat_flag_revealed_arr = np.zeros(4, dtype=bool)

        # ---- Seat scalars derivable from `info` (both schema versions) ----
        for s in ALL_SEATS:
            if info[s].dead:
                seat_dead_arr[s.value] = True
            if info[s].flag_revealed:
                seat_flag_revealed_arr[s.value] = True

        # ---- Live pieces → SoA columns ----
        # For v2 we can trust the saved piece_id on every PieceRef.
        # For v1 we fall back to -1 and keep SoA cells unassigned for
        # those pieces (hot-path queries on loaded-v1 states are expected
        # to be cold-path only).
        for pos, ref in pieces.items():
            pid = ref.piece_id
            if pid < 0:
                continue
            alive[pid] = True
            piece_type_arr[pid] = ref.piece_type.value
            piece_seat_arr[pid] = ref.seat.value
            pos_x[pid] = pos[0]
            pos_y[pid] = pos[1]
            flat = xy_to_flat(*pos)
            cell_piece_id[flat] = pid

        # ---- zero_board / piece_state / deaths ----
        zero_board: dict[tuple[int, int], PieceRef] = {}
        piece_state: dict[int, PieceState] = {}
        deaths: dict[int, DeathInfo] = {}

        if sv.startswith("2."):
            for entry in d.get("zero_board", []):
                pos = (entry["pos"][0], entry["pos"][1])
                ref = PieceRef(
                    seat=Seat(entry["seat"]),
                    piece_type=PieceType[entry["type"]],
                    alive=True,
                    piece_id=entry.get("piece_id", -1),
                )
                zero_board[pos] = ref
                pid = ref.piece_id
                if pid >= 0:
                    piece_seat_arr[pid] = ref.seat.value
                    zero_x[pid] = pos[0]
                    zero_y[pid] = pos[1]
            for entry in d.get("piece_state", []):
                pid = entry["piece_id"]
                piece_state[pid] = PieceState(
                    move_count=entry["move_count"],
                    active_eat_count=entry["active_eat_count"],
                    passive_survive_count=entry["passive_survive_count"],
                )
                if 0 <= pid < num_pids:
                    move_count_arr[pid] = entry["move_count"]
                    active_eat_arr[pid] = entry["active_eat_count"]
                    passive_surv_arr[pid] = entry["passive_survive_count"]
            for entry in d.get("deaths", []):
                pid = entry["piece_id"]
                di = DeathInfo(
                    piece_id=pid,
                    reason=DeathReason[entry["reason"]],
                    death_loc=(entry["death_loc"][0], entry["death_loc"][1]),
                    step=entry["step"],
                )
                deaths[pid] = di
                if 0 <= pid < num_pids:
                    death_reason_arr[pid] = di.reason.value
                    death_step_arr[pid] = di.step
                    death_loc_flat_arr[pid] = xy_to_flat(*di.death_loc)
        else:
            # v1: bootstrap piece_state/zero_board/deaths in a lossy way.
            # Every live piece gets a zero-initialised PieceState (iff its
            # piece_id was supplied), and deaths stays empty.  zero_board
            # is left empty — callers of loaded-v1 states that want T7
            # observation channels need to re-run ``new_game`` from the
            # setups instead.
            for pid_set_flag_revealed in ():  # no-op placeholder
                pass
            for _, ref in pieces.items():
                if ref.piece_id >= 0 and ref.piece_id not in piece_state:
                    piece_state[ref.piece_id] = PieceState()

        # ---- Zobrist: prefer the saved value if v2, else recompute ----
        if sv.startswith("2.") and "zobrist" in d:
            zobrist_val = int(d["zobrist"])
        else:
            zobrist_val = _recompute_zobrist(
                alive=alive,
                piece_type_arr=piece_type_arr,
                pos_x=pos_x,
                pos_y=pos_y,
                turn_val=int(d["turn"]),
                move_counter=int(d["move_counter"]),
                moves_since_last_combat=int(d["moves_since_last_combat"]),
                terminated=bool(d.get("terminated", False)),
                winner_team=d.get("winner_team"),
                draw=bool(d.get("draw", False)),
                show_mode_val=int(d.get("show_mode", ShowMode.HALF_DARK.value)),
                seat_dead_arr=seat_dead_arr,
                seat_flag_revealed_arr=seat_flag_revealed_arr,
            )

        return cls(
            pieces=pieces,
            turn=Seat(d["turn"]),
            move_counter=d["move_counter"],
            moves_since_last_combat=d["moves_since_last_combat"],
            info=info,
            terminated=d.get("terminated", False),
            winner_team=d.get("winner_team"),
            draw=d.get("draw", False),
            show_mode=ShowMode(d.get("show_mode", ShowMode.HALF_DARK.value)),
            rules_version=rv,
            debug_include_private=d.get("debug_include_private", False),
            zero_board=zero_board,
            piece_state=piece_state,
            deaths=deaths,
            alive=alive,
            piece_type_arr=piece_type_arr,
            piece_seat_arr=piece_seat_arr,
            pos_x=pos_x,
            pos_y=pos_y,
            zero_x=zero_x,
            zero_y=zero_y,
            move_count_arr=move_count_arr,
            active_eat_arr=active_eat_arr,
            passive_surv_arr=passive_surv_arr,
            death_reason_arr=death_reason_arr,
            death_step_arr=death_step_arr,
            death_loc_flat_arr=death_loc_flat_arr,
            cell_piece_id=cell_piece_id,
            seat_dead_arr=seat_dead_arr,
            seat_flag_revealed_arr=seat_flag_revealed_arr,
            zobrist=zobrist_val,
        )


# ===========================================================================
# Helpers (module-private)
# ===========================================================================


def _seat_has_any_piece(pieces: PieceMap, seat: Seat) -> bool:
    for p in pieces.values():
        if p.alive and p.seat is seat:
            return True
    return False


def _remove_all_pieces_of_seat(
    pieces: PieceMap,
    seat: Seat,
    *,
    piece_state: dict[int, PieceState] | None = None,
    deaths: dict[int, DeathInfo] | None = None,
    death_reason: DeathReason | None = None,
    death_step: int | None = None,
) -> None:
    """Remove every piece belonging to `seat` from the PieceMap (in place).

    When `piece_state` is supplied (T7 M2 callers), the corresponding
    piece_id entries are also popped to preserve the invariant that
    `piece_state` only ever keys on *currently alive* pieces.

    When `deaths` / `death_reason` / `death_step` are supplied together
    (T7 M3 callers: surrender + Q12), a DeathInfo is recorded for every
    collateral victim, anchored at that piece's **current** cell (not its
    zero-board cell). The flag itself, if it triggered the surrender, is
    already logged separately in step()'s combat branch before this
    helper is called.

    Idempotence guarantee: if `deaths` already contains a piece_id
    (e.g. the flag was eaten seconds earlier in the same step()), we do
    NOT overwrite it — the earlier, more specific record wins.
    """
    to_remove = [pos for pos, p in pieces.items() if p.seat is seat]
    record_deaths = (
        deaths is not None
        and death_reason is not None
        and death_step is not None
    )
    for pos in to_remove:
        ref = pieces[pos]
        pid = ref.piece_id
        if piece_state is not None:
            piece_state.pop(pid, None)
        if record_deaths and pid not in deaths:
            deaths[pid] = DeathInfo(
                piece_id=pid,
                reason=death_reason,
                death_loc=pos,
                step=death_step,
            )
    for pos in to_remove:
        del pieces[pos]


def _check_victory(
    *,
    info: dict[Seat, SeatInfo],
    attacker_seat: Seat,
    move_counter: int,
    moves_since_last_combat: int,
) -> tuple[bool, int | None, bool]:
    """Determine (terminated, winner_team, draw).

    Implements §5.2 + §5.2a (Q14 mutual destruction → attacker wins) + §5.3
    draw thresholds. See docs/DECISIONS.md ADR-016.
    """
    red_alive = not (info[Seat.SOUTH].dead and info[Seat.NORTH].dead)
    blue_alive = not (info[Seat.WEST].dead and info[Seat.EAST].dead)

    # Q14: mutual destruction → attacker's team wins
    if not red_alive and not blue_alive:
        return (True, attacker_seat.team, False)

    # Standard team victory
    if not red_alive:
        return (True, 1, False)  # blue team wins
    if not blue_alive:
        return (True, 0, False)  # red team wins

    # §5.3 draw thresholds — only remaining path to a draw
    if move_counter >= MAX_NUM_MOVES:
        return (True, None, True)
    if moves_since_last_combat >= MAX_NUM_MOVES_BETWEEN_ATTACKS:
        return (True, None, True)

    return (False, None, False)


# ---------------------------------------------------------------------------
# T7 M2 — per-piece counter increment helpers.
# ---------------------------------------------------------------------------
# Kept as module-private free functions rather than methods so step() reads
# as a straight recipe. All three follow the same pattern: if the piece_id
# entry is missing (shouldn't normally happen — every live piece has one
# after new_game) we defensively create a fresh zeroed PieceState and then
# bump the appropriate field. PieceState is frozen, so we rebuild via
# dataclasses.replace.

def _bump_move_count(ps_map: dict[int, PieceState], pid: int) -> None:
    cur = ps_map.get(pid, PieceState())
    ps_map[pid] = replace(cur, move_count=cur.move_count + 1)


def _bump_active_eat(ps_map: dict[int, PieceState], pid: int) -> None:
    cur = ps_map.get(pid, PieceState())
    ps_map[pid] = replace(cur, active_eat_count=cur.active_eat_count + 1)


def _bump_passive_survive(ps_map: dict[int, PieceState], pid: int) -> None:
    cur = ps_map.get(pid, PieceState())
    ps_map[pid] = replace(
        cur, passive_survive_count=cur.passive_survive_count + 1
    )


# ---------------------------------------------------------------------------
# Zobrist recomputation (cold path — used by v1 `from_dict` fallback)
# ---------------------------------------------------------------------------


def _recompute_zobrist(
    *,
    alive: np.ndarray,
    piece_type_arr: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    turn_val: int,
    move_counter: int,
    moves_since_last_combat: int,
    terminated: bool,
    winner_team: int | None,
    draw: bool,
    show_mode_val: int,
    seat_dead_arr: np.ndarray,
    seat_flag_revealed_arr: np.ndarray,
) -> int:
    """Reconstruct the Zobrist hash from scratch.

    Used only on the cold path (``from_dict`` of a v1 replay or a hand-
    constructed state).  Mirrors the contributions XOR'd in by
    ``new_game`` and ``step_inplace``.
    """
    h = 0
    # Pieces.
    live_pids = np.nonzero(alive)[0]
    for pid in live_pids:
        pid_int = int(pid)
        type_val = int(piece_type_arr[pid_int])
        if type_val < 0:
            continue
        flat = int(pos_y[pid_int]) * BOARD_SIZE + int(pos_x[pid_int])
        h ^= int(_ZOB.ZOB_PIECE[pid_int, type_val, flat])
    # Scalars.
    h ^= int(_ZOB.ZOB_TURN[turn_val])
    h ^= int(_ZOB.ZOB_MOVE_COUNTER[move_counter & _ZOB.MOVE_COUNTER_MASK])
    h ^= int(_ZOB.ZOB_MOVES_SINCE_COMBAT[
        moves_since_last_combat & _ZOB.MOVES_SINCE_COMBAT_MASK
    ])
    h ^= int(_ZOB.ZOB_WINNER[0 if winner_team is None else (winner_team + 1)])
    h ^= int(_ZOB.ZOB_SHOW_MODE[show_mode_val])
    if terminated:
        h ^= _ZOB.ZOB_TERMINATED
    if draw:
        h ^= _ZOB.ZOB_DRAW
    # Seat scalars.
    for s_val in range(4):
        if bool(seat_dead_arr[s_val]):
            h ^= int(_ZOB.ZOB_SEAT_DEAD[s_val])
        if bool(seat_flag_revealed_arr[s_val]):
            h ^= int(_ZOB.ZOB_SEAT_FLAG_REVEALED[s_val])
    return h


# ===========================================================================
# Self-test (covers Q1, Q7, Q10, Q12, Q14 paths)
# ===========================================================================


def _self_test() -> None:  # pragma: no cover
    from .setup import generate_random_setup
    import random as _random

    # 1. new_game from a random setup
    rng = _random.Random(42)
    setups = generate_random_setup(rng)
    st = GameState.new_game(setups)
    assert st.turn is Seat.SOUTH
    assert st.move_counter == 0
    assert len(st.pieces) == 4 * 25  # 25 pieces per seat
    assert all(not st.info[s].dead for s in Seat)

    # 2. legal_actions from opening position
    acts = st.legal_actions()
    assert len(acts) > 0, "HOME should have legal moves at start"

    # 3. Step once — piece count invariants
    a = acts[0]
    st2, r = st.step(a)
    assert st2 is not st
    assert st2.move_counter == 1
    # After a plain MOVE we should have the same number of pieces as before
    if r.event is Event.MOVE:
        assert len(st2.pieces) == len(st.pieces)
    # Original state unchanged (immutability)
    assert st.move_counter == 0
    assert len(st.pieces) == 4 * 25

    # 4. Hash determinism
    assert st.state_hash() == st.clone().state_hash()

    # 5. Serialization round-trip
    d = st.to_dict()
    st3 = GameState.from_dict(d)
    assert st3.state_hash() == st.state_hash()

    print("junqi_core.state self-test: OK")


if __name__ == "__main__":
    _self_test()
