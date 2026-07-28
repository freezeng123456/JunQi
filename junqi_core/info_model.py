
"""Information model & belief tensors for 4-player Junqi.

This module implements the per-observer belief state and the deterministic
inference rules described in docs/INFERENCE.md. Each living seat has its own
`BeliefTensor` that tracks, for every *enemy* piece on the board, a probability
distribution over the 12 concrete piece types.

Own and teammate pieces (under Q11 / HALF_DARK) are always tracked as one-hot
distributions matching their true types — the observer simply knows them.

Top-level API:
    BeliefTensor.initial(state, observer, show_mode)
        → seed belief from the opening position using the constraint-aware
          per-slot prior (see §2 of INFERENCE.md).

    belief.update(prev_state, new_state, result)
        → apply all deterministic R1–R10 updates for a single `step()` edge.

    belief.get(pos) → ndarray[12]
        → query a single cell's probability vector (zero vector if no piece).

    belief.to_world_tensor() → ndarray[17, 17, 12]
        → render as world-frame tensor suitable for NN input.

Reference: docs/INFERENCE.md v0.1.0; tied to RULES_VERSION 1.1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .board import BOARD_SIZE, index_to_pos, xy_to_flat
from .move_gen import PieceRef
from .rules import (
    ALL_PLACEABLE_PIECES,
    ALL_SEATS,
    BACK_TWO_ROWS_INDICES,
    CAMP_INDICES,
    FRONT_ROW_INDICES,
    PIECE_COUNTS,
    STRONGHOLD_INDICES,
    Event,
    PieceType,
    Seat,
    ShowMode,
    TOTAL_PIECES_PER_SEAT,
    same_team,
)
from .state import GameState, MoveResult

INFERENCE_VERSION: Final[str] = "0.1.0"

# ===========================================================================
# Type / constant tables
# ===========================================================================

# All 12 concrete piece types (indexing order used for probability vectors).
TRACKED_TYPES: Final[tuple[PieceType, ...]] = tuple(
    t for t in ALL_PLACEABLE_PIECES if t not in (PieceType.NONE, PieceType.DARK)
)
NUM_TRACKED_TYPES: Final[int] = len(TRACKED_TYPES)
_TYPE_TO_IDX: Final[dict[PieceType, int]] = {t: i for i, t in enumerate(TRACKED_TYPES)}

# Per-index admissible types (derived from C1–C4 in RULES §1.3 + camp rule).
# Index `i ∈ [0, 30)` → tuple of PieceType that MAY appear at slot `i`.
# Camp indices have no piece ever (not represented in belief map).
_ADMISSIBLE_BY_INDEX: list[tuple[PieceType, ...]] = []
for _i in range(30):
    if _i in CAMP_INDICES:
        _ADMISSIBLE_BY_INDEX.append(())
    elif _i in STRONGHOLD_INDICES:
        # Stronghold hosts either JUNQI or any non-special movable piece.
        # Constraint: exactly one of the two strongholds holds JUNQI; the
        # other holds some other piece (which cannot be DILEI or ZHADAN
        # because both are restricted to back-two-rows AND stronghold is
        # in back row BUT the flag-must-be-in-one-stronghold rule allows
        # the other stronghold slot to hold any non-flag piece).
        # Simplification for the per-slot prior: all non-flag types allowed,
        # plus JUNQI. A more refined joint prior would enforce exactly-one
        # flag, but the per-slot approximation is sufficient here.
        types: list[PieceType] = [PieceType.JUNQI]
        for t in TRACKED_TYPES:
            if t is PieceType.JUNQI:
                continue
            # Stronghold is in back-row (index 26 or 28 → row 5), so DILEI
            # is allowed. ZHADAN is allowed (not front row).
            types.append(t)
        _ADMISSIBLE_BY_INDEX.append(tuple(types))
    else:
        # Non-camp, non-stronghold: piece-type specific constraints
        is_front = _i in FRONT_ROW_INDICES
        is_back_two = _i in BACK_TWO_ROWS_INDICES
        types = []
        for t in TRACKED_TYPES:
            if t is PieceType.JUNQI:
                continue  # JUNQI only on strongholds
            if t is PieceType.DILEI and not is_back_two:
                continue  # DILEI only back two rows
            if t is PieceType.ZHADAN and is_front:
                continue  # ZHADAN never on front row
            types.append(t)
        _ADMISSIBLE_BY_INDEX.append(tuple(types))

ADMISSIBLE_BY_INDEX: Final[tuple[tuple[PieceType, ...], ...]] = tuple(
    _ADMISSIBLE_BY_INDEX
)

# ===========================================================================
# Prior construction
# ===========================================================================


def _per_slot_prior_vector(index_local: int) -> np.ndarray:
    """Return P(type | slot=index_local) as a length-12 vector.

    Per INFERENCE.md §2.1: `p[t] ∝ PIECE_COUNTS[t]` for `t` admissible at this
    slot, zero otherwise; normalize so the vector sums to 1.
    """
    vec = np.zeros(NUM_TRACKED_TYPES, dtype=np.float32)
    allowed = ADMISSIBLE_BY_INDEX[index_local]
    if not allowed:
        return vec  # empty (camp): should never be queried
    for t in allowed:
        vec[_TYPE_TO_IDX[t]] = float(PIECE_COUNTS[t])
    s = vec.sum()
    if s == 0:
        raise RuntimeError(
            f"no admissible types at slot {index_local}; prior is degenerate"
        )
    return vec / s


# Precomputed 30×12 initial-prior table (seat-local slot → distribution).
_INITIAL_PRIOR_TABLE: Final[np.ndarray] = np.stack(
    [_per_slot_prior_vector(i) for i in range(30)], axis=0
)


def one_hot(piece_type: PieceType) -> np.ndarray:
    """Return a one-hot probability vector for a known type."""
    vec = np.zeros(NUM_TRACKED_TYPES, dtype=np.float32)
    vec[_TYPE_TO_IDX[piece_type]] = 1.0
    return vec


def _local_index_from_world(seat: Seat, pos: tuple[int, int]) -> int | None:
    """Return the seat-local index (0..29) of `pos` if it belongs to `seat`'s
    own 30-slot rectangle, else None.  Used only for initial-prior seeding.
    """
    # Linear scan — only called at game start, so O(30) is fine.
    for i in range(30):
        if index_to_pos(seat, i) == pos:
            return i
    return None


# ===========================================================================
# BeliefTensor
# ===========================================================================


@dataclass(slots=True)
class BeliefTensor:
    """Per-observer belief state for a single 4-player Junqi game.

    Tracks probability distributions over `TRACKED_TYPES` for every *currently
    alive* piece on the board. Own and teammate pieces are stored as one-hot
    vectors reflecting their true types (Q11 in HALF_DARK; everyone in BRIGHT).
    """

    observer: Seat                                  # whose view this belief is
    show_mode: ShowMode
    probs: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    # Remaining inventory per enemy seat: type → count still unrevealed on board
    remaining: dict[Seat, dict[PieceType, int]] = field(default_factory=dict)
    # Version stamp (for future replay compatibility)
    version: str = INFERENCE_VERSION
    # ---- Phase 0.4 M3 / ADR-120 tensor mirror ------------------------
    # probs_arr: (num_pids, 12) float32 mirror of ``probs``, indexed by
    # piece_id; rebuilt from the dict at the end of ``initial`` and
    # every ``update``.  All zero for pids that are dead / have no
    # belief entry.  This is the hot-path input that
    # ``ObservationBuilder`` consumes to vectorize the prob_teammate /
    # belief_left_side / belief_right_side / bucket_group writers.
    probs_arr: np.ndarray = field(
        default_factory=lambda: np.zeros((0, NUM_TRACKED_TYPES), dtype=np.float32)
    )
    # remaining_arr: (4, 12) int16; indexed by Seat.value, 12 columns in
    # TRACKED_TYPES order.  Rows for own/teammate (never tracked in
    # ``remaining`` dict) stay zero.  Same sync cadence as ``probs_arr``.
    remaining_arr: np.ndarray = field(
        default_factory=lambda: np.zeros((4, NUM_TRACKED_TYPES), dtype=np.int16)
    )
    # ---- Phase 0.4 M6 / lazy sync ------------------------------------
    # ``update()`` used to eagerly refresh ``probs_arr`` / ``remaining_arr``
    # on every call; profiling under JunqiEnv (M6) showed this alone cost
    # ~28 % of per-step wall-clock because ``update()`` fires once per
    # seat (4×/play) while the tensor mirror is only read when an obs is
    # actually built for that seat.
    #
    # The lazy contract is:
    #   * ``update()`` sets ``_dirty_state`` to the new ``GameState`` and
    #     returns without touching ``probs_arr`` / ``remaining_arr``.
    #   * Before any read of ``probs_arr`` / ``remaining_arr`` the caller
    #     (currently only the obs builder's ``_fill_into``) invokes
    #     :meth:`ensure_synced`, which is a no-op when nothing is dirty.
    #   * ``initial()`` keeps the eager sync because the game state and
    #     the belief are born together — there is no "dirty" window.
    _dirty_state: GameState | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def initial(
        cls,
        state: GameState,
        observer: Seat,
        *,
        show_mode: ShowMode | None = None,
    ) -> BeliefTensor:
        """Seed belief from the opening `state`.

        `show_mode` defaults to `state.show_mode`. Observer's own and
        teammate's pieces are one-hot; enemy pieces use the per-slot prior.
        In BRIGHT mode all pieces are one-hot.
        """
        mode = show_mode if show_mode is not None else state.show_mode
        b = cls(observer=observer, show_mode=mode)

        for pos, piece in state.pieces.items():
            if not piece.alive:
                continue
            if mode is ShowMode.BRIGHT or _observer_sees_truth(
                observer, piece.seat, mode
            ):
                b.probs[pos] = one_hot(piece.piece_type)
            else:
                # Enemy piece — use constraint-aware per-slot prior.
                local = _local_index_from_world(piece.seat, pos)
                if local is None:
                    # Piece is not in its home slot anymore → should not
                    # happen at game start, but fall back to uniform over
                    # all tracked types.
                    b.probs[pos] = np.full(
                        NUM_TRACKED_TYPES, 1.0 / NUM_TRACKED_TYPES, dtype=np.float32
                    )
                else:
                    b.probs[pos] = _INITIAL_PRIOR_TABLE[local].copy()

        # Initialize remaining-inventory for each enemy seat.
        for s in ALL_SEATS:
            if s is observer or same_team(observer, s):
                continue
            b.remaining[s] = dict(PIECE_COUNTS)

        # ADR-120: seed the tensor mirrors so downstream observation
        # consumers (ObservationBuilder.build) can read them immediately
        # even before the first step().
        b._sync_tensors(state)

        return b

    # ------------------------------------------------------------------
    # Update: apply R1–R9 for a single step
    # ------------------------------------------------------------------

    def update(
        self,
        prev_state: GameState,
        new_state: GameState,
        result: MoveResult,
    ) -> None:
        """Apply deterministic inferences for the edge
        `prev_state --result--> new_state` IN PLACE.
        """
        src = result.src
        dst = result.dst
        event = result.event

        # Capture the pre-step identities of attacker/defender (if known)
        prev_src_piece = prev_state.pieces.get(src)
        prev_dst_piece = prev_state.pieces.get(dst)
        assert prev_src_piece is not None, "step() must have an attacker at src"

        # Pull the attacker seat (we know it from result.seat) and defender
        # seat (if the cell was occupied).
        attacker_seat = result.seat
        defender_seat = prev_dst_piece.seat if prev_dst_piece is not None else None

        # Precompute the attacker's pre-move distribution (copy so we can
        # reuse it for R1 migration even after delete).
        attacker_belief = self.probs.get(src)
        defender_belief = (
            self.probs.get(dst) if prev_dst_piece is not None else None
        )

        # -----------------------------------------------------------------
        # R3 + Q7: SILING flag reveal — both sides (if any) that revealed
        #         SILING had their attacker/defender turn into SILING one-hot
        # -----------------------------------------------------------------
        if result.flag_reveal_src:
            # Attacker WAS SILING; if observer didn't know that, upgrade.
            # (attacker is already dead; update remaining inventory.)
            self._reveal_src_siling(attacker_seat)
        if result.flag_reveal_dst:
            assert defender_seat is not None
            self._reveal_dst_siling(defender_seat)

        # -----------------------------------------------------------------
        # R5/R7: GONGB signature (attacker eats DILEI)
        # -----------------------------------------------------------------
        gongb_revealed = False
        if (
            event is Event.EAT
            and prev_dst_piece is not None
            and prev_dst_piece.piece_type is PieceType.DILEI
        ):
            # Only GONGB can eat DILEI → attacker is provably GONGB (unless
            # observer already knew, e.g., because attacker is own/teammate).
            # Update attacker's belief to GONGB one-hot.
            if attacker_belief is not None and not _is_one_hot(attacker_belief):
                attacker_belief = one_hot(PieceType.GONGB)
                self.probs[src] = attacker_belief
                self._decrement_remaining(attacker_seat, PieceType.GONGB)
                gongb_revealed = True

        # Also: if we just observed a KILLED event where the defender
        # (stationary, on back-row cell) survives → defender is at least
        # DILEI-strong. We leave this as a soft hint; the per-slot prior
        # already biased back-row cells toward DILEI/higher. (TODO R5 soft)

        # -----------------------------------------------------------------
        # R4 + R6: Flag capture / stronghold EAT deductions
        # -----------------------------------------------------------------
        if result.flag_captured:
            # R4: defender's entire remaining inventory is wiped (they die).
            assert defender_seat is not None
            self.remaining.pop(defender_seat, None)
        elif (
            event is Event.EAT
            and prev_dst_piece is not None
            and dst in _stronghold_positions_of(defender_seat)  # type: ignore[arg-type]
        ):
            # R6: the flag is guaranteed to be at the OTHER stronghold of
            # defender_seat (since dst was a stronghold but NOT the flag).
            assert defender_seat is not None
            strongholds = _stronghold_positions_of(defender_seat)
            other = strongholds[0] if strongholds[1] == dst else strongholds[1]
            if other in self.probs and not _is_one_hot(self.probs[other]):
                self.probs[other] = one_hot(PieceType.JUNQI)
                # JUNQI count stays as 1 in the remaining inventory — it is
                # still alive; no decrement.

        # -----------------------------------------------------------------
        # R1: piece migration (must run AFTER reveals, so the revealed
        # identities get copied to the destination)
        # -----------------------------------------------------------------
        # Refresh attacker_belief in case R5/R7 modified self.probs[src]
        attacker_belief = self.probs.get(src, attacker_belief)

        if event is Event.MOVE:
            if src in self.probs:
                self.probs[dst] = self.probs.pop(src)
        elif event is Event.EAT:
            # attacker wins; migrates into dst cell
            if defender_belief is not None:
                self._on_piece_removed(defender_seat, defender_belief)
            if src in self.probs:
                self.probs[dst] = self.probs.pop(src)
            elif attacker_belief is not None:
                self.probs[dst] = attacker_belief.copy()
                self.probs.pop(src, None)
        elif event is Event.KILLED:
            # attacker dies; dst keeps its piece
            if attacker_belief is not None:
                self._on_piece_removed(attacker_seat, attacker_belief)
            self.probs.pop(src, None)
            # dst unchanged (its belief stays)
        elif event is Event.BOMB:
            if attacker_belief is not None:
                self._on_piece_removed(attacker_seat, attacker_belief)
            if defender_belief is not None:
                self._on_piece_removed(defender_seat, defender_belief)
            self.probs.pop(src, None)
            self.probs.pop(dst, None)
        else:
            raise AssertionError(f"unknown event {event}")

        # -----------------------------------------------------------------
        # R9: Q12 seat death — remove any belief entries for seats that
        # became dead in this step.
        # -----------------------------------------------------------------
        for seat in result.seats_died_this_step:
            self._purge_seat(seat, new_state)

        # -----------------------------------------------------------------
        # I5 consistency sweep: every belief key should still correspond to
        # a live piece in new_state.pieces.
        # -----------------------------------------------------------------
        stale = [pos for pos in self.probs if pos not in new_state.pieces]
        for pos in stale:
            del self.probs[pos]

        # Sanity: every live piece should have a belief entry
        for pos, piece in new_state.pieces.items():
            if pos not in self.probs:
                # Missing entry — synthesize a reasonable fallback (the
                # observer should have seen the migration). In practice
                # this should not fire; we're defensive.
                if self.show_mode is ShowMode.BRIGHT or _observer_sees_truth(
                    self.observer, piece.seat, self.show_mode
                ):
                    self.probs[pos] = one_hot(piece.piece_type)
                else:
                    # Uniform fallback — should be rare.
                    self.probs[pos] = np.full(
                        NUM_TRACKED_TYPES,
                        1.0 / NUM_TRACKED_TYPES,
                        dtype=np.float32,
                    )

        # ADR-120 + M6 lazy sync: mark the tensor mirror stale.  The
        # actual refresh is deferred until an observation build asks for
        # the data (ensure_synced), which saves ~100 µs per seat per
        # step when there is no concurrent obs build.
        self._dirty_state = new_state

    def ensure_synced(self, state: GameState | None = None) -> None:
        """Refresh ``probs_arr`` / ``remaining_arr`` if they are stale.

        No-op when ``update()`` has not been called since the last sync.

        Parameters
        ----------
        state
            Optional override for the state-of-record.  When ``None``
            (the default) the last state passed to :meth:`update` is
            used.  Pass ``state`` explicitly if you want to sync against
            a different state (e.g. a forked copy).
        """
        target = state if state is not None else self._dirty_state
        if target is None:
            return  # nothing to sync
        self._sync_tensors(target)
        self._dirty_state = None

    # ------------------------------------------------------------------
    # Query / rendering
    # ------------------------------------------------------------------

    def get(self, pos: tuple[int, int]) -> np.ndarray:
        """Return the probability vector at `pos`, or a zero vector if empty."""
        v = self.probs.get(pos)
        if v is None:
            return np.zeros(NUM_TRACKED_TYPES, dtype=np.float32)
        return v

    def to_world_tensor(self) -> np.ndarray:
        """Render belief as an `[BOARD_SIZE, BOARD_SIZE, 12]` tensor."""
        t = np.zeros(
            (BOARD_SIZE, BOARD_SIZE, NUM_TRACKED_TYPES), dtype=np.float32
        )
        for (x, y), v in self.probs.items():
            t[x, y] = v
        return t

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_piece_removed(
        self, seat: Seat | None, belief_vec: np.ndarray
    ) -> None:
        """Update `remaining[seat]` when a piece is removed from the board.

        If belief was one-hot we can decrement exactly; otherwise we skip
        (the inventory remains conservatively high).
        """
        if seat is None:
            return
        if seat not in self.remaining:
            return  # own/teammate — inventory not tracked
        if _is_one_hot(belief_vec):
            t = TRACKED_TYPES[int(np.argmax(belief_vec))]
            self._decrement_remaining(seat, t)

    def _decrement_remaining(self, seat: Seat, t: PieceType) -> None:
        inv = self.remaining.get(seat)
        if inv is None:
            return
        if inv.get(t, 0) > 0:
            inv[t] -= 1

    def _reveal_src_siling(self, attacker_seat: Seat) -> None:
        """After flag_reveal_src: attacker was SILING and is dead.

        The attacker piece is already in `self.probs[src]`; R1 will delete
        it. We just need to update `remaining` inventory.
        """
        self._decrement_remaining(attacker_seat, PieceType.SILING)

    def _reveal_dst_siling(self, defender_seat: Seat) -> None:
        self._decrement_remaining(defender_seat, PieceType.SILING)

    def _purge_seat(self, seat: Seat, new_state: GameState) -> None:
        """Remove all belief entries for pieces that belonged to a dead seat
        (R9 cleanup). The dead seat's pieces should already be gone from
        new_state.pieces; we just defensively clear ours."""
        if seat in self.remaining:
            self.remaining.pop(seat)
        # Drop any orphan belief entries (positions where the piece was
        # owned by `seat`). We don't know ownership from the belief alone,
        # so rely on new_state.pieces (the dead seat's pieces are gone).
        stale = [pos for pos in self.probs if pos not in new_state.pieces]
        for pos in stale:
            del self.probs[pos]

    # ------------------------------------------------------------------
    # Phase 0.4 M3 / ADR-120 — tensor mirror sync.
    # ------------------------------------------------------------------

    def _sync_tensors(self, state: GameState) -> None:
        """Refresh ``probs_arr`` and ``remaining_arr`` from the dict state.

        Called at the end of :meth:`initial` and :meth:`update`.  Uses
        ``state.cell_piece_id`` (ADR-117 SoA mirror) to resolve each
        ``(x, y)`` belief entry to its piece_id; the mapping is O(1).

        Cost: O(N) where N is the number of live pieces (<=100).  For a
        typical mid-game this adds ~30 us per update; observation build
        time is dropped by >>1000 us in exchange (see
        ``tools/benchmark_phase04_m3.py``).
        """
        num_pids = state.alive.shape[0]
        # Grow / reset the piece-indexed buffer on first use or whenever
        # the state's pid budget changes.
        if self.probs_arr.shape != (num_pids, NUM_TRACKED_TYPES):
            self.probs_arr = np.zeros(
                (num_pids, NUM_TRACKED_TYPES), dtype=np.float32
            )
        else:
            self.probs_arr.fill(0.0)

        cell_piece_id = state.cell_piece_id
        for (x, y), vec in self.probs.items():
            flat = y * BOARD_SIZE + x
            pid = int(cell_piece_id[flat])
            if 0 <= pid < num_pids:
                self.probs_arr[pid] = vec

        # Remaining inventory: (4, 12) int16.
        self.remaining_arr.fill(0)
        for seat, inv in self.remaining.items():
            row = self.remaining_arr[seat.value]
            for pt, cnt in inv.items():
                idx = _TYPE_TO_IDX.get(pt)
                if idx is not None and cnt:
                    row[idx] = cnt

# ===========================================================================
# Helpers (module-private)
# ===========================================================================


def _observer_sees_truth(
    observer: Seat, owner: Seat, show_mode: ShowMode
) -> bool:
    """True iff the observer can see the true type of `owner`'s pieces."""
    if show_mode is ShowMode.BRIGHT:
        return True
    if observer is owner:
        return True
    # HALF_DARK / DARK: teammate pieces are visible iff HALF_DARK (Q11)
    if show_mode is ShowMode.HALF_DARK and same_team(observer, owner):
        return True
    return False


def _is_one_hot(vec: np.ndarray, eps: float = 1e-6) -> bool:
    if vec.shape != (NUM_TRACKED_TYPES,):
        return False
    maxv = float(np.max(vec))
    return maxv > 1.0 - eps


def _stronghold_positions_of(seat: Seat | None) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return the two stronghold positions (world-frame) of `seat`."""
    assert seat is not None
    strongholds = sorted(STRONGHOLD_INDICES)  # [26, 28]
    return (
        index_to_pos(seat, strongholds[0]),
        index_to_pos(seat, strongholds[1]),
    )


# ===========================================================================
# Self-test
# ===========================================================================


def _self_test() -> None:  # pragma: no cover
    import random as _random
    from .setup import generate_random_setup

    rng = _random.Random(0)
    setups = generate_random_setup(rng)
    st = GameState.new_game(setups)

    # Observer HOME (red). Teammate OPPS is one-hot (Q11). Enemies RIGHT+LEFT
    # have per-slot priors.
    b = BeliefTensor.initial(st, Seat.SOUTH)

    # Invariants
    for pos, vec in b.probs.items():
        assert vec.shape == (NUM_TRACKED_TYPES,)
        s = float(vec.sum())
        assert abs(s - 1.0) < 1e-5, f"belief at {pos} sums to {s}"
        piece = st.pieces[pos]
        if piece.seat is Seat.SOUTH or piece.seat is Seat.NORTH:
            assert _is_one_hot(vec), f"own/teammate at {pos} not one-hot"
        else:
            # enemy — must have >0 prob on TRUE type
            true_idx = _TYPE_TO_IDX[piece.piece_type]
            assert vec[true_idx] > 0, f"enemy at {pos} has 0 prob on true type"

    # Remaining inventory only for enemies
    assert Seat.SOUTH not in b.remaining
    assert Seat.NORTH not in b.remaining
    assert Seat.WEST in b.remaining
    assert Seat.EAST in b.remaining
    for seat in (Seat.WEST, Seat.EAST):
        assert b.remaining[seat] == dict(PIECE_COUNTS)

    # Render tensor
    t = b.to_world_tensor()
    assert t.shape == (BOARD_SIZE, BOARD_SIZE, NUM_TRACKED_TYPES)
    assert t.dtype == np.float32

    print(f"junqi_core.info_model self-test: OK ({len(b.probs)} pieces tracked)")


if __name__ == "__main__":
    _self_test()
