
"""Canonical rule definitions for 4-player Junqi (四国军棋).

This module is the single source of truth for:
  - Piece type enum (`PieceType`) — smaller enum value = stronger piece
  - Seat enum (`Seat`) with team membership
  - Combat event enum (`Event`)
  - Per-piece counts in a setup
  - Cell role constants (camp / stronghold / rail indices)
  - Per-setup placement hard constraints (C1..C5)
  - Draw thresholds (Q10)

No game state lives here — see `state.py`. No geometry — see `board.py`.
Everything below is data, not behavior.

Strict invariants:
  * `PieceType.JUNQI < PieceType.GONGB` must always hold.
  * `Seat.SOUTH.team == Seat.NORTH.team`; `Seat.WEST.team == Seat.EAST.team`.
  * Changing `RULES_VERSION` MAJOR bumps semantic rule changes and invalidates
    all existing golden tests / replays of that older major.
"""

from __future__ import annotations

from enum import IntEnum, unique
from types import MappingProxyType
from typing import Final, Mapping

# ===========================================================================
# Rule version (see ADR-015)
# ===========================================================================

RULES_VERSION: Final[str] = "1.1.0"
RULES_MAJOR: Final[int] = 1

# ===========================================================================
# Piece types (ChessType enum)
# ===========================================================================
# Enum values intentionally match legacy_engine/src/junqi.h::ChessType so that
# the Python library can directly consume legacy binary replay streams.
#
# CRITICAL: smaller value == stronger piece (inherited legacy convention).
#   SILING (5) > JUNZH (6) > SHIZH (7) > ... > GONGB (13)
# Thus "A eats B" corresponds to `A.value < B.value` for ranked pieces.


@unique
class PieceType(IntEnum):
    """Piece type. Lower enum value = stronger. See legacy ChessType."""

    # Non-combat / structural
    NONE = 0      # empty cell or camp
    DARK = 1      # legacy "unknown enemy" marker (not used in new engine)

    # Special immobile
    JUNQI = 2     # 军旗 Flag — immobile, in stronghold only
    DILEI = 3     # 地雷 Landmine — immobile, back two rows only
    ZHADAN = 4    # 炸弹 Bomb — mobile, not front row

    # Ranked combatants (strongest first)
    SILING = 5    # 司令 Field Marshal
    JUNZH = 6     # 军长 General
    SHIZH = 7     # 师长 Division Commander
    LVZH = 8      # 旅长 Brigadier
    TUANZH = 9    # 团长 Colonel
    YINGZH = 10   # 营长 Major
    LIANZH = 11   # 连长 Captain
    PAIZH = 12    # 排长 Lieutenant
    GONGB = 13    # 工兵 Engineer — weakest but can eat mines and do rail-BFS

    # --- Convenience predicates ---
    @property
    def is_empty(self) -> bool:
        """True iff this is a NONE / DARK placeholder (not a real piece)."""
        return self in (PieceType.NONE, PieceType.DARK)

    @property
    def is_immobile(self) -> bool:
        """True iff this piece cannot move (JUNQI, DILEI)."""
        return self in (PieceType.JUNQI, PieceType.DILEI)

    @property
    def is_mine(self) -> bool:
        return self is PieceType.DILEI

    @property
    def is_bomb(self) -> bool:
        return self is PieceType.ZHADAN

    @property
    def is_flag(self) -> bool:
        return self is PieceType.JUNQI

    @property
    def is_engineer(self) -> bool:
        return self is PieceType.GONGB

    @property
    def is_siling(self) -> bool:
        return self is PieceType.SILING

    @property
    def is_ranked_combatant(self) -> bool:
        """True iff in [SILING, GONGB] — pieces that compare by rank."""
        return PieceType.SILING <= self <= PieceType.GONGB


# Tuple of all 12 legal piece types that appear on the board (excluding NONE/DARK)
ALL_PLACEABLE_PIECES: Final[tuple[PieceType, ...]] = (
    PieceType.JUNQI,
    PieceType.DILEI,
    PieceType.ZHADAN,
    PieceType.SILING,
    PieceType.JUNZH,
    PieceType.SHIZH,
    PieceType.LVZH,
    PieceType.TUANZH,
    PieceType.YINGZH,
    PieceType.LIANZH,
    PieceType.PAIZH,
    PieceType.GONGB,
)

assert len(ALL_PLACEABLE_PIECES) == 12, "must be exactly 12 placeable types"

# ===========================================================================
# Piece counts per seat (C5 constraint)
# ===========================================================================

PIECE_COUNTS: Final[Mapping[PieceType, int]] = MappingProxyType({
    PieceType.JUNQI:  1,
    PieceType.DILEI:  3,
    PieceType.ZHADAN: 2,
    PieceType.SILING: 1,
    PieceType.JUNZH:  1,
    PieceType.SHIZH:  2,
    PieceType.LVZH:   2,
    PieceType.TUANZH: 2,
    PieceType.YINGZH: 2,
    PieceType.LIANZH: 3,
    PieceType.PAIZH:  3,
    PieceType.GONGB:  3,
})

# Total pieces placed by each seat: 1+3+2+1+1+2+2+2+2+3+3+3 = 25.
TOTAL_PIECES_PER_SEAT: Final[int] = sum(PIECE_COUNTS.values())
assert TOTAL_PIECES_PER_SEAT == 25

# Each seat occupies a 5×6 = 30-cell rectangle, of which 5 cells are camps
# (always empty). So 30 - 5 = 25 slots hold the 25 pieces.
SLOTS_PER_SEAT: Final[int] = 30
CAMPS_PER_SEAT: Final[int] = 5
NON_CAMP_SLOTS_PER_SEAT: Final[int] = SLOTS_PER_SEAT - CAMPS_PER_SEAT
assert NON_CAMP_SLOTS_PER_SEAT == TOTAL_PIECES_PER_SEAT == 25


# ===========================================================================
# Seats
# ===========================================================================


@unique
class Seat(IntEnum):
    """Seat index. See RULES.md §0 for board layout.

    The four seats are named after their canonical (world-frame) compass
    position on the 17×17 board with origin at top-left and y-axis pointing
    *down* (standard image convention):

        SOUTH (0): bottom of the board (y ∈ [11, 16]), x ∈ [6, 10]
        WEST  (1): left side of the board (x ∈ [0,  5]), y ∈ [6, 10]
        NORTH (2): top of the board (y ∈ [0, 5]), x ∈ [6, 10]
        EAST  (3): right side of the board (x ∈ [11, 16]), y ∈ [6, 10]

    The enum integer values are deliberately identical to the legacy engine's
    `ChessDir` enum (HOME=0, RIGHT=1, OPPS=2, LEFT=3), so binary replays and
    on-wire protocols do NOT need re-encoding. Only the Python-visible names
    changed. See `LEGACY_DIR_TO_SEAT` below for the explicit mapping.

    Turn order is SOUTH → WEST → NORTH → EAST → SOUTH... (matches legacy
    turn order HOME → RIGHT → OPPS → LEFT → ...).
    """

    SOUTH = 0
    WEST = 1
    NORTH = 2
    EAST = 3

    @property
    def team(self) -> int:
        """Team id: 0 for SOUTH/NORTH (red), 1 for WEST/EAST (blue)."""
        return int(self) % 2

    @property
    def teammate(self) -> Seat:
        """The seat on my team (opposite corner)."""
        return Seat((int(self) + 2) % 4)

    @property
    def next_seat(self) -> Seat:
        """The seat acting after me in turn order (SOUTH→WEST→NORTH→EAST→SOUTH)."""
        return Seat((int(self) + 1) % 4)

    @property
    def left_side_enemy(self) -> Seat:
        """Enemy seat that appears on the LEFT half of MY canonical image.

        Computation: `Seat((int(self) + 1) % 4)`.

        After `world_to_canonical` rotates the board so I (the acting seat)
        sit at SOUTH, this enemy's pieces occupy canonical x ∈ [0, 5] — the
        image left half. In turn order, this is also the seat that acts
        *immediately after* me (same as `self.next_seat`).

        Example: `Seat.SOUTH.left_side_enemy` is `Seat.WEST` — which is at
        world x ∈ [0, 5] (the board's left), so it's already on the left
        half without any rotation. Under canonical rotation for `SOUTH`
        this is identity, so the name "left side" is geometrically
        faithful in all four rotations.
        """
        return Seat((int(self) + 1) % 4)

    @property
    def right_side_enemy(self) -> Seat:
        """Enemy seat that appears on the RIGHT half of MY canonical image.

        Computation: `Seat((int(self) + 3) % 4)`.

        After `world_to_canonical` rotates the board so I sit at SOUTH, this
        enemy's pieces occupy canonical x ∈ [11, 16] — the image right half.
        In turn order, this is the enemy who acts *just before* my teammate
        (i.e. three steps after me = one step before me modulo 4).

        Example: `Seat.SOUTH.right_side_enemy` is `Seat.EAST` — which is at
        world x ∈ [11, 16] (the board's right), so it's already on the
        right half without any rotation.
        """
        return Seat((int(self) + 3) % 4)


NUM_SEATS: Final[int] = 4
# Seats in canonical turn order (also matches legacy ChessDir integer values).
ALL_SEATS: Final[tuple[Seat, ...]] = (Seat.SOUTH, Seat.WEST, Seat.NORTH, Seat.EAST)
TEAM_RED: Final[tuple[Seat, Seat]] = (Seat.SOUTH, Seat.NORTH)
TEAM_BLUE: Final[tuple[Seat, Seat]] = (Seat.WEST, Seat.EAST)

# -----------------------------------------------------------------------
# Legacy ChessDir ↔ Seat mapping
# -----------------------------------------------------------------------
# The legacy engine uses `enum ChessDir {HOME, RIGHT, OPPS, LEFT}` with
# integer values (0, 1, 2, 3). Those names originate in a first-person-view
# labelling ("my left enemy", "my opponent"). We deliberately renamed our
# Python Seat enum to cardinal directions (SOUTH/WEST/NORTH/EAST) to match
# the fixed board geometry, which eliminates an entire class of left/right
# confusion bugs in the canonical rotation pipeline. See ADR-111 in
# docs/DECISIONS.md for the full rationale.
#
# Use this dict when reading legacy binary replays or talking to the legacy
# C engine via ctypes / libjunqicore.so (see tools/legacy_spot_check.py).
LEGACY_DIR_TO_SEAT: Final[Mapping[int, Seat]] = MappingProxyType({
    0: Seat.SOUTH,   # legacy HOME
    1: Seat.WEST,    # legacy RIGHT (first-person "right enemy", on board LEFT)
    2: Seat.NORTH,   # legacy OPPS
    3: Seat.EAST,    # legacy LEFT  (first-person "left enemy",  on board RIGHT)
})
SEAT_TO_LEGACY_DIR: Final[Mapping[Seat, int]] = MappingProxyType({
    seat: legacy_dir for legacy_dir, seat in LEGACY_DIR_TO_SEAT.items()
})


def same_team(a: Seat, b: Seat) -> bool:
    """True iff two seats are teammates (or the same seat)."""
    return a.team == b.team


# ===========================================================================
# Combat events (what is broadcast after each action, see §3.3)
# ===========================================================================


@unique
class Event(IntEnum):
    """Result of a single move. The ONLY info directly exposed to all seats."""

    MOVE = 1      # moving piece relocates to empty cell
    EAT = 2       # src wins, dst dies (flag captures also use EAT + flag_captured=True)
    BOMB = 3      # mutual death
    KILLED = 4    # src dies, dst lives


# ===========================================================================
# Death reasons (T7 / ADR-114) — per-piece historical death attribution
# ===========================================================================
# Unlike `Event` (which describes the combat *outcome* relative to the two
# participants), `DeathReason` classifies **why a specific piece died**.
# It is written into `GameState.death_info[piece_id]` at the moment of death
# and never mutates afterwards. Observation channel group D
# (`death_reason_{ours,theirs}`) reads these reasons at each piece's
# `death_loc` anchor. See docs/PHASE_0.3_T7_TODO.md §1.3.
#
# Design decision (D-2, 2026-04-21): mutual deaths — whether caused by a bomb,
# a landmine collision, or two same-rank combatants — are ALL recorded as
# `MUTUAL`. We intentionally do NOT split mutual-with-mine/bomb into a
# separate bucket; the model can learn that distinction from other channels
# (e.g. `dead_at_zero`, `death_reason` anchor location, piece rank channels).


@unique
class DeathReason(IntEnum):
    """Why a given piece died. Recorded once, at the moment of death."""

    KILLED_BY_ENEMY = 0     # attacked by an enemy piece and lost (non-mutual)
    HIT_MINE_OR_BOMB = 1    # attacker triggered an opposing mine/bomb; dies alone
    MUTUAL = 2              # both sides die in the same combat (BOMB/same-rank)


def classify_death_reason(
    *,
    own_piece: PieceType,
    opponent_piece: PieceType,
    event: Event,
    own_is_attacker: bool,
) -> DeathReason:
    """Classify why `own_piece` died in a combat resolved as `event`.

    Parameters
    ----------
    own_piece
        The piece that is (about to be / just) dead, from the caller's POV.
    opponent_piece
        The other side of the combat.
    event
        The `Event` returned by `resolve_combat`. Must be one of
        EAT / KILLED / BOMB (MOVE has no death and is not a valid input).
    own_is_attacker
        True iff `own_piece` is the src (attacker) in this combat.

    Returns
    -------
    DeathReason
        - BOMB event → `MUTUAL` (D-2 decision; unified regardless of cause)
        - EAT  event → attacker's opponent died → for the loser side:
            * If the loser is a mine/bomb → this branch is never reached,
              since mine/bomb eaten by engineer or flag eaten yields EAT but
              the *loser* (mine/bomb/flag) was the stationary defender —
              from loser's POV, reason = KILLED_BY_ENEMY.
            * Otherwise → `KILLED_BY_ENEMY`.
        - KILLED event → attacker died; if defender is a mine/bomb the
          attacker's death is `HIT_MINE_OR_BOMB`, else `KILLED_BY_ENEMY`
          (e.g. a lower-rank attacker meeting a stronger ranked defender).
    """
    if event is Event.MOVE:
        raise ValueError("Event.MOVE has no death; classify_death_reason not applicable")
    if event not in (Event.EAT, Event.KILLED, Event.BOMB):
        raise ValueError(f"unexpected event {event!r}")

    # Mutual death — always MUTUAL, irrespective of mine/bomb involvement.
    if event is Event.BOMB:
        return DeathReason.MUTUAL

    # At this point event is EAT or KILLED (exactly one side dies).
    # Determine whether `own_piece` is the loser of this combat.
    if event is Event.EAT:
        loser_is_attacker = False   # attacker ate defender; defender died.
    else:  # Event.KILLED
        loser_is_attacker = True    # attacker died; defender survived.

    if own_is_attacker != loser_is_attacker:
        # `own_piece` is not the loser, so it shouldn't die in this call.
        raise AssertionError(
            f"classify_death_reason called on non-losing piece "
            f"(event={event!r}, own_is_attacker={own_is_attacker})"
        )

    # `own_piece` is the loser. Attribute the death.
    # - If the OTHER side is a mine/bomb AND `own_piece` attacked into it,
    #   the death is HIT_MINE_OR_BOMB (walked into a stationary trap).
    if own_is_attacker and (opponent_piece.is_mine or opponent_piece.is_bomb):
        return DeathReason.HIT_MINE_OR_BOMB

    # Otherwise (including: engineer eats mine? — that's EAT not KILLED, so
    # engineer doesn't die; flag-eating — eaten piece is the flag, reason is
    # KILLED_BY_ENEMY since an enemy captured it) it's a plain enemy kill.
    return DeathReason.KILLED_BY_ENEMY


# ===========================================================================
# Cell roles within a seat's 30-slot rectangle
# ===========================================================================
# Camps: fixed indices {6, 8, 12, 16, 18} in each seat's local numbering.
# Strongholds: {26, 28} (the flag must sit in one of these two).
# Rails: see board.py for the full 17×17 world-coordinate classification.
#
# These indices derive from legacy_engine/src/junqi.c::SetBoardCamp.

CAMP_INDICES: Final[frozenset[int]] = frozenset({6, 8, 12, 16, 18})
STRONGHOLD_INDICES: Final[frozenset[int]] = frozenset({26, 28})

# Within a seat's own 5×6 layout (see RULES.md §1.2):
#   Row 0: indices 0..4   (front row, closest to center)
#   Row 1: indices 5..9
#   ...
#   Row 5: indices 25..29 (back row, closest to own edge)
FRONT_ROW_INDICES: Final[frozenset[int]] = frozenset(range(0, 5))
BACK_TWO_ROWS_INDICES: Final[frozenset[int]] = frozenset(range(20, 30))


def index_row(i: int) -> int:
    """Return row index (0..5) of a seat-local piece index."""
    if not 0 <= i < SLOTS_PER_SEAT:
        raise ValueError(f"piece index {i} out of range [0,{SLOTS_PER_SEAT})")
    return i // 5


def index_col(i: int) -> int:
    """Return column index (0..4) of a seat-local piece index."""
    if not 0 <= i < SLOTS_PER_SEAT:
        raise ValueError(f"piece index {i} out of range [0,{SLOTS_PER_SEAT})")
    return i % 5


def is_camp_index(i: int) -> bool:
    """True iff local index `i` is a camp cell."""
    return i in CAMP_INDICES


def is_stronghold_index(i: int) -> bool:
    """True iff local index `i` is a stronghold (must host the flag for exactly one)."""
    return i in STRONGHOLD_INDICES


def is_front_row_index(i: int) -> bool:
    """True iff `i` is in the front row (bombs forbidden)."""
    return i in FRONT_ROW_INDICES


def is_back_two_rows_index(i: int) -> bool:
    """True iff `i` is in the back two rows (the only legal zone for landmines)."""
    return i in BACK_TWO_ROWS_INDICES


# ===========================================================================
# Draw thresholds (Q10 / ADR-010)
# ===========================================================================

MAX_NUM_MOVES: Final[int] = 4000
MAX_NUM_MOVES_BETWEEN_ATTACKS: Final[int] = 200

# ===========================================================================
# Show modes (see RULES.md §4)
# ===========================================================================


@unique
class ShowMode(IntEnum):
    """Visibility setting for non-own pieces."""

    BRIGHT = 0       # all pieces visible to all seats (明棋, training curriculum stage 1)
    DARK = 1         # only own pieces visible (暗棋, full imperfect-info)
    HALF_DARK = 2    # own + teammate visible; enemies hidden (Q11, training default)


# ===========================================================================
# Combat table (§3.1)
# ===========================================================================


def resolve_combat(
    attacker: PieceType,
    defender: PieceType,
) -> Event:
    """Pure-function resolver for a single combat.

    Preconditions:
      - `attacker` is a mobile piece (not NONE, not JUNQI, not DILEI).
      - `defender` may be anything EXCEPT NONE/DARK (if dst is empty, caller
        should return Event.MOVE without calling this function).

    Returns the Event that should be broadcast. Does NOT mutate state.

    This function encodes the full combat table from RULES.md §3.1 in one place.
    """
    if attacker in (PieceType.NONE, PieceType.DARK):
        raise ValueError(f"invalid attacker type {attacker!r}")
    if defender in (PieceType.NONE, PieceType.DARK):
        raise ValueError(f"invalid defender type {defender!r} (empty cell should skip combat)")
    if attacker.is_immobile:
        raise ValueError(f"immobile attacker {attacker!r} cannot initiate combat")

    # Flag capture: any piece eats the flag.
    if defender.is_flag:
        return Event.EAT

    # Mine: only engineer eats it; everyone else dies.
    if defender.is_mine:
        return Event.EAT if attacker.is_engineer else Event.KILLED

    # Bomb: mutual death regardless of direction.
    if attacker.is_bomb or defender.is_bomb:
        return Event.BOMB

    # Both ranked combatants — compare by rank (smaller enum = stronger).
    if not (attacker.is_ranked_combatant and defender.is_ranked_combatant):
        # Covers the "attacker is the flag or the mine" pre-condition violations
        # that we already guarded against; reaching here is a bug.
        raise AssertionError(
            f"unreachable combat combination: {attacker!r} vs {defender!r}"
        )

    if attacker.value == defender.value:
        return Event.BOMB  # same rank → mutual death
    if attacker.value < defender.value:
        return Event.EAT   # attacker stronger
    return Event.KILLED    # attacker weaker


def siling_reveals_src(attacker: PieceType, defender: PieceType, event: Event) -> bool:
    """True iff src-seat's flag should be revealed after this combat (Q7 / §3.2).

    The rule: "whichever side's SILING dies, that side's flag is revealed".
    A SILING dies when it loses a KILLED (vs stronger... but SILING is the
    strongest ranked combatant, so it only dies in KILLED vs a MINE, or in BOMB
    against a same-rank SILING or against any ZHADAN).

    We check src-side SILING death: src dies iff event in {KILLED, BOMB}.
    """
    if not attacker.is_siling:
        return False
    return event in (Event.KILLED, Event.BOMB)


def siling_reveals_dst(attacker: PieceType, defender: PieceType, event: Event) -> bool:
    """True iff dst-seat's flag should be revealed after this combat (Q7 / §3.2)."""
    if not defender.is_siling:
        return False
    # dst dies iff event in {EAT, BOMB}.
    return event in (Event.EAT, Event.BOMB)


# ===========================================================================
# Tiny self-test (runs only when module is executed directly)
# ===========================================================================


def _self_test() -> None:  # pragma: no cover - manual sanity check
    # Strength ordering
    assert PieceType.SILING < PieceType.JUNZH < PieceType.GONGB
    # Team membership
    assert Seat.SOUTH.team == Seat.NORTH.team == 0
    assert Seat.WEST.team == Seat.EAST.team == 1
    assert Seat.SOUTH.teammate is Seat.NORTH
    # Side-of-image enemy properties
    assert Seat.SOUTH.left_side_enemy is Seat.WEST
    assert Seat.SOUTH.right_side_enemy is Seat.EAST
    assert Seat.WEST.left_side_enemy is Seat.NORTH
    assert Seat.WEST.right_side_enemy is Seat.SOUTH
    # Legacy ChessDir mapping (integer identity preserved)
    assert int(Seat.SOUTH) == 0 and LEGACY_DIR_TO_SEAT[0] is Seat.SOUTH
    assert int(Seat.WEST) == 1 and LEGACY_DIR_TO_SEAT[1] is Seat.WEST
    assert int(Seat.NORTH) == 2 and LEGACY_DIR_TO_SEAT[2] is Seat.NORTH
    assert int(Seat.EAST) == 3 and LEGACY_DIR_TO_SEAT[3] is Seat.EAST
    # Combat smoke tests
    assert resolve_combat(PieceType.SILING, PieceType.JUNZH) is Event.EAT
    assert resolve_combat(PieceType.GONGB, PieceType.DILEI) is Event.EAT
    assert resolve_combat(PieceType.PAIZH, PieceType.DILEI) is Event.KILLED
    assert resolve_combat(PieceType.SILING, PieceType.ZHADAN) is Event.BOMB
    assert resolve_combat(PieceType.SILING, PieceType.SILING) is Event.BOMB
    assert resolve_combat(PieceType.PAIZH, PieceType.GONGB) is Event.EAT
    # Q7 deduction
    e = resolve_combat(PieceType.SILING, PieceType.ZHADAN)
    assert siling_reveals_src(PieceType.SILING, PieceType.ZHADAN, e) is True
    assert siling_reveals_dst(PieceType.SILING, PieceType.ZHADAN, e) is False
    e = resolve_combat(PieceType.ZHADAN, PieceType.SILING)
    assert siling_reveals_src(PieceType.ZHADAN, PieceType.SILING, e) is False
    assert siling_reveals_dst(PieceType.ZHADAN, PieceType.SILING, e) is True
    e = resolve_combat(PieceType.SILING, PieceType.SILING)
    assert siling_reveals_src(PieceType.SILING, PieceType.SILING, e) is True
    assert siling_reveals_dst(PieceType.SILING, PieceType.SILING, e) is True

    # T7 / ADR-114 — DeathReason classification smoke tests
    # 1. Ranked combatant walks into stronger defender → attacker KILLED_BY_ENEMY.
    e = resolve_combat(PieceType.PAIZH, PieceType.SILING)  # KILLED
    assert classify_death_reason(
        own_piece=PieceType.PAIZH, opponent_piece=PieceType.SILING,
        event=e, own_is_attacker=True,
    ) is DeathReason.KILLED_BY_ENEMY
    # 2. Ranked combatant walks into a mine → attacker HIT_MINE_OR_BOMB.
    e = resolve_combat(PieceType.PAIZH, PieceType.DILEI)   # KILLED (non-engineer)
    assert classify_death_reason(
        own_piece=PieceType.PAIZH, opponent_piece=PieceType.DILEI,
        event=e, own_is_attacker=True,
    ) is DeathReason.HIT_MINE_OR_BOMB
    # 3. Same-rank bomb-out → both MUTUAL.
    e = resolve_combat(PieceType.SILING, PieceType.SILING)  # BOMB
    assert classify_death_reason(
        own_piece=PieceType.SILING, opponent_piece=PieceType.SILING,
        event=e, own_is_attacker=True,
    ) is DeathReason.MUTUAL
    assert classify_death_reason(
        own_piece=PieceType.SILING, opponent_piece=PieceType.SILING,
        event=e, own_is_attacker=False,
    ) is DeathReason.MUTUAL
    # 4. Ranked piece EAT → defender dies, reason KILLED_BY_ENEMY.
    e = resolve_combat(PieceType.SILING, PieceType.JUNZH)   # EAT
    assert classify_death_reason(
        own_piece=PieceType.JUNZH, opponent_piece=PieceType.SILING,
        event=e, own_is_attacker=False,
    ) is DeathReason.KILLED_BY_ENEMY
    # 5. Mine attacker into bomb defender → attacker HIT_MINE_OR_BOMB.
    e = resolve_combat(PieceType.LIANZH, PieceType.ZHADAN)  # BOMB (mutual)
    assert classify_death_reason(
        own_piece=PieceType.LIANZH, opponent_piece=PieceType.ZHADAN,
        event=e, own_is_attacker=True,
    ) is DeathReason.MUTUAL  # MUTUAL wins over mine-attribution per D-2.

    print("junqi_core.rules self-test: OK")


if __name__ == "__main__":
    _self_test()
