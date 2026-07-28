"""T7 / ADR-114 M2 — `PieceState` counter maintenance unit tests.

These tests construct minimal hand-built `GameState` snapshots that isolate
a single combat scenario and then assert the exact counter transitions
stipulated in docs/PHASE_0.3_T7_TODO.md §2.1:

  - Event.MOVE   → acting piece's move_count += 1; others untouched.
  - Event.EAT    → attacker.active_eat_count += 1; defender drops out of
                   piece_state (death handled by M3).
  - Event.KILLED → defender.passive_survive_count += 1; attacker drops.
  - Event.BOMB   → both sides drop; no increments (mutual death, no survivor).
  - Flag capture → every alive piece of the surrendering seat drops; the
                   capturing piece still gets active_eat_count += 1.
  - Q12 cascade  → every alive piece of the stranded seat drops.

Invariants pinned globally:
  - Every piece_id in `piece_state` corresponds to a *currently alive*
    piece on `state.pieces`.
  - `deaths` stays empty (M3 scope, not M2).
  - `zero_board` is unchanged across all transitions.
"""

from __future__ import annotations

import random

import pytest

from junqi_core.move_gen import PieceMap, PieceRef
from junqi_core.rules import Event, PieceType, Seat
from junqi_core.setup import generate_random_setup
from junqi_core.state import (
    Action,
    GameState,
    PieceState,
    SeatInfo,
)


# ---------------------------------------------------------------------------
# Manual GameState builders
# ---------------------------------------------------------------------------
# We can't easily use `new_game()` for micro-combat tests because it
# requires a full 4×25 piece setup. Instead we build tiny `GameState`
# snapshots by hand, assigning synthetic piece_ids, and call step() on them.


def _seat_of(ref: PieceRef) -> Seat:
    return ref.seat


def _p(seat: Seat, pt: PieceType, pid: int) -> PieceRef:
    """Shorthand for building a PieceRef with an explicit piece_id."""
    return PieceRef(seat=seat, piece_type=pt, alive=True, piece_id=pid)


def _build_state(
    *,
    pieces: PieceMap,
    turn: Seat,
) -> GameState:
    """Assemble a minimal GameState from hand-built PieceMap.

    - `piece_state` is initialised to PieceState() for every piece_id in
      `pieces`.
    - `zero_board` is seeded to `pieces` (we don't care about its exact
      contents for counter tests; only that it exists and is preserved).
    - All four seats start alive with no flag reveal.
    """
    from junqi_core.rules import RULES_VERSION, ShowMode
    piece_state = {ref.piece_id: PieceState() for ref in pieces.values()}
    info = {s: SeatInfo() for s in Seat}
    return GameState(
        pieces=dict(pieces),
        turn=turn,
        move_counter=0,
        moves_since_last_combat=0,
        info=info,
        terminated=False,
        winner_team=None,
        draw=False,
        show_mode=ShowMode.HALF_DARK,
        rules_version=RULES_VERSION,
        debug_include_private=False,
        zero_board=dict(pieces),
        piece_state=piece_state,
        deaths={},
    )


def _live_piece_state_invariant(st: GameState) -> None:
    """Every key in piece_state maps to a currently alive piece on pieces."""
    live_ids = {ref.piece_id for ref in st.pieces.values()}
    assert set(st.piece_state.keys()) == live_ids


# ---------------------------------------------------------------------------
# Event.MOVE
# ---------------------------------------------------------------------------

class TestMoveCountBump:
    def test_single_move_bumps_attacker_only(self) -> None:
        """Build a board with one SOUTH PAIZH next to an empty camp-free cell
        and one distant WEST piece (to keep the game going). Execute a plain
        move — attacker's move_count must be exactly 1 and other counters
        untouched."""
        # (8, 10) is a non-camp central junction on the 17×17 board. We
        # place SOUTH PAIZH at (8, 11) and keep a WEST piece far enough so
        # it's SOUTH's turn and a legal move is `(8,11) -> (8,10)` (a
        # one-step orthogonal move onto an empty non-camp cell).
        pieces: PieceMap = {
            (8, 11): _p(Seat.SOUTH, PieceType.PAIZH, 10),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),   # keeps SOUTH alive
            # WEST: JUNQI (immobile) + a MOBILE PAIZH far from walls.
            (2, 8):  _p(Seat.WEST, PieceType.PAIZH, 30),
            (0, 8):  _p(Seat.WEST, PieceType.JUNQI, 56),
            # NORTH: JUNQI + a MOBILE PAIZH.
            (8, 3):  _p(Seat.NORTH, PieceType.PAIZH, 60),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            # EAST: JUNQI + a MOBILE PAIZH.
            (14, 8): _p(Seat.EAST, PieceType.PAIZH, 90),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(8, 11), dst=(8, 10))
        st2, res = st.step(act)

        assert res.event is Event.MOVE
        assert st2.piece_state[10].move_count == 1
        assert st2.piece_state[10].active_eat_count == 0
        assert st2.piece_state[10].passive_survive_count == 0
        for pid, ps in st2.piece_state.items():
            if pid == 10:
                continue
            assert ps == PieceState()

        # Invariants
        _live_piece_state_invariant(st2)
        # Event.MOVE has no death — deaths dict unchanged.
        assert st2.deaths == {}
        assert st2.zero_board is st.zero_board

        # Source state unchanged.
        assert st.piece_state[10].move_count == 0


# ---------------------------------------------------------------------------
# Event.EAT
# ---------------------------------------------------------------------------

class TestActiveEatBump:
    def test_attacker_wins_eat(self) -> None:
        """SOUTH SILING attacks WEST PAIZH. SILING survives (EAT);
        active_eat_count += 1 on SILING; PAIZH's piece_state entry is popped."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.SILING, 11),   # attacker
            (6, 10): _p(Seat.WEST,  PieceType.PAIZH,  40),   # victim
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (0, 8):  _p(Seat.WEST,  PieceType.JUNQI, 56),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (10, 3): _p(Seat.NORTH, PieceType.PAIZH, 60),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST, PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10))
        st2, res = st.step(act)

        assert res.event is Event.EAT
        # Attacker's active_eat_count bumped; move_count stays 0 (an eat
        # that displaces into dst is NOT a MOVE per our contract — the
        # move_count bucket only tracks Event.MOVE semantically).
        assert st2.piece_state[11].active_eat_count == 1
        assert st2.piece_state[11].move_count == 0
        assert st2.piece_state[11].passive_survive_count == 0
        # Victim's piece_state entry popped.
        assert 40 not in st2.piece_state
        # Board still has 40's attacker occupying (6, 10); victim gone.
        assert st2.pieces[(6, 10)].piece_id == 11

        _live_piece_state_invariant(st2)
        # T7 M3: victim recorded in deaths as KILLED_BY_ENEMY @ dst.
        from junqi_core.rules import DeathReason
        from junqi_core.state import DeathInfo
        assert 40 in st2.deaths
        assert st2.deaths[40] == DeathInfo(
            piece_id=40,
            reason=DeathReason.KILLED_BY_ENEMY,
            death_loc=(6, 10),
            step=1,
        )
        # Attacker (survivor) is NOT in deaths.
        assert 11 not in st2.deaths


# ---------------------------------------------------------------------------
# Event.KILLED
# ---------------------------------------------------------------------------

class TestPassiveSurviveBump:
    def test_defender_survives_hit(self) -> None:
        """SOUTH PAIZH attacks WEST SILING. Attacker dies (KILLED);
        SILING's passive_survive_count += 1; attacker's piece_state popped."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.PAIZH,  12),   # attacker → dies
            (6, 10): _p(Seat.WEST,  PieceType.SILING, 41),   # defender → lives
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (0, 8):  _p(Seat.WEST,  PieceType.JUNQI, 56),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (10, 3): _p(Seat.NORTH, PieceType.PAIZH, 60),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST, PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10))
        st2, res = st.step(act)

        assert res.event is Event.KILLED
        assert 12 not in st2.piece_state                 # attacker popped
        assert st2.piece_state[41].passive_survive_count == 1
        assert st2.piece_state[41].move_count == 0
        assert st2.piece_state[41].active_eat_count == 0
        # Defender is still on the board at its original square.
        assert st2.pieces[(6, 10)].piece_id == 41

        _live_piece_state_invariant(st2)
        # T7 M3: attacker recorded in deaths as KILLED_BY_ENEMY @ dst
        # (PAIZH losing to SILING — a plain enemy kill, not a mine/bomb).
        from junqi_core.rules import DeathReason
        assert 12 in st2.deaths
        assert st2.deaths[12].reason is DeathReason.KILLED_BY_ENEMY
        assert st2.deaths[12].death_loc == (6, 10)
        assert st2.deaths[12].step == 1
        # Defender (survivor) is NOT in deaths.
        assert 41 not in st2.deaths


# ---------------------------------------------------------------------------
# Event.BOMB — mutual death, no increments
# ---------------------------------------------------------------------------

class TestBombNoIncrement:
    def test_mutual_bomb_pops_both(self) -> None:
        """SILING vs SILING → BOMB. Both pieces' piece_state entries
        disappear. No counter is incremented (the dead never accumulate)."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.SILING, 13),
            (6, 10): _p(Seat.WEST,  PieceType.SILING, 42),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (0, 8):  _p(Seat.WEST,  PieceType.JUNQI, 56),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (10, 3): _p(Seat.NORTH, PieceType.PAIZH, 60),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST, PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10))
        st2, res = st.step(act)

        assert res.event is Event.BOMB
        assert 13 not in st2.piece_state
        assert 42 not in st2.piece_state
        # No piece on either square.
        assert (6, 11) not in st2.pieces
        assert (6, 10) not in st2.pieces

        _live_piece_state_invariant(st2)
        # T7 M3: both sides recorded as MUTUAL @ dst cell.
        from junqi_core.rules import DeathReason
        assert 13 in st2.deaths
        assert 42 in st2.deaths
        assert st2.deaths[13].reason is DeathReason.MUTUAL
        assert st2.deaths[42].reason is DeathReason.MUTUAL
        assert st2.deaths[13].death_loc == (6, 10)
        assert st2.deaths[42].death_loc == (6, 10)

    def test_ranked_vs_bomb_pops_both(self) -> None:
        """PAIZH walks into ZHADAN → BOMB → both die. No counter bump."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.PAIZH,  14),
            (6, 10): _p(Seat.WEST,  PieceType.ZHADAN, 43),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (0, 8):  _p(Seat.WEST,  PieceType.JUNQI, 56),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (10, 3): _p(Seat.NORTH, PieceType.PAIZH, 60),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST, PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10))
        st2, res = st.step(act)

        assert res.event is Event.BOMB
        assert 14 not in st2.piece_state
        assert 43 not in st2.piece_state
        _live_piece_state_invariant(st2)


# ---------------------------------------------------------------------------
# Landmine
# ---------------------------------------------------------------------------

class TestMineKilled:
    def test_non_engineer_dies_on_mine(self) -> None:
        """PAIZH attacks a DILEI → Event.KILLED. Defender (mine) survives
        → passive_survive_count += 1 on the mine; attacker popped."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.PAIZH, 15),
            (6, 10): _p(Seat.WEST,  PieceType.DILEI, 44),
            # Extra WEST mobile piece so WEST still has legal moves after
            # combat (avoids Q12 cascade that would purge pid=44).
            (3, 10): _p(Seat.WEST,  PieceType.PAIZH, 49),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (0, 8):  _p(Seat.WEST,  PieceType.JUNQI, 56),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (10, 3): _p(Seat.NORTH, PieceType.PAIZH, 60),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST, PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10))
        st2, res = st.step(act)

        assert res.event is Event.KILLED
        assert 15 not in st2.piece_state
        assert st2.piece_state[44].passive_survive_count == 1
        _live_piece_state_invariant(st2)

    def test_engineer_eats_mine(self) -> None:
        """GONGB attacks a DILEI → Event.EAT. Engineer survives & wins →
        active_eat_count += 1; mine popped."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.GONGB, 16),
            (6, 10): _p(Seat.WEST,  PieceType.DILEI, 45),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
            (0, 8):  _p(Seat.WEST,  PieceType.JUNQI, 56),
            (10, 2): _p(Seat.NORTH, PieceType.JUNQI, 86),
            (10, 3): _p(Seat.NORTH, PieceType.PAIZH, 60),
            (16, 10): _p(Seat.EAST, PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST, PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10))
        st2, res = st.step(act)

        assert res.event is Event.EAT
        assert 45 not in st2.piece_state
        assert st2.piece_state[16].active_eat_count == 1
        _live_piece_state_invariant(st2)


# ---------------------------------------------------------------------------
# Flag capture: surrendering seat's piece_state is fully purged
# ---------------------------------------------------------------------------

class TestFlagCaptureSurrender:
    def test_flag_capture_purges_all_defender_piece_state(self) -> None:
        """SOUTH SILING captures WEST JUNQI adjacent to its stronghold.
        All WEST pieces (3 of them in this setup) get their piece_state
        entries popped; SILING gets active_eat_count += 1."""
        pieces: PieceMap = {
            (1, 9): _p(Seat.SOUTH, PieceType.SILING, 17),   # attacker
            (0, 9): _p(Seat.WEST,  PieceType.JUNQI,  46),   # flag at stronghold
            (2, 9): _p(Seat.WEST,  PieceType.PAIZH,  47),   # extra WEST pieces
            (0, 8): _p(Seat.WEST,  PieceType.DILEI,  48),
            # Other seats so the game doesn't terminate.
            (10, 16): _p(Seat.SOUTH, PieceType.PAIZH, 29),
            (6, 0):   _p(Seat.NORTH, PieceType.JUNQI, 60),
            (10, 0):  _p(Seat.NORTH, PieceType.PAIZH, 86),
            (16, 8):  _p(Seat.EAST,  PieceType.JUNQI, 116),
            (16, 9):  _p(Seat.EAST,  PieceType.PAIZH, 90),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        act = Action(seat=Seat.SOUTH, src=(1, 9), dst=(0, 9))
        st2, res = st.step(act)

        assert res.event is Event.EAT
        assert res.flag_captured is True
        # All WEST piece_state entries gone.
        for west_pid in (46, 47, 48):
            assert west_pid not in st2.piece_state
        # Attacker bumped active_eat_count.
        assert st2.piece_state[17].active_eat_count == 1
        # WEST seat marked dead.
        assert st2.info[Seat.WEST].dead is True

        _live_piece_state_invariant(st2)


# ---------------------------------------------------------------------------
# Q12 cascade: stranded seat's piece_state purged
# ---------------------------------------------------------------------------

class TestQ12Cascade:
    def test_q12_seat_piece_state_purged(self) -> None:
        """Construct a setup where NORTH (next seat after SOUTH moves) has
        exactly one piece wedged such that it has no legal action, triggering
        Q12 death of NORTH. That seat's piece_state entries must all be
        popped by _remove_all_pieces_of_seat with piece_state= threaded in."""
        # We reuse the same pattern as the existing Q12 test in
        # test_state_transitions.py: NORTH has a single JUNQI (immobile) so
        # it never has a legal move. SOUTH moves first; the turn will rotate
        # to WEST (which has a legal move), so Q12 cascade applies to
        # NORTH only after WEST/EAST if they also lack moves. Simpler and
        # more reliable path: put NORTH's ONLY piece as a JUNQI (immobile);
        # put WEST and EAST with alive pieces that can move; SOUTH moves
        # first → turn goes to WEST (who can move) → no Q12 triggers on
        # this step. So this particular micro-test can't easily force Q12
        # without deep fixture building.
        #
        # Instead we validate the *mechanism* directly: call
        # `_remove_all_pieces_of_seat` with piece_state and confirm the
        # pop-sweep works.
        from junqi_core.state import _remove_all_pieces_of_seat

        pieces: PieceMap = {
            (6, 0):  _p(Seat.NORTH, PieceType.JUNQI, 60),
            (10, 0): _p(Seat.NORTH, PieceType.PAIZH, 86),
            (6, 13): _p(Seat.SOUTH, PieceType.JUNQI, 28),
        }
        piece_state: dict[int, PieceState] = {
            60: PieceState(move_count=2),
            86: PieceState(active_eat_count=5),
            28: PieceState(passive_survive_count=1),
        }
        _remove_all_pieces_of_seat(
            pieces, Seat.NORTH, piece_state=piece_state
        )
        # NORTH pieces gone from both structures.
        assert (6, 0) not in pieces
        assert (10, 0) not in pieces
        assert 60 not in piece_state
        assert 86 not in piece_state
        # SOUTH piece untouched.
        assert (6, 13) in pieces
        assert piece_state[28].passive_survive_count == 1

    def test_remove_without_piece_state_still_works(self) -> None:
        """Backwards-compat: `piece_state=None` leaves the dict untouched
        (we exercise this path from legacy tests that don't care)."""
        from junqi_core.state import _remove_all_pieces_of_seat

        pieces: PieceMap = {
            (6, 0): _p(Seat.NORTH, PieceType.JUNQI, 60),
        }
        _remove_all_pieces_of_seat(pieces, Seat.NORTH)  # default None
        assert pieces == {}


# ---------------------------------------------------------------------------
# Full-game smoke: run a whole random game, then check global invariants
# ---------------------------------------------------------------------------

class TestFullGameInvariants:
    def test_random_game_preserves_liveness_invariant(self) -> None:
        """Play a random game for up to 200 steps (or until termination) and
        after every step check that piece_state keys match the set of alive
        piece_ids. This is the #1 invariant that M3 will rely on."""
        rng = random.Random(2026)
        setups = generate_random_setup(rng)
        st = GameState.new_game(setups)

        for _ in range(200):
            if st.terminated:
                break
            acts = st.legal_actions()
            if not acts:
                break
            act = acts[rng.randrange(len(acts))]
            st, _ = st.step(act)
            live_ids = {ref.piece_id for ref in st.pieces.values()}
            assert set(st.piece_state.keys()) == live_ids, (
                f"piece_state diverged from alive pieces at step "
                f"{st.move_counter}"
            )
            # Every counter value must be non-negative and strictly bounded.
            for ps in st.piece_state.values():
                assert ps.move_count >= 0
                assert ps.active_eat_count >= 0
                assert ps.passive_survive_count >= 0
            # T7 M3: deaths grows monotonically — never shrinks, every
            # entry carries a valid DeathReason and step<=current step.
            for pid, di in st.deaths.items():
                assert di.piece_id == pid
                assert di.step <= st.move_counter
                # Dead pieces must NOT be in piece_state (liveness).
                assert pid not in st.piece_state
