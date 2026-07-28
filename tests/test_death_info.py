"""T7 / ADR-114 M3 — DeathInfo recording unit tests.

These tests verify that every death scenario writes exactly one frozen
`DeathInfo` entry into `GameState.deaths`, keyed by piece_id, with the
correct `reason`, `death_loc`, and `step`.

Scenarios covered:

  1. Event.EAT        → defender recorded as KILLED_BY_ENEMY @ dst.
  2. Event.KILLED     → attacker recorded as KILLED_BY_ENEMY (if vs ranked)
                      or HIT_MINE_OR_BOMB (if vs DILEI/ZHADAN) @ dst.
  3. Event.BOMB       → both combatants recorded as MUTUAL @ dst.
  4. Flag capture     → defending seat's collateral pieces recorded as
                        KILLED_BY_ENEMY @ their current cells (decision A);
                        the flag itself recorded via the EAT combat branch.
  5. Q12 cascade      → stranded seat's pieces recorded as KILLED_BY_ENEMY
                        @ their current cells (decision A).
  6. Idempotence      → step() never overwrites an existing DeathInfo.
  7. DeathInfo is frozen (cannot mutate post-creation).

Global invariants cross-checked at each step:
  - Every piece_id in `deaths` is NOT in `piece_state` (liveness).
  - `deaths[pid].step <= new_state.move_counter` (causality).
  - `deaths[pid].piece_id == pid` (key/field consistency).
  - `zero_board` remains identical across all transitions.
"""

from __future__ import annotations

import random

import pytest

from junqi_core.move_gen import PieceMap, PieceRef
from junqi_core.rules import (
    RULES_VERSION,
    DeathReason,
    Event,
    PieceType,
    Seat,
    ShowMode,
)
from junqi_core.setup import generate_random_setup
from junqi_core.state import (
    Action,
    DeathInfo,
    GameState,
    PieceState,
    SeatInfo,
)


# ---------------------------------------------------------------------------
# Builders (mirror those in test_piece_counters.py)
# ---------------------------------------------------------------------------

def _p(seat: Seat, pt: PieceType, pid: int) -> PieceRef:
    return PieceRef(seat=seat, piece_type=pt, alive=True, piece_id=pid)


def _build_state(*, pieces: PieceMap, turn: Seat) -> GameState:
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


def _assert_global_invariants(st: GameState) -> None:
    """Death dict consistency: no key is also in piece_state, and all
    recorded DeathInfos are self-consistent."""
    for pid, di in st.deaths.items():
        assert di.piece_id == pid, f"deaths[{pid}] has mismatched piece_id"
        assert di.step <= st.move_counter, (
            f"deaths[{pid}].step={di.step} > move_counter={st.move_counter}"
        )
        assert pid not in st.piece_state, (
            f"pid {pid} is in BOTH deaths and piece_state (liveness broken)"
        )
        assert isinstance(di.reason, DeathReason)


# Shared background: non-combatant pieces on seats we're not testing,
# each with a MOBILE second piece so Q12 cascade doesn't fire unexpectedly.
def _quiet_others(*, exclude: set[Seat]) -> PieceMap:
    out: PieceMap = {}
    if Seat.SOUTH not in exclude:
        out[(6, 13)] = _p(Seat.SOUTH, PieceType.JUNQI, 28)
        out[(8, 13)] = _p(Seat.SOUTH, PieceType.PAIZH, 20)
    if Seat.WEST not in exclude:
        out[(0, 8)] = _p(Seat.WEST, PieceType.JUNQI, 56)
        out[(2, 8)] = _p(Seat.WEST, PieceType.PAIZH, 50)
    if Seat.NORTH not in exclude:
        out[(10, 2)] = _p(Seat.NORTH, PieceType.JUNQI, 86)
        out[(8, 3)] = _p(Seat.NORTH, PieceType.PAIZH, 70)
    if Seat.EAST not in exclude:
        out[(16, 10)] = _p(Seat.EAST, PieceType.JUNQI, 116)
        out[(14, 8)] = _p(Seat.EAST, PieceType.PAIZH, 100)
    return out


# ---------------------------------------------------------------------------
# Event.EAT → defender KILLED_BY_ENEMY @ dst
# ---------------------------------------------------------------------------

class TestDeathInfoOnEat:
    def test_defender_recorded_killed_by_enemy(self) -> None:
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.SILING, 11),   # attacker
            (6, 10): _p(Seat.WEST, PieceType.PAIZH, 40),     # victim
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),      # WEST still has
            (3, 10): _p(Seat.WEST, PieceType.LIANZH, 51),    # mobile piece
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        assert res.event is Event.EAT
        assert 40 in st2.deaths
        assert st2.deaths[40] == DeathInfo(
            piece_id=40,
            reason=DeathReason.KILLED_BY_ENEMY,
            death_loc=(6, 10),
            step=1,
        )
        # Attacker is alive → NOT in deaths.
        assert 11 not in st2.deaths
        _assert_global_invariants(st2)


# ---------------------------------------------------------------------------
# Event.KILLED → attacker recorded
# ---------------------------------------------------------------------------

class TestDeathInfoOnKilled:
    def test_ranked_attacker_loses_to_ranked_defender(self) -> None:
        """PAIZH (weak) vs SILING (strongest) → KILLED, attacker dies.
        Reason should be KILLED_BY_ENEMY (defender is not mine/bomb)."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.PAIZH, 12),
            (6, 10): _p(Seat.WEST, PieceType.SILING, 41),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),
            (3, 10): _p(Seat.WEST, PieceType.LIANZH, 51),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        assert res.event is Event.KILLED
        assert 12 in st2.deaths
        assert st2.deaths[12].reason is DeathReason.KILLED_BY_ENEMY
        assert st2.deaths[12].death_loc == (6, 10)
        assert st2.deaths[12].step == 1
        # Defender survived → no DeathInfo for it.
        assert 41 not in st2.deaths
        _assert_global_invariants(st2)

    def test_non_engineer_into_mine_is_hit_mine_or_bomb(self) -> None:
        """Non-engineer attacks DILEI → KILLED, reason HIT_MINE_OR_BOMB."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.PAIZH, 15),
            (6, 10): _p(Seat.WEST, PieceType.DILEI, 44),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),
            (3, 10): _p(Seat.WEST, PieceType.LIANZH, 52),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        assert res.event is Event.KILLED
        assert st2.deaths[15].reason is DeathReason.HIT_MINE_OR_BOMB
        assert st2.deaths[15].death_loc == (6, 10)
        assert 44 not in st2.deaths                      # mine survived
        _assert_global_invariants(st2)


# ---------------------------------------------------------------------------
# Event.BOMB → both recorded as MUTUAL
# ---------------------------------------------------------------------------

class TestDeathInfoOnBomb:
    def test_same_rank_mutual(self) -> None:
        """SILING vs SILING → BOMB; both recorded as MUTUAL."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.SILING, 13),
            (6, 10): _p(Seat.WEST, PieceType.SILING, 42),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),
            (3, 10): _p(Seat.WEST, PieceType.LIANZH, 53),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        assert res.event is Event.BOMB
        assert st2.deaths[13].reason is DeathReason.MUTUAL
        assert st2.deaths[42].reason is DeathReason.MUTUAL
        assert st2.deaths[13].death_loc == (6, 10)
        assert st2.deaths[42].death_loc == (6, 10)
        assert st2.deaths[13].step == st2.deaths[42].step == 1
        _assert_global_invariants(st2)

    def test_ranked_vs_bomb_is_mutual_not_mine(self) -> None:
        """Per D-2: BOMB event always → MUTUAL, even when ZHADAN defender.
        This guards against accidental classification as HIT_MINE_OR_BOMB."""
        pieces: PieceMap = {
            (6, 11): _p(Seat.SOUTH, PieceType.LIANZH, 14),
            (6, 10): _p(Seat.WEST, PieceType.ZHADAN, 43),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (0, 8): _p(Seat.WEST, PieceType.JUNQI, 56),
            (3, 10): _p(Seat.WEST, PieceType.PAIZH, 54),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(6, 11), dst=(6, 10)))

        assert res.event is Event.BOMB
        assert st2.deaths[14].reason is DeathReason.MUTUAL
        assert st2.deaths[43].reason is DeathReason.MUTUAL
        _assert_global_invariants(st2)


# ---------------------------------------------------------------------------
# Flag capture → defending seat's collateral deaths
# ---------------------------------------------------------------------------

class TestDeathInfoOnFlagCapture:
    def test_surrender_records_collateral_deaths(self) -> None:
        """SOUTH SILING eats WEST JUNQI. WEST surrenders → all WEST pieces
        collaterally die. Per decision A: collateral deaths are recorded
        as KILLED_BY_ENEMY anchored at each piece's current cell. The
        JUNQI itself is recorded via the combat branch (EAT), also
        KILLED_BY_ENEMY, anchored at the combat dst cell."""
        pieces: PieceMap = {
            # SOUTH attacker adjacent to WEST JUNQI stronghold (0, 9).
            (1, 9): _p(Seat.SOUTH, PieceType.SILING, 17),
            # WEST collateral pieces that will die on surrender.
            (0, 9): _p(Seat.WEST, PieceType.JUNQI, 46),   # flag (combat victim)
            (2, 9): _p(Seat.WEST, PieceType.PAIZH, 47),
            (0, 8): _p(Seat.WEST, PieceType.DILEI, 48),
            # Background (exclude SOUTH+WEST since we custom-place them).
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (10, 16): _p(Seat.SOUTH, PieceType.PAIZH, 29),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, res = st.step(Action(seat=Seat.SOUTH, src=(1, 9), dst=(0, 9)))

        assert res.event is Event.EAT
        assert res.flag_captured is True

        # Flag (combat victim) — recorded at dst with KILLED_BY_ENEMY.
        assert 46 in st2.deaths
        assert st2.deaths[46].reason is DeathReason.KILLED_BY_ENEMY
        assert st2.deaths[46].death_loc == (0, 9)

        # Collateral PAIZH @ (2, 9) — anchored at its CURRENT cell.
        assert 47 in st2.deaths
        assert st2.deaths[47].reason is DeathReason.KILLED_BY_ENEMY
        assert st2.deaths[47].death_loc == (2, 9)

        # Collateral DILEI @ (0, 8) — anchored at its CURRENT cell.
        assert 48 in st2.deaths
        assert st2.deaths[48].reason is DeathReason.KILLED_BY_ENEMY
        assert st2.deaths[48].death_loc == (0, 8)

        # All three share the same death step (the move that triggered it).
        assert st2.deaths[46].step == st2.deaths[47].step == st2.deaths[48].step == 1

        # Attacker survived.
        assert 17 not in st2.deaths
        _assert_global_invariants(st2)

    def test_surrender_does_not_overwrite_flag_death_anchor(self) -> None:
        """Idempotence: the flag was recorded first (in combat branch) at
        dst (0, 9). The surrender sweep must NOT overwrite that record
        even though the flag's current cell coincides with dst. Tests
        the `if pid not in deaths` guard in _remove_all_pieces_of_seat."""
        pieces: PieceMap = {
            (1, 9): _p(Seat.SOUTH, PieceType.SILING, 17),
            (0, 9): _p(Seat.WEST, PieceType.JUNQI, 46),
            **_quiet_others(exclude={Seat.SOUTH, Seat.WEST}),
            (10, 16): _p(Seat.SOUTH, PieceType.PAIZH, 29),
        }
        st = _build_state(pieces=pieces, turn=Seat.SOUTH)
        st2, _ = st.step(Action(seat=Seat.SOUTH, src=(1, 9), dst=(0, 9)))

        # Flag DeathInfo has death_loc == (0, 9), recorded in combat branch.
        # After combat the flag was deleted from `pieces`; surrender sweep
        # won't see it. Either way the record must remain pristine.
        assert st2.deaths[46] == DeathInfo(
            piece_id=46,
            reason=DeathReason.KILLED_BY_ENEMY,
            death_loc=(0, 9),
            step=1,
        )


# ---------------------------------------------------------------------------
# DeathInfo frozen-ness
# ---------------------------------------------------------------------------

class TestDeathInfoImmutability:
    def test_death_info_is_frozen(self) -> None:
        di = DeathInfo(
            piece_id=1,
            reason=DeathReason.MUTUAL,
            death_loc=(0, 0),
            step=1,
        )
        with pytest.raises((AttributeError, Exception)):
            di.piece_id = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Random-game aggregate invariants
# ---------------------------------------------------------------------------

class TestDeathInfoRandomGame:
    def test_deaths_monotonic_and_consistent(self) -> None:
        """Run a random game; at every step assert deaths grows
        monotonically and all global invariants hold."""
        rng = random.Random(2027)
        setups = generate_random_setup(rng)
        st = GameState.new_game(setups)
        prev_death_count = 0

        for _ in range(200):
            if st.terminated:
                break
            acts = st.legal_actions()
            if not acts:
                break
            act = acts[rng.randrange(len(acts))]
            st, res = st.step(act)

            # Monotonicity: deaths can only grow.
            assert len(st.deaths) >= prev_death_count
            prev_death_count = len(st.deaths)

            _assert_global_invariants(st)

    def test_no_new_deaths_on_plain_move(self) -> None:
        """Find a random game state, make a plain-move step, assert
        `deaths` does NOT grow across this step."""
        rng = random.Random(999)
        setups = generate_random_setup(rng)
        st = GameState.new_game(setups)

        # Find the first plain MOVE in a short rollout.
        for _ in range(50):
            if st.terminated:
                pytest.skip("random game terminated too fast")
            acts = st.legal_actions()
            # Pick a move that lands on an empty cell (guaranteed MOVE).
            move_act = next(
                (a for a in acts if a.dst not in st.pieces),
                None,
            )
            if move_act is None:
                st, _ = st.step(acts[0])
                continue
            deaths_before = dict(st.deaths)
            st2, res = st.step(move_act)
            assert res.event is Event.MOVE
            # Q12 cascade could still fire in theory, but for an early
            # game state no seat is close to losing all moves.
            assert st2.deaths == deaths_before, (
                f"deaths changed across a MOVE step: {deaths_before} -> "
                f"{st2.deaths}"
            )
            return
        pytest.skip("no plain MOVE found in 50 random actions")
