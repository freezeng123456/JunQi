"""T7 / ADR-114 piece_id identity system unit tests (M1 only)."""

from __future__ import annotations

import random

import pytest

from junqi_core.move_gen import PieceRef
from junqi_core.rules import (
    ALL_SEATS,
    PIECE_COUNTS,
    PieceType,
    SLOTS_PER_SEAT,
    Seat,
)
from junqi_core.setup import (
    INVALID_PIECE_ID,
    assign_piece_ids,
    generate_random_setup,
)
from junqi_core.state import DeathInfo, GameState, PieceState


def _make_state(seed: int = 42) -> GameState:
    rng = random.Random(seed)
    setups = generate_random_setup(rng)
    return GameState.new_game(setups)


class TestAssignPieceIds:
    def test_encoding_rule(self) -> None:
        rng = random.Random(0)
        setups = generate_random_setup(rng)
        ids = assign_piece_ids(setups)
        for (seat, slot), pid in ids.items():
            assert pid == seat.value * SLOTS_PER_SEAT + slot
            assert 0 <= pid < 4 * SLOTS_PER_SEAT

    def test_uniqueness(self) -> None:
        rng = random.Random(1)
        ids = assign_piece_ids(generate_random_setup(rng))
        all_ids = list(ids.values())
        assert len(set(all_ids)) == len(all_ids)

    def test_count_is_100(self) -> None:
        rng = random.Random(2)
        ids = assign_piece_ids(generate_random_setup(rng))
        expected_total = sum(PIECE_COUNTS.values()) * len(ALL_SEATS)
        assert expected_total == 100
        assert len(ids) == expected_total

    def test_camp_slots_absent(self) -> None:
        rng = random.Random(3)
        setups = generate_random_setup(rng)
        ids = assign_piece_ids(setups)
        for seat in ALL_SEATS:
            for slot, pt in enumerate(setups[seat.value]):
                if pt is PieceType.NONE:
                    assert (seat, slot) not in ids
                else:
                    assert (seat, slot) in ids

    def test_invalid_sentinel(self) -> None:
        assert INVALID_PIECE_ID == -1
        ids = assign_piece_ids(generate_random_setup(random.Random(4)))
        assert INVALID_PIECE_ID not in ids.values()

    def test_wrong_lineup_length_raises(self) -> None:
        bad = (
            [PieceType.NONE] * 30,
            [PieceType.NONE] * 30,
            [PieceType.NONE] * 30,
            [PieceType.NONE] * 29,
        )
        with pytest.raises(ValueError, match="lineup for EAST"):
            assign_piece_ids(bad)


class TestNewGameTrackingFields:
    def test_every_piece_has_a_piece_id(self) -> None:
        st = _make_state()
        for ref in st.pieces.values():
            assert isinstance(ref, PieceRef)
            assert ref.piece_id != INVALID_PIECE_ID
            assert 0 <= ref.piece_id < 4 * SLOTS_PER_SEAT

    def test_zero_board_matches_initial_pieces(self) -> None:
        st = _make_state(seed=7)
        assert set(st.pieces.keys()) == set(st.zero_board.keys())
        for pos in st.pieces:
            a, b = st.pieces[pos], st.zero_board[pos]
            assert a.seat is b.seat
            assert a.piece_type is b.piece_type
            assert a.piece_id == b.piece_id

    def test_piece_state_one_entry_per_live_piece(self) -> None:
        st = _make_state(seed=8)
        expected_ids = {ref.piece_id for ref in st.pieces.values()}
        assert set(st.piece_state.keys()) == expected_ids
        assert len(st.piece_state) == 4 * 25

    def test_initial_counters_are_zero(self) -> None:
        st = _make_state(seed=9)
        for ps in st.piece_state.values():
            assert ps == PieceState(
                move_count=0,
                active_eat_count=0,
                passive_survive_count=0,
            )

    def test_deaths_empty(self) -> None:
        st = _make_state(seed=10)
        assert st.deaths == {}

    def test_piece_id_uniqueness_on_board(self) -> None:
        st = _make_state(seed=11)
        ids = [ref.piece_id for ref in st.pieces.values()]
        assert len(set(ids)) == len(ids)

    def test_piece_id_encoding_recovers_seat(self) -> None:
        st = _make_state(seed=12)
        for ref in st.zero_board.values():
            assert ref.piece_id // SLOTS_PER_SEAT == ref.seat.value


class TestZeroBoardImmutability:
    def test_zero_board_unchanged_after_step(self) -> None:
        st = _make_state(seed=13)
        snapshot = dict(st.zero_board)
        st2, _ = st.step(st.legal_actions()[0])
        assert set(st2.zero_board.keys()) == set(snapshot.keys())
        for pos, ref in snapshot.items():
            assert st2.zero_board[pos] is ref
        assert dict(st.zero_board) == snapshot

    def test_clone_shares_zero_board(self) -> None:
        st = _make_state(seed=14)
        twin = st.clone()
        assert twin.zero_board is st.zero_board

    def test_step_preserves_zero_board_identity(self) -> None:
        st = _make_state(seed=15)
        st2, _ = st.step(st.legal_actions()[0])
        assert st2.zero_board is st.zero_board


class TestCloneIsolation:
    def test_clone_piece_state_independent(self) -> None:
        st = _make_state(seed=16)
        twin = st.clone()
        victim_id = next(iter(twin.piece_state.keys()))
        del twin.piece_state[victim_id]
        assert victim_id in st.piece_state
        assert victim_id not in twin.piece_state

    def test_clone_deaths_independent(self) -> None:
        from junqi_core.rules import DeathReason
        st = _make_state(seed=17)
        twin = st.clone()
        twin.deaths[9999] = DeathInfo(
            piece_id=9999,
            reason=DeathReason.KILLED_BY_ENEMY,
            death_loc=(0, 0),
            step=1,
        )
        assert 9999 not in st.deaths
        assert 9999 in twin.deaths

    def test_clone_pieces_independent(self) -> None:
        st = _make_state(seed=18)
        twin = st.clone()
        victim_pos = next(iter(twin.pieces.keys()))
        del twin.pieces[victim_pos]
        assert victim_pos in st.pieces


class TestStepTransparency:
    def test_step_preserves_live_piece_state_entries(self) -> None:
        st = _make_state(seed=19)
        before_ids = set(st.piece_state.keys())
        st2, _ = st.step(st.legal_actions()[0])
        for ref in st2.pieces.values():
            assert ref.piece_id in st2.piece_state
        assert set(st2.piece_state.keys()) <= before_ids

    def test_m3_deaths_grow_on_combat_empty_on_move(self) -> None:
        """M3 contract (flips the previous M1 pin test):

        After any step, the `deaths` dict is invariant for Event.MOVE, and
        for any combat event exactly the dying piece(s) receive a fresh
        DeathInfo entry with `step == new_state.move_counter`.
        """
        from junqi_core.rules import DeathReason, Event

        st = _make_state(seed=20)
        st2, res = st.step(st.legal_actions()[0])

        if res.event is Event.MOVE:
            # Plain move → no deaths produced by this step.
            assert st2.deaths == st.deaths == {}
        elif res.event is Event.EAT:
            # Exactly the defender gets a DeathInfo.
            assert len(st2.deaths) == len(st.deaths) + 1
        elif res.event is Event.KILLED:
            assert len(st2.deaths) == len(st.deaths) + 1
        elif res.event is Event.BOMB:
            # Both combatants get DeathInfo.
            assert len(st2.deaths) == len(st.deaths) + 2
        else:  # pragma: no cover
            raise AssertionError(f"unexpected event {res.event}")

        # Every newly recorded death is tagged with the CURRENT step #.
        new_ids = set(st2.deaths.keys()) - set(st.deaths.keys())
        for pid in new_ids:
            di = st2.deaths[pid]
            assert di.step == st2.move_counter
            assert di.reason in DeathReason

    def test_m2_move_bumps_move_count_only(self) -> None:
        """M2 contract (flips the previous M1 pin test):

        After a plain MOVE, the acting piece's move_count goes 0 → 1;
        every *other* live piece's counters stay untouched at zero;
        active_eat / passive_survive remain 0 for all pieces (pure move).
        """
        st = _make_state(seed=21)
        # Seed 21's first legal action happens to be a plain move (asserted
        # below). If a future rng/ordering change makes it a combat, this
        # test deliberately fails loud — at which point pick another seed.
        act = st.legal_actions()[0]
        src_ref = st.pieces[act.src]
        acting_pid = src_ref.piece_id

        st2, res = st.step(act)
        assert res.event.name == "MOVE", (
            f"seed 21 first action is expected to be a plain MOVE, "
            f"got {res.event!r}"
        )

        # Acting piece's move_count bumped by 1, other counters zero.
        assert st2.piece_state[acting_pid].move_count == 1
        assert st2.piece_state[acting_pid].active_eat_count == 0
        assert st2.piece_state[acting_pid].passive_survive_count == 0

        # Every *other* piece keeps all-zero counters.
        for pid, ps in st2.piece_state.items():
            if pid == acting_pid:
                continue
            assert ps.move_count == 0
            assert ps.active_eat_count == 0
            assert ps.passive_survive_count == 0

        # Source state is untouched (immutability).
        assert st.piece_state[acting_pid].move_count == 0
