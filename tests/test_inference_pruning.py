"""Do not turn mine/flag exceptions into unconditional rank facts."""
from junqi_core.move_gen import PieceRef
from junqi_core.rules import Event, PieceType, Seat, ShowMode
from junqi_core.state import Action, GameState, SeatInfo


def game(pieces, turn=Seat.SOUTH):
    return GameState(pieces=pieces, turn=turn, move_counter=0,
                     moves_since_last_combat=0, info={s: SeatInfo() for s in Seat},
                     terminated=False, winner_team=None, draw=False, show_mode=ShowMode.DARK)


def test_hidden_mine_survival_does_not_imply_ordinary_rank():
    from tests.test_feature_information_boundaries import position
    st = position(PieceType.PAIZH, PieceType.DILEI)
    after, result = st.step(Action(seat=Seat.SOUTH, src=(2, 7), dst=(1, 7)))
    assert result.event is Event.KILLED
    pid = after.pieces[(1, 7)].piece_id
    assert after.combat_memory.rank_floor[Seat.SOUTH.value, pid] == 0


def test_rank_does_not_propagate_through_hidden_mine_chain():
    pieces = {
        (6, 3): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (6, 2): PieceRef(Seat.WEST, PieceType.LIANZH),
        (6, 1): PieceRef(Seat.NORTH, PieceType.DILEI),
        (7, 1): PieceRef(Seat.EAST, PieceType.GONGB),
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st = game(pieces)
    for seat, src, dst, expected in (
        (Seat.SOUTH, (6, 3), (6, 2), Event.KILLED),
        (Seat.WEST, (6, 2), (6, 1), Event.KILLED),
        (Seat.NORTH, (10, 5), (9, 5), Event.MOVE),
        (Seat.EAST, (7, 1), (6, 1), Event.EAT),
    ):
        st, result = st.step(Action(seat=seat, src=src, dst=dst))
        assert result.event is expected
    pid = st.pieces[(6, 1)].piece_id
    assert st.combat_memory.rank_floor[Seat.SOUTH.value, pid] == 0
    assert not st.combat_memory.not_gongb[Seat.SOUTH.value, pid]
    # The causal kill history itself remains useful and must not be deleted.
    assert st.combat_memory.chain_pid_lo[Seat.SOUTH.value, pid] != 0


def test_capturing_flag_does_not_exclude_engineer():
    pieces = {
        (1, 7): PieceRef(Seat.SOUTH, PieceType.GONGB),
        (0, 7): PieceRef(Seat.WEST, PieceType.JUNQI),
        (10, 11): PieceRef(Seat.SOUTH, PieceType.PAIZH),
        (5, 6): PieceRef(Seat.WEST, PieceType.PAIZH),
        (10, 5): PieceRef(Seat.NORTH, PieceType.PAIZH),
        (11, 8): PieceRef(Seat.EAST, PieceType.PAIZH),
    }
    st, result = game(pieces).step(Action(seat=Seat.SOUTH, src=(1, 7), dst=(0, 7)))
    assert result.event is Event.EAT and result.flag_captured
    pid = st.pieces[(0, 7)].piece_id
    assert not st.combat_memory.not_gongb[Seat.WEST.value, pid]
