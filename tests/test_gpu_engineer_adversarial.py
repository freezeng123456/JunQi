"""tests/test_gpu_engineer_adversarial.py — engineer endgame stress tests.

Tests adversarial states that random play rarely reaches:
  * Two seats dead, engineer alone on cleared rail → BFS reaches the full
    rail component (up to 72 of the 73 cells)
  * Verifies GPU emits no duplicate actions (multiset comparison, not set)
  * Verifies 80-slot mask count matches dense CPU count

This test caught a critical bug: in cyclic rail components, the far end of
the BFS chain wraps back to the OTHER ortho rail neighbor of src.  The kernel
was skipping only k=0 of each BFS chain, but in a 16-cycle, k=14 of one chain
is also an ortho neighbor of src.  Fixed by checking ortho-membership, not
k-index.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import junqi_cuda as _cuda
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda not available",
)

from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays
from junqi_core.rules import PieceType, Seat


@pytest.fixture(scope="module", autouse=True)
def _init():
    _cuda.init_tables()


def _construct_lone_engineer(src_x: int, src_y: int) -> JunqiEnv:
    """Build a state: 2 seats dead, SOUTH has only 1 engineer at (src_x, src_y)."""
    env = JunqiEnv()
    env.reset(seed=0xC0FFEE)
    st = env.state

    # Kill WEST + EAST
    st.info[Seat.WEST] = type(st.info[Seat.WEST])(dead=True, flag_revealed=False)
    st.info[Seat.EAST] = type(st.info[Seat.EAST])(dead=True, flag_revealed=False)
    st.seat_dead_arr[Seat.WEST.value] = True
    st.seat_dead_arr[Seat.EAST.value] = True

    # Clear all non-SOUTH-engineer pieces from board
    from junqi_core.rules import PieceType as PT
    south_engineer = None
    for pid in range(120):
        if st.piece_seat_arr[pid] == Seat.SOUTH.value and \
           st.piece_type_arr[pid] == PT.GONGB.value and st.alive[pid]:
            south_engineer = pid
            break
    assert south_engineer is not None

    for pid in range(120):
        if pid == south_engineer:
            continue
        if st.alive[pid]:
            x, y = int(st.pos_x[pid]), int(st.pos_y[pid])
            if 0 <= x < 17 and 0 <= y < 17:
                flat = y * 17 + x
                if st.cell_piece_id[flat] == pid:
                    st.cell_piece_id[flat] = -1
            st.alive[pid] = False
            st.pos_x[pid] = -1
            st.pos_y[pid] = -1

    # Place engineer at desired location
    old_flat = int(st.pos_y[south_engineer]) * 17 + int(st.pos_x[south_engineer])
    if 0 <= old_flat < 289 and st.cell_piece_id[old_flat] == south_engineer:
        st.cell_piece_id[old_flat] = -1
    st.pos_x[south_engineer] = src_x
    st.pos_y[south_engineer] = src_y
    st.cell_piece_id[src_y * 17 + src_x] = south_engineer

    st.turn = Seat.SOUTH
    return env


@pytest.mark.parametrize(
    "src_x,src_y",
    [
        (8,  1),   # top rail mid-row — max BFS reach, 19 total
        (6,  1),   # corner of rail component
        (8,  5),   # mid-row other side
        (8, 15),   # bottom rail mid-row (symmetric)
        (1,  8),   # vertical rail mid
        (15, 8),   # vertical rail mid other side
    ],
)
def test_engineer_endgame_multiset_parity(src_x: int, src_y: int) -> None:
    """At (src_x, src_y) with empty rail: GPU must match CPU as a MULTISET.

    This test used to compare sets (via np.sort + array_equal), which masked
    a real bug: the GPU emitted (7,1) twice for engineer at (8,1) because
    the BFS cycle wrapped back to the other ortho neighbor at the END of the
    first chain direction.  Multiset (count-aware) comparison catches it.
    """
    env = _construct_lone_engineer(src_x, src_y)
    st = env.state

    # CPU reference
    cpu_aids = st.legal_action_ids(Seat.SOUTH)

    # GPU via dense kernel
    sd = _pack_state_arrays([env])
    gs = _cuda.DeviceGameStateBatch(1)
    gs.copy_from_host(sd)
    acting = np.array([Seat.SOUTH.value], dtype=np.int8)
    ids, cnt = _cuda.legal_action_ids_batch(gs, acting)
    c = int(cnt[0])

    # CRITICAL: count parity (multiset, not set)
    assert c == len(cpu_aids), (
        f"({src_x},{src_y}): GPU count {c} != CPU count {len(cpu_aids)}  "
        f"(the set-equality test would have missed duplicate emissions)"
    )

    # No duplicates in GPU output
    gpu_list = ids[0, :c].tolist()
    assert len(gpu_list) == len(set(gpu_list)), (
        f"({src_x},{src_y}): GPU emitted duplicates: "
        f"{[a for a in set(gpu_list) if gpu_list.count(a) > 1]}"
    )

    # Sets identical
    np.testing.assert_array_equal(
        np.sort(ids[0, :c]),
        np.sort(cpu_aids.astype(np.int32)),
    )


@pytest.mark.parametrize("src_x,src_y", [(8, 1), (6, 1), (8, 5), (8, 15), (1, 8), (15, 8)])
def test_engineer_endgame_mask_matches_cpu(src_x: int, src_y: int) -> None:
    """80-slot mask True-count == CPU action count for the lone engineer."""
    env = _construct_lone_engineer(src_x, src_y)
    st = env.state
    cpu_aids = st.legal_action_ids(Seat.SOUTH)

    sd = _pack_state_arrays([env])
    gs = _cuda.DeviceGameStateBatch(1)
    gs.copy_from_host(sd)
    acting = np.array([Seat.SOUTH.value], dtype=np.int8)

    mask = _cuda.legal_action_mask_batch(gs, acting)
    total_true = int(mask[0].sum())
    assert total_true == len(cpu_aids), (
        f"mask True count {total_true} != CPU count {len(cpu_aids)}"
    )

    # Engineer uses slots 8..79 for BFS destinations (up to 72 cells);
    # non-engineer curve slots 56..67 are unused by the engineer path and
    # reserved slots 68..79 are always zero for non-engineers.  For this
    # lone-engineer scenario we only require that the total count matches.


def test_engineer_bfs_slots_within_mid_range_budget() -> None:
    """Engineer BFS uses at most 72 slots (slots 8..79, i.e. 73-1 rail cells)."""
    # Test across many src positions on rail.
    from junqi_core._movegen_tables import ENGINEER_RAIL_NEIGHBORS
    from junqi_core.board import NUM_CELLS

    slots_per_piece = _cuda.SLOTS_PER_PIECE
    for src_flat in range(NUM_CELLS):
        if len(ENGINEER_RAIL_NEIGHBORS[src_flat]) == 0:
            continue
        src_x, src_y = src_flat % 17, src_flat // 17
        env = _construct_lone_engineer(src_x, src_y)

        sd = _pack_state_arrays([env])
        gs = _cuda.DeviceGameStateBatch(1)
        gs.copy_from_host(sd)
        acting = np.array([Seat.SOUTH.value], dtype=np.int8)
        mask = _cuda.legal_action_mask_batch(gs, acting)

        # Find the engineer's row in the mask
        for pid in range(120):
            row = mask[0, pid]
            if row.any():
                bfs_slots = int(row[8:slots_per_piece].sum())
                assert bfs_slots <= 72, (
                    f"Engineer at ({src_x},{src_y}): "
                    f"BFS used {bfs_slots} slots (limit 72)"
                )


def test_all_kernel_variants_multiset_parity_endgame() -> None:
    """At the worst-case state, dense / CSR / mask all agree (no duplicates)."""
    env = _construct_lone_engineer(8, 1)
    sd = _pack_state_arrays([env])
    gs = _cuda.DeviceGameStateBatch(1)
    gs.copy_from_host(sd)
    acting = np.array([Seat.SOUTH.value], dtype=np.int8)

    ids_d, cnt_d = _cuda.legal_action_ids_batch(gs, acting)
    offs_c, vals_c = _cuda.legal_action_ids_batch_csr(gs, acting)
    mask = _cuda.legal_action_mask_batch(gs, acting)

    c = int(cnt_d[0])
    # All three formats must report the same count
    assert c == int(offs_c[1]), "CSR offset[1] must equal dense count"
    assert c == int(mask[0].sum()), "mask count must equal dense count"

    # No duplicates in any format
    dense_list = ids_d[0, :c].tolist()
    assert len(dense_list) == len(set(dense_list)), "dense has duplicates"
    csr_list = vals_c[:c].tolist()
    assert len(csr_list) == len(set(csr_list)), "CSR has duplicates"
