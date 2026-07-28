"""tools/build_engineer_worst_case.py — directly construct a 2-dead endgame
state and verify GPU kernel matches CPU reference even at the theoretical max.

This is the stress test that random play cannot reach: 2 seats eliminated,
engineer on cleared rail, all 15 BFS cells reachable.
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import random
from junqi_rl.env import JunqiEnv
from junqi_core.rules import PieceType, Seat
from junqi_core.state import GameState


def find_engineer_of_seat(st: GameState, seat: Seat) -> int | None:
    for pid in range(120):
        if (st.alive[pid]
                and st.piece_seat_arr[pid] == seat.value
                and st.piece_type_arr[pid] == PieceType.GONGB.value):
            return pid
    return None


def build_two_dead_scenario() -> JunqiEnv:
    """Seed an env, then manually modify state to kill 2 seats and clear rails."""
    env = JunqiEnv()
    env.reset(seed=0xC0FFEE)
    st = env.state

    # Strategy: kill WEST and EAST (seats 1 and 3), leaving SOUTH (0) + NORTH (2) team.
    # We clear ALL WEST and EAST pieces, then find a SOUTH engineer on an interior rail.

    # Mark seats as dead
    st.info[Seat.WEST]  = type(st.info[Seat.WEST])(dead=True, flag_revealed=False)
    st.info[Seat.EAST]  = type(st.info[Seat.EAST])(dead=True, flag_revealed=False)
    st.seat_dead_arr[Seat.WEST.value] = True
    st.seat_dead_arr[Seat.EAST.value] = True

    # Remove all WEST and EAST pieces from the board
    for pid in range(120):
        if st.piece_seat_arr[pid] in (Seat.WEST.value, Seat.EAST.value):
            if st.alive[pid]:
                # Remove from board
                x, y = int(st.pos_x[pid]), int(st.pos_y[pid])
                if 0 <= x < 17 and 0 <= y < 17:
                    flat = y * 17 + x
                    if st.cell_piece_id[flat] == pid:
                        st.cell_piece_id[flat] = -1
                st.alive[pid] = False
                st.pos_x[pid] = -1
                st.pos_y[pid] = -1

    # Also clear all NORTH (teammate) pieces — we want the SOUTH engineer alone on its rail
    for pid in range(120):
        if st.piece_seat_arr[pid] == Seat.NORTH.value and st.alive[pid]:
            x, y = int(st.pos_x[pid]), int(st.pos_y[pid])
            if 0 <= x < 17 and 0 <= y < 17:
                flat = y * 17 + x
                if st.cell_piece_id[flat] == pid:
                    st.cell_piece_id[flat] = -1
            st.alive[pid] = False
            st.pos_x[pid] = -1
            st.pos_y[pid] = -1

    # For SOUTH, remove ALL pieces EXCEPT one engineer.
    south_engineer = find_engineer_of_seat(st, Seat.SOUTH)
    assert south_engineer is not None

    for pid in range(120):
        if (st.piece_seat_arr[pid] == Seat.SOUTH.value
                and st.alive[pid]
                and pid != south_engineer):
            x, y = int(st.pos_x[pid]), int(st.pos_y[pid])
            if 0 <= x < 17 and 0 <= y < 17:
                flat = y * 17 + x
                if st.cell_piece_id[flat] == pid:
                    st.cell_piece_id[flat] = -1
            st.alive[pid] = False
            st.pos_x[pid] = -1
            st.pos_y[pid] = -1

    # Move the engineer to a central rail cell with maximum BFS reach.
    # From earlier analysis, src (8,1) yields 15 BFS cells + 2 ortho + 2 diag = 19.
    # But SOUTH pieces start at y≥11, so we need to actually re-place.
    # Clear engineer's old position first.
    old_flat = int(st.pos_y[south_engineer]) * 17 + int(st.pos_x[south_engineer])
    if 0 <= old_flat < 289 and st.cell_piece_id[old_flat] == south_engineer:
        st.cell_piece_id[old_flat] = -1
    # Place at (8, 1) — North team's rail row, but a cleared board means it's reachable
    # by our engineer via rail (SOUTH engineer can roam freely on rails).
    # We're just using the state for GPU legal-action generation test, not playing
    # a legal sequence of moves.
    target_x, target_y = 8, 1
    st.pos_x[south_engineer] = target_x
    st.pos_y[south_engineer] = target_y
    target_flat = target_y * 17 + target_x
    st.cell_piece_id[target_flat] = south_engineer

    # Set turn to SOUTH
    st.turn = Seat.SOUTH
    return env


def main():
    env = build_two_dead_scenario()
    st = env.state
    print(f"Dead seats: {[s.name for s, info in st.info.items() if info.dead]}")
    print(f"Total alive pieces: {int(st.alive.sum())}")
    eng = None
    for pid in range(120):
        if (st.alive[pid]
                and st.piece_seat_arr[pid] == Seat.SOUTH.value
                and st.piece_type_arr[pid] == PieceType.GONGB.value):
            eng = pid; break
    print(f"Engineer pid={eng} at ({int(st.pos_x[eng])}, {int(st.pos_y[eng])})")

    # Count CPU legal moves for this engineer
    aids = st.legal_action_ids(Seat.SOUTH)
    src_flat = int(st.pos_y[eng]) * 17 + int(st.pos_x[eng])
    n_moves_cpu = int(((aids.astype(np.int64) // 289) == src_flat).sum())
    print(f"\nCPU legal moves for engineer at ({int(st.pos_x[eng])},{int(st.pos_y[eng])}): {n_moves_cpu}")
    print(f"Total CPU legal actions (all pieces): {len(aids)}")

    # Now compare with GPU
    try:
        import junqi_cuda as jc
        from junqi_rl.env_gpu import _pack_state_arrays
        jc.init_tables()
        sd = _pack_state_arrays([env])
        gs = jc.DeviceGameStateBatch(1)
        gs.copy_from_host(sd)
        acting = np.array([Seat.SOUTH.value], dtype=np.int8)
        ids, cnt = jc.legal_action_ids_batch(gs, acting)
        c = int(cnt[0])
        print(f"\nGPU returned: {c} actions")
        gpu_sorted = np.sort(ids[0, :c])
        cpu_sorted = np.sort(aids.astype(np.int32))
        print(f"CPU set size: {len(cpu_sorted)}, GPU set size: {len(gpu_sorted)}")
        if not np.array_equal(gpu_sorted, cpu_sorted):
            gpu_only = np.setdiff1d(gpu_sorted, cpu_sorted)
            cpu_only = np.setdiff1d(cpu_sorted, gpu_sorted)
            print(f"  GPU only: {gpu_only}")
            print(f"  CPU only: {cpu_only}")
        else:
            print("  GPU == CPU (bit-identical)")

        # Now the 32-slot mask test
        mask = jc.legal_action_mask_batch(gs, acting)
        mask_slot_count = int(mask[0, eng].sum())
        print(f"\n32-slot mask for engineer: {mask_slot_count} slots True")
        print(f"  mask[eng] = {mask[0, eng].astype(np.int8)}")
        print(f"  slots used: {sorted(int(i) for i in np.where(mask[0, eng])[0])}")
        if mask_slot_count > 16:
            print(f"  WARNING: engineer slot count {mask_slot_count} exceeds slot 8-23 budget (16)!")
        elif mask_slot_count == n_moves_cpu - 4 and n_moves_cpu >= 4:
            print(f"  OK: mask_slot_count ({mask_slot_count}) + 4 ortho ≈ CPU n_moves ({n_moves_cpu})")
    except ImportError:
        print("junqi_cuda not available")


if __name__ == "__main__":
    main()
