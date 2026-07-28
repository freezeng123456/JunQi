"""tools/measure_engineer_adversarial.py — worst-case engineer mobility.

Construct adversarial board states directly (not through random play) to
measure the true upper bound on engineer action count.
"""
from __future__ import annotations

import numpy as np
from junqi_rl.env import JunqiEnv
from junqi_core.rules import PieceType, Seat
from junqi_core.move_gen import _engineer_dests_soa
from junqi_core.board import NUM_CELLS, xy_to_flat
from junqi_core._movegen_tables import ENGINEER_RAIL_NEIGHBORS


def measure_pure_bfs():
    """Pure BFS reachability on rail subgraph — no other pieces on board.

    This gives the absolute theoretical maximum engineer BFS destinations
    (ignoring the 1-step orthogonal / diagonal-via-camp moves).
    """
    rail_flags = np.array([len(ENGINEER_RAIL_NEIGHBORS[f]) > 0 for f in range(NUM_CELLS)])
    print(f"Total rail cells: {rail_flags.sum()}")

    # Place engineer on each rail cell in turn, everything else empty.
    # Count BFS-reachable destinations.
    empty = np.ones(NUM_CELLS, dtype=bool)
    # Non-rail cells are treated as "not a destination" by the BFS
    # (ENGINEER_RAIL_NEIGHBORS gives 0 for non-rail).
    enemy_attackable = np.zeros(NUM_CELLS, dtype=bool)

    counts = []
    for src in range(NUM_CELLS):
        if not rail_flags[src]:
            continue
        empty[src] = False  # engineer occupies src
        dests = _engineer_dests_soa(src, empty, enemy_attackable)
        counts.append((src, len(dests)))
        empty[src] = True

    max_cnt = max(c for _, c in counts)
    print(f"Max pure-BFS destinations (empty rail, single engineer): {max_cnt}")
    src_best = [s for s, c in counts if c == max_cnt][0]
    print(f"  src flat={src_best}  xy=({src_best%17}, {src_best//17})")

    # But this is JUST BFS. The kernel ALSO emits:
    #   - 4 orthogonal 1-step (3a) — may overlap with BFS k=0 (suppressed)
    #   - 4 diagonal via camp (3b) — does NOT overlap with BFS (off-rail)
    # So max TOTAL engineer moves = BFS_max - rail_ortho_dup + camp_diag_max

    # For src (8,1), diagonal via camp: check if any of (7,0)(9,0)(7,2)(9,2) is camp
    # For engineer on mid-board, no camps are adjacent; camps are near the edges.
    # Let me compute for every src.
    from junqi_core.board import xy_to_flat, is_on_board
    from junqi_core.rules import Seat

    # Build camp set the same way tables.cu does
    def seat_pos(seat, i):
        col, row = i % 5, i // 5
        if seat == 0: return (10 - col, 11 + row)
        if seat == 1: return (5 - row, 10 - col)
        if seat == 2: return (6 + col, 5 - row)
        if seat == 3: return (11 + row, 6 + col)

    CAMP_IDX = {6, 8, 12, 16, 18}
    camps = set()
    on_board = set()
    for seat in range(4):
        for i in range(30):
            x, y = seat_pos(seat, i)
            on_board.add((x, y))
            if i in CAMP_IDX:
                camps.add((x, y))

    print(f"Camp cells: {len(camps)}")

    # Now worst case total for engineer:
    diag = [(1,1),(1,-1),(-1,1),(-1,-1)]
    ortho = [(1,0),(-1,0),(0,1),(0,-1)]
    max_total = 0
    max_where = None
    for src_flat, bfs_cnt in counts:
        sx, sy = src_flat % 17, src_flat // 17
        # Ortho moves — 4 at most, but overlap with BFS k=0 (same cell).
        # Ortho rail neighbors: the cells BFS would reach at k=0.
        # Non-rail ortho neighbors are EXTRA (not in BFS).
        # BFS covers all orthogonal rail cells in range; extra ortho = non-rail neighbors.
        ortho_rail = 0
        ortho_nonrail = 0
        for dx, dy in ortho:
            nx, ny = sx+dx, sy+dy
            if (nx, ny) in on_board:
                nf = ny*17 + nx
                if rail_flags[nf]:
                    ortho_rail += 1
                else:
                    ortho_nonrail += 1
        # Diagonal via camp
        diag_legal = 0
        for dx, dy in diag:
            nx, ny = sx+dx, sy+dy
            if (nx, ny) in on_board:
                if (sx, sy) in camps or (nx, ny) in camps:
                    diag_legal += 1

        # Total = BFS destinations (includes rail orthos, k>0 rail cells) + non-rail orthos + camp diagonals
        # Actually BFS includes ortho rail cells at k=0, and we kept them because
        # the kernel emits them via section 3a (ortho) and suppresses BFS k=0.
        # Net effect: BFS-emitted cells (k>0) + 4 ortho (some overlap with BFS k=0) + camp diagonals.
        # Simplify: total distinct destinations = BFS_full (includes all rail reachable)
        #                                       + non-rail ortho neighbors
        #                                       + camp-diagonal neighbors (non-rail)
        total = bfs_cnt + ortho_nonrail + diag_legal
        if total > max_total:
            max_total = total
            max_where = (src_flat, sx, sy, bfs_cnt, ortho_nonrail, diag_legal)

    print(f"\nWorst-case total engineer moves (theoretical, all rail clear):")
    print(f"  max={max_total} at src_flat={max_where[0]} ({max_where[1]},{max_where[2]})")
    print(f"    BFS dests: {max_where[3]}")
    print(f"    non-rail ortho: {max_where[4]}")
    print(f"    camp diag: {max_where[5]}")
    return max_total


def measure_via_state_construction():
    """Reach a 2-seat-dead state by manually marking seats as dead."""
    import random
    # Play a game long enough to have pieces in varied positions
    env = JunqiEnv()
    env.reset(seed=42)
    rng = random.Random(42)
    for _ in range(400):
        if env.state.terminated: break
        aids = env.legal_action_ids()
        if aids.size == 0: break
        env._step_game_only(int(rng.choice(aids)))

    # Now manually inspect: find all engineers and measure their moves
    st = env.state
    print(f"\nGame state after 400 moves:")
    print(f"  Turn: {st.turn}")
    print(f"  Dead seats: {[s.name for s, info in st.info.items() if info.dead]}")

    # Engineer mobility for each seat
    for seat in [Seat.SOUTH, Seat.WEST, Seat.NORTH, Seat.EAST]:
        if st.info[seat].dead: continue
        aids = st.legal_action_ids(seat)
        for pid in range(120):
            if not st.alive[pid]: continue
            if st.piece_seat_arr[pid] != seat.value: continue
            if st.piece_type_arr[pid] != PieceType.GONGB.value: continue
            src_flat = int(st.pos_y[pid]) * 17 + int(st.pos_x[pid])
            n_moves = int(((aids.astype(np.int64) // 289) == src_flat).sum())
            if n_moves > 5:
                print(f"  Engineer pid={pid} seat={seat.name} src=({src_flat%17},{src_flat//17}): {n_moves} moves")


def main():
    print("=" * 60)
    print("1. Pure BFS theoretical maximum (all rails empty)")
    print("=" * 60)
    max_total = measure_pure_bfs()

    print()
    print("=" * 60)
    print("2. Realistic state after extended random play")
    print("=" * 60)
    measure_via_state_construction()


if __name__ == "__main__":
    main()
