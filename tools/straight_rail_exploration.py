"""tools/straight_rail_exploration.py — compute straight-rail rays using graph + axis constraint."""

from junqi_core.rail_topology import RAIL_ADJ, IS_RAILWAY, BOARD_SIZE, NC
from collections import deque

def xy(flat): return flat % BOARD_SIZE, flat // BOARD_SIZE
def f(x, y): return y * BOARD_SIZE + x

def straight_rail_bfs(src_flat, fixed_axis):
    """BFS from src along rail graph, only visiting cells that share `fixed_axis` with src.

    fixed_axis = 'x' means destinations must have dst.x == src.x (vertical rail column).
    fixed_axis = 'y' means destinations must have dst.y == src.y (horizontal rail row).

    Returns list of reachable cells (excluding src) sorted by distance.
    """
    sx, sy = xy(src_flat)
    visited = {src_flat}
    q = deque([src_flat])
    result = []
    while q:
        cur = q.popleft()
        for nb in RAIL_ADJ[cur]:
            if nb in visited: continue
            nx, ny = xy(nb)
            if fixed_axis == 'x' and nx != sx: continue
            if fixed_axis == 'y' and ny != sy: continue
            visited.add(nb)
            result.append(nb)
            q.append(nb)
    return result

# Test from (6, 5) on x=6
src = f(6, 5)
print(f"From (6,5) same-x rail reach:")
for n in sorted(straight_rail_bfs(src, 'x')):
    print(f"  {xy(n)}")

print(f"\nFrom (6,1) same-x rail reach:")
for n in sorted(straight_rail_bfs(f(6,1), 'x')):
    print(f"  {xy(n)}")

print(f"\nFrom (8,1) same-x rail reach:")
for n in sorted(straight_rail_bfs(f(8,1), 'x')):
    print(f"  {xy(n)}")
