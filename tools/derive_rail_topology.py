"""Derive the authoritative rail cell set + adjacency from legacy_engine.

Legacy sources:
  * legacy_engine/src/junqi.c::SetChess       — piece position for each seat
  * legacy_engine/src/junqi.c::SetBoardRailway — per-seat i<25 && (row 0/4 || col 0/4)
  * legacy_engine/src/junqi.c::InitNineGrid    — 9 central cells at (10-(i%3)*2, 6+(i/3)*2)
                                                 marked isRailway = 1 AND isNineGrid = 1
  * legacy_engine/src/junqi.c::InitBoardGraph — adjacency = ortho rails; for NineGrid also
                                                 +2 ortho jumps
  * legacy_engine/src/junqi.c::AddSpcNode      — 4 special diagonal edges at corners

This script is the single source of truth.  No changes to legacy files.
"""
from collections import deque

BS = 17

# --- Step 1: rail flags per cell ---
# 30 cells per seat, indexed i=0..29 (row = i/5, col = i%5):
#   HOME  (seat 0, SOUTH): x = 10 - i%5, y = 11 + i/5
#   RIGHT (seat 1, WEST):  x = 5  - i/5, y = 10 - i%5
#   OPPS  (seat 2, NORTH): x = 6  + i%5, y = 5  - i/5
#   LEFT  (seat 3, EAST):  x = 11 + i/5, y = 6  + i%5
# SetBoardRailway: i<25 && (i/5==0 || i/5==4 || i%5==0 || i%5==4)

def seat_pos(dir_, i):
    if dir_ == 0:   # HOME / SOUTH
        return (10 - i % 5, 11 + i // 5)
    if dir_ == 1:   # RIGHT / WEST
        return (5 - i // 5, 10 - i % 5)
    if dir_ == 2:   # OPPS / NORTH
        return (6 + i % 5, 5 - i // 5)
    if dir_ == 3:   # LEFT / EAST
        return (11 + i // 5, 6 + i % 5)
    assert False

is_railway = [[0]*BS for _ in range(BS)]
is_ninegrid = [[0]*BS for _ in range(BS)]
is_on_board = [[0]*BS for _ in range(BS)]
is_camp = [[0]*BS for _ in range(BS)]
is_stronghold = [[0]*BS for _ in range(BS)]
owner = [[None]*BS for _ in range(BS)]

# Camp indices (from legacy SetBoardCamp): i in {6,8,12,16,18}
CAMP_IDX = {6, 8, 12, 16, 18}
# Stronghold indices (from legacy): i in {26, 28}
STRONGHOLD_IDX = {26, 28}

for dir_ in range(4):
    for i in range(30):
        x, y = seat_pos(dir_, i)
        is_on_board[x][y] = 1
        owner[x][y] = dir_
        if i in CAMP_IDX:
            is_camp[x][y] = 1
        if i in STRONGHOLD_IDX:
            is_stronghold[x][y] = 1
        # SetBoardRailway
        if i < 25 and (i // 5 == 0 or i // 5 == 4 or i % 5 == 0 or i % 5 == 4):
            is_railway[x][y] = 1

# NineGrid: 9 cells at (10-(i%3)*2, 6+(i/3)*2) for i=0..8
for i in range(9):
    x = 10 - (i % 3) * 2
    y = 6 + (i // 3) * 2
    is_on_board[x][y] = 1
    is_railway[x][y] = 1
    is_ninegrid[x][y] = 1

# --- Step 2: print the rail map ---
print("=" * 60)
print("Rail map from legacy_engine (R = rail, N = 9-grid rail):")
print("=" * 60)
for y in range(BS):
    row = []
    for x in range(BS):
        if is_ninegrid[x][y]:
            row.append('N')
        elif is_railway[x][y]:
            row.append('R')
        elif is_on_board[x][y]:
            row.append('.')
        else:
            row.append(' ')
    print(f"  y={y:2d}: " + ' '.join(row))

rail_count = sum(is_railway[x][y] for x in range(BS) for y in range(BS))
ninegrid_count = sum(is_ninegrid[x][y] for x in range(BS) for y in range(BS))
print(f"\nTotal rail cells: {rail_count}  (of which NineGrid: {ninegrid_count})")

# --- Step 3: build the adjacency graph exactly per legacy InitBoardGraph ---
# For each rail vertex v:
#   add ortho neighbors that are also rails
#   if v is NineGrid: also add 2-step ortho neighbors that are (rail AND NineGrid)
# Then AddSpcNode adds 4 corner diagonal edges.

rail_adj = {(x, y): set()
            for y in range(BS) for x in range(BS) if is_railway[x][y]}

# Helper to add a rail edge (symmetric)
def add_edge(a, b):
    if a in rail_adj and b in rail_adj:
        rail_adj[a].add(b)
        rail_adj[b].add(a)

# Ortho
for (x, y) in list(rail_adj):
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < BS and 0 <= ny < BS and is_railway[nx][ny]:
            add_edge((x,y), (nx,ny))

# NineGrid 2-step ortho
# Important: legacy AddAdjNode(..., isNineGrid=1) checks the target cell's isNineGrid.
# So the 2-step jumps only connect NineGrid cell to NineGrid cell (on the same ortho ray).
for (x, y) in list(rail_adj):
    if not is_ninegrid[x][y]:
        continue
    for dx, dy in [(2,0),(-2,0),(0,2),(0,-2)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < BS and 0 <= ny < BS and is_ninegrid[nx][ny]:
            add_edge((x,y), (nx,ny))

# AddSpcNode: 4 special corner edges
# Decoded from legacy C (see analysis above)
SPC_EDGES = [
    ((10, 11), (11, 10)),   # i=10, j=11 : (i,j) ↔ (j,i)
    ((6, 11),  (5, 10)),    # i=6,  j=11 : (i,j) ↔ (i-1, j-1)
    ((6, 5),   (5, 6)),     # i=6,  j=5  : (i,j) ↔ (j,i)
    ((11, 6),  (10, 5)),    # i=11, j=6  : (i,j) ↔ (i-1, j-1)
]
for a, b in SPC_EDGES:
    add_edge(a, b)

# --- Step 4: examine the graph ---
# Degree distribution
from collections import Counter
deg_dist = Counter(len(v) for v in rail_adj.values())
print(f"\nDegree distribution: {dict(sorted(deg_dist.items()))}")

# High-degree cells (junctions)
print("Junctions (degree ≥ 3):")
for cell, nbrs in sorted(rail_adj.items()):
    if len(nbrs) >= 3:
        print(f"  {cell}: degree {len(nbrs)}, neighbors={sorted(nbrs)}")

# Connected components
seen = set()
components = []
for c in rail_adj:
    if c in seen: continue
    comp = set()
    q = deque([c])
    while q:
        cur = q.popleft()
        if cur in comp: continue
        comp.add(cur)
        for nb in rail_adj[cur]:
            if nb not in comp:
                q.append(nb)
    components.append(comp)
    seen |= comp

print(f"\nConnected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"  component {i}: {len(comp)} cells")

# --- Step 5: BFS reachability from a lone engineer on empty board ---
# For every rail cell, compute BFS size (subtract src itself for 'destinations').
print("\nEngineer BFS reach (lone engineer, empty board):")
max_reach = 0
max_src = None
# Histogram of reach counts
reach_dist = Counter()
for src in rail_adj:
    visited = {src}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nb in rail_adj[cur]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    r = len(visited) - 1
    reach_dist[r] += 1
    if r > max_reach:
        max_reach = r
        max_src = src

print(f"  Max BFS reach: {max_reach} cells from src={max_src}")
print(f"  Reach distribution: {dict(sorted(reach_dist.items()))}")

# Export adjacency
print(f"\nTotal rail edges (undirected): {sum(len(v) for v in rail_adj.values()) // 2}")
