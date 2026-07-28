"""Analyze engineer BFS: how many rail cells, what's the topology, etc."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from junqi_core._movegen_tables import (
    ENGINEER_RAIL_NEIGHBORS, ENGINEER_RAIL_NEIGHBORS_PAD,
    IS_RAIL_FLAT, NUM_CELLS
)

# Count rail cells
rail_cells = np.nonzero(IS_RAIL_FLAT)[0]
print(f"Number of rail cells: {len(rail_cells)}")
print(f"Rail cell indices: {rail_cells}")
print(f"\nENGINEER_RAIL_NEIGHBORS_PAD shape: {ENGINEER_RAIL_NEIGHBORS_PAD.shape}")
print(f"dtype: {ENGINEER_RAIL_NEIGHBORS_PAD.dtype}")

# Check max neighbors per rail cell
max_nbr = max(len(ENGINEER_RAIL_NEIGHBORS[f]) for f in rail_cells)
print(f"Max rail neighbors per cell: {max_nbr}")

# Print the rail cell adjacency structure
print("\nRail adjacency (first few):")
for i, f in enumerate(rail_cells[:10]):
    nbrs = ENGINEER_RAIL_NEIGHBORS[f]
    print(f"  cell {f} ({f%17},{f//17}): neighbors = {nbrs}")

# Build connectivity matrix: adj[i,j] = True iff cell_i rail-adjacent to cell_j
# where i,j are indices into rail_cells array
R = len(rail_cells)
rail_to_idx = np.full(NUM_CELLS, -1, dtype=np.int32)
rail_to_idx[rail_cells] = np.arange(R, dtype=np.int32)

adj = np.zeros((R, R), dtype=bool)
for i, f in enumerate(rail_cells):
    for nb in ENGINEER_RAIL_NEIGHBORS[f]:
        j = int(rail_to_idx[nb])
        if j >= 0:
            adj[i, j] = True

# Sanity: symmetric?
print(f"\nadjacency matrix symmetric: {np.all(adj == adj.T)}")
print(f"R (rail cells): {R}")
print(f"adjacency density: {adj.sum()}/{R*R} = {adj.mean():.3f}")

# Try BFS: for each starting rail cell, how many BFS hops to reach all?
# Simulate: frontier = {src}; while frontier: expand
import time
t0 = time.perf_counter()
NUM_BFS = 1000

for _ in range(NUM_BFS):
    for src_i in range(min(R, 5)):
        visited = np.zeros(R, dtype=bool)
        visited[src_i] = True
        frontier = visited.copy()
        while True:
            new_reach = adj[frontier].any(axis=0) & ~visited
            if not new_reach.any():
                break
            visited |= new_reach
            frontier = new_reach
t1 = time.perf_counter()
print(f"\n{NUM_BFS} x 5 numpy BFS steps: {(t1-t0)*1000:.2f}ms = {(t1-t0)/NUM_BFS/5*1000:.3f}ms each")

# Now measure the current Python BFS
from junqi_core.move_gen import _engineer_dests_soa
import numpy as np

# Create empty and enemy_attackable (all empty for benchmark)
empty_all = IS_RAIL_FLAT.copy().astype(bool)
enemy_none = np.zeros(NUM_CELLS, dtype=bool)

t0 = time.perf_counter()
for _ in range(1000):
    for src_f in rail_cells[:5]:
        _engineer_dests_soa(int(src_f), empty_all, enemy_none)
elapsed = time.perf_counter() - t0
print(f"\n1000 x 5 Python BFS calls: {elapsed*1000:.2f}ms = {elapsed/1000/5*1000:.3f}ms each")
