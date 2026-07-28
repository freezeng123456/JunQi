"""Analyze rail topology for fast engineer BFS."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from junqi_core._movegen_tables import (
    ENGINEER_RAIL_NEIGHBORS, IS_RAIL_FLAT, NUM_CELLS
)

RAIL_CELLS = np.nonzero(IS_RAIL_FLAT)[0].astype(np.int32)
R = len(RAIL_CELLS)
RAIL_TO_IDX = np.full(NUM_CELLS, -1, dtype=np.int32)
RAIL_TO_IDX[RAIL_CELLS] = np.arange(R, dtype=np.int32)

# Print degree of each rail cell
print("Rail cell degrees:")
for i, f in enumerate(RAIL_CELLS.tolist()):
    nbrs = ENGINEER_RAIL_NEIGHBORS[f]
    print(f"  rail_idx={i:2d} flat={f:3d} ({f%17},{f//17}) deg={len(nbrs)} nbrs={nbrs}")

# Find connected components
adj_mat = np.zeros((R, R), dtype=bool)
for i, f in enumerate(RAIL_CELLS.tolist()):
    for nb in ENGINEER_RAIL_NEIGHBORS[f]:
        j = int(RAIL_TO_IDX[nb])
        if j >= 0:
            adj_mat[i, j] = True

visited = np.zeros(R, dtype=bool)
components = []
for start in range(R):
    if visited[start]:
        continue
    comp = []
    stack = [start]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        comp.append(v)
        for w in np.nonzero(adj_mat[v])[0].tolist():
            if not visited[w]:
                stack.append(w)
    components.append(comp)

print(f"\nConnected components: {len(components)}")
for ci, comp in enumerate(components):
    cells = [int(RAIL_CELLS[i]) for i in comp]
    print(f"  Component {ci}: {len(comp)} cells: flat={cells[:8]}...")
