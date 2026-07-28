"""Inspect curve rail structure."""
import sys
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
import junqi_core._movegen_tables as _T

# Each curve has 12 cells. Let's see the adjacency within each curve
for cid, cells in sorted(_T.CURVE_CELLS.items()):
    print(f"\nCurve {cid}: {len(cells)} cells")
    for flat in cells:
        nbrs = _T.CURVE_NEIGHBORS[int(flat)]
        x, y = int(flat) % 17, int(flat) // 17
        print(f"  ({x:2d},{y:2d}) flat={int(flat):3d}  nbrs={nbrs}")
    # Check if it's a simple chain (max degree 2)
    degrees = [len(_T.CURVE_NEIGHBORS[int(f)]) for f in cells]
    print(f"  Degrees: {degrees}")

# How many curve cells have degree 1 (chain endpoints)?
# How many have degree 2 (chain middle)?
all_degrees = []
for cid, cells in _T.CURVE_CELLS.items():
    for flat in cells:
        all_degrees.append(len(_T.CURVE_NEIGHBORS[int(flat)]))
print(f"\nAll curve degrees: min={min(all_degrees)} max={max(all_degrees)} unique={set(all_degrees)}")
