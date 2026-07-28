"""tools/debug_curve_rails.py — print all curve rail cells."""

from junqi_core._movegen_tables import CURVE_CELLS, CURVE_NEIGHBORS
from junqi_core.board import BOARD_SIZE


def xy(f):
    return f % BOARD_SIZE, f // BOARD_SIZE


for cid in sorted(CURVE_CELLS.keys()):
    cells = [xy(int(f)) for f in CURVE_CELLS[cid]]
    print(f"Curve {cid}: {cells}")

print()
# Neighbours within curve for key cells
for pt in ((6, 11), (10, 11), (6, 5), (10, 5), (5, 10), (11, 10), (5, 6), (11, 6)):
    f = pt[1] * BOARD_SIZE + pt[0]
    nbrs = [xy(int(n)) for n in CURVE_NEIGHBORS[f]]
    print(f"  {pt}: curve_nbrs={nbrs}")
