"""tools/debug_6_11_rays.py — print rail rays from (6,11)."""

from junqi_core._movegen_tables import STRAIGHT_RAIL_RAYS, ENGINEER_RAIL_NEIGHBORS, CURVE_ID_OF
from junqi_core.board import xy_to_flat, BOARD_SIZE


def xy(f):
    return f % BOARD_SIZE, f // BOARD_SIZE


flat = xy_to_flat(6, 11)
dirs = ['+x', '-x', '+y', '-y']
print(f"(6,11) flat={flat} curve_id={int(CURVE_ID_OF[flat])}")
print("STRAIGHT_RAIL_RAYS:")
for d, ray in zip(dirs, STRAIGHT_RAIL_RAYS[flat]):
    print(f"  {d}: {[xy(c) for c in ray]}")
print("ENGINEER_RAIL_NEIGHBORS:", [xy(n) for n in ENGINEER_RAIL_NEIGHBORS[flat]])
