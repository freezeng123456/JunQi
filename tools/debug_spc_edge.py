"""tools/debug_spc_edge.py — Inspect rail-graph edges at SpcNode corners."""

from junqi_core.rail_topology import RAIL_ADJ, BOARD_SIZE, IS_RAILWAY


def xy(f):
    return f % BOARD_SIZE, f // BOARD_SIZE


for pt in ((6, 11), (5, 10), (10, 11), (11, 10), (6, 5), (5, 6), (11, 6), (10, 5)):
    f = pt[1] * BOARD_SIZE + pt[0]
    print(f"  {pt} flat={f} rail={IS_RAILWAY[f]} adj={[xy(n) for n in RAIL_ADJ[f]]}")
