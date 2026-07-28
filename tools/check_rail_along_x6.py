"""tools/check_rail_along_x6.py — check which cells along x=6 are rail."""
from junqi_core.rail_topology import IS_RAILWAY, IS_NINEGRID
def f(x, y): return y * 17 + x
for y in range(17):
    print(f"  (6,{y}): rail={IS_RAILWAY[f(6,y)]}  9grid={IS_NINEGRID[f(6,y)]}")
print()
print("Along x=8:")
for y in range(17):
    print(f"  (8,{y}): rail={IS_RAILWAY[f(8,y)]}  9grid={IS_NINEGRID[f(8,y)]}")
