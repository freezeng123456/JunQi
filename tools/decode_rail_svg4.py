"""Decode rail topology v4 — establish SVG<->grid mapping, derive full adjacency.

From v3 we have the paired center lines:
   Horizontal rail center y-values: [69.7, 227.0, 296.3, 399.0, 501.7, 574.1, 731.4]
   Vertical rail center x-values:   [68.6, 225.9, 295.4, 398.2, 500.9, 574.1, 731.4]

Map these to grid rows/cols of the 17x17 board:
   North's rails:   y=1 (row 1), y=5 (row 5)
   Central:         y=6 (row 6), y=8 (row 8=center), y=10 (row 10)
   South's rails:   y=11 (row 11), y=15 (row 15)
Similarly for x.

There are 7 horizontal + 7 vertical = 14 rail centerlines. Since each family
produces 7 center-lines and the board has 17 rows, spacing = (731.4-69.7)/(15-1) ~ 47.3.
"""

def map_svg_to_grid():
    # SVG centerline y values
    y_centers = [69.7, 227.0, 296.3, 399.0, 501.7, 574.1, 731.4]
    x_centers = [68.6, 225.9, 295.4, 398.2, 500.9, 574.1, 731.4]
    # Expected grid rows: 1, 5, 6, 8, 10, 11, 15
    # (y=8 is the board center, y=6 & y=10 bracket it; y=1/5 are North rails;
    #  y=11/15 are South rails)
    expected_rows = [1, 5, 6, 8, 10, 11, 15]
    expected_cols = [1, 5, 6, 8, 10, 11, 15]
    # Verify proportional spacing
    for i, (yc, r) in enumerate(zip(y_centers, expected_rows)):
        if i > 0:
            dy_svg = yc - y_centers[i-1]
            dr     = r  - expected_rows[i-1]
            print(f"  y: {y_centers[i-1]:.1f}→{yc:.1f}  dy={dy_svg:.1f}   grid row {expected_rows[i-1]}→{r}  dr={dr}")
    return y_centers, x_centers, expected_rows, expected_cols

print("=" * 60)
print("SVG centerline ↔ grid row mapping check:")
print("=" * 60)
map_svg_to_grid()

# From the v3 output, the rail segments (in grid coordinates):
# Horizontal rails:
#   y=1  (SVG y=69.7):  x 6..10 (North row 1)   [path4973]
#   y=5  (SVG y=227):   x 6..10 (North row 5)   [path4969]
#   y=6  (SVG y=296.3): x 1..5, x 6..10, x 11..15  — THREE SEGMENTS at y=6
#   y=8  (SVG y=399):   x 6..10 (central middle, 1-segment)  [path6918]
#   y=10 (SVG y=501.7): x 1..5, x 6..10, x 11..15 — three segments
#   y=11 (SVG y=574.1): x 6..10 (South row 11)
#   y=15 (SVG y=731.4): x 6..10 (South row 15)

# Vertical rails:
#   x=1  (SVG x=68.6):  y 6..10 (West col 1)
#   x=5  (SVG x=225.9): y 6..10 (West col 5)
#   x=6  (SVG x=295.4): y 1..5, y 6..10, y 11..15 (three segments)
#   x=8  (SVG x=398.2): y 6..10 (central middle, 1-segment) [path6912]
#   x=10 (SVG x=500.9): y 1..5, y 6..10, y 11..15 (three segments)
#   x=11 (SVG x=574.1): y 6..10 (East col 11)
#   x=15 (SVG x=731.4): y 6..10 (East col 15)

print()
print("=" * 60)
print("Rail topology in grid coordinates:")
print("=" * 60)

# Build the full set of rail cells from the segments
rail_cells = set()

# Horizontal rails (y fixed, x ranges)
h_rails = [
    (1,  6, 10),  (5,  6, 10),
    (6,  1, 5),   (6,  6, 10),  (6,  11, 15),
    (8,  6, 10),                # central row — partial? check!
    (10, 1, 5),   (10, 6, 10),  (10, 11, 15),
    (11, 6, 10),  (15, 6, 10),
]
# Vertical rails (x fixed, y ranges)
v_rails = [
    (1,  6, 10),  (5,  6, 10),
    (6,  1, 5),   (6,  6, 10),  (6,  11, 15),
    (8,  6, 10),                # central column
    (10, 1, 5),   (10, 6, 10),  (10, 11, 15),
    (11, 6, 10),  (15, 6, 10),
]

for (y, x_lo, x_hi) in h_rails:
    for x in range(x_lo, x_hi + 1):
        rail_cells.add((x, y))
for (x, y_lo, y_hi) in v_rails:
    for y in range(y_lo, y_hi + 1):
        rail_cells.add((x, y))

print(f"Total rail cells: {len(rail_cells)}")

# Visualize on 17x17 grid
print("\n17×17 rail map ('R' = rail cell):")
for y in range(17):
    row = []
    for x in range(17):
        row.append('R' if (x, y) in rail_cells else '.')
    print(f"  y={y:2d}: " + ' '.join(row))

# Build adjacency list: for each rail cell, find orthogonal neighbors
# that are ALSO on a rail AND on the same rail segment.
# Since a rail cell lies on a horizontal and/or vertical line, its rail
# neighbors are the adjacent cells along those lines.
rail_adj = {cell: set() for cell in rail_cells}
# Add horizontal neighbors
for (y, x_lo, x_hi) in h_rails:
    for x in range(x_lo, x_hi):
        if (x, y) in rail_cells and (x+1, y) in rail_cells:
            rail_adj[(x, y)].add((x+1, y))
            rail_adj[(x+1, y)].add((x, y))
for (x, y_lo, y_hi) in v_rails:
    for y in range(y_lo, y_hi):
        if (x, y) in rail_cells and (x, y+1) in rail_cells:
            rail_adj[(x, y)].add((x, y+1))
            rail_adj[(x, y+1)].add((x, y))

# Check degree distribution
from collections import Counter
deg_dist = Counter(len(v) for v in rail_adj.values())
print(f"\nRail degree distribution: {dict(deg_dist)}")

# Print degree-3 or degree-4 cells (the key junctions)
print("Degree ≥ 3 cells (rail junctions):")
for cell, nbrs in sorted(rail_adj.items()):
    if len(nbrs) >= 3:
        print(f"  {cell}: degree {len(nbrs)}, neighbors={sorted(nbrs)}")

# Connected components check
from collections import deque
seen = set()
components = []
for c in rail_cells:
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
