"""Decode the SVG rail topology from the reference gameboard.

Black paths (stroke:#000000, stroke-width:8) = engineer rail tracks.
This is what the user called "深色的轨道".

Approach:
  1. Extract all black path d="..." endpoints from SVG.
  2. Extract all rect <x,y> to find the grid offset + spacing.
  3. Map path endpoints to grid (x, y) cells.
  4. Build the adjacency list of engineer-reachable edges.
"""
import re
from xml.etree import ElementTree as ET

SVG = '/data/home/freezeng/data/workspace/coding/Junqi_Gameboard.svg'

tree = ET.parse(SVG)
root = tree.getroot()

# Collect all black paths + their effective transform.
# Many paths are inside a <g transform="translate(...)"> layer.
# Walk the tree tracking ancestor transforms.
def walk_with_transform(elem, parent_tf=(0.0, 0.0)):
    tf = parent_tf
    t = elem.get('transform')
    if t:
        m = re.match(r'translate\(\s*([-0-9.e]+)\s*,\s*([-0-9.e]+)\s*\)', t)
        if m:
            tf = (parent_tf[0] + float(m.group(1)),
                  parent_tf[1] + float(m.group(2)))
    yield elem, tf
    for child in elem:
        yield from walk_with_transform(child, tf)

def parse_path_endpoints(d: str, tf=(0.0, 0.0)):
    """Parse SVG path d attribute into list of absolute (x, y) points."""
    # Supports M, m, L, l, H, h, V, v (simplified).
    tokens = re.findall(r'[MmLlHhVvZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d)
    pts = []
    cur = [0.0, 0.0]
    i = 0
    cmd = None
    while i < len(tokens):
        t = tokens[i]
        if t in 'MmLlHhVvZz':
            cmd = t
            i += 1
            continue
        try:
            n = float(t)
        except ValueError:
            i += 1; continue
        if cmd in ('M', 'L'):
            cur = [n, float(tokens[i+1])]
            pts.append(tuple(cur))
            i += 2
            cmd = 'L' if cmd == 'M' else cmd
        elif cmd in ('m', 'l'):
            if not pts:
                cur = [n, float(tokens[i+1])]
            else:
                cur = [cur[0] + n, cur[1] + float(tokens[i+1])]
            pts.append(tuple(cur))
            i += 2
            cmd = 'l' if cmd == 'm' else cmd
        elif cmd == 'H':
            cur = [n, cur[1]]; pts.append(tuple(cur)); i += 1
        elif cmd == 'h':
            cur = [cur[0] + n, cur[1]]; pts.append(tuple(cur)); i += 1
        elif cmd == 'V':
            cur = [cur[0], n]; pts.append(tuple(cur)); i += 1
        elif cmd == 'v':
            cur = [cur[0], cur[1] + n]; pts.append(tuple(cur)); i += 1
        else:
            i += 1
    # Apply transform
    return [(x + tf[0], y + tf[1]) for (x, y) in pts]


# Gather all black rail paths (= engineer tracks)
black_segs = []  # list of (pt_start, pt_end)
for elem, tf in walk_with_transform(root):
    if elem.tag.split('}')[-1] != 'path':
        continue
    style = elem.get('style', '')
    if 'stroke:#000000' not in style or 'stroke-width:8' not in style:
        continue
    d = elem.get('d', '')
    if not d:
        continue
    pts = parse_path_endpoints(d, tf)
    for i in range(len(pts) - 1):
        black_segs.append((pts[i], pts[i+1]))

print(f"Total black rail segments: {len(black_segs)}")

# Also gather all rect positions to map to grid.
rect_positions = []
for elem, tf in walk_with_transform(root):
    if elem.tag.split('}')[-1] != 'rect':
        continue
    try:
        x = float(elem.get('x', '0')) + tf[0]
        y = float(elem.get('y', '0')) + tf[1]
        rect_positions.append((x, y, elem.get('id', '')))
    except Exception:
        pass

# Build a grid from rect positions (17x17).
xs = sorted(set(round(x, 1) for x, y, _ in rect_positions))
ys = sorted(set(round(y, 1) for x, y, _ in rect_positions))
print(f"Unique rect x: {len(xs)}  y: {len(ys)}")
print(f"x range: {min(xs):.1f} .. {max(xs):.1f}")
print(f"y range: {min(ys):.1f} .. {max(ys):.1f}")

# We expect 17 distinct x and 17 distinct y
# Compute spacing
if len(xs) > 1:
    dx = (max(xs) - min(xs)) / (len(xs) - 1)
    dy = (max(ys) - min(ys)) / (len(ys) - 1)
    print(f"dx: {dx:.2f}  dy: {dy:.2f}")

x_base, y_base = min(xs), min(ys)

def svg_to_grid(px, py):
    """Map an SVG point (path endpoint) to grid (x, y) coordinates 0..16.

    Rect centers are at (x_base + i*dx + cell_size/2, ...).
    But path endpoints may snap to rect corners or midpoints.
    """
    if len(xs) < 2:
        return None
    dx = (max(xs) - min(xs)) / (len(xs) - 1)
    dy = (max(ys) - min(ys)) / (len(ys) - 1)
    # cell_size is roughly dx (same as spacing)
    # Rect's nominal width = dx; path endpoints may be at rect center + some offset
    # Let's try: gx = round((px - (x_base - dx/2)) / dx - 0.5)
    # Simpler: find nearest (x, y) in rect_positions.
    # Use dx as tolerance.
    gx = round((px - x_base) / dx)
    gy = round((py - y_base) / dy)
    return (gx, gy), dx, dy

print("\nSample path endpoint mappings:")
for i, (p1, p2) in enumerate(black_segs[:6]):
    g1 = svg_to_grid(*p1)
    g2 = svg_to_grid(*p2)
    print(f"  seg {i}: SVG {p1} -> grid {g1[0] if g1 else None},  SVG {p2} -> grid {g2[0] if g2 else None}")
