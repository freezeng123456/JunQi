"""Decode rail topology v3 — pair-aware, derive logical track center lines."""
import re
from xml.etree import ElementTree as ET

SVG = '/data/home/freezeng/data/workspace/coding/Junqi_Gameboard.svg'

tree = ET.parse(SVG)
root = tree.getroot()

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
    tokens = re.findall(r'[MmLlHhVvZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', d)
    pts = []
    cur = [0.0, 0.0]
    i = 0
    cmd = None
    while i < len(tokens):
        t = tokens[i]
        if t in 'MmLlHhVvZz':
            cmd = t; i += 1; continue
        try: n = float(t)
        except ValueError: i += 1; continue
        if cmd in ('M','L'):
            cur=[n, float(tokens[i+1])]; pts.append(tuple(cur)); i+=2
            cmd='L' if cmd=='M' else cmd
        elif cmd in ('m','l'):
            if not pts:
                cur=[n, float(tokens[i+1])]
            else:
                cur=[cur[0]+n, cur[1]+float(tokens[i+1])]
            pts.append(tuple(cur)); i+=2
            cmd='l' if cmd=='m' else cmd
        elif cmd=='H': cur=[n, cur[1]]; pts.append(tuple(cur)); i+=1
        elif cmd=='h': cur=[cur[0]+n, cur[1]]; pts.append(tuple(cur)); i+=1
        elif cmd=='V': cur=[cur[0], n]; pts.append(tuple(cur)); i+=1
        elif cmd=='v': cur=[cur[0], cur[1]+n]; pts.append(tuple(cur)); i+=1
        else: i+=1
    return [(x+tf[0], y+tf[1]) for (x,y) in pts]

black_segs = []
for elem, tf in walk_with_transform(root):
    if elem.tag.split('}')[-1] != 'path': continue
    style = elem.get('style', '')
    if 'stroke:#000000' not in style or 'stroke-width:8' not in style: continue
    d = elem.get('d', '')
    if not d: continue
    pts = parse_path_endpoints(d, tf)
    for i in range(len(pts)-1):
        black_segs.append((pts[i], pts[i+1]))

# Also look at yellow segments (the thin yellow lines between black ones)
# These are the INTERIOR rail segments — connecting adjacent cells on a rail.
# Because the svg uses two parallel lines to show "train tracks", the actual
# CENTERLINE between them is where pieces sit.

# Observe: black segments come in pairs forming rectangles. Each "rectangle"
# is actually a rail segment. We need to find the centerline of each rail.

# Classify segments as horizontal or vertical.
horiz, vert = [], []
for p1, p2 in black_segs:
    if abs(p1[1] - p2[1]) < 1:  # same y → horizontal
        horiz.append((min(p1[0], p2[0]), max(p1[0], p2[0]), p1[1]))
    elif abs(p1[0] - p2[0]) < 1:  # same x → vertical
        vert.append((min(p1[1], p2[1]), max(p1[1], p2[1]), p1[0]))

print(f"Horizontal segs: {len(horiz)}")
for s in sorted(horiz, key=lambda s: (s[2], s[0])):
    print(f"  y={s[2]:7.2f}  x={s[0]:7.2f} .. {s[1]:7.2f}")
print(f"Vertical segs: {len(vert)}")
for s in sorted(vert, key=lambda s: (s[2], s[0])):
    print(f"  x={s[2]:7.2f}  y={s[0]:7.2f} .. {s[1]:7.2f}")

# For each rail segment (horizontal pair + vertical pair), find centerline
# Cluster horizontal segments by nearby y: pairs of y are the train tracks
print("\nClustering horizontal segments (by y coordinate)...")
ys_h = sorted(set(round(s[2], 1) for s in horiz))
print(f"Unique y values for horizontal: {ys_h}")
# Pair consecutive close y's
paired_y = []
i = 0
while i < len(ys_h):
    if i + 1 < len(ys_h) and abs(ys_h[i+1] - ys_h[i]) < 20:
        paired_y.append((ys_h[i] + ys_h[i+1]) / 2.0)
        i += 2
    else:
        paired_y.append(ys_h[i])
        i += 1
print(f"Horizontal rail center lines (y): {paired_y}")

xs_v = sorted(set(round(s[2], 1) for s in vert))
print(f"Unique x for vertical: {xs_v}")
paired_x = []
i = 0
while i < len(xs_v):
    if i + 1 < len(xs_v) and abs(xs_v[i+1] - xs_v[i]) < 20:
        paired_x.append((xs_v[i] + xs_v[i+1]) / 2.0)
        i += 2
    else:
        paired_x.append(xs_v[i])
        i += 1
print(f"Vertical rail center lines (x): {paired_x}")
