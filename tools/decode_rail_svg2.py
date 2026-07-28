"""Decode rail topology v2 — hand-tuned grid mapping."""
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
        black_segs.append((pts[i], pts[i+1], elem.get('id', 'anon')))

print(f"Total black rail segments: {len(black_segs)}")

# Print sorted by segment coordinates to see structure
for pt1, pt2, id_ in sorted(black_segs):
    print(f"  [{id_:>12s}] ({pt1[0]:7.2f},{pt1[1]:7.2f}) -> ({pt2[0]:7.2f},{pt2[1]:7.2f})")
