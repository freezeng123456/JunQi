"""tools/measure_rail_limits.py — measure max rail-ray length and max rail degree."""

from collections import Counter, deque

from junqi_core.rail_topology import RAIL_ADJ, IS_RAILWAY, BOARD_SIZE, NC


def xy(flat):
    return flat % BOARD_SIZE, flat // BOARD_SIZE


def axis_bfs(src, axis):
    """BFS constrained to same axis as src."""
    sx, sy = xy(src)
    visited = {src}
    q = deque([src])
    while q:
        c = q.popleft()
        for nb in RAIL_ADJ[c]:
            if nb in visited:
                continue
            nx, ny = xy(nb)
            if axis == 'x' and nx != sx:
                continue
            if axis == 'y' and ny != sy:
                continue
            visited.add(nb)
            q.append(nb)
    return len(visited) - 1  # exclude src


def main():
    max_ray = 0
    max_src = -1
    max_axis = None
    ray_lens = []
    for src in range(NC):
        if not IS_RAILWAY[src]:
            continue
        for axis in ('x', 'y'):
            n = axis_bfs(src, axis)
            ray_lens.append(n)
            if n > max_ray:
                max_ray = n
                max_src = src
                max_axis = axis

    print(f'Max same-axis BFS reach (excluding src): {max_ray} cells '
          f'from src={xy(max_src)} axis={max_axis}')
    print('Histogram of same-axis reach counts:')
    for k, v in sorted(Counter(ray_lens).items()):
        print(f'  {k:2d} cells: {v} occurrences')

    max_deg = 0
    deg_hist = Counter()
    for f in range(NC):
        if not IS_RAILWAY[f]:
            continue
        d = len(RAIL_ADJ[f])
        deg_hist[d] += 1
        if d > max_deg:
            max_deg = d
    print(f'\nMax rail node degree: {max_deg}')
    print(f'Degree histogram: {sorted(deg_hist.items())}')


if __name__ == '__main__':
    main()
