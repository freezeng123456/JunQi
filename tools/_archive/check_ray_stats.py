"""Check actual straight rail ray lengths."""
import sys
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
import junqi_core._movegen_tables as _T

rays_pad = _T.STRAIGHT_RAIL_RAYS_PAD  # (289, 4, 12)
# For each src cell that is a rail cell, count the actual ray lengths
actual_lengths = []
for flat in range(289):
    if not _T.IS_RAIL_FLAT[flat]:
        continue
    for k in range(4):
        ray = rays_pad[flat, k]
        l = int((ray >= 0).sum())
        actual_lengths.append(l)

actual_lengths = np.array(actual_lengths)
print(f"Total rays: {len(actual_lengths)}")
print(f"Mean length: {actual_lengths.mean():.1f}")
print(f"Median length: {np.median(actual_lengths):.1f}")
print(f"Max length: {actual_lengths.max()}")
print(f"Distribution:")
for l in range(13):
    count = (actual_lengths == l).sum()
    if count:
        print(f"  len={l:2d}: {count:4d} ({count/len(actual_lengths)*100:.1f}%)")
print(f"\nTrailing -1 fraction: {(rays_pad[_T.IS_RAIL_FLAT] == -1).sum() / rays_pad[_T.IS_RAIL_FLAT].size * 100:.1f}%")
