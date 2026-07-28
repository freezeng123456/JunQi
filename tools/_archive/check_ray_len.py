"""Check max straight rail ray length."""
import sys
sys.path.insert(0, "/data/home/freezeng/data/workspace/JunQi")
import numpy as np
import junqi_core._movegen_tables as _T

rays_pad = _T.STRAIGHT_RAIL_RAYS_PAD
print(f"STRAIGHT_RAIL_RAYS_PAD shape: {rays_pad.shape}")
print(f"Max ray len (L): {rays_pad.shape[-1]}")

# Check actual max occupied length
max_len = 0
for flat in range(289):
    for k in range(4):
        ray = rays_pad[flat, k]
        actual_len = int((ray >= 0).sum())
        if actual_len > max_len:
            max_len = actual_len
print(f"Actual max ray length used: {max_len}")

# Check _SRAYS_L
from junqi_core.move_gen import _SRAYS_L
print(f"_SRAYS_L = {_SRAYS_L}")
