"""Micro-benchmark of build_legal_mask_batch_gpu only."""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.gpu_collector import build_legal_mask_batch_gpu


def main():
    N = 1024
    world = GpuRollout(num_envs=N)
    world.reset(seed_base=0)
    turn = world.state.copy_turn_to_host()
    acting = np.asarray(turn, dtype=np.int8).reshape(N)
    done = np.zeros(N, dtype=bool)

    # Warmup
    for _ in range(3):
        build_legal_mask_batch_gpu(world, acting, done)

    STEPS = 50
    # Breakdown
    t_kernel = t_scatter = 0.0
    for _ in range(STEPS):
        t0 = time.perf_counter()
        ids, counts = world.legal_actions_dense(acting)
        t1 = time.perf_counter()
        # Replicate scatter body
        from junqi_rl.training.gpu_collector import _ROTATE_LUT_STACK
        from junqi_rl.action_lut import FLAT_ACTION_DIM
        N2, K = ids.shape
        col_idx = np.arange(K, dtype=np.int32)[None, :]
        valid = (col_idx < counts[:, None]) & (~done)[:, None]
        safe_ids = np.where(valid, ids, 0).astype(np.int64, copy=False)
        seat_idx = acting.astype(np.int64, copy=False)[:, None]
        can_ids = _ROTATE_LUT_STACK[seat_idx, safe_ids]
        mask = np.zeros((N2, FLAT_ACTION_DIM), dtype=bool)
        row_idx = np.repeat(np.arange(N2, dtype=np.int32), K)
        flat_valid = valid.ravel()
        if flat_valid.any():
            mask[row_idx[flat_valid], can_ids.ravel()[flat_valid]] = True
        t2 = time.perf_counter()
        t_kernel += t1 - t0
        t_scatter += t2 - t1

    print(f"N={N} STEPS={STEPS}")
    print(f"  legal_actions_dense (GPU + D2H):  {t_kernel/STEPS*1e3:7.2f} ms/call")
    print(f"  scatter to dense mask (numpy):    {t_scatter/STEPS*1e3:7.2f} ms/call")


if __name__ == "__main__":
    main()
