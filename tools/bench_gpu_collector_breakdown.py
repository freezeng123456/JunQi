"""tools/bench_gpu_collector_breakdown.py — component-level timer."""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def main():
    N = 1024
    device = torch.device("cuda")
    world = GpuRollout(num_envs=N)
    world.reset(seed_base=0)

    cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=2, depth=2,
                         embed_dim=64, n_head=4, ff_factor=2, dropout=0.0)
    policy = JunqiNet(cfg).to(device).eval()

    from junqi_rl.training.gpu_collector import (
        build_legal_mask_batch_gpu, _per_seat_terminal_rewards,
        _UNROTATE_LUT_STACK,
    )

    # Warmup
    for _ in range(3):
        world.build_all_seat_observations_torch()
        torch.cuda.synchronize()

    STEPS = 20

    # --- Component timers (isolate each stage) ---
    t_obs = t_legal = t_policy = t_step = t_conv = 0.0

    done_flags = np.zeros(N, dtype=bool)
    env_idx_t = torch.arange(N, device=device)

    torch.cuda.synchronize()
    for _ in range(STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        turns = world.state.copy_turn_to_host()
        turns = np.asarray(turns, dtype=np.int8).reshape(N)
        acting_seats = np.where(done_flags, np.int8(0), turns).astype(np.int8)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        sp_full, gl_full = world.build_all_seat_observations_torch()
        acting_t = torch.from_numpy(acting_seats).to(device, dtype=torch.long)
        sp_t = sp_full[env_idx_t, acting_t].contiguous()
        gl_t = gl_full[env_idx_t, acting_t].contiguous()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        mask = build_legal_mask_batch_gpu(world, acting_seats, done_flags)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        lm_t = torch.from_numpy(mask).to(device)
        with torch.no_grad():
            a_can, lp, v = policy.act(sp_t, gl_t, lm_t)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        a_can_np = a_can.cpu().numpy().astype(np.int32)
        a_world = _UNROTATE_LUT_STACK[
            acting_seats.astype(np.int64), a_can_np.astype(np.int64)
        ].astype(np.int32)
        result = world.step(a_world)
        torch.cuda.synchronize()
        t5 = time.perf_counter()

        t_obs    += (t2 - t1) - (t1 - t0)
        t_legal  += t3 - t2
        t_policy += t4 - t3
        t_step   += t5 - t4
        t_conv   += t1 - t0

    total_kernel = t_obs + t_legal + t_policy + t_step + t_conv
    print(f"N={N} steps={STEPS}")
    print(f"  copy_turn    {t_conv*1e3:7.1f} ms  ({t_conv/STEPS*1e3:.2f} ms/step)")
    print(f"  obs+slice    {t_obs*1e3:7.1f} ms  ({t_obs/STEPS*1e3:.2f} ms/step)")
    print(f"  legal mask   {t_legal*1e3:7.1f} ms  ({t_legal/STEPS*1e3:.2f} ms/step)")
    print(f"  policy.act   {t_policy*1e3:7.1f} ms  ({t_policy/STEPS*1e3:.2f} ms/step)")
    print(f"  step_batch   {t_step*1e3:7.1f} ms  ({t_step/STEPS*1e3:.2f} ms/step)")
    print(f"  TOTAL        {total_kernel*1e3:7.1f} ms   → {N*STEPS/total_kernel:,.0f} env·steps/s")


if __name__ == "__main__":
    main()
