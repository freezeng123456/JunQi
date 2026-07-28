"""Benchmark GPU step_batch vs CPU BatchedGameState.step_batch."""

from __future__ import annotations

import os
import random
import sys
import time

# Make the .so at the repo root importable when running from tools/ directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

import junqi_cuda as _cuda
from junqi_core.batched_state import BatchedGameState
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from junqi_rl.gpu_world import _upload_zobrist_tables


def _new_game(seed: int) -> GameState:
    rng = random.Random(seed)
    return GameState.new_game(generate_random_setup(rng))


def _pack_from_batched(b: BatchedGameState) -> dict:
    N = b.num_envs
    flat = lambda a: a.reshape(-1)
    alv = flat(b.alive); psa = flat(b.piece_seat_arr); pta = flat(b.piece_type_arr)
    px = flat(b.pos_x); py = flat(b.pos_y)
    cpip = np.where(alv, py.astype(np.int16)*17 + px.astype(np.int16), np.int16(-1)).astype(np.int16, copy=False)
    return {
        "cell_piece_id_per_piece": cpip,
        "piece_seat_arr": psa.astype(np.int8),
        "piece_type_arr": pta.astype(np.int8),
        "alive": alv.astype(bool),
        "pos_x": px.astype(np.int8), "pos_y": py.astype(np.int8),
        "zero_x": np.zeros_like(px, dtype=np.int8), "zero_y": np.zeros_like(py, dtype=np.int8),
        "move_count_arr": flat(b.move_count_arr).astype(np.int16),
        "active_eat_arr": flat(b.active_eat_arr).astype(np.int16),
        "passive_surv_arr": flat(b.passive_surv_arr).astype(np.int16),
        "death_reason_arr": flat(b.death_reason_arr).astype(np.int8),
        "death_step_arr": flat(b.death_step_arr).astype(np.int16),
        "death_loc_flat_arr": flat(b.death_loc_flat_arr).astype(np.int16),
        "cell_piece_id": flat(b.cell_piece_id).astype(np.int16),
        "seat_dead_arr": flat(b.seat_dead_arr).astype(bool),
        "seat_flag_revealed_arr": flat(b.seat_flag_revealed_arr).astype(bool),
        "turn": b.turn.astype(np.int8),
        "zobrist": b.zobrist.astype(np.int64),
        "move_counter": b.move_counter.astype(np.int32),
        "moves_since_last_combat": b.moves_since_last_combat.astype(np.int32),
    }


def main() -> None:
    _cuda.init_tables()
    _upload_zobrist_tables()

    for N in (64, 256, 1024, 4096):
        # Build fresh CPU batch
        states = [_new_game(i) for i in range(N)]
        b = BatchedGameState.from_game_states(states)

        # Push to GPU
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(_pack_from_batched(b))
        gs.copy_termination_from_host(
            b.terminated.astype(bool), b.winner_team.astype(np.int8), b.draw.astype(bool),
        )

        # Pre-generate an action list
        rng = random.Random(0xBEEF)
        b_copy = b  # shared; we'll reset by re-pushing
        steps = 20

        # ---------- CPU benchmark ----------
        t0 = time.perf_counter()
        for _ in range(steps):
            if b_copy.terminated.all(): break
            ids = b_copy.legal_action_ids_batch()
            actions = np.zeros(N, dtype=np.int32)
            for i in range(N):
                if b_copy.terminated[i] or ids[i].size == 0:
                    continue
                actions[i] = int(rng.choice(ids[i]))
            b_copy.step_batch(actions)
        t1 = time.perf_counter()
        cpu_envsteps = steps * N / (t1 - t0)

        # ---------- GPU benchmark (re-push state, same action sequence) ----------
        rng2 = random.Random(0xBEEF)
        states2 = [_new_game(i) for i in range(N)]
        b2 = BatchedGameState.from_game_states(states2)
        gs2 = _cuda.DeviceGameStateBatch(N)
        gs2.copy_from_host(_pack_from_batched(b2))
        gs2.copy_termination_from_host(
            b2.terminated.astype(bool), b2.winner_team.astype(np.int8), b2.draw.astype(bool),
        )

        t0 = time.perf_counter()
        for _ in range(steps):
            if b2.terminated.all(): break
            # We still use CPU legal-action enumeration for action picking here,
            # because this benchmark isolates step_batch; legal-action GPU is
            # a separate path. The CPU legal call + action pick is excluded
            # below by measuring only the GPU step time.
            ids = b2.legal_action_ids_batch()
            actions = np.zeros(N, dtype=np.int32)
            for i in range(N):
                if b2.terminated[i] or ids[i].size == 0:
                    continue
                actions[i] = int(rng2.choice(ids[i]))
            # Step CPU in lockstep so we can pick from fresh legal list.
            b2.step_batch(actions)

            # Meanwhile, step the GPU on the same actions; we only time GPU.
            tg0 = time.perf_counter()
            gs2.step_batch(actions)
            tg1 = time.perf_counter()
            # Accumulate GPU time only.
            if not hasattr(main, "_gpu_accum"): main._gpu_accum = 0.0
            main._gpu_accum += (tg1 - tg0)
        t_total = time.perf_counter() - t0
        gpu_step_time = main._gpu_accum
        main._gpu_accum = 0.0

        gpu_envsteps = steps * N / gpu_step_time

        print(f"N={N:5d}  CPU full-loop {cpu_envsteps:10,.0f} env·steps/s    "
              f"GPU step_batch kernel-only {gpu_envsteps:10,.0f} env·steps/s  "
              f"(speedup {gpu_envsteps/cpu_envsteps:.1f}×)")


if __name__ == "__main__":
    main()
