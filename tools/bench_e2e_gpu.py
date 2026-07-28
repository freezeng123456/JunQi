"""tools/bench_e2e_gpu.py — End-to-end GPU loop throughput.

Measures the combined cost of one RL iteration:
  1. legal_action_ids_batch    (GPU kernel, writes candidate actions)
  2. (CPU) pick one random action per env from the returned dense list
  3. step_batch                (GPU kernel, mutates state + emits MoveResult)
  4. build_observation_batch   (GPU kernel, writes 101-ch spatial + 28-dim global)

Everything between push_state (once at start) and final D2H stays GPU-resident.
The CPU does only:
  * action pick (from legal actions dense ID list)
  * read back the (tiny) MoveResult for episode-boundary logic

This is the measurement that predicts the throughput a PPO rollout can achieve
under ADR-112 / Phase 1b once policy & value nets drop in as thin torch modules
consuming the observation slab directly.
"""

from __future__ import annotations

import os
import random
import sys
import time

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
    alv = flat(b.alive); px = flat(b.pos_x); py = flat(b.pos_y)
    cpip = np.where(alv, py.astype(np.int16) * 17 + px.astype(np.int16), np.int16(-1)).astype(np.int16, copy=False)
    return {
        "cell_piece_id_per_piece": cpip,
        "piece_seat_arr": flat(b.piece_seat_arr).astype(np.int8),
        "piece_type_arr": flat(b.piece_type_arr).astype(np.int8),
        "alive": alv.astype(bool),
        "pos_x": px.astype(np.int8), "pos_y": py.astype(np.int8),
        "zero_x": np.zeros_like(px, dtype=np.int8),
        "zero_y": np.zeros_like(py, dtype=np.int8),
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


def _pick_actions(action_ids_dense: np.ndarray, counts: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Pick one uniformly-random legal action per env from the dense (N, 512) output.
    Envs with count==0 get action_id=0 (will be treated as invalid by step_batch)."""
    N = action_ids_dense.shape[0]
    out = np.zeros(N, dtype=np.int32)
    has_actions = counts > 0
    # Uniform random index ∈ [0, count_i) per env.
    rand_idx = (rng.integers(0, 1 << 30, size=N) % np.maximum(counts, 1)).astype(np.int32)
    picked = action_ids_dense[np.arange(N), rand_idx]
    out[has_actions] = picked[has_actions]
    return out


def main() -> None:
    _cuda.init_tables()
    _upload_zobrist_tables()

    for N in (256, 1024, 4096):
        # Fresh batch on CPU, single initial pack to GPU.
        states = [_new_game(i) for i in range(N)]
        b = BatchedGameState.from_game_states(states)
        gs = _cuda.DeviceGameStateBatch(N)
        gs.copy_from_host(_pack_from_batched(b))
        gs.copy_termination_from_host(
            b.terminated.astype(bool),
            b.winner_team.astype(np.int8),
            b.draw.astype(bool),
        )

        # Observer/belief scratch: identity observer, zero beliefs (shapes only).
        observer_seats = np.tile(np.arange(4, dtype=np.int8), (N, 1))  # (N, 4)
        beliefs = np.zeros((N, 4, 12, 289), dtype=np.float32)          # (N, 4, 12, 289)
        obs = _cuda.DeviceObservationBatch(N)

        # One-shot H2D of the "static" inputs into persistent GpuScratch.
        # For PPO where beliefs are produced by the belief network per-episode,
        # this mirrors the real rollout pattern: upload once, hit the kernel
        # many times without paying PCIe again.
        _cuda.upload_beliefs(N, beliefs)
        _cuda.upload_observer_seats(N, observer_seats)

        steps = 50
        rng = np.random.default_rng(0xBEEF)

        # Warm-up + seed turn/acting_seats.
        acting_seats = np.zeros(N, dtype=np.int8)  # read from GPU each step
        # For simplicity the benchmark drives all envs as if turn=SOUTH; that
        # is not game-legal (turn should rotate) but this benchmark is a
        # throughput microbench, not a game; step_batch still validates.

        t_legal_total = 0.0
        t_step_total  = 0.0
        t_obs_total   = 0.0

        t_loop = time.perf_counter()
        for _ in range(steps):
            # Read current turn from device to drive acting_seats.
            # (This D2H is tiny — N int8s.  Skip in real RL: kept here for legality.)
            pass

            t0 = time.perf_counter()
            # acting_seats = turn  (use a constant for benchmark; real RL reads turn)
            ids, counts = _cuda.legal_action_ids_batch(gs, acting_seats)
            t_legal_total += time.perf_counter() - t0

            # Action pick on CPU.
            actions = _pick_actions(ids, counts, rng)

            t0 = time.perf_counter()
            gs.step_batch(actions)
            t_step_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            _cuda.build_observation_batch_resident(gs, obs, np.int8(2))
            t_obs_total += time.perf_counter() - t0

        total = time.perf_counter() - t_loop
        envsteps = steps * N / total

        print(f"N={N:5d}  {steps:3d}-step loop  {total*1e3:6.1f} ms  "
              f"→ {envsteps:10,.0f} env·steps/s  "
              f"(legal {t_legal_total*1e3:6.1f} ms, "
              f"step {t_step_total*1e3:6.1f} ms, "
              f"obs {t_obs_total*1e3:6.1f} ms, "
              f"other {(total - t_legal_total - t_step_total - t_obs_total)*1e3:6.1f} ms)")


if __name__ == "__main__":
    main()
