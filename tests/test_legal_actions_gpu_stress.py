"""tests/test_legal_actions_gpu_stress.py — stress parity on deeper game states.

Phase 1b success criteria §4 specifies: "GPU results bit-identical with CPU on
50 random games @ N=32".  The existing ``test_legal_actions_gpu.py`` uses 32
envs of varying depth (0-49 steps); this stress test goes further:

  * 50 random games
  * Each game advanced 0, 25, 50, 100, 200 random steps (covers opening,
    midgame, endgame, near-draw phases).
  * Every GPU legal-action set checked against CPU for bit-identical parity.

Also exercises less-common states:
  * Fresh games (all seats at their starting positions).
  * States after 200 steps (deep midgame, pieces scattered).
  * States where a seat is dead (count must be 0 for that env when that seat
    is acting).

Runs automatically when junqi_cuda is available; skipped otherwise.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

try:
    import junqi_cuda as _cuda
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda extension not available or no CUDA-capable GPU",
)

from junqi_core.move_gen import generate_legal_action_ids_batch
from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays


_MAX_ACTIONS_PER_ENV = 512


def _advance(env: JunqiEnv, steps: int, rng: random.Random) -> None:
    for _ in range(steps):
        if env.state.terminated:
            break
        aids = env.legal_action_ids()
        if aids.size == 0:
            break
        env._step_game_only(int(rng.choice(aids)))


def _cpu_aids(envs: list[JunqiEnv]) -> list[np.ndarray]:
    out = []
    for env in envs:
        st = env.state
        if st.terminated:
            out.append(np.empty(0, dtype=np.int32))
            continue
        acting = st.turn
        if st.info[acting].dead:
            out.append(np.empty(0, dtype=np.int32))
            continue
        aids = generate_legal_action_ids_batch(
            st.cell_piece_id, st.piece_seat_arr, st.piece_type_arr,
            st.alive, st.pos_x, st.pos_y, acting.value,
        )
        out.append(aids.astype(np.int32))
    return out


def _gpu_aids(envs: list[JunqiEnv]) -> tuple[np.ndarray, np.ndarray]:
    N = len(envs)
    acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
    state_dict = _pack_state_arrays(envs)
    gpu_state = _cuda.DeviceGameStateBatch(N)
    gpu_state.copy_from_host(state_dict)
    return _cuda.legal_action_ids_batch(gpu_state, acting)


@pytest.fixture(scope="module", autouse=True)
def _init_tables():
    _cuda.init_tables()


@pytest.mark.parametrize("steps", [0, 25, 50, 100, 200])
def test_50_games_various_depths(steps: int) -> None:
    """50 random games advanced by `steps` random moves; GPU == CPU bit-for-bit."""
    N = 50
    envs = []
    for i in range(N):
        rng = random.Random(0x50500 + i + steps * 1000)
        env = JunqiEnv()
        env.reset(seed=0x60600 + i + steps * 1000)
        _advance(env, steps, rng)
        envs.append(env)

    cpu_sets = _cpu_aids(envs)
    gpu_ids, gpu_counts = _gpu_aids(envs)

    for i, cpu in enumerate(cpu_sets):
        cnt = int(gpu_counts[i])
        assert cnt == len(cpu), (
            f"steps={steps} env {i}: GPU count={cnt}, CPU count={len(cpu)}"
        )
        if cnt == 0:
            continue

        # Multiset check: GPU must have no duplicate action IDs
        gpu_slice = gpu_ids[i, :cnt].tolist()
        assert len(gpu_slice) == len(set(gpu_slice)), (
            f"steps={steps} env {i}: GPU has duplicate actions "
            f"{[a for a in set(gpu_slice) if gpu_slice.count(a) > 1]}"
        )

        gpu_sorted = np.sort(gpu_ids[i, :cnt])
        cpu_sorted = np.sort(cpu.astype(np.int32))
        np.testing.assert_array_equal(
            gpu_sorted, cpu_sorted,
            err_msg=(
                f"steps={steps} env {i}: action set mismatch\n"
                f"  GPU only: {np.setdiff1d(gpu_sorted, cpu_sorted)}\n"
                f"  CPU only: {np.setdiff1d(cpu_sorted, gpu_sorted)}"
            ),
        )


def test_large_batch_N256() -> None:
    """N=256 mixed-depth batch parity check."""
    N = 256
    envs = []
    for i in range(N):
        rng = random.Random(0x70000 + i)
        env = JunqiEnv()
        env.reset(seed=0x80000 + i)
        _advance(env, (i * 3) % 75, rng)
        envs.append(env)

    cpu_sets = _cpu_aids(envs)
    gpu_ids, gpu_counts = _gpu_aids(envs)

    mismatches = 0
    for i, cpu in enumerate(cpu_sets):
        cnt = int(gpu_counts[i])
        if cnt != len(cpu):
            mismatches += 1
            continue
        if cnt == 0:
            continue
        gpu_sorted = np.sort(gpu_ids[i, :cnt])
        cpu_sorted = np.sort(cpu.astype(np.int32))
        if not np.array_equal(gpu_sorted, cpu_sorted):
            mismatches += 1
    assert mismatches == 0, f"N=256 batch: {mismatches}/{N} envs mismatched"


def test_full_game_trajectory() -> None:
    """Play a random game to completion, checking parity at each step."""
    rng = random.Random(0x99999)
    env = JunqiEnv()
    env.reset(seed=0x99998)

    for step in range(300):
        if env.state.terminated:
            break
        # Parity check at this state
        cpu = env.legal_action_ids()
        state_dict = _pack_state_arrays([env])
        gpu_state = _cuda.DeviceGameStateBatch(1)
        gpu_state.copy_from_host(state_dict)
        acting = np.array([env.state.turn.value], dtype=np.int8)
        ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)

        cnt = int(counts[0])
        assert cnt == len(cpu), (
            f"step {step}: GPU count={cnt}, CPU count={len(cpu)}"
        )
        if cnt == 0:
            break
        gpu_sorted = np.sort(ids[0, :cnt])
        cpu_sorted = np.sort(cpu.astype(np.int32))
        np.testing.assert_array_equal(
            gpu_sorted, cpu_sorted,
            err_msg=f"step {step}: set mismatch",
        )

        # Advance
        env._step_game_only(int(rng.choice(cpu)))


def test_all_seats_across_depths() -> None:
    """For each seat, at several depths, GPU == CPU for that seat's moves."""
    from junqi_core.rules import ALL_SEATS
    for base_seed in (11, 22, 33, 44):
        env = JunqiEnv()
        env.reset(seed=base_seed)
        rng = random.Random(base_seed + 1)
        for depth in (0, 20, 40):
            _advance(env, depth - (0 if depth == 0 else 20), rng)
            state_dict = _pack_state_arrays([env])
            gpu_state = _cuda.DeviceGameStateBatch(1)
            gpu_state.copy_from_host(state_dict)
            for seat in ALL_SEATS:
                cpu = env.state.legal_action_ids(seat)
                acting = np.array([seat.value], dtype=np.int8)
                ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)
                cnt = int(counts[0])
                assert cnt == len(cpu), (
                    f"seed={base_seed} depth={depth} seat={seat}: "
                    f"GPU count={cnt}, CPU count={len(cpu)}"
                )
                if cnt == 0:
                    continue
                gpu_sorted = np.sort(ids[0, :cnt])
                cpu_sorted = np.sort(cpu.astype(np.int32))
                np.testing.assert_array_equal(
                    gpu_sorted, cpu_sorted,
                    err_msg=f"seed={base_seed} depth={depth} seat={seat}: mismatch",
                )
