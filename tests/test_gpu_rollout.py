"""tests/test_gpu_rollout.py — GpuRollout smoke + parity tests (Phase 1b)."""

from __future__ import annotations

import random

import numpy as np
import pytest

try:
    import junqi_cuda as _cuda  # type: ignore[import]
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda extension not available or no CUDA-capable GPU",
)

from junqi_core.batched_state import BatchedGameState
from junqi_core.observation import OBS_CHANNELS
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from junqi_rl.gpu_rollout import GpuRollout


def _new_game(seed: int) -> GameState:
    rng = random.Random(seed)
    return GameState.new_game(generate_random_setup(rng))


def _matching_cpu_batch(seed_base: int, N: int) -> BatchedGameState:
    states = [_new_game(seed_base + i) for i in range(N)]
    return BatchedGameState.from_game_states(states)


def _pick_actions_from_legal_ids(
    ids: np.ndarray, counts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Pick a uniformly-random legal action per env from (N, 512) dense list."""
    N = ids.shape[0]
    out = np.zeros(N, dtype=np.int32)
    has = counts > 0
    if has.any():
        idxs = rng.integers(0, 1 << 30, size=N) % np.maximum(counts, 1)
        picked = ids[np.arange(N), idxs.astype(np.int32)]
        out[has] = picked[has]
    return out


def test_rollout_reset_roundtrip() -> None:
    """reset() produces a device state that round-trips back to the same CPU state."""
    N = 8
    rollout = GpuRollout(num_envs=N)
    rollout.reset(seed_base=0)

    # Pull from device and compare with CPU-constructed reference.
    device = rollout.state.copy_to_host()
    term = rollout.state.copy_termination_to_host()
    b_ref = _matching_cpu_batch(0, N)

    # Reshape device flat views to match BatchedGameState.
    assert np.array_equal(
        device["piece_type_arr"].reshape((N, 120)), b_ref.piece_type_arr
    )
    assert np.array_equal(
        device["alive"].reshape((N, 120)), b_ref.alive
    )
    assert np.array_equal(
        device["cell_piece_id"].reshape((N, 289)), b_ref.cell_piece_id
    )
    assert np.array_equal(device["turn"].reshape((N,)), b_ref.turn)
    assert np.array_equal(device["zobrist"].reshape((N,)), b_ref.zobrist)
    assert np.array_equal(term["terminated"], b_ref.terminated)
    assert np.array_equal(term["draw"],       b_ref.draw)


def test_rollout_step_matches_cpu() -> None:
    """A sequence of steps produces the same state as CPU BatchedGameState."""
    N = 8
    rollout = GpuRollout(num_envs=N)
    rollout.reset(seed_base=123)
    b_cpu = _matching_cpu_batch(123, N)

    rng = np.random.default_rng(0xFEED)
    for _ in range(20):
        if b_cpu.terminated.all():
            break
        ids = b_cpu.legal_action_ids_batch()
        actions = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if b_cpu.terminated[i] or ids[i].size == 0:
                continue
            actions[i] = int(rng.choice(ids[i]))

        cpu_result = b_cpu.step_batch(actions)
        gpu_result = rollout.step(actions)
        # Results must match.
        for key in ("valid", "event", "terminated", "winner_team", "draw", "flag_captured"):
            assert np.array_equal(getattr(cpu_result, key), gpu_result[key]), (
                f"result.{key} mismatch"
            )
        # State must match.
        d = rollout.state.copy_to_host()
        term = rollout.state.copy_termination_to_host()
        assert np.array_equal(d["zobrist"].reshape((N,)),  b_cpu.zobrist)
        assert np.array_equal(d["turn"].reshape((N,)),     b_cpu.turn)
        assert np.array_equal(term["terminated"],          b_cpu.terminated)
        assert np.array_equal(term["winner_team"],         b_cpu.winner_team)


def test_rollout_observation_shape() -> None:
    """build_all_seat_observations returns the correct shapes."""
    N = 4
    rollout = GpuRollout(num_envs=N)
    rollout.reset(seed_base=7)
    spatial, global_ = rollout.build_all_seat_observations()
    assert spatial.shape == (N, 4, OBS_CHANNELS, 17, 17)
    assert global_.shape == (N, 4, 28)
    assert spatial.dtype == np.float32
    assert global_.dtype == np.float32


def test_rollout_legal_action_match_cpu() -> None:
    """legal_actions_dense multiset matches CPU BatchedGameState for initial state."""
    N = 4
    rollout = GpuRollout(num_envs=N)
    rollout.reset(seed_base=42)
    b_cpu = _matching_cpu_batch(42, N)

    acting_seats = b_cpu.turn.astype(np.int8, copy=False)
    ids, counts = rollout.legal_actions_dense(acting_seats)

    cpu_ids_batch = b_cpu.legal_action_ids_batch()
    for i in range(N):
        gpu_set = set(ids[i, :int(counts[i])].tolist())
        cpu_set = set(cpu_ids_batch[i].tolist())
        assert gpu_set == cpu_set, f"env {i}: legal-action set mismatch"


def test_rollout_step_and_observation_together() -> None:
    """Full step + observation loop runs without crashing, masks are valid."""
    N = 16
    rollout = GpuRollout(num_envs=N)
    rollout.reset(seed_base=1)

    b_cpu = _matching_cpu_batch(1, N)  # just for action picking
    rng = np.random.default_rng(0xABCD)

    for _ in range(10):
        ids = b_cpu.legal_action_ids_batch()
        actions = np.zeros(N, dtype=np.int32)
        for i in range(N):
            if b_cpu.terminated[i] or ids[i].size == 0:
                continue
            actions[i] = int(rng.choice(ids[i]))
        b_cpu.step_batch(actions)
        rollout.step(actions)
        sp, gl = rollout.build_all_seat_observations()
        assert np.isfinite(sp).all()
        assert np.isfinite(gl).all()
