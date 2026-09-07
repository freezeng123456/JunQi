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


def test_evaluation_pool_does_not_replace_training_pool() -> None:
    """A fixed eval pool must not become the training setup distribution."""
    import junqi_rl.gpu_rollout as gpu_rollout_mod

    N = 8
    # The device pool is process-global and uploaded once, so force a rebuild
    # to make this test independent of which rollout was constructed first.
    gpu_rollout_mod._reset_pool_uploaded = False
    gpu_rollout_mod._training_setup_pool = None

    try:
        rollout = GpuRollout(
            num_envs=N, mixed_setup=True, mixed_own_team_styles=("T",),
        )
        rollout.reset(seed_base=0)

        def own_lineups() -> set[tuple[int, ...]]:
            host = rollout.state.copy_to_host()
            per_seat = host["piece_type_arr"].reshape(N, 4, 30)
            return {tuple(row) for row in per_seat[:, 0, :]}

        def restart_every_env(seed: int) -> None:
            rollout.state.copy_termination_from_host(
                np.ones(N, dtype=bool),
                np.zeros(N, dtype=np.int8),
                np.zeros(N, dtype=bool),
            )
            rollout.reset_terminated_device(seed=seed)

        restart_every_env(1)
        assert len(own_lineups()) == 1, "mixed_setup should fix the own-team lineup"

        rollout.upload_fixed_evaluation_setup_pool(seed=20_260_817)
        restart_every_env(2)
        assert len(own_lineups()) > 1, "eval pool should be uniform for all seats"

        assert rollout.restore_training_setup_pool() > 0
        restart_every_env(3)
        assert len(own_lineups()) == 1, "training pool must survive an evaluation"
    finally:
        gpu_rollout_mod._reset_pool_uploaded = False
        gpu_rollout_mod._training_setup_pool = None
        GpuRollout(num_envs=1)

@pytest.mark.parametrize("device_step", [False, True])
def test_custom_move_limit_is_per_rollout_and_survives_reset(device_step):
    import torch

    short = GpuRollout(num_envs=2, max_num_moves=1)
    longer = GpuRollout(num_envs=2, max_num_moves=3)
    for world in (short, longer):
        world.reset(seed_base=100)

    def step(world):
        if device_step:
            acting = world.turn_torch().clone()
            mask = world.legal_mask_canonical_torch_device(acting)
            actions = mask.long().argmax(-1).to(torch.int32)
            world.step_device_torch(actions, acting)
        else:
            acting = world.state.copy_to_host()["turn"].reshape(-1)
            ids, counts = world.legal_actions_dense(acting)
            assert (counts > 0).all()
            world.step(ids[:, 0].copy())

    step(short)
    step(longer)
    assert short.read_termination()["draw"].all()
    assert not longer.read_termination()["terminated"].any()
    step(longer)
    step(longer)
    assert longer.read_termination()["draw"].all()
    for world in (short, longer):
        world.reset_terminated_device(seed=103)
        assert not world.read_termination()["terminated"].any()
    step(short)
    step(longer)
    assert short.read_termination()["draw"].all()
    assert not longer.read_termination()["terminated"].any()


def test_gpu_evaluation_honors_move_limit_and_cache():
    import torch
    from junqi_rl.analysis.random_eval import evaluate_vs_random_gpu, evaluate_head_to_head_gpu

    class Policy:
        def eval(self):
            return self

        def act_greedy(self, sp, gl, mask):
            return mask.long().argmax(-1)

    for limit in (1, 3, 1):
        for evaluator, args in [
            (evaluate_vs_random_gpu, (Policy(),)),
            (evaluate_head_to_head_gpu, (Policy(), Policy())),
        ]:
            metrics = evaluator(
                *args, num_envs=2, num_games=2, max_moves=limit, device="cuda", seed=101
            )
            prefix = "eval" if evaluator is evaluate_vs_random_gpu else "h2h"
            assert metrics[f"{prefix}/draw_rate"] == 1
            assert metrics[f"{prefix}/ongoing_rate"] == 0
            assert metrics[f"{prefix}/avg_game_len"] == limit
