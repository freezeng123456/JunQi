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


def test_dark_missing_belief_fallback_keeps_teammate_unknown():
    """Exercise native I5 recovery, not CPU-uploaded posterior parity."""
    import torch
    from junqi_core.observation import CHANNEL_LAYOUT

    rollout = GpuRollout(num_envs=1)
    rollout.reset(seed_base=42)
    rollout.upload_beliefs(np.zeros((1, 4, 12, 289), dtype=np.float32))
    acting = rollout.turn_torch().clone()
    mask = rollout.legal_mask_canonical_torch_device(acting)
    # Policy masks and device stepping both use canonical compact IDs.
    action_id = int(mask[0].nonzero()[0, 0].item())
    result = rollout.step_device_torch(torch.tensor([action_id], dtype=torch.int32, device='cuda'), acting)
    rollout.update_beliefs_device(result, acting)
    spatial, _ = rollout.build_all_seat_observations()
    teammate = spatial[0, 0, CHANNEL_LAYOUT['prob_teammate']]
    occupied = teammate.sum(axis=0) > 0
    assert occupied.any()
    assert np.all(teammate[:, occupied].max(axis=0) < 1.0)


def test_native_public_mine_and_engineer_inference_chain():
    """Native step + belief updates must broadcast the entire Q7 inference chain."""
    import torch
    from junqi_core.board import FLAT_TO_COMPACT, NUM_ON_BOARD_CELLS
    from junqi_core.info_model import BeliefTensor
    from junqi_core.observation import CHANNEL_LAYOUT, build_observation
    from junqi_core.rotation import world_to_canonical
    from junqi_core.rules import Seat
    from junqi_core.state import Action
    from tests.test_feature_information_boundaries import public_mine_sequence
    from tests.test_gpu_combat_memory_parity import _push_state

    state, actions = public_mine_sequence()
    rollout = GpuRollout(num_envs=1)
    rollout.reset(seed_base=42)
    rollout.state = _push_state(BatchedGameState.from_game_states([state]))
    beliefs = {s: BeliefTensor.initial(state, s) for s in Seat}
    initial = np.zeros((1, 4, 12, 289), dtype=np.float32)
    for observer, belief in beliefs.items():
        for (x, y), vector in belief.probs.items():
            initial[0, observer.value, :, y * 17 + x] = vector
    rollout.upload_beliefs(initial)

    for seat, src, dst in actions:
        sc = world_to_canonical(*src, seat)
        dc = world_to_canonical(*dst, seat)
        action_id = (int(FLAT_TO_COMPACT[sc[1] * 17 + sc[0]]) * NUM_ON_BOARD_CELLS
                     + int(FLAT_TO_COMPACT[dc[1] * 17 + dc[0]]))
        acting = torch.tensor([seat.value], device='cuda', dtype=torch.int8)
        result = rollout.step_device_torch(
            torch.tensor([action_id], device='cuda', dtype=torch.int32), acting)
        rollout.update_beliefs_device(result, acting)
        after, cpu_result = state.step(Action(seat=seat, src=src, dst=dst))
        for belief in beliefs.values():
            belief.update(state, after, cpu_result)
        state = after
        spatial, global_ = rollout.build_all_seat_observations()
        for observer, belief in beliefs.items():
            expected = build_observation(state, belief, observer)
            for name in ('piece_own', 'prob_teammate', 'belief_left_side', 'belief_right_side'):
                sl = CHANNEL_LAYOUT[name]
                np.testing.assert_allclose(spatial[0, observer.value, sl], expected.spatial[sl], atol=1e-6)
            np.testing.assert_allclose(global_[0, observer.value], expected.global_, atol=1e-5)
    pid = state.pieces[(1, 7)].piece_id
    assert rollout.state.cm_copy_to_host()['is_gongb'].reshape(1, 4, 120)[0, :, pid].all()


def test_beliefs_and_observer_mapping_are_owned_by_each_rollout():
    import torch
    first = GpuRollout(num_envs=2)
    first.reset(seed_base=55)
    expected = tuple(x.copy() for x in first.build_all_seat_observations())
    rules = first.rule_beliefs_torch().clone()
    for n in (1, 3, 2):
        other = GpuRollout(num_envs=n)
        other.reset(seed_base=900)
        other.set_observer_seats(np.tile(np.array([3, 2, 1, 0], dtype=np.int8), (n, 1)))
        other.upload_beliefs(np.zeros((n, 4, 12, 289), dtype=np.float32))
        other.build_all_seat_observations()
        actual = first.build_all_seat_observations()
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        torch.testing.assert_close(first.rule_beliefs_torch(), rules, rtol=0, atol=0)


def test_bulk_reset_clears_history_and_combat_memory():
    import torch
    used = GpuRollout(num_envs=2)
    used.reset(seed_base=33)
    used.rule_beliefs_torch()  # both soft and independent rule buffers reset
    for _ in range(40):
        acting = used.turn_torch().clone()
        mask = used.legal_mask_canonical_torch_device(acting)
        action = mask.long().argmax(-1).to(torch.int32)
        result = used.step_device_torch(action, acting)
        used.update_beliefs_device(result, acting)
    used.reset(seed_base=81)
    fresh = GpuRollout(num_envs=2)
    fresh.reset(seed_base=81)
    for a, b in zip(used.build_all_seat_observations(), fresh.build_all_seat_observations()):
        np.testing.assert_array_equal(a, b)
    for key, value in used.state.cm_copy_to_host().items():
        np.testing.assert_array_equal(value, fresh.state.cm_copy_to_host()[key])
    torch.testing.assert_close(used.rule_beliefs_torch(), fresh.rule_beliefs_torch(), rtol=0, atol=0)


def test_native_neural_refresh_keeps_public_mine_then_engineer_facts():
    import torch
    from junqi_core.board import FLAT_TO_COMPACT
    from junqi_core.info_model import BeliefTensor, TRACKED_TYPES
    from junqi_core.rotation import world_to_canonical
    from junqi_core.rules import PieceType, Seat
    from junqi_rl.belief.inference import refresh_beliefs_neural
    from tests.test_belief_refresh_constraints import ConstantNet
    from tests.test_feature_information_boundaries import public_mine_sequence
    from tests.test_gpu_combat_memory_parity import _push_state

    state, actions = public_mine_sequence()
    world = GpuRollout(num_envs=1)
    world.state = _push_state(BatchedGameState.from_game_states([state]))
    prior = np.zeros((1, 4, 12, 289), dtype=np.float32)
    for seat in Seat:
        belief = BeliefTensor.initial(state, seat)
        for (x, y), vector in belief.probs.items():
            prior[0, seat.value, :, y*17+x] = vector
    world.upload_beliefs(prior)
    # Establish the independent rule buffer before injecting neural certainty.
    refresh_beliefs_neural(world, ConstantNet(0).cuda(), empty_cache=False)
    for index, (seat, src, dst) in enumerate(actions):
        sx, sy = world_to_canonical(*src, seat)
        dx, dy = world_to_canonical(*dst, seat)
        action_id = int(FLAT_TO_COMPACT[sy*17+sx])*129 + int(FLAT_TO_COMPACT[dy*17+dx])
        acting = torch.tensor([seat.value], device='cuda', dtype=torch.int8)
        result = world.step_device_torch(torch.tensor([action_id], device='cuda', dtype=torch.int32), acting)
        world.update_beliefs_device(result, acting)
        refresh_beliefs_neural(world, ConstantNet().cuda(), empty_cache=False)
        rule = world.rule_beliefs_torch().cpu().numpy()
        soft = world._beliefs
        assert np.count_nonzero(soft[rule == 0]) == 0
        kind = PieceType.GONGB if index == 2 else PieceType.DILEI
        assert np.all(soft[0, :, TRACKED_TYPES.index(kind), 7*17+1] == 1)


def test_actor_only_observation_does_not_allocate_all_seat_output():
    world = GpuRollout(num_envs=2)
    world.reset(seed_base=801)
    world.build_acting_seat_observation_torch(world.turn_torch())
    assert world._obs_all is None
    expected = world.build_all_seat_observations()
    assert world._obs_all is not None
    current = world.build_all_seat_observations()
    for a, b in zip(expected, current):
        np.testing.assert_array_equal(a, b)
