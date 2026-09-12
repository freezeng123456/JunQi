from __future__ import annotations

import os
import socket

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import junqi_cuda as _cuda  # type: ignore[import]

    _HAS_CUDA = torch.cuda.is_available() and _cuda.get_gpu_count() > 0
except ImportError:
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not _HAS_CUDA,
        reason="compact rollout history requires junqi_cuda and a CUDA GPU",
    ),
]

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.rollout_storage import (
    COMPACT_HISTORY_BYTES_PER_TRANSITION,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ddp_compact_history_worker(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        num_envs = 2
        num_steps = 2
        rollout = GpuRollout(num_envs=num_envs, device_id=rank)
        # Seed 800 reaches a public combat-memory feature on the second step.
        # This catches accidental acting-observer-only history snapshots.
        rollout.reset(seed_base=800)
        history = rollout.create_rollout_history(num_steps)

        expected_spatial = []
        expected_global = []
        expected_legal = []
        expected_seats = []
        for step in range(num_steps):
            acting = rollout.turn_torch().clone()
            spatial, global_ = (
                rollout.build_acting_seat_observation_torch(acting)
            )
            legal = rollout.legal_mask_canonical_torch_device(acting)
            history.snapshot(rollout.state, acting, step)
            expected_spatial.append(spatial.to(torch.bfloat16).clone())
            expected_global.append(global_.to(torch.bfloat16).clone())
            expected_legal.append(legal.clone())
            expected_seats.append(acting.clone())

            actions = legal.to(torch.int8).argmax(dim=-1).to(torch.int32)
            result = rollout.step_device_torch(actions, acting)
            rollout.update_beliefs_device(result, acting)
            rollout.reset_terminated_device(seed=9000 + rank * 100 + step)

        indices = torch.arange(
            num_envs * num_steps,
            device=f"cuda:{rank}",
            dtype=torch.int64,
        )
        seats = torch.stack(expected_seats).reshape(-1)
        spatial_actual, global_actual, legal_actual = history.reconstruct(
            indices,
            seats,
            dtype=torch.bfloat16,
        )
        assert spatial_actual.device.index == rank
        assert torch.equal(
            spatial_actual,
            torch.stack(expected_spatial).reshape_as(spatial_actual),
        )
        assert torch.equal(
            global_actual,
            torch.stack(expected_global).reshape_as(global_actual),
        )
        assert torch.equal(
            legal_actual,
            torch.stack(expected_legal).reshape_as(legal_actual),
        )

        # A collective after reconstruction proves both ranks reached the same
        # point without sharing history allocations or deadlocking NCCL.
        reached = torch.tensor([1], device=f"cuda:{rank}")
        dist.all_reduce(reached)
        assert int(reached.item()) == world_size
    finally:
        dist.destroy_process_group()


def test_compact_history_reconstructs_observation_and_legal_mask() -> None:
    num_envs = 4
    num_steps = 4
    rollout = GpuRollout(num_envs=num_envs)
    rollout.reset(seed_base=77)
    history = rollout.create_rollout_history(num_steps)

    expected_spatial = []
    expected_global = []
    expected_legal = []
    expected_seats = []

    for step in range(num_steps):
        acting = rollout.turn_torch().clone()
        spatial, global_ = rollout.build_acting_seat_observation_torch(acting)
        legal = rollout.legal_mask_canonical_torch_device(acting)

        history.snapshot(rollout.state, acting, step)
        expected_spatial.append(spatial.to(torch.bfloat16).clone())
        expected_global.append(global_.to(torch.bfloat16).clone())
        expected_legal.append(legal.clone())
        expected_seats.append(acting.clone())

        actions = legal.to(torch.int8).argmax(dim=-1).to(torch.int32)
        result = rollout.step_device_torch(actions, acting)
        rollout.update_beliefs_device(result, acting)
        rollout.reset_terminated_device(seed=1000 + step)

    indices = torch.tensor(
        [0, num_envs + 1, 2 * num_envs + 2, 3 * num_envs + 3],
        device="cuda",
        dtype=torch.int64,
    )
    all_seats = torch.stack(expected_seats).reshape(-1)
    reconstructed = history.reconstruct(
        indices,
        all_seats.index_select(0, indices),
        dtype=torch.bfloat16,
    )
    spatial_actual, global_actual, legal_actual = reconstructed

    spatial_expected = (
        torch.stack(expected_spatial)
        .reshape(
            num_steps * num_envs,
            *expected_spatial[0].shape[1:],
        )
        .index_select(0, indices)
    )
    global_expected = (
        torch.stack(expected_global)
        .reshape(num_steps * num_envs, -1)
        .index_select(0, indices)
    )
    legal_expected = (
        torch.stack(expected_legal)
        .reshape(num_steps * num_envs, -1)
        .index_select(0, indices)
    )

    assert torch.equal(spatial_actual, spatial_expected)
    assert torch.equal(global_actual, global_expected)
    assert torch.equal(legal_actual, legal_expected)
    assert history.history_bytes == (
        num_steps * num_envs * COMPACT_HISTORY_BYTES_PER_TRANSITION
    )


@pytest.mark.skipif(
    not _HAS_CUDA or _cuda.get_gpu_count() < 2,
    reason="two-GPU compact-history DDP test requires at least two CUDA GPUs",
)
def test_compact_history_is_independent_across_two_ddp_ranks() -> None:
    world_size = 2
    mp.spawn(
        _ddp_compact_history_worker,
        args=(world_size, _free_port()),
        nprocs=world_size,
        join=True,
    )


def test_compressed_history_retains_all_features_through_long_play_and_reuse():
    """Full input equivalence, including the derived reverse capture channels."""
    from junqi_rl.networks.combat_features import CombatOutcomeHead
    num_envs, num_steps = 4, 128
    world = GpuRollout(num_envs=num_envs)
    world.reset(seed_base=800)
    history = world.create_rollout_history(num_steps)
    head = CombatOutcomeHead().cuda()
    spatial_rows, global_rows, legal_rows, seats, chosen, features = [], [], [], [], [], []
    generator = torch.Generator(device='cuda').manual_seed(912)
    for step in range(num_steps):
        acting = world.turn_torch().clone()
        sp, gl = world.build_acting_seat_observation_torch(acting)
        legal = world.legal_mask_canonical_torch_device(acting)
        action = torch.multinomial(legal.float(), 1, generator=generator).squeeze(1).int()
        spatial_rows.append(sp.clone())
        global_rows.append(gl.clone())
        legal_rows.append(legal.clone())
        seats.append(acting)
        chosen.append(action)
        features.append(head.for_actions(sp, action, legal).clone())
        history.snapshot(world.state, acting, step)
        result = world.step_device_torch(action, acting)
        world.update_beliefs_device(result, acting)
        world.reset_terminated_device(seed=9900 + step)
    expected = [torch.cat(x) for x in (spatial_rows, global_rows, legal_rows)]
    all_seats, actions, expected_features = map(torch.cat, (seats, chosen, features))
    from junqi_core.observation import CHANNEL_LAYOUT
    reverse = expected[0][:, CHANNEL_LAYOUT['cm_eaten_by_pid']]
    assert reverse.count_nonzero() > 0, 'trace must exercise reverse capture memory'
    order = torch.randperm(num_envs * num_steps, device='cuda', generator=generator)
    # Grow, shrink, duplicate and revisit samples to catch stale replay rows.
    for indices in (order[:32], order[:5], order[:64], order[:1].expand(9), order[-31:]):
        sp, gl, legal = history.reconstruct(indices, all_seats[indices], dtype=torch.float32)
        for actual, reference in zip((sp, gl, legal), expected):
            torch.testing.assert_close(actual, reference[indices], rtol=0, atol=0)
        torch.testing.assert_close(head.for_actions(sp, actions[indices], legal),
                                   expected_features[indices], rtol=0, atol=0)
    assert history.history_bytes == num_envs * num_steps * COMPACT_HISTORY_BYTES_PER_TRANSITION
