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
        rollout.reset(seed_base=700 + rank * 100)
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
