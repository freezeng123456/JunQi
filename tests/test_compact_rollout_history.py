from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

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
