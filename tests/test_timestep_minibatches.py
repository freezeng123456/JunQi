"""Timestep batching with row-local and rollout-global advantage filters."""
from __future__ import annotations

import numpy as np
import pytest

from junqi_rl.training.rollout import timestep_keep_env_indices


def test_keeps_topk_within_each_row():
    # T=4, N=8, rate=0.25 -> k=2 per row.
    abs_adv = np.array(
        [
            [10.0, 9.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.2, 0.3, 8.0, 7.0, 0.4, 0.4, 0.4, 0.4],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    rows = timestep_keep_env_indices(abs_adv, rate=0.25, thresh=0.01)
    assert len(rows) == 4
    assert set(rows[0].tolist()) == {0, 1}
    assert set(rows[1].tolist()) == {2, 3}
    assert set(rows[2].tolist()) == {6, 7}
    assert rows[3].size == 2


def test_thresh_can_empty_a_row():
    abs_adv = np.array(
        [
            [10.0, 9.0, 1.0, 1.0],
            [0.001, 0.001, 0.001, 0.001],
        ],
        dtype=np.float64,
    )
    rows = timestep_keep_env_indices(abs_adv, rate=0.5, thresh=0.01)
    assert set(rows[0].tolist()) == {0, 1}
    assert rows[1].size == 0


def test_own_mask_excludes_envs():
    abs_adv = np.ones((1, 4)) * 5.0
    own = np.array([[True, False, True, False]])
    rows = timestep_keep_env_indices(
        abs_adv, rate=1.0, thresh=0.01, own_mask=own,
    )
    assert set(rows[0].tolist()) == {0, 2}


def test_does_not_change_row_count():
    rng = np.random.default_rng(0)
    abs_adv = rng.random((7, 16)) + 0.05
    rows = timestep_keep_env_indices(abs_adv, rate=0.25, thresh=0.01)
    assert len(rows) == 7
    assert all(r.size == 4 for r in rows)


def test_cpu_buffer_timestep_emits_one_batch_per_row():
    from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
    from junqi_rl.training.rollout import FLAT_ACTION_DIM, RolloutBuffer

    T, N = 4, 8
    buf = RolloutBuffer(
        num_envs=N,
        steps_per_env=T,
        adv_filt_rate=0.25,
        adv_filt_thresh=0.01,
        minibatch_group="timestep",
    )
    for _t in range(T):
        buf.add(
            obs_spatial=np.zeros((N, OBS_CHANNELS, 17, 17), dtype=np.float32),
            obs_global=np.zeros((N, OBS_GLOBAL_DIMS), dtype=np.float32),
            legal_mask=np.ones((N, FLAT_ACTION_DIM), dtype=bool),
            actions=np.zeros(N, dtype=np.int32),
            log_probs=np.zeros(N, dtype=np.float32),
            values=np.zeros(N, dtype=np.float32),
            rewards=np.zeros(N, dtype=np.float32),
            dones=np.zeros(N, dtype=bool),
            seats=np.zeros(N, dtype=np.int8),
        )
    raw = np.zeros((T, N), dtype=np.float32)
    raw[0, 0], raw[0, 1] = 10.0, 9.0
    raw[1, 2], raw[1, 3] = 8.0, 7.0
    raw[2, 6], raw[2, 7] = 6.0, 5.0
    raw[3, :] = 0.5
    buf.advantages_ = raw
    buf.returns_ = raw.copy()
    batches = list(buf.minibatches(batch_size=512, shuffle=True))
    assert len(batches) == T
    sizes = [int(b.actions.shape[0]) for b in batches]
    assert sizes == [2, 2, 2, 2]
    assert buf._last_n_policy == 8
    assert buf._last_n_empty_steps == 0.0


def test_cpu_buffer_rollout_scope_uses_one_threshold_for_all_rows():
    """A weak row is empty instead of receiving its own top-25% quota."""
    from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
    from junqi_rl.training.rollout import FLAT_ACTION_DIM, RolloutBuffer

    T, N = 2, 4
    buf = RolloutBuffer(
        num_envs=N,
        steps_per_env=T,
        adv_filt_rate=0.25,
        adv_filt_thresh=0.01,
        adv_filter_scope="rollout",
        minibatch_group="timestep",
    )
    for _t in range(T):
        buf.add(
            obs_spatial=np.zeros((N, OBS_CHANNELS, 17, 17), dtype=np.float32),
            obs_global=np.zeros((N, OBS_GLOBAL_DIMS), dtype=np.float32),
            legal_mask=np.ones((N, FLAT_ACTION_DIM), dtype=bool),
            actions=np.zeros(N, dtype=np.int32),
            log_probs=np.zeros(N, dtype=np.float32),
            values=np.zeros(N, dtype=np.float32),
            rewards=np.zeros(N, dtype=np.float32),
            dones=np.zeros(N, dtype=bool),
            seats=np.zeros(N, dtype=np.int8),
        )

    # The rollout-wide 0.75 quantile is 92.5. Both survivors are in row 0;
    # row 1 is intentionally empty. A per-row top-25% filter would emit one
    # sample from each row instead.
    raw = np.array(
        [[100.0, -100.0, 90.0, -90.0], [1.0, -1.0, 0.5, -0.5]],
        dtype=np.float32,
    )
    buf.advantages_ = raw
    buf.returns_ = raw.copy()

    batches = list(buf.minibatches(batch_size=512, shuffle=True))
    assert len(batches) == 1
    assert int(batches[0].actions.shape[0]) == 2
    assert buf._last_n_policy == 2
    assert buf._last_thresh_used == pytest.approx(92.5)
    assert buf._last_kept_min == 2.0
    assert buf._last_kept_max == 2.0
    assert buf._last_n_empty_steps == 1.0


def test_cpu_rollout_value_scope_keeps_all_rows_for_value_loss():
    """Policy stays top-25%, while value sees every valid transition."""
    import torch

    from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
    from junqi_rl.training.rollout import FLAT_ACTION_DIM, RolloutBuffer

    T, N = 2, 4
    buf = RolloutBuffer(
        num_envs=N,
        steps_per_env=T,
        adv_filt_rate=0.25,
        adv_filt_thresh=0.01,
        adv_filter_scope="rollout",
        value_sample_scope="all_valid",
        minibatch_group="timestep",
    )
    for _t in range(T):
        buf.add(
            obs_spatial=np.zeros((N, OBS_CHANNELS, 17, 17), dtype=np.float32),
            obs_global=np.zeros((N, OBS_GLOBAL_DIMS), dtype=np.float32),
            legal_mask=np.ones((N, FLAT_ACTION_DIM), dtype=bool),
            actions=np.zeros(N, dtype=np.int32),
            log_probs=np.zeros(N, dtype=np.float32),
            values=np.zeros(N, dtype=np.float32),
            rewards=np.zeros(N, dtype=np.float32),
            dones=np.zeros(N, dtype=bool),
            seats=np.zeros(N, dtype=np.int8),
        )

    raw = np.array(
        [[100.0, -100.0, 90.0, -90.0], [1.0, -1.0, 0.5, -0.5]],
        dtype=np.float32,
    )
    buf.advantages_ = raw
    buf.returns_ = raw.copy()

    batches = list(buf.minibatches(batch_size=512, shuffle=False))

    # Ataraxos policy selection still keeps only the two |A|=100 samples.
    # The value path, however, emits all four samples in both rows. The weak
    # second row is value-only instead of disappearing from the epoch.
    assert len(batches) == T
    assert [int(batch.actions.shape[0]) for batch in batches] == [2, 0]
    assert batches[0].value_only_mask.tolist() == [False, False]
    assert batches[1].value_only_mask.tolist() == []
    assert batches[0].policy_value_indices.tolist() == [0, 1]
    assert batches[1].policy_value_indices.tolist() == []
    assert torch.equal(batches[0].value_returns.cpu(), torch.from_numpy(raw[0]))
    assert torch.equal(batches[1].value_returns.cpu(), torch.from_numpy(raw[1]))
    assert buf._last_n_policy == 2
    assert buf._last_n_value == T * N


def test_gpu_tensor_rollout_selector_is_device_independent():
    """Exercise the GPU buffer's tensor selector without requiring CUDA."""
    import torch

    from junqi_rl.training.rollout_gpu import _rollout_advantage_keep_mask

    raw = torch.tensor(
        [[100.0, -100.0, 90.0, -90.0], [1.0, -1.0, 0.5, -0.5]],
    )
    valid = torch.ones_like(raw, dtype=torch.bool)
    keep, threshold = _rollout_advantage_keep_mask(
        raw,
        valid,
        keep_rate=0.25,
        min_thresh=0.01,
    )

    assert threshold == pytest.approx(92.5)
    assert torch.equal(
        keep,
        torch.tensor(
            [[True, True, False, False], [False, False, False, False]],
        ),
    )
