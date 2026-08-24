"""Per-collect-row advantage filter (Ataraxos Appendix D.4)."""
from __future__ import annotations

import numpy as np

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
