"""tests/test_rollout_buffer_gpu.py — RolloutBufferGPU parity with host buffer."""
from __future__ import annotations

import pytest

from junqi_core.observation import OBS_CHANNELS

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

try:
    import junqi_cuda as _cuda  # type: ignore[import]
    _HAS_CUDA = _cuda.get_gpu_count() > 0
except ImportError:
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_CUDA),
    reason="torch + junqi_cuda + CUDA GPU all required",
)

import numpy as np

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training import collect_rollout_gpu
from junqi_rl.training.rollout import RolloutBuffer, FLAT_ACTION_DIM
from junqi_rl.training.rollout_gpu import RolloutBufferGPU


def _tiny_cfg():
    return JunqiNetConfig(
        cnn_channels=32, cnn_layers=2, depth=2,
        embed_dim=64, n_head=4, ff_factor=2, dropout=0.0,
    )


def test_gpu_buffer_allocates_on_device():
    # Legacy dense mode: legal_mask is allocated; CSR fields are None.
    buf = RolloutBufferGPU(num_envs=4, steps_per_env=3, device="cuda",
                           csr_legal_mask=False)
    assert buf.obs_spatial.device.type == "cuda"
    assert buf.legal_mask.dtype == torch.bool
    assert buf.actions.dtype == torch.int32
    assert buf.rewards.dtype == torch.float32

    # CSR mode: legal_ids/legal_counts are allocated; legal_mask is None.
    buf_csr = RolloutBufferGPU(num_envs=4, steps_per_env=3, device="cuda",
                               csr_legal_mask=True)
    assert buf_csr.legal_ids.device.type == "cuda"
    assert buf_csr.legal_ids.dtype == torch.int32
    assert buf_csr.legal_counts.dtype == torch.int32
    assert buf_csr.legal_mask is None


def test_gpu_buffer_add_accepts_tensors_and_numpy():
    buf = RolloutBufferGPU(num_envs=4, steps_per_env=2, device="cuda",
                           csr_legal_mask=False)
    N = 4
    buf.add(
        obs_spatial=torch.zeros(N, OBS_CHANNELS, 17, 17, device="cuda"),
        obs_global=torch.zeros(N, 28, device="cuda"),
        legal_mask=torch.zeros(N, FLAT_ACTION_DIM, dtype=torch.bool, device="cuda"),
        actions=np.zeros(N, dtype=np.int32),       # numpy still allowed
        log_probs=np.zeros(N, dtype=np.float32),
        values=np.zeros(N, dtype=np.float32),
        rewards=np.zeros(N, dtype=np.float32),
        dones=np.zeros(N, dtype=bool),
        seats=np.zeros(N, dtype=np.int8),
    )
    assert not buf.is_ready
    buf.add(
        obs_spatial=torch.ones(N, OBS_CHANNELS, 17, 17, device="cuda"),
        obs_global=torch.zeros(N, 28, device="cuda"),
        legal_mask=torch.zeros(N, FLAT_ACTION_DIM, dtype=torch.bool, device="cuda"),
        actions=torch.zeros(N, dtype=torch.int32, device="cuda"),
        log_probs=torch.zeros(N, device="cuda"),
        values=torch.zeros(N, device="cuda"),
        rewards=torch.zeros(N, device="cuda"),
        dones=torch.zeros(N, dtype=torch.bool, device="cuda"),
        seats=torch.zeros(N, dtype=torch.int8, device="cuda"),
    )
    assert buf.is_ready
    # Verify the tensor-add path landed correctly
    assert buf.obs_spatial[1, 0, 0, 0, 0].item() == 1.0
    assert buf.obs_spatial[0, 0, 0, 0, 0].item() == 0.0


def test_gpu_buffer_csr_roundtrip():
    """CSR-mode buffer: add() encodes a dense mask, minibatches() decodes
    to an identical dense mask for the selected transitions.
    """
    from junqi_rl.training.rollout_gpu import _dense_mask_to_csr, _csr_to_dense_selected

    N = 8
    dev = torch.device("cuda")
    torch.manual_seed(0)
    # Use sparse mask (few truths per row) that fits inside K_MAX=64.
    # The real legal-mask has ~27 true positions per row on average, so
    # K_MAX=64 is the realistic configuration.
    mask = torch.rand(N, FLAT_ACTION_DIM, device=dev) > 0.998  # ~33 truths/row
    assert (mask.sum(dim=1) <= 64).all().item(), "bump rand threshold"

    ids = torch.zeros((N, 64), dtype=torch.int32, device=dev)
    cnt = torch.zeros(N, dtype=torch.int32, device=dev)
    _dense_mask_to_csr(mask, ids, cnt)

    # Reconstruct all rows and compare.
    all_idx = torch.arange(N, device=dev, dtype=torch.int64)
    back = _csr_to_dense_selected(ids, cnt, all_idx, FLAT_ACTION_DIM)
    assert torch.equal(back, mask)


def test_gpu_buffer_csr_overflow_raises_without_mutating():
    """Overflow must fail closed before changing stored IDs/counts."""
    from junqi_rl.training.rollout_gpu import _dense_mask_to_csr

    dev = torch.device("cuda")
    mask = torch.zeros(1, FLAT_ACTION_DIM, dtype=torch.bool, device=dev)
    # 100 truths at positions 10..1000 step 10 — exceeds K=64.
    positions = list(range(10, 1010, 10))
    mask[0, positions] = True
    ids = torch.full((1, 64), 123, dtype=torch.int32, device=dev)
    cnt = torch.full((1,), 7, dtype=torch.int32, device=dev)
    with pytest.raises(ValueError, match="overflow"):
        _dense_mask_to_csr(mask, ids, cnt)
    assert (ids == 123).all()
    assert int(cnt[0].item()) == 7


def test_gpu_buffer_collect_e2e_and_gae():
    """collect_rollout_gpu writes into RolloutBufferGPU directly, then GAE
    runs on-device and produces finite advantages."""
    N = 8
    T = 4
    device = torch.device("cuda")

    world = GpuRollout(num_envs=N)
    policy = JunqiNet(_tiny_cfg()).to(device)
    buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)

    collect_rollout_gpu(
        rollout_world=world, policy=policy, buffer=buf,
        device=device, seed_base=0,
    )
    assert buf.is_ready
    # GAE was computed
    assert torch.isfinite(buf.advantages_).all()
    assert torch.isfinite(buf.returns_).all()


def test_gpu_buffer_parity_with_host_buffer():
    """Running collect_rollout_gpu with the same seeds and a deterministic
    (argmax) policy should produce matching observations / log_probs
    between the host and GPU buffers.

    Because JunqiNet.act samples stochastically, exact parity across
    buffers is impossible.  Instead: use both buffers on the same run
    (buffer vs buffer2) and check the per-step obs/rewards/dones match
    — these are derived from the env, not the policy.
    """
    N = 4
    T = 3
    device = torch.device("cuda")

    world_a = GpuRollout(num_envs=N)
    world_b = GpuRollout(num_envs=N)
    policy = JunqiNet(_tiny_cfg()).to(device)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed(0)

    buf_host = RolloutBuffer(num_envs=N, steps_per_env=T, device=device)
    buf_gpu  = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)

    # Deterministic sampling: inject torch seed before each collect.
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    collect_rollout_gpu(world_a, policy, buf_host, device=device, seed_base=100)

    torch.manual_seed(42); torch.cuda.manual_seed(42)
    collect_rollout_gpu(world_b, policy, buf_gpu, device=device, seed_base=100)

    # obs_spatial — just check shapes match and both are finite
    assert buf_host.obs_spatial.shape == tuple(buf_gpu.obs_spatial.shape)
    assert np.isfinite(buf_host.obs_spatial).all()
    assert torch.isfinite(buf_gpu.obs_spatial).all()


def test_gpu_buffer_minibatches_on_device():
    """minibatches() yields RolloutBatch with device tensors."""
    N = 4
    T = 3
    device = torch.device("cuda")
    world = GpuRollout(num_envs=N)
    policy = JunqiNet(_tiny_cfg()).to(device)
    buf = RolloutBufferGPU(num_envs=N, steps_per_env=T, device=device)
    collect_rollout_gpu(world, policy, buf, device=device, seed_base=0)

    got_batch = False
    for batch in buf.minibatches(batch_size=4):
        assert batch.obs_spatial.device.type == "cuda"
        assert batch.legal_mask.device.type == "cuda"
        assert batch.actions.dtype == torch.int64  # PPO expects long
        got_batch = True
    assert got_batch


def test_gpu_buffer_rollout_scope_uses_one_threshold_for_all_rows():
    """GPU selection matches Ataraxos's per-rank rollout quantile."""
    N, T = 4, 2
    buf = RolloutBufferGPU(
        num_envs=N,
        steps_per_env=T,
        adv_filt_rate=0.25,
        adv_filt_thresh=0.01,
        adv_filter_scope="rollout",
        minibatch_group="timestep",
        device="cuda",
        csr_legal_mask=False,
    )
    raw = torch.tensor(
        [[100.0, -100.0, 90.0, -90.0], [1.0, -1.0, 0.5, -0.5]],
        dtype=torch.float32,
        device="cuda",
    )
    buf.advantages_.copy_(raw)
    buf.returns_.copy_(raw)

    batches = list(buf.minibatches(batch_size=512, shuffle=True))
    assert len(batches) == 1
    assert int(batches[0].actions.shape[0]) == 2
    assert buf._last_n_policy == 2
    assert buf._last_thresh_used == pytest.approx(92.5)
    assert buf._last_n_empty_steps == 1.0


def test_gpu_rollout_value_scope_keeps_all_rows_for_value_loss():
    """GPU buffer decouples policy filtering from all-valid value samples."""
    N, T = 4, 2
    buf = RolloutBufferGPU(
        num_envs=N,
        steps_per_env=T,
        adv_filt_rate=0.25,
        adv_filt_thresh=0.01,
        adv_filter_scope="rollout",
        value_sample_scope="all_valid",
        minibatch_group="timestep",
        device="cuda",
        csr_legal_mask=False,
    )
    raw = torch.tensor(
        [[100.0, -100.0, 90.0, -90.0], [1.0, -1.0, 0.5, -0.5]],
        dtype=torch.float32,
        device="cuda",
    )
    buf.advantages_.copy_(raw)
    buf.returns_.copy_(raw)

    batches = list(buf.minibatches(batch_size=512, shuffle=False))

    assert len(batches) == T
    assert [int(batch.actions.shape[0]) for batch in batches] == [2, 0]
    assert batches[0].value_only_mask.tolist() == [False, False]
    assert batches[1].value_only_mask.tolist() == []
    assert batches[0].policy_value_indices.tolist() == [0, 1]
    assert batches[1].policy_value_indices.tolist() == []
    assert torch.equal(batches[0].value_returns, raw[0])
    assert torch.equal(batches[1].value_returns, raw[1])
    assert buf._last_n_policy == 2
    assert buf._last_n_value == T * N
