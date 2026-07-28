"""tests/test_gpu_collector.py — GPU PPO collector parity + helper tests.

Exercises the pure-numpy building blocks of
:mod:`junqi_rl.training.gpu_collector` without requiring torch (since the
training package as a whole depends on torch, we load the collector module
via ``importlib`` to bypass its ``__init__`` side-effects when torch is
absent).  The full :func:`collect_rollout_gpu` loop is tested separately
under a torch-gated marker; here we focus on the GPU-dependent helpers.
"""

from __future__ import annotations

import importlib.util
import os

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

from junqi_rl.action_lut import FLAT_ACTION_DIM, UNROTATE_LUT
from junqi_rl.gpu_rollout import GpuRollout


def _load_gpu_collector():
    """Import ``gpu_collector`` without triggering torch-dependent siblings."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "junqi_rl", "training",
        "gpu_collector.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_jrl_gpu_collector_test", os.path.abspath(path),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_per_seat_terminal_rewards_basic() -> None:
    """+1/-1/0 reward shaping matches the seat→team table."""
    mod = _load_gpu_collector()
    # 4 envs:
    #   0: terminated win (team 0), acting SOUTH(0) → +1
    #   1: terminated win (team 0), acting WEST(1)  → -1
    #   2: terminated draw → 0
    #   3: non-terminal → 0
    term = np.array([True, True, True, False])
    win  = np.array([0, 0, -1, -1], dtype=np.int8)
    draw = np.array([False, False, True, False])
    acting = np.array([0, 1, 2, 3], dtype=np.int8)
    r = mod._per_seat_terminal_rewards(term, win, draw, acting)
    assert r.dtype == np.float32
    assert r.tolist() == [1.0, -1.0, 0.0, 0.0]


def test_per_seat_terminal_rewards_non_terminal_zero() -> None:
    """Non-terminal envs always get 0 regardless of winner/draw arrays."""
    mod = _load_gpu_collector()
    N = 8
    term = np.zeros(N, dtype=bool)      # nobody terminated
    win  = np.full(N, 0, dtype=np.int8) # nonsense data
    draw = np.ones(N, dtype=bool)
    acting = np.arange(N, dtype=np.int8) % 4
    r = mod._per_seat_terminal_rewards(term, win, draw, acting)
    assert (r == 0.0).all()


def test_build_legal_mask_batch_gpu_shape_and_consistency() -> None:
    """Canonical-frame legal mask contains exactly the kernel's action count,
    and un-rotating the set-bits recovers the world-frame action set."""
    mod = _load_gpu_collector()
    N = 8
    r = GpuRollout(num_envs=N)
    r.reset(seed_base=0)

    term = r.read_termination()["terminated"].astype(bool)
    # Use the device turn for acting seats (no env is terminated yet).
    host = r.state.copy_to_host()
    acting = host["turn"].reshape(N).astype(np.int8)

    mask = mod.build_legal_mask_batch_gpu(r, acting, term)
    assert mask.shape == (N, FLAT_ACTION_DIM)
    assert mask.dtype == bool

    # Compare to raw GPU kernel output.
    ids, counts = r.legal_actions_dense(acting)
    for i in range(N):
        k = int(counts[i])
        # Same cardinality (mask is dense, dedup is irrelevant for legal moves).
        assert mask[i].sum() == k, (
            f"env {i}: mask-set={mask[i].sum()} kernel count={k}"
        )
        # Mask ↔ world-frame set: un-rotating the mask's True bits yields
        # the same multiset the kernel returned.
        # Post-compact-refactor: UNROTATE_LUT produces compact-world ids;
        # expand through COMPACT_TO_FLAT to recover world-full ids.
        from junqi_core.board import COMPACT_TO_FLAT, NUM_ON_BOARD_CELLS
        can_set_bits = np.flatnonzero(mask[i]).astype(np.int32)
        compact_world = UNROTATE_LUT[int(acting[i])][can_set_bits]
        src_c = compact_world // NUM_ON_BOARD_CELLS
        dst_c = compact_world %  NUM_ON_BOARD_CELLS
        c2f = np.asarray(COMPACT_TO_FLAT, dtype=np.int64)
        world_recovered = (c2f[src_c] * 289 + c2f[dst_c]).astype(np.int32)
        assert set(world_recovered.tolist()) == set(
            ids[i, :k].tolist()
        ), f"env {i}: rotated set mismatch"


def test_build_legal_mask_batch_gpu_terminated_skipped() -> None:
    """Envs flagged terminated get an all-False mask row."""
    mod = _load_gpu_collector()
    N = 4
    r = GpuRollout(num_envs=N)
    r.reset(seed_base=7)

    host = r.state.copy_to_host()
    acting = host["turn"].reshape(N).astype(np.int8)
    # Mark half the envs as terminated.
    term = np.array([False, True, False, True])
    mask = mod.build_legal_mask_batch_gpu(r, acting, term)

    assert mask[1].sum() == 0
    assert mask[3].sum() == 0
    assert mask[0].sum() > 0
    assert mask[2].sum() > 0


@pytest.mark.skipif(
    not pytest.importorskip("torch", reason="torch not installed"),
    reason="torch not installed",
)
def test_build_legal_mask_torch_parity_with_numpy() -> None:
    """GPU-resident mask builder produces bit-identical results to the
    numpy reference for the same inputs."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    mod = _load_gpu_collector()
    device = torch.device("cuda")

    N = 16
    r = GpuRollout(num_envs=N)
    r.reset(seed_base=333)
    host = r.state.copy_to_host()
    acting = host["turn"].reshape(N).astype(np.int8)
    term = np.array([i % 3 == 0 for i in range(N)])   # mix of terminated

    mask_np = mod.build_legal_mask_batch_gpu(r, acting, term)
    mask_t = mod.build_legal_mask_batch_gpu_torch(r, acting, term, device)
    assert mask_t.device.type == "cuda"
    assert mask_t.shape == mask_np.shape
    assert np.array_equal(mask_t.cpu().numpy(), mask_np)


def test_reset_envs_inplace_clears_terminated_slots() -> None:
    """_reset_envs_inplace overwrites only the selected envs and clears their
    termination flags."""
    mod = _load_gpu_collector()
    N = 6
    r = GpuRollout(num_envs=N)
    r.reset(seed_base=42)

    # Capture initial zobrist for a control group.
    pre = r.state.copy_to_host()["zobrist"].reshape(N).copy()

    # Fake-terminate envs 1 and 4.
    term_mask = np.array([False, True, False, False, True, False])
    r.state.copy_termination_from_host(
        term_mask.astype(bool),
        np.array([-1, 0, -1, -1, 1, -1], dtype=np.int8),
        np.zeros(N, dtype=bool),
    )
    mod._reset_envs_inplace(r, reset_mask=term_mask, seed_base=100)

    post_term = r.state.copy_termination_to_host()
    assert post_term["terminated"].tolist() == [False] * N, (
        "reset slots must have terminated cleared"
    )
    assert post_term["winner_team"].tolist() == [-1] * N
    assert post_term["draw"].tolist() == [False] * N

    # Unreset envs keep their zobrist untouched.
    post_z = r.state.copy_to_host()["zobrist"].reshape(N)
    for i in range(N):
        if not term_mask[i]:
            assert post_z[i] == pre[i], (
                f"env {i}: zobrist should be unchanged when not reset"
            )
