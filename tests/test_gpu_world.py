"""tests/test_gpu_world.py — GpuWorld facade tests.

Covers the three output formats (dense / CSR / mask) via the high-level
``GpuWorld`` API and the persistent scratch lifecycle.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

try:
    import junqi_cuda as _cuda
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda not available",
)

from junqi_rl import GpuWorld, JunqiEnv


def _make_envs(N: int, seed_base: int = 0xABCD) -> list[JunqiEnv]:
    envs = []
    for i in range(N):
        rng = random.Random(seed_base + i)
        env = JunqiEnv()
        env.reset(seed=seed_base + i)
        for _ in range(i % 30):
            if env.state.terminated:
                break
            aids = env.legal_action_ids()
            if aids.size == 0:
                break
            env._step_game_only(int(rng.choice(aids)))
        envs.append(env)
    return envs


class TestGpuWorldBasics:
    def test_construction(self):
        world = GpuWorld(num_envs=8)
        assert world.num_envs == 8
        assert "num_envs=8" in repr(world)

    def test_invalid_num_envs(self):
        with pytest.raises(ValueError):
            GpuWorld(num_envs=0)

    def test_push_state_size_mismatch(self):
        world = GpuWorld(num_envs=8)
        envs = _make_envs(4)
        with pytest.raises(ValueError, match="got 4 envs, expected 8"):
            world.push_state_from_envs(envs)

    def test_legal_actions_input_validation(self):
        world = GpuWorld(num_envs=4)
        envs = _make_envs(4)
        world.push_state_from_envs(envs)
        # Wrong shape
        with pytest.raises(ValueError, match="shape"):
            world.legal_actions_dense(np.zeros(2, dtype=np.int8))
        # Wrong dtype
        with pytest.raises(ValueError, match="int8"):
            world.legal_actions_dense(np.zeros(4, dtype=np.int32))


class TestGpuWorldOutputFormats:
    """All three output formats must agree on the legal-action set per env."""

    def test_dense_csr_mask_agree(self):
        N = 16
        world = GpuWorld(num_envs=N)
        envs = _make_envs(N)
        world.push_state_from_envs(envs)

        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        ids_d, cnt_d = world.legal_actions_dense(acting)
        offs_c, vals_c = world.legal_actions_csr(acting)
        mask = world.legal_actions_mask(acting)

        # Counts parity
        np.testing.assert_array_equal(np.diff(offs_c), cnt_d)
        np.testing.assert_array_equal(mask.sum(axis=(1, 2)).astype(np.int32), cnt_d)

        # Set parity (dense vs CSR)
        for i in range(N):
            c = int(cnt_d[i])
            d_sorted = np.sort(ids_d[i, :c])
            c_sorted = np.sort(vals_c[offs_c[i]:offs_c[i + 1]])
            np.testing.assert_array_equal(d_sorted, c_sorted)

    def test_lite_upload_path(self):
        N = 8
        world = GpuWorld(num_envs=N)
        envs = _make_envs(N, seed_base=0xFEED)

        # Compare full-upload vs lite-upload — same legal actions.
        world.push_state_from_envs(envs)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        ids_full, cnt_full = world.legal_actions_dense(acting)

        # Same world, different upload path — re-seed to guarantee clean state.
        world2 = GpuWorld(num_envs=N, init_tables=False)
        world2.push_state_lite(envs)
        ids_lite, cnt_lite = world2.legal_actions_dense(acting)

        np.testing.assert_array_equal(cnt_full, cnt_lite)
        for i in range(N):
            c = int(cnt_full[i])
            np.testing.assert_array_equal(
                np.sort(ids_full[i, :c]),
                np.sort(ids_lite[i, :c]),
            )


class TestGpuWorldLifecycle:
    def test_release_scratch_and_reuse(self):
        N = 4
        world = GpuWorld(num_envs=N)
        envs = _make_envs(N)
        world.push_state_from_envs(envs)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

        r1_ids, r1_cnt = world.legal_actions_dense(acting)
        world.release_scratch()
        r2_ids, r2_cnt = world.legal_actions_dense(acting)

        np.testing.assert_array_equal(r1_cnt, r2_cnt)
        for i in range(N):
            c = int(r1_cnt[i])
            np.testing.assert_array_equal(
                np.sort(r1_ids[i, :c]), np.sort(r2_ids[i, :c])
            )

    def test_multiple_worlds_coexist(self):
        """Two GpuWorld instances can share the same GPU."""
        w1 = GpuWorld(num_envs=4)
        w2 = GpuWorld(num_envs=8, init_tables=False)

        envs1 = _make_envs(4, seed_base=111)
        envs2 = _make_envs(8, seed_base=222)
        w1.push_state_from_envs(envs1)
        w2.push_state_from_envs(envs2)

        ac1 = np.array([e.state.turn.value for e in envs1], dtype=np.int8)
        ac2 = np.array([e.state.turn.value for e in envs2], dtype=np.int8)

        ids1, cnt1 = w1.legal_actions_dense(ac1)
        ids2, cnt2 = w2.legal_actions_dense(ac2)

        assert ids1.shape == (4, 512)
        assert ids2.shape == (8, 512)


class TestGpuWorldObservation:
    def test_observation_roundtrip(self):
        """Build observation through GpuWorld — shape and dtype correct."""
        N = 2
        world = GpuWorld(num_envs=N)
        envs = _make_envs(N, seed_base=321)
        world.push_state_from_envs(envs)

        from junqi_rl.env_gpu import _build_belief_batch
        bel = _build_belief_batch(envs)
        observer_seats = np.tile(np.arange(4, dtype=np.int8), (N, 1))

        spatial, global_ = world.build_observation(bel, observer_seats, show_mode=2)
        from junqi_core.observation import OBS_CHANNELS
        assert spatial.shape == (N, 4, OBS_CHANNELS, 17, 17)
        assert spatial.dtype == np.float32
        assert global_.shape == (N, 4, 28)
        assert global_.dtype == np.float32
