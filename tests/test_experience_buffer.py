"""Phase 0.4 M7 — ExperienceBuffer ring, capacity rollover, torch bridge."""

from __future__ import annotations

import numpy as np
import pytest

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl import ExperienceBuffer
# ExperienceBuffer stores CPU-side world-frame legal masks (289×289 = 83521),
# not the compact 16641 LUT export from ``junqi_rl.action_lut``.
from junqi_rl.buffer import FLAT_ACTION_DIM

BOARD = 17


def _mk_obs(i: int) -> tuple[np.ndarray, np.ndarray]:
    sp = np.full((OBS_CHANNELS, BOARD, BOARD), float(i), dtype=np.float32)
    gl = np.full((OBS_GLOBAL_DIMS,), float(i), dtype=np.float32)
    return sp, gl


# ---------------------------------------------------------------------------
# 1. Construction + shape contract
# ---------------------------------------------------------------------------


class TestShapes:
    def test_default_shapes(self) -> None:
        buf = ExperienceBuffer(capacity=8)
        assert buf.obs_spatial.shape  == (8, OBS_CHANNELS, BOARD, BOARD)
        assert buf.obs_global.shape   == (8, OBS_GLOBAL_DIMS)
        assert buf.action_id.shape    == (8,)
        assert buf.reward.shape       == (8, 4)
        assert buf.done.shape         == (8,)
        assert buf.seat.shape         == (8,)
        assert buf.value_target.shape == (8,)
        assert buf.size == 0
        assert not buf.full
        assert buf.legal_mask is None
        assert buf.legal_ids is None

    def test_dense_mode_allocates_mask(self) -> None:
        buf = ExperienceBuffer(capacity=4, legal_mask_mode="dense")
        assert buf.legal_mask is not None
        assert buf.legal_mask.shape == (4, FLAT_ACTION_DIM)
        assert buf.legal_mask.dtype == bool

    def test_sparse_mode_allocates_list(self) -> None:
        buf = ExperienceBuffer(capacity=5, legal_mask_mode="sparse")
        assert buf.legal_ids is not None
        assert len(buf.legal_ids) == 5
        for a in buf.legal_ids:
            assert a.dtype == np.int32

    def test_invalid_capacity(self) -> None:
        with pytest.raises(ValueError):
            ExperienceBuffer(capacity=0)

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError):
            ExperienceBuffer(capacity=4, legal_mask_mode="whatever")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. Append + retrieval
# ---------------------------------------------------------------------------


class TestAppend:
    def test_append_fills_row(self) -> None:
        buf = ExperienceBuffer(capacity=4)
        sp, gl = _mk_obs(3)
        row = buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=1234, reward=(1, -1, 0, 0), done=False, seat=2,
        )
        assert row == 0
        assert buf.size == 1
        assert buf.action_id[0] == 1234
        np.testing.assert_array_equal(buf.reward[0], [1, -1, 0, 0])
        assert not buf.done[0]
        assert buf.seat[0] == 2
        np.testing.assert_array_equal(buf.obs_spatial[0], sp)
        np.testing.assert_array_equal(buf.obs_global[0], gl)

    def test_append_copies_obs(self) -> None:
        """Mutating the caller's obs after append must not affect slab."""
        buf = ExperienceBuffer(capacity=2)
        sp, gl = _mk_obs(5)
        buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=1, reward=(0, 0, 0, 0), done=False, seat=0,
        )
        sp.fill(99.0)  # caller mutation
        assert buf.obs_spatial[0].sum() == 5.0 * OBS_CHANNELS * BOARD * BOARD

    def test_wrap_around(self) -> None:
        """Writing capacity+1 steps overwrites slot 0."""
        C = 3
        buf = ExperienceBuffer(capacity=C)
        for i in range(C + 1):
            sp, gl = _mk_obs(i)
            buf.append_step(
                obs_spatial=sp, obs_global=gl,
                action_id=i, reward=(i, i, i, i), done=(i == C),
                seat=i % 4,
            )
        # Buffer must report full + capacity size.
        assert buf.full
        assert buf.size == C
        # Slot 0 now holds the last write (i==C).
        assert buf.action_id[0] == C
        # iter_ordered must yield oldest first (i.e., slot 1 → 2 → 0).
        ordered = list(buf.iter_ordered())
        assert ordered == [1, 2, 0]


# ---------------------------------------------------------------------------
# 3. Legal-mask modes
# ---------------------------------------------------------------------------


class TestLegalMask:
    def test_dense_roundtrip(self) -> None:
        buf = ExperienceBuffer(capacity=2, legal_mask_mode="dense")
        mask = np.zeros(FLAT_ACTION_DIM, dtype=bool)
        mask[[1, 10, 100, FLAT_ACTION_DIM - 1]] = True
        sp, gl = _mk_obs(0)
        buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=10, reward=(0, 0, 0, 0), done=False, seat=0,
            legal=mask,
        )
        assert buf.legal_mask is not None
        np.testing.assert_array_equal(buf.legal_mask[0], mask)

    def test_sparse_roundtrip_copies(self) -> None:
        buf = ExperienceBuffer(capacity=2, legal_mask_mode="sparse")
        ids = np.array([5, 17, 200], dtype=np.int32)
        sp, gl = _mk_obs(0)
        buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=17, reward=(0, 0, 0, 0), done=False, seat=1,
            legal=ids,
        )
        # Caller mutation must not leak into the stored copy.
        ids[0] = -1
        assert buf.legal_ids is not None
        np.testing.assert_array_equal(buf.legal_ids[0], [5, 17, 200])

    def test_none_mode_rejects_legal(self) -> None:
        buf = ExperienceBuffer(capacity=1, legal_mask_mode="none")
        sp, gl = _mk_obs(0)
        with pytest.raises(ValueError, match="legal_mask_mode='none'"):
            buf.append_step(
                obs_spatial=sp, obs_global=gl,
                action_id=0, reward=(0, 0, 0, 0), done=False, seat=0,
                legal=np.zeros(FLAT_ACTION_DIM, dtype=bool),
            )


# ---------------------------------------------------------------------------
# 4. Reset + sampling + torch bridge
# ---------------------------------------------------------------------------


class TestSamplingAndReset:
    def test_reset_preserves_allocation_but_resets_size(self) -> None:
        buf = ExperienceBuffer(capacity=4)
        sp, gl = _mk_obs(7)
        buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=1, reward=(1, 0, 0, 0), done=False, seat=0,
        )
        before_id_obj = id(buf.obs_spatial)
        buf.reset()
        assert buf.size == 0
        assert not buf.full
        # Underlying slab is NOT reallocated.
        assert id(buf.obs_spatial) == before_id_obj
        # But logical content is gone on the next append (new write at 0).

    def test_zero_resets_content(self) -> None:
        buf = ExperienceBuffer(capacity=2)
        sp, gl = _mk_obs(9)
        buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=1, reward=(1, 0, 0, 0), done=False, seat=0,
        )
        buf.zero_()
        assert buf.size == 0
        assert not buf.obs_spatial.any()
        assert not buf.reward.any()

    def test_sample_indices_empty_raises(self) -> None:
        buf = ExperienceBuffer(capacity=4)
        with pytest.raises(ValueError):
            buf.sample_indices(3)

    def test_sample_indices_in_range(self) -> None:
        buf = ExperienceBuffer(capacity=8)
        for i in range(5):
            sp, gl = _mk_obs(i)
            buf.append_step(
                obs_spatial=sp, obs_global=gl,
                action_id=i, reward=(i, 0, 0, 0), done=False, seat=0,
            )
        idx = buf.sample_indices(16, rng=np.random.default_rng(0))
        assert idx.shape == (16,)
        assert idx.min() >= 0 and idx.max() < buf.size


class TestTorchBridge:
    def test_as_torch_shapes_and_values(self) -> None:
        torch = pytest.importorskip("torch")
        buf = ExperienceBuffer(capacity=4)
        for i in range(3):
            sp, gl = _mk_obs(i)
            buf.append_step(
                obs_spatial=sp, obs_global=gl,
                action_id=i, reward=(i, 0, 0, 0), done=(i == 2), seat=i % 4,
            )
        out = buf.as_torch()
        assert out["obs_spatial"].shape == (3, OBS_CHANNELS, BOARD, BOARD)
        assert out["obs_global"].shape  == (3, OBS_GLOBAL_DIMS)
        assert out["action_id"].shape   == (3,)
        assert out["reward"].shape      == (3, 4)
        # Content check.
        for i in range(3):
            assert out["action_id"][i].item() == i
            assert out["done"][i].item() == (i == 2)

    def test_as_torch_include_legal_sparse(self) -> None:
        torch = pytest.importorskip("torch")
        buf = ExperienceBuffer(capacity=2, legal_mask_mode="sparse")
        sp, gl = _mk_obs(0)
        buf.append_step(
            obs_spatial=sp, obs_global=gl,
            action_id=0, reward=(0, 0, 0, 0), done=False, seat=0,
            legal=np.array([1, 2, 3], dtype=np.int32),
        )
        out = buf.as_torch(include_legal=True)
        assert "legal_ids" in out
        assert len(out["legal_ids"]) == 1
        assert out["legal_ids"][0].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# 5. End-to-end: capture rollout from VectorJunqiEnv into the buffer
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_append_from_vector_env(self) -> None:
        """Drive a 4-env rollout through the buffer for 10 steps."""
        from junqi_rl import VectorJunqiEnv

        N = 4
        venv = VectorJunqiEnv(num_envs=N)
        venv.reset(seed_base=0)
        buf = ExperienceBuffer(capacity=64)
        rng = np.random.default_rng(0)

        for _step in range(10):
            actions = np.zeros(N, dtype=np.int32)
            acting_seats = np.zeros(N, dtype=np.int8)
            for i, env in enumerate(venv.envs):
                if venv.done[i]:
                    continue
                ids = env.legal_action_ids()
                if ids.size:
                    actions[i] = int(ids[rng.integers(int(ids.size))])
                acting_seats[i] = int(env.current_seat().value)
            sp, gl, rwd, done, _infos = venv.step(actions)
            for i in range(N):
                if venv.done[i] and not done[i]:
                    continue  # was already done pre-step; skip
                s_idx = int(acting_seats[i])
                buf.append_step(
                    obs_spatial=sp[i, s_idx],
                    obs_global=gl[i, s_idx],
                    action_id=int(actions[i]),
                    reward=rwd[i].astype(np.float32),
                    done=bool(done[i]),
                    seat=s_idx,
                )
        # At least some rows must have been recorded.
        assert buf.size > 0
        # Every recorded action must be in range.
        assert (buf.action_id[:buf.size] >= 0).all()
        assert (buf.action_id[:buf.size] < FLAT_ACTION_DIM).all()
