"""tests/test_gpu_legal_lite.py — parity of the lite upload path.

Validates that ``DeviceGameStateBatch.copy_from_host_legal_lite`` + the scatter
kernel produce identical results to the full ``copy_from_host`` path for the
``legal_action_ids_batch`` kernel.
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

from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays, _pack_state_arrays_lite


def _make_envs(N: int, seed_base: int = 0xABCDEF) -> list[JunqiEnv]:
    envs = []
    for i in range(N):
        rng = random.Random(seed_base + i)
        env = JunqiEnv()
        env.reset(seed=seed_base + i)
        for _ in range(i % 40):
            if env.state.terminated:
                break
            aids = env.legal_action_ids()
            if aids.size == 0:
                break
            env._step_game_only(int(rng.choice(aids)))
        envs.append(env)
    return envs


@pytest.fixture(scope="module", autouse=True)
def _init():
    _cuda.init_tables()


def test_lite_matches_full_upload() -> None:
    """Lite and full upload paths produce identical GPU legal-action results."""
    N = 48
    envs = _make_envs(N)
    acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

    # Full path
    sd = _pack_state_arrays(envs)
    gs1 = _cuda.DeviceGameStateBatch(N)
    gs1.copy_from_host(sd)
    ids_full, cnt_full = _cuda.legal_action_ids_batch(gs1, acting)

    # Lite path
    psa, pta, alv, px, py_, cpi = _pack_state_arrays_lite(envs)
    gs2 = _cuda.DeviceGameStateBatch(N)
    gs2.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)
    ids_lite, cnt_lite = _cuda.legal_action_ids_batch(gs2, acting)

    # Counts identical
    np.testing.assert_array_equal(cnt_full, cnt_lite)

    # Sets identical per env
    for i in range(N):
        c = int(cnt_full[i])
        full_sorted = np.sort(ids_full[i, :c])
        lite_sorted = np.sort(ids_lite[i, :c])
        np.testing.assert_array_equal(
            full_sorted, lite_sorted,
            err_msg=f"env {i}: full vs lite mismatch",
        )


def test_lite_parity_various_batch_sizes() -> None:
    """Lite path works for N=1, 8, 64, 256."""
    for N in (1, 8, 64, 256):
        envs = _make_envs(N, seed_base=0x1_0000 + N)
        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

        sd = _pack_state_arrays(envs)
        gs1 = _cuda.DeviceGameStateBatch(N)
        gs1.copy_from_host(sd)
        ids_full, cnt_full = _cuda.legal_action_ids_batch(gs1, acting)

        psa, pta, alv, px, py_, cpi = _pack_state_arrays_lite(envs)
        gs2 = _cuda.DeviceGameStateBatch(N)
        gs2.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)
        ids_lite, cnt_lite = _cuda.legal_action_ids_batch(gs2, acting)

        assert np.array_equal(cnt_full, cnt_lite), f"N={N}: counts differ"
        for i in range(N):
            c = int(cnt_full[i])
            if c == 0:
                continue
            assert np.array_equal(
                np.sort(ids_full[i, :c]), np.sort(ids_lite[i, :c])
            ), f"N={N} env {i}: lite mismatch"


def test_lite_repeated_calls_consistent() -> None:
    """Calling lite twice on the same state gives identical output both times."""
    N = 16
    envs = _make_envs(N, seed_base=0xFEED)
    acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)

    gs = _cuda.DeviceGameStateBatch(N)
    psa, pta, alv, px, py_, cpi = _pack_state_arrays_lite(envs)

    gs.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)
    ids1, cnt1 = _cuda.legal_action_ids_batch(gs, acting)

    gs.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)
    ids2, cnt2 = _cuda.legal_action_ids_batch(gs, acting)

    np.testing.assert_array_equal(cnt1, cnt2)
    for i in range(N):
        c = int(cnt1[i])
        np.testing.assert_array_equal(
            np.sort(ids1[i, :c]), np.sort(ids2[i, :c])
        )
