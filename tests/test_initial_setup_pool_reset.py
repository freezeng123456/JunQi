from __future__ import annotations

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


def test_reset_from_setup_pool_uses_current_uploaded_pool() -> None:
    import junqi_rl.gpu_rollout as gpu_rollout_mod
    from junqi_rl.gpu_rollout import GpuRollout

    n_env = 16
    rollout = GpuRollout(num_envs=n_env)
    previous = None if gpu_rollout_mod._training_setup_pool is None else gpu_rollout_mod._training_setup_pool.copy()
    pool = gpu_rollout_mod._build_setup_pool(64, seed=91_337)

    try:
        _cuda.upload_setup_pool(pool)
        gpu_rollout_mod._training_setup_pool = pool
        rollout.reset_from_setup_pool(seed=12_345)

        host = rollout.state.copy_to_host()
        actual = host["piece_type_arr"].reshape(n_env, 120)
        allowed = {row.tobytes() for row in pool}
        assert all(row.tobytes() in allowed for row in actual)

        term = rollout.state.copy_termination_to_host()
        assert not bool(term["terminated"].any())
        assert not bool(term["draw"].any())
        assert bool((term["winner_team"] == -1).all())
    finally:
        if previous is not None:
            _cuda.upload_setup_pool(previous)
            gpu_rollout_mod._training_setup_pool = previous
