from __future__ import annotations

import pytest

from junqi_rl.training.rollout_storage import (
    COMPACT_HISTORY_BYTES_PER_TRANSITION,
    estimate_rollout_storage,
)


def test_full_observation_storage_matches_tensor_schema() -> None:
    estimate = estimate_rollout_storage(
        num_envs=1,
        steps_per_env=1,
        storage_mode="full_obs",
    )

    assert estimate.observation_bytes == (412 * 17 * 17 + 28) * 2
    assert estimate.legal_bytes == 256 * 4 + 4
    assert estimate.bytes_per_transition == 239_278


def test_compact_history_accounts_for_observer_slices_only() -> None:
    estimate = estimate_rollout_storage(
        num_envs=1,
        steps_per_env=1,
        storage_mode="compact_history",
    )

    assert COMPACT_HISTORY_BYTES_PER_TRANSITION == 24_203
    assert estimate.observation_bytes == 24_203
    assert estimate.legal_bytes == 0
    assert estimate.bytes_per_transition == 24_229


def test_compact_history_reduces_baseline_rollout_by_about_tenfold() -> None:
    full = estimate_rollout_storage(
        num_envs=128,
        steps_per_env=512,
        storage_mode="full_obs",
    )
    compact = estimate_rollout_storage(
        num_envs=128,
        steps_per_env=512,
        storage_mode="compact_history",
    )

    assert full.total_gib == pytest.approx(14.60, abs=0.02)
    assert compact.total_gib == pytest.approx(1.48, abs=0.02)
    assert compact.total_bytes / full.total_bytes < 0.11


@pytest.mark.parametrize("storage_mode", ["full_obs", "compact_history"])
def test_storage_estimate_rejects_non_positive_shape(storage_mode: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_rollout_storage(
            num_envs=0,
            steps_per_env=512,
            storage_mode=storage_mode,
        )
