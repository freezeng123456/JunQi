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

    # Derived from the layout rather than hardcoded: the channel count has
    # moved once already (412 -> 317 when piece_id's 120 one-hot planes
    # compacted to piece_slot's 25), and a magic total here just fails without
    # saying what the schema now is.
    from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS

    expected_obs = (OBS_CHANNELS * 17 * 17 + OBS_GLOBAL_DIMS) * 2
    expected_legal = 256 * 4 + 4
    assert estimate.observation_bytes == expected_obs
    assert estimate.legal_bytes == expected_legal
    assert estimate.bytes_per_transition == expected_obs + expected_legal + 26


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


def test_compact_history_is_much_smaller_than_full_observations() -> None:
    """The saving is large, but the exact factor tracks OBS_CHANNELS.

    This used to assert a flat 14.60 GiB and a ratio under 0.11, both of
    which encoded a 412-channel observation. Compacting piece_id to
    piece_slot took the layout to 317 channels, so full_obs shrank and the
    ratio moved with it. The invariant worth holding is that compact_history
    stores an observer slice instead of a stacked tensor, which is an order
    of magnitude either way; the GiB figure belongs to the schema and is
    derived from it here.
    """
    n_envs, steps = 128, 512
    full = estimate_rollout_storage(
        num_envs=n_envs, steps_per_env=steps, storage_mode="full_obs",
    )
    compact = estimate_rollout_storage(
        num_envs=n_envs, steps_per_env=steps, storage_mode="compact_history",
    )

    transitions = n_envs * steps
    assert full.total_bytes == full.bytes_per_transition * transitions
    assert compact.total_bytes == compact.bytes_per_transition * transitions
    assert compact.total_gib == pytest.approx(1.48, abs=0.02)
    assert compact.total_bytes / full.total_bytes < 0.2


@pytest.mark.parametrize("storage_mode", ["full_obs", "compact_history"])
def test_storage_estimate_rejects_non_positive_shape(storage_mode: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        estimate_rollout_storage(
            num_envs=0,
            steps_per_env=512,
            storage_mode=storage_mode,
        )
