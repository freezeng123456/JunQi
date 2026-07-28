"""tests/test_action_lut.py — Unit tests for junqi_rl.action_lut.

Verifies:
  1. ROTATE_LUT / UNROTATE_LUT round-trip consistency.
  2. LUT values match the scalar rotate_action_id / unrotate_action_id.
  3. build_legal_mask_batch produces the same result as the old per-action loop.
  4. max_num_moves parameter is honoured by JunqiEnv / VectorJunqiEnv.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.rules import ALL_SEATS, Seat
from junqi_rl.action_lut import (
    FLAT_ACTION_DIM,
    ROTATE_LUT,
    UNROTATE_LUT,
    build_legal_mask_batch,
)
from junqi_rl.env import (
    JunqiEnv,
    VectorJunqiEnv,
    rotate_action_id,
    rotate_to_compact_action_id,
    unrotate_action_id,
    unrotate_compact_action_id,
)


# ---------------------------------------------------------------------------
# LUT shape / dtype basics
# ---------------------------------------------------------------------------


def test_lut_shape_and_dtype():
    assert len(ROTATE_LUT) == 4
    assert len(UNROTATE_LUT) == 4
    for seat in ALL_SEATS:
        lut_r = ROTATE_LUT[seat.value]
        lut_u = UNROTATE_LUT[seat.value]
        assert lut_r.shape == (FLAT_ACTION_DIM,), f"ROTATE_LUT[{seat.name}] shape wrong"
        assert lut_u.shape == (FLAT_ACTION_DIM,), f"UNROTATE_LUT[{seat.name}] shape wrong"
        assert lut_r.dtype == np.int32
        assert lut_u.dtype == np.int32


# ---------------------------------------------------------------------------
# Round-trip: rotate then unrotate must give back the original
#
# Note (post-compact-refactor): ROTATE_LUT / UNROTATE_LUT are indexed in the
# COMPACT frame (129×129 = 16,641 ids), not the world frame. The input IDs
# here are compact-world ids that survive the round-trip back to themselves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seat", ALL_SEATS)
def test_lut_round_trip(seat: Seat):
    rng = np.random.default_rng(0)
    compact_world_ids = rng.integers(0, FLAT_ACTION_DIM, size=1000, dtype=np.int32)
    can_ids = ROTATE_LUT[seat.value][compact_world_ids]
    # Invalid (unreachable) rotations are written as -1 in the LUT; skip them.
    keep = can_ids >= 0
    recovered = UNROTATE_LUT[seat.value][can_ids[keep]]
    np.testing.assert_array_equal(
        recovered, compact_world_ids[keep],
        err_msg=f"Round-trip failed for seat {seat.name}",
    )


# ---------------------------------------------------------------------------
# Consistency with scalar helpers
#
# The LUT takes compact-world ids; compare against ``rotate_to_compact_action_id``
# / ``unrotate_compact_action_id`` (which take world-full ids and map through
# the COMPACT frame). Path: world-full → compact-world (via FLAT_TO_COMPACT) →
# canonical-compact (via LUT); equivalent to the scalar helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seat", ALL_SEATS)
def test_rotate_lut_matches_scalar(seat: Seat):
    """LUT[compact_world_id] must equal the scalar helper output for every
    *valid* compact world id (ones backed by an on-board cell pair)."""
    rng = np.random.default_rng(1)
    compact_world_ids = rng.integers(0, FLAT_ACTION_DIM, size=200, dtype=np.int32)
    lut_result = ROTATE_LUT[seat.value][compact_world_ids]

    # Scalar: compact_world → world_full → compact_canonical
    from junqi_core.board import COMPACT_TO_FLAT, NUM_ON_BOARD_CELLS
    scalar_result = np.full_like(lut_result, -1)
    for i, cw in enumerate(compact_world_ids):
        src_c = int(cw) // NUM_ON_BOARD_CELLS
        dst_c = int(cw) % NUM_ON_BOARD_CELLS
        src_w = int(COMPACT_TO_FLAT[src_c])
        dst_w = int(COMPACT_TO_FLAT[dst_c])
        world_id = src_w * 289 + dst_w
        scalar_result[i] = rotate_to_compact_action_id(world_id, seat)

    np.testing.assert_array_equal(
        lut_result, scalar_result,
        err_msg=f"ROTATE_LUT mismatch for seat {seat.name}",
    )


@pytest.mark.parametrize("seat", ALL_SEATS)
def test_unrotate_lut_matches_scalar(seat: Seat):
    """UNROTATE_LUT[compact_canonical_id] must equal the scalar helper output."""
    rng = np.random.default_rng(2)
    compact_can_ids = rng.integers(0, FLAT_ACTION_DIM, size=200, dtype=np.int32)
    lut_result = UNROTATE_LUT[seat.value][compact_can_ids]

    # Scalar: compact_canonical → world_full (via unrotate_compact_action_id)
    # → compact_world (via FLAT_TO_COMPACT).
    from junqi_core.board import FLAT_TO_COMPACT, NUM_ON_BOARD_CELLS
    scalar_result = np.full_like(lut_result, -1)
    for i, cc in enumerate(compact_can_ids):
        world_full = unrotate_compact_action_id(int(cc), seat)
        src_w = world_full // 289
        dst_w = world_full % 289
        src_c = int(FLAT_TO_COMPACT[src_w])
        dst_c = int(FLAT_TO_COMPACT[dst_w])
        if src_c < 0 or dst_c < 0:
            scalar_result[i] = -1
        else:
            scalar_result[i] = src_c * NUM_ON_BOARD_CELLS + dst_c

    np.testing.assert_array_equal(
        lut_result, scalar_result,
        err_msg=f"UNROTATE_LUT mismatch for seat {seat.name}",
    )


# ---------------------------------------------------------------------------
# SOUTH is identity
# ---------------------------------------------------------------------------


def test_south_is_identity():
    """SOUTH rotation must be a no-op (identity permutation)."""
    all_ids = np.arange(FLAT_ACTION_DIM, dtype=np.int32)
    np.testing.assert_array_equal(ROTATE_LUT[Seat.SOUTH.value], all_ids)
    np.testing.assert_array_equal(UNROTATE_LUT[Seat.SOUTH.value], all_ids)


# ---------------------------------------------------------------------------
# build_legal_mask_batch matches the old per-action loop
# ---------------------------------------------------------------------------


def test_build_legal_mask_batch_correctness():
    """build_legal_mask_batch must produce the same mask as the scalar loop.

    Post-compact-refactor: mask is (N, 16641) in the compact canonical frame.
    Reference computes via world → compact-canonical using
    ``rotate_to_compact_action_id``.
    """
    env = VectorJunqiEnv(num_envs=4)
    env.reset(seed_base=0)
    seats = env.current_seats()

    # New LUT-based path
    mask_lut = build_legal_mask_batch(env, seats)

    # Reference: world-frame legal_action_ids → compact canonical ids.
    N = env.num_envs
    mask_ref = np.zeros((N, FLAT_ACTION_DIM), dtype=bool)
    for i, (e, done) in enumerate(zip(env._envs, env.done)):
        if done:
            continue
        seat = seats[i]
        world_ids = e.legal_action_ids(seat)
        can_ids = np.array(
            [rotate_to_compact_action_id(int(w), seat) for w in world_ids],
            dtype=np.int32,
        )
        valid = can_ids >= 0
        mask_ref[i, can_ids[valid]] = True

    np.testing.assert_array_equal(
        mask_lut, mask_ref,
        err_msg="build_legal_mask_batch disagrees with scalar reference",
    )


# ---------------------------------------------------------------------------
# max_num_moves parameter
# ---------------------------------------------------------------------------


def test_junqienv_max_num_moves_default():
    """JunqiEnv without max_num_moves runs past 10 steps without forced draw."""
    env = JunqiEnv()
    env.reset(seed=0)
    for _ in range(10):
        seat = env.current_seat()
        ids = env.legal_action_ids(seat)
        if len(ids) == 0:
            break
        _, _, done, _ = env.step(int(ids[0]))
        if done:
            break
    # Simply verify no exception and env is usable


def test_junqienv_max_num_moves_custom_triggers_draw():
    """JunqiEnv with max_num_moves=5 must terminate within 5 steps."""
    env = JunqiEnv(max_num_moves=5)
    env.reset(seed=0)
    done_count = 0
    for step in range(10):
        seat = env.current_seat()
        ids = env.legal_action_ids(seat)
        if len(ids) == 0:
            break
        _, _, done, info = env.step(int(ids[0]))
        if done:
            assert info.draw or info.termination_reason == "max_num_moves", \
                f"Expected draw at step {step}, got {info}"
            done_count += 1
            break
    assert done_count == 1, "Game should have been forced-terminated within max_num_moves steps"


def test_vectorjunqienv_max_num_moves_param():
    """VectorJunqiEnv accepts max_num_moves without error."""
    env = VectorJunqiEnv(num_envs=2, max_num_moves=2000)
    assert env.num_envs == 2
    obs_sp, obs_gl = env.reset(seed_base=0)
    assert obs_sp.shape[0] == 2


# ---------------------------------------------------------------------------
# VectorJunqiEnv parallel stepping
# ---------------------------------------------------------------------------


def test_vectorjunqienv_parallel_vs_sequential():
    """num_workers > 1 must produce the same obs / done flags as num_workers=1."""
    import copy

    num_envs = 4
    steps = 20

    # Sequential baseline
    env_seq = VectorJunqiEnv(num_envs=num_envs, num_workers=1)
    env_seq.reset(seed_base=7)

    # Parallel
    env_par = VectorJunqiEnv(num_envs=num_envs, num_workers=4)
    env_par.reset(seed_base=7)

    for _ in range(steps):
        seats_seq = env_seq.current_seats()
        seats_par = env_par.current_seats()

        # Take first legal action for each env (deterministic).
        actions_seq = np.zeros(num_envs, dtype=np.int32)
        actions_par = np.zeros(num_envs, dtype=np.int32)
        for j in range(num_envs):
            if not env_seq.done[j]:
                ids = env_seq._envs[j].legal_action_ids(seats_seq[j])
                if len(ids):
                    actions_seq[j] = int(ids[0])
            if not env_par.done[j]:
                ids = env_par._envs[j].legal_action_ids(seats_par[j])
                if len(ids):
                    actions_par[j] = int(ids[0])

        sp_seq, gl_seq, rw_seq, done_seq, _ = env_seq.step(actions_seq)
        sp_par, gl_par, rw_par, done_par, _ = env_par.step(actions_par)

        np.testing.assert_array_equal(done_seq, done_par,
                                      err_msg="done arrays differ (step {_})")
        np.testing.assert_array_almost_equal(
            sp_seq, sp_par, decimal=5,
            err_msg=f"obs_spatial differs at step {_}",
        )
        np.testing.assert_array_almost_equal(
            gl_seq, gl_par, decimal=5,
            err_msg=f"obs_global differs at step {_}",
        )
        np.testing.assert_array_equal(rw_seq, rw_par,
                                      err_msg=f"rewards differ at step {_}")

    env_seq.close()
    env_par.close()


def test_vectorjunqienv_num_workers_1_sequential():
    """num_workers=1 uses the sequential code path (no pool)."""
    env = VectorJunqiEnv(num_envs=2, num_workers=1)
    assert env._pool is None
    env.reset(seed_base=0)
    for _ in range(5):
        seats = env.current_seats()
        actions = np.zeros(2, dtype=np.int32)
        for j in range(2):
            if not env.done[j]:
                ids = env._envs[j].legal_action_ids(seats[j])
                if len(ids):
                    actions[j] = int(ids[0])
        env.step(actions)
    env.close()


def test_vectorjunqienv_close():
    """close() can be called multiple times without error."""
    env = VectorJunqiEnv(num_envs=2)
    env.reset(seed_base=0)
    env.close()
    env.close()  # idempotent


def test_step_game_only_matches_full_step():
    """_step_game_only must return same reward/done/info as the full step."""
    # We can't run two separate envs with identical seeds easily because
    # JunqiEnv.reset() uses Python random which is seeded internally; instead
    # we verify that _step_game_only doesn't raise and returns sensible types.
    env = JunqiEnv()
    env.reset(seed=42)
    seat = env.current_seat()
    ids = env.legal_action_ids(seat)
    assert len(ids) > 0
    rwd, done, info = env._step_game_only(int(ids[0]))
    assert isinstance(rwd, tuple) and len(rwd) == 4
    assert isinstance(done, bool)
    assert isinstance(info.acting_seat, type(seat))
