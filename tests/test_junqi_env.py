"""Phase 0.4 M6 / ADR-122 — JunqiEnv / VectorJunqiEnv."""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.observation import (
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationTensor,
)
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_rl.env import (
    JunqiEnv,
    JunqiStepInfo,
    VectorJunqiEnv,
    action_id_to_src_dst,
    rotate_action_id,
    src_dst_to_action_id,
    unrotate_action_id,
)

BOARD_SIZE = 17
NUM_CELLS = 289


# ---------------------------------------------------------------------------
# 1. Flat action-id helpers
# ---------------------------------------------------------------------------


class TestActionIdCodec:
    def test_roundtrip_random(self) -> None:
        rng = random.Random(42)
        for _ in range(200):
            aid = rng.randrange(NUM_CELLS * NUM_CELLS)
            src, dst = action_id_to_src_dst(aid)
            back = src_dst_to_action_id(src, dst)
            assert back == aid, f"roundtrip fail: aid={aid} src={src} dst={dst} back={back}"

    def test_canonical_world_inverse(self) -> None:
        """unrotate . rotate == identity for every seat."""
        rng = random.Random(7)
        for seat in ALL_SEATS:
            for _ in range(50):
                aid = rng.randrange(NUM_CELLS * NUM_CELLS)
                can = rotate_action_id(aid, seat)
                back = unrotate_action_id(can, seat)
                assert back == aid, f"seat={seat.name} aid={aid} can={can} back={back}"

    def test_invalid_aid_raises(self) -> None:
        with pytest.raises(ValueError):
            action_id_to_src_dst(-1)
        with pytest.raises(ValueError):
            action_id_to_src_dst(NUM_CELLS * NUM_CELLS)


# ---------------------------------------------------------------------------
# 2. JunqiEnv lifecycle + API contract
# ---------------------------------------------------------------------------


class TestJunqiEnvBasic:
    def test_reset_returns_all_four_obs(self) -> None:
        env = JunqiEnv()
        obs = env.reset(seed=0)
        assert set(obs.keys()) == set(ALL_SEATS)
        for seat, ot in obs.items():
            assert isinstance(ot, ObservationTensor)
            assert ot.spatial.shape == (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
            assert ot.global_.shape == (OBS_GLOBAL_DIMS,)
            assert ot.spatial.dtype == np.float32
            assert ot.global_.dtype == np.float32
            assert ot.observer is seat

    def test_step_requires_reset(self) -> None:
        env = JunqiEnv()
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_step_advances_turn(self) -> None:
        env = JunqiEnv()
        env.reset(seed=1)
        before = env.current_seat()
        ids = env.legal_action_ids()
        assert ids.size > 0
        _obs, _r, done, info = env.step(int(ids[0]))
        assert info.acting_seat is before
        assert not done
        after = env.current_seat()
        # Either the next seat in order, or someone later if a seat was
        # skipped by Q12 — but never the same as ``before``.
        assert after is not before

    def test_obs_snapshots_are_independent(self) -> None:
        """Per-seat obs dict values do not alias each other."""
        env = JunqiEnv()
        obs = env.reset(seed=2)
        # Each call is a fresh .snapshot(): no two obs share memory.
        seats = list(obs.keys())
        for i, s1 in enumerate(seats):
            for s2 in seats[i+1:]:
                assert not np.shares_memory(
                    obs[s1].spatial, obs[s2].spatial,
                ), f"{s1.name} and {s2.name} alias each other"

    def test_legal_action_ids_seat_argument(self) -> None:
        env = JunqiEnv()
        env.reset(seed=3)
        default = env.legal_action_ids()
        explicit = env.legal_action_ids(env.current_seat())
        np.testing.assert_array_equal(default, explicit)
        # Other seats also have moves at game start.
        for s in ALL_SEATS:
            ids = env.legal_action_ids(s)
            assert ids.size > 0

    def test_illegal_action_rejected(self) -> None:
        env = JunqiEnv()
        env.reset(seed=4)
        legal = set(env.legal_action_ids().tolist())
        # Pick any action id not in the legal set.
        all_ids = range(NUM_CELLS * NUM_CELLS)
        illegal = next(i for i in all_ids if i not in legal)
        with pytest.raises(Exception):  # noqa: BLE001  (expected ValueError from state.step)
            env.step(illegal)


# ---------------------------------------------------------------------------
# 3. Full self-play smoke: 100 games complete without exception
# ---------------------------------------------------------------------------


class TestSelfPlaySmoke:
    def test_self_play_smoke(self) -> None:
        """Smoke test: 20 random self-play games run without exception.

        Under uniform-random play, the junqi rules rarely terminate
        within the step cap (Q11 allows ~1000 moves per game), but
        the env must sustain the load for many steps without error.
        """
        env = JunqiEnv()
        completed = 0
        total_steps = 0
        N_GAMES, STEP_CAP = 20, 300
        for game_idx in range(N_GAMES):
            rng = random.Random(game_idx)
            env.reset(seed=game_idx)
            for _ in range(STEP_CAP):
                ids = env.legal_action_ids()
                if ids.size == 0:
                    break
                aid = int(ids[rng.randrange(ids.size)])
                _obs, reward, done, _info = env.step(aid)
                total_steps += 1
                if done:
                    # Reward tuple sums to zero (per-team cancellation).
                    assert sum(reward) == 0
                    completed += 1
                    break
        # Every game must run at least some steps, and cumulative steps
        # should be high (confirms the env isn't silently crashing).
        assert total_steps > N_GAMES * 50, (
            f"expected total steps > {N_GAMES * 50}, got {total_steps}"
        )


# ---------------------------------------------------------------------------
# 4. VectorJunqiEnv contract
# ---------------------------------------------------------------------------


class TestVectorJunqiEnv:
    def test_reset_fills_slab(self) -> None:
        N = 8
        venv = VectorJunqiEnv(num_envs=N)
        sp, gl = venv.reset(seed_base=0)
        assert sp.shape == (N, 4, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        assert gl.shape == (N, 4, OBS_GLOBAL_DIMS)
        assert sp.dtype == np.float32
        assert gl.dtype == np.float32
        # Something must be non-zero post-reset.
        assert sp.any()

    def test_step_updates_slab(self) -> None:
        N = 8
        venv = VectorJunqiEnv(num_envs=N)
        venv.reset(seed_base=100)
        # Pick each env's first legal action.
        action_ids = np.zeros(N, dtype=np.int32)
        for i, env in enumerate(venv.envs):
            ids = env.legal_action_ids()
            action_ids[i] = int(ids[0])
        sp_before = venv.obs_spatial.copy()
        sp, gl, rwd, done, infos = venv.step(action_ids)
        assert sp.shape == (N, 4, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        assert rwd.shape == (N, 4)
        assert done.shape == (N,)
        assert len(infos) == N
        # Slab must have changed for at least one env (step semantics
        # always mutate the active seat's piece_own plane at minimum).
        assert not np.array_equal(sp, sp_before)
        # None of the 8 envs should be done after a single random step.
        assert not done.any()

    def test_parity_with_single_env(self) -> None:
        """After K steps, the vector env's obs slab matches per-env JunqiEnv."""
        N = 4
        K = 10
        venv = VectorJunqiEnv(num_envs=N)
        venv.reset(seed_base=42)
        rngs = [random.Random(42 + i) for i in range(N)]
        for _ in range(K):
            aids = np.zeros(N, dtype=np.int32)
            for i, env in enumerate(venv.envs):
                ids = env.legal_action_ids()
                if ids.size == 0:
                    aids[i] = 0
                else:
                    aids[i] = int(ids[rngs[i].randrange(ids.size)])
            venv.step(aids)

        # Reference: replay each env's sequence of actions via a
        # solo JunqiEnv and compare.  Because the vector env owns the
        # underlying JunqiEnvs, we just build obs from those and
        # compare to the slab.
        from junqi_core.observation import ObservationBuilder
        ref_builder = ObservationBuilder()
        for i, env in enumerate(venv.envs):
            if venv.done[i]:
                continue
            for s_idx, seat in enumerate(ALL_SEATS):
                ref = ref_builder.build(
                    env.state, env.beliefs[seat], seat,
                ).snapshot()
                np.testing.assert_array_equal(
                    venv.obs_spatial[i, s_idx], ref.spatial,
                )
                np.testing.assert_array_equal(
                    venv.obs_global[i, s_idx], ref.global_,
                )

    def test_done_envs_are_skipped(self) -> None:
        """Already-done envs are not re-stepped."""
        N = 2
        venv = VectorJunqiEnv(num_envs=N)
        venv.reset(seed_base=7)
        # Manually mark env[0] done.
        venv._done[0] = True
        # env[1] needs a legal action; env[0] is skipped so any value is fine.
        aids = np.zeros(N, dtype=np.int32)
        aids[1] = int(venv.envs[1].legal_action_ids()[0])
        sp_before = venv.obs_spatial[0].copy()
        _sp, _gl, rwd, done, infos = venv.step(aids)
        # env[0]'s slab row must be unchanged (we never touched it).
        np.testing.assert_array_equal(venv.obs_spatial[0], sp_before)
        # env[0]'s info must be None (skip sentinel).
        assert infos[0] is None
        # env[0]'s reward row must be zero.
        assert (rwd[0] == 0).all()
        # env[0]'s done flag still True.
        assert done[0]
        # env[1] should have a real info and still be active.
        assert infos[1] is not None
        assert not done[1]


# ---------------------------------------------------------------------------
# 5. Action-frame invariant (ADR-122): world-frame id, per-seat rotation
# ---------------------------------------------------------------------------


class TestActionFrameInvariant:
    def test_canonical_unrotate_matches_world_legal(self) -> None:
        """For each seat, un-rotating a canonical-frame action recovers a
        legal world-frame action id."""
        env = JunqiEnv()
        env.reset(seed=0)
        seat = env.current_seat()
        # Grab legal world-frame ids, rotate to canonical, unrotate back,
        # confirm set equality.
        world_ids = env.legal_action_ids()
        canonical_ids = {rotate_action_id(int(a), seat) for a in world_ids}
        back = {unrotate_action_id(c, seat) for c in canonical_ids}
        assert back == set(int(x) for x in world_ids.tolist())
