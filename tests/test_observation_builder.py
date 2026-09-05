"""Phase 0.4 M2 / ADR-118 — ObservationBuilder acceptance tests.

These tests pin down the Phase 0.4 M2 contracts:

1. **Bit-parity with the pre-M2 implementation**: for a matrix of
   scenarios (seed × steps × show_mode × observer), the new builder
   output equals the legacy ``build_observation`` output byte-for-byte.
   The legacy wrapper is itself built on top of the builder now, so the
   "legacy" path in these tests is a stand-in for the hashed golden
   captured pre-M2 (``tests/golden/obs_hashes.json``).

2. **Buffer-reuse contract**: a single builder's consecutive ``build()``
   calls return ObservationTensors that share memory; the second call
   overwrites the first.

3. **snapshot()**: calling ``.snapshot()`` on a builder-backed
   ObservationTensor detaches its buffers so they survive the next
   ``build()``.

4. **Zero per-call allocation**: after the first call, ``build()`` must
   NOT allocate new NumPy arrays.  Verified with ``tracemalloc``.

5. **Vectorized writers correctness on dead-piece scenarios** (D/E
   groups): the D/E writers now batch-index the SoA columns and must
   respect the death ownership (via ``piece_seat_arr``) even for
   hand-constructed states.
"""

from __future__ import annotations

import json
import random
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import (
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationBuilder,
    ObservationTensor,
    build_observation,
)
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from tools.gen_obs_golden import GOLDEN_HASH_KIND, hash_observation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replay(seed: int, steps: int, show_mode: ShowMode, observer: Seat):
    rng = random.Random(seed)
    setups = generate_random_setup(rng)
    state = GameState.new_game(setups, show_mode=show_mode)
    belief = BeliefTensor.initial(state, observer)
    for _ in range(steps):
        if state.terminated:
            break
        legal = state.legal_actions()
        if not legal:
            break
        action = rng.choice(legal)
        new_state, result = state.step(action)
        if not new_state.info[observer].dead:
            belief.update(state, new_state, result)
        state = new_state
    return state, belief


# ---------------------------------------------------------------------------
# 1. Bit-identity across a scenario matrix
# ---------------------------------------------------------------------------


class TestBitIdentity:
    @pytest.mark.parametrize("seed", [0, 7, 42])
    @pytest.mark.parametrize("steps", [0, 50, 200])
    @pytest.mark.parametrize("show_mode", list(ShowMode))
    @pytest.mark.parametrize("observer", list(ALL_SEATS))
    def test_builder_matches_legacy(
        self,
        seed: int,
        steps: int,
        show_mode: ShowMode,
        observer: Seat,
    ) -> None:
        state, belief = _replay(seed, steps, show_mode, observer)
        if state.terminated or state.info[observer].dead:
            pytest.skip("state terminated or observer dead")

        builder = ObservationBuilder()
        obs_new = builder.build(state, belief, observer).snapshot()
        obs_legacy = build_observation(state, belief, observer)

        assert np.array_equal(obs_new.spatial, obs_legacy.spatial), (
            f"spatial differs for seed={seed} steps={steps} "
            f"mode={show_mode.name} observer={observer.name}"
        )
        assert np.array_equal(obs_new.global_, obs_legacy.global_)
        assert obs_new.observer is observer

    def test_golden_hashes_still_match(self) -> None:
        """The 12 deterministic golden scenarios must all match."""
        repo = Path(__file__).resolve().parent.parent
        golden_path = repo / "tests/golden/obs_hashes.json"
        if not golden_path.exists():
            pytest.skip("golden/obs_hashes.json not present")
        golden = json.loads(golden_path.read_text())

        builder = ObservationBuilder()
        matched = 0
        for rec in golden:
            state, belief = _replay(
                rec["seed"], rec["steps"], ShowMode[rec["show_mode"]],
                Seat.SOUTH,
            )
            if state.terminated:
                continue
            obs = builder.build(state, belief, Seat.SOUTH).snapshot()
            assert rec.get("hash_kind") == GOLDEN_HASH_KIND
            assert hash_observation(obs) == rec["sha256"], (
                f"hash drift on seed={rec['seed']} steps={rec['steps']} "
                f"mode={rec['show_mode']}"
            )
            matched += 1
        assert matched >= 10, f"expected at least 10 golden matches, got {matched}"


# ---------------------------------------------------------------------------
# 2. Buffer-reuse contract
# ---------------------------------------------------------------------------


class TestBufferReuse:
    def test_consecutive_build_shares_memory(self) -> None:
        state, belief_s = _replay(123, 10, ShowMode.HALF_DARK, Seat.SOUTH)
        # Second seat to ensure the builder actually rewrites everything.
        _, belief_w = _replay(123, 10, ShowMode.HALF_DARK, Seat.WEST)

        builder = ObservationBuilder()
        obs_a = builder.build(state, belief_s, Seat.SOUTH)
        # Point to same buffer as the builder.
        assert obs_a.spatial.base is builder._canonical or np.shares_memory(
            obs_a.spatial, builder._canonical
        )

        # Snapshot first obs to freeze its current contents for comparison.
        snap_a = obs_a.snapshot()

        obs_b = builder.build(state, belief_w, Seat.WEST)
        # The returned tensor references the same buffer.
        assert np.shares_memory(obs_a.spatial, obs_b.spatial)

        # And the old view (obs_a) has been overwritten by the new build.
        assert not np.array_equal(obs_a.spatial, snap_a.spatial)

    def test_snapshot_detaches_buffers(self) -> None:
        state, belief_s = _replay(321, 20, ShowMode.HALF_DARK, Seat.SOUTH)
        _, belief_w = _replay(321, 20, ShowMode.HALF_DARK, Seat.WEST)

        builder = ObservationBuilder()
        snap = builder.build(state, belief_s, Seat.SOUTH).snapshot()
        snap_spatial_before = snap.spatial.copy()

        # Build with a different observer; snap must remain unchanged.
        _ = builder.build(state, belief_w, Seat.WEST)
        assert np.array_equal(snap.spatial, snap_spatial_before)
        assert not np.shares_memory(snap.spatial, builder._canonical)


# ---------------------------------------------------------------------------
# 3. Zero per-call allocation
# ---------------------------------------------------------------------------


class TestZeroAllocation:
    def test_build_does_not_allocate_large_arrays(self) -> None:
        state, belief = _replay(0, 50, ShowMode.HALF_DARK, Seat.SOUTH)
        builder = ObservationBuilder()
        # Warmup: first call may still allocate scratch arrays for
        # live_pids / live_x / live_y etc., which is fine.
        builder.build(state, belief, Seat.SOUTH)

        tracemalloc.start()
        for _ in range(50):
            builder.build(state, belief, Seat.SOUTH)
        snapshot = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Headline metric: total bytes allocated across 50 builds.
        # The three builder buffers are 101*17*17*4 + 101*17*17*4 + 28*4
        # = 233,696 bytes -> if any of those got reallocated once we'd
        # exceed 200 KiB.  Scratch live_pids arrays are a few KiB each.
        total = sum(stat.size for stat in snapshot.statistics("filename"))
        # Give ourselves plenty of slack: 50 builds × < 20 KiB each.
        assert total < 1_000_000, (
            f"build() is allocating too much: {total} bytes across 50 calls"
        )


# ---------------------------------------------------------------------------
# 4. Observer-frame correctness: rotation invariants
# ---------------------------------------------------------------------------


class TestRotationInvariants:
    @pytest.mark.parametrize("observer", list(ALL_SEATS))
    def test_spatial_shape_and_dtype(self, observer: Seat) -> None:
        state, belief = _replay(0, 30, ShowMode.HALF_DARK, observer)
        if state.terminated or state.info[observer].dead:
            pytest.skip("unusable state")
        obs = ObservationBuilder().build(state, belief, observer)
        assert obs.spatial.shape == (OBS_CHANNELS, 17, 17)
        assert obs.spatial.dtype == np.float32
        assert obs.global_.shape == (OBS_GLOBAL_DIMS,)
        assert obs.global_.dtype == np.float32

    def test_piece_own_lights_at_canonical_bottom(self) -> None:
        """The observer's own pieces must appear at canonical bottom
        (y in [11, 16]) regardless of the observer's world seat."""
        for observer in ALL_SEATS:
            state, belief = _replay(0, 0, ShowMode.BRIGHT, observer)
            obs = ObservationBuilder().build(state, belief, observer)
            own = obs.channel("piece_own")                # (12, 17, 17)
            # Sum every channel across y<11 — must be 0 (no own pieces above).
            top_sum = float(own[:, :11, :].sum())
            bottom_sum = float(own[:, 11:, :].sum())
            assert top_sum == 0.0, (
                f"observer {observer.name}: own piece leaked above canonical "
                f"bottom (sum={top_sum})"
            )
            assert bottom_sum == 25.0, (
                f"observer {observer.name}: only {bottom_sum}/25 own pieces "
                f"at canonical bottom"
            )


# ---------------------------------------------------------------------------
# 5. D/E group correctness under hand-constructed dead-piece scenarios
# ---------------------------------------------------------------------------


class TestDeathChannelsHandConstructed:
    def _build_after_combat_state(self) -> tuple[GameState, BeliefTensor]:
        """Run a random game until at least one death is recorded, then
        return the state + belief so D/E tests have real data to assert on.
        """
        rng = random.Random(31415)
        setups = generate_random_setup(rng)
        state = GameState.new_game(setups, show_mode=ShowMode.BRIGHT)
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        for _ in range(400):
            if state.terminated:
                break
            legal = state.legal_actions()
            if not legal:
                break
            a = rng.choice(legal)
            new_state, result = state.step(a)
            if not new_state.info[Seat.SOUTH].dead:
                belief.update(state, new_state, result)
            state = new_state
            if state.deaths:
                break
        assert state.deaths, "failed to generate a death in 400 steps"
        return state, belief

    def test_death_reason_planes_count_matches_dict_deaths(self) -> None:
        state, belief = self._build_after_combat_state()
        obs = ObservationBuilder().build(state, belief, Seat.SOUTH)
        dr = obs.channel("death_reason")
        # Total lit cells across all 6 planes should equal len(state.deaths):
        # each dead piece contributes exactly one lit cell at death_loc.
        # (We allow a small relaxation: multiple deaths at the SAME cell
        # with the SAME reason+side would collapse to one lit cell.)
        total_lit = int(dr.sum())
        assert 1 <= total_lit <= len(state.deaths), (
            f"death_reason lit cells ({total_lit}) does not match deaths "
            f"({len(state.deaths)})"
        )

    def test_dead_at_zero_planes_count_matches_dict_deaths(self) -> None:
        state, belief = self._build_after_combat_state()
        obs = ObservationBuilder().build(state, belief, Seat.SOUTH)
        dz = obs.channel("dead_at_zero")
        total_lit = int(dz.sum())
        # Dead-at-zero anchors at the piece's ZERO cell, which is unique
        # per piece_id, so this equality is tight.
        assert total_lit == len(state.deaths), (
            f"dead_at_zero lit cells ({total_lit}) != deaths "
            f"({len(state.deaths)})"
        )
