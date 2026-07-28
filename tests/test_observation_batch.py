"""Phase 0.4 M5 / ADR-121 + ADR-124 — batch observation builder & torch bridge."""

from __future__ import annotations

import random

import numpy as np
import pytest

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import (
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationBuilder,
)
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState

BOARD_SIZE = 17


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fresh_state(seed: int, *, steps: int = 0) -> GameState:
    rng = random.Random(seed)
    state = GameState.new_game(
        generate_random_setup(rng), show_mode=ShowMode.HALF_DARK
    )
    for _ in range(steps):
        if state.terminated:
            break
        la = state.legal_actions()
        if not la:
            break
        a = rng.choice(la)
        state, _ = state.step(a)
    return state


def _belief_for(state: GameState, observer: Seat) -> BeliefTensor:
    return BeliefTensor.initial(state, observer)


# ---------------------------------------------------------------------------
# 1. build_into shape / dtype contract
# ---------------------------------------------------------------------------


class TestBuildInto:
    def test_build_into_writes_into_caller_buffer(self) -> None:
        state = _fresh_state(0)
        observer = state.turn
        belief = _belief_for(state, observer)

        builder = ObservationBuilder()
        out_s = np.zeros((OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        out_g = np.zeros(OBS_GLOBAL_DIMS, dtype=np.float32)
        builder.build_into(state, belief, observer, out_s, out_g)

        # Reference via the ordinary .build() path.
        ref = builder.build(state, belief, observer)

        np.testing.assert_array_equal(out_s, ref.spatial)
        np.testing.assert_array_equal(out_g, ref.global_)

    def test_build_into_rejects_wrong_shape(self) -> None:
        state = _fresh_state(1)
        observer = state.turn
        belief = _belief_for(state, observer)
        builder = ObservationBuilder()

        bad_s = np.zeros((OBS_CHANNELS, 10, 17), dtype=np.float32)
        out_g = np.zeros(OBS_GLOBAL_DIMS, dtype=np.float32)
        with pytest.raises(ValueError, match="out_spatial shape"):
            builder.build_into(state, belief, observer, bad_s, out_g)

        good_s = np.zeros((OBS_CHANNELS, 17, 17), dtype=np.float32)
        bad_g = np.zeros(99, dtype=np.float32)
        with pytest.raises(ValueError, match="out_global shape"):
            builder.build_into(state, belief, observer, good_s, bad_g)

    def test_build_into_rejects_wrong_dtype(self) -> None:
        state = _fresh_state(2)
        observer = state.turn
        belief = _belief_for(state, observer)
        builder = ObservationBuilder()

        bad_s = np.zeros((OBS_CHANNELS, 17, 17), dtype=np.float64)
        out_g = np.zeros(OBS_GLOBAL_DIMS, dtype=np.float32)
        with pytest.raises(TypeError, match="out_spatial must be float32"):
            builder.build_into(state, belief, observer, bad_s, out_g)


# ---------------------------------------------------------------------------
# 2. Batch API equivalence
# ---------------------------------------------------------------------------


class TestBatchEquivalence:
    def test_batch_matches_per_state(self) -> None:
        """64-way batch output must match per-state build_into."""
        N = 64
        states: list[GameState] = []
        beliefs: list[BeliefTensor] = []
        observers: list[Seat] = []
        for i in range(N):
            state = _fresh_state(seed=i * 7, steps=i % 20)
            observer = ALL_SEATS[i % 4]
            if state.info[observer].dead:
                observer = state.turn
            states.append(state)
            beliefs.append(_belief_for(state, observer))
            observers.append(observer)

        builder = ObservationBuilder()

        # Batch path.
        out_s = np.zeros(
            (N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        out_g = np.zeros((N, OBS_GLOBAL_DIMS), dtype=np.float32)
        builder.build_observations_batch(states, beliefs, observers, out_s, out_g)

        # Per-state reference.  (A fresh builder so internal scratch is
        # independent; the writer is deterministic.)
        ref_builder = ObservationBuilder()
        for i in range(N):
            ref_s = np.zeros(
                (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
            )
            ref_g = np.zeros(OBS_GLOBAL_DIMS, dtype=np.float32)
            ref_builder.build_into(states[i], beliefs[i], observers[i], ref_s, ref_g)
            np.testing.assert_array_equal(out_s[i], ref_s)
            np.testing.assert_array_equal(out_g[i], ref_g)

    def test_batch_length_mismatch_raises(self) -> None:
        state = _fresh_state(3)
        belief = _belief_for(state, state.turn)
        builder = ObservationBuilder()
        out_s = np.zeros(
            (1, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        out_g = np.zeros((1, OBS_GLOBAL_DIMS), dtype=np.float32)
        with pytest.raises(ValueError, match="batch length mismatch"):
            builder.build_observations_batch(
                [state, state], [belief], [state.turn], out_s, out_g,
            )


# ---------------------------------------------------------------------------
# 3. Torch bridge (skipped if torch not installed)
# ---------------------------------------------------------------------------


def _import_torch_or_skip():
    import importlib
    try:
        return importlib.import_module("torch")
    except ImportError:
        pytest.skip("torch not installed")


class TestTorchBridge:
    def test_numpy_view_shares_storage(self) -> None:
        torch = _import_torch_or_skip()
        from junqi_core.observation import numpy_view_of_torch_cpu

        t = torch.zeros(4, 5, dtype=torch.float32)
        a = numpy_view_of_torch_cpu(t)
        assert a.shape == (4, 5)
        assert a.dtype == np.float32
        a[1, 2] = 42.0
        # Writing through numpy must be visible in the tensor.
        assert float(t[1, 2]) == 42.0

    def test_torch_batch_equals_numpy_batch(self) -> None:
        torch = _import_torch_or_skip()

        N = 16
        states: list[GameState] = []
        beliefs: list[BeliefTensor] = []
        observers: list[Seat] = []
        for i in range(N):
            state = _fresh_state(seed=100 + i, steps=i % 15)
            observer = ALL_SEATS[i % 4]
            if state.info[observer].dead:
                observer = state.turn
            states.append(state)
            beliefs.append(_belief_for(state, observer))
            observers.append(observer)

        builder = ObservationBuilder()

        # Numpy reference.
        ref_s = np.zeros(
            (N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        ref_g = np.zeros((N, OBS_GLOBAL_DIMS), dtype=np.float32)
        builder.build_observations_batch(states, beliefs, observers, ref_s, ref_g)

        # Torch path.
        t_s = torch.zeros(
            (N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32
        )
        t_g = torch.zeros((N, OBS_GLOBAL_DIMS), dtype=torch.float32)
        builder.build_observations_batch_torch(
            states, beliefs, observers, t_s, t_g,
        )
        np.testing.assert_array_equal(t_s.numpy(), ref_s)
        np.testing.assert_array_equal(t_g.numpy(), ref_g)

    def test_torch_bridge_rejects_gpu(self) -> None:
        torch = _import_torch_or_skip()
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        from junqi_core.observation import numpy_view_of_torch_cpu

        t = torch.zeros(2, 2, dtype=torch.float32, device="cuda")
        with pytest.raises(ValueError, match="CPU tensors"):
            numpy_view_of_torch_cpu(t)

    def test_torch_bridge_rejects_noncontig(self) -> None:
        torch = _import_torch_or_skip()
        from junqi_core.observation import numpy_view_of_torch_cpu

        t = torch.zeros(4, 8, dtype=torch.float32)
        view = t[:, ::2]  # non-contiguous
        with pytest.raises(ValueError, match="contiguous"):
            numpy_view_of_torch_cpu(view)
