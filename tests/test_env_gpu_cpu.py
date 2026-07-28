"""tests/test_env_gpu_cpu.py — CPU-side unit tests for env_gpu.py helpers.

These tests exercise the Python helper functions (_build_belief_tensor_one,
_build_belief_batch, _pack_state_arrays) that run on CPU, WITHOUT requiring
the junqi_cuda CUDA extension to be compiled.  They validate:

1. Belief tensor scatter: shape, dtype, live-cell validity.
2. State packing: cell_piece_id_per_piece correctness (alive→flat, dead→-1).
3. Dtype and shape constraints that the CUDA bindings enforce at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import (
    _build_belief_batch,
    _build_belief_tensor_one,
    _pack_state_arrays,
)
from junqi_core.board import BOARD_SIZE, NUM_CELLS
from junqi_core.info_model import NUM_TRACKED_TYPES
from junqi_core.rules import ShowMode


@pytest.fixture
def single_env():
    env = JunqiEnv(show_mode=ShowMode.HALF_DARK)
    env.reset(seed=42)
    return env


@pytest.fixture
def two_envs():
    envs = [JunqiEnv(show_mode=ShowMode.HALF_DARK) for _ in range(2)]
    for i, e in enumerate(envs):
        e.reset(seed=i * 100)
    return envs


# ---------------------------------------------------------------------------
# Belief tensor tests
# ---------------------------------------------------------------------------

class TestBuildBeliefTensorOne:
    def test_shape(self, single_env):
        bel = _build_belief_tensor_one(single_env)
        assert bel.shape == (4, NUM_TRACKED_TYPES, NUM_CELLS)

    def test_dtype(self, single_env):
        bel = _build_belief_tensor_one(single_env)
        assert bel.dtype == np.float32

    def test_values_in_range(self, single_env):
        bel = _build_belief_tensor_one(single_env)
        assert np.all(bel >= 0.0)
        assert np.all(bel <= 1.0 + 1e-6)

    def test_empty_cells_are_zero(self, single_env):
        """Cells with no piece should have zero belief for all seats/types."""
        state = single_env.state
        bel = _build_belief_tensor_one(single_env)
        empty_cells = np.where(state.cell_piece_id < 0)[0]
        # For all seats and types, empty cells must be zero
        assert np.all(bel[:, :, empty_cells] == 0.0)


class TestBuildBeliefBatch:
    def test_shape(self, two_envs):
        batch = _build_belief_batch(two_envs)
        assert batch.shape == (2, 4, NUM_TRACKED_TYPES, NUM_CELLS)

    def test_dtype(self, two_envs):
        batch = _build_belief_batch(two_envs)
        assert batch.dtype == np.float32

    def test_c_contiguous(self, two_envs):
        batch = _build_belief_batch(two_envs)
        assert batch.flags['C_CONTIGUOUS']


# ---------------------------------------------------------------------------
# State packing tests
# ---------------------------------------------------------------------------

class TestPackStateArrays:
    def test_keys_present(self, two_envs):
        d = _pack_state_arrays(two_envs)
        required_keys = {
            "cell_piece_id_per_piece", "piece_seat_arr", "piece_type_arr",
            "alive", "pos_x", "pos_y", "zero_x", "zero_y",
            "move_count_arr", "active_eat_arr", "passive_surv_arr",
            "death_reason_arr", "death_step_arr", "death_loc_flat_arr",
            "cell_piece_id", "seat_dead_arr", "seat_flag_revealed_arr",
            "turn", "zobrist", "move_counter", "moves_since_last_combat",
        }
        assert required_keys <= set(d.keys())

    def test_piece_array_shapes(self, two_envs):
        N = len(two_envs)
        d = _pack_state_arrays(two_envs)
        for key in ("cell_piece_id_per_piece", "move_count_arr", "active_eat_arr",
                    "passive_surv_arr", "death_step_arr", "death_loc_flat_arr"):
            assert d[key].shape == (N * 120,), f"{key} shape mismatch"
        for key in ("piece_seat_arr", "piece_type_arr",
                    "pos_x", "pos_y", "zero_x", "zero_y", "death_reason_arr"):
            assert d[key].shape == (N * 120,), f"{key} shape mismatch"
        assert d["alive"].shape == (N * 120,)
        assert d["cell_piece_id"].shape == (N * 289,)
        assert d["seat_dead_arr"].shape == (N * 4,)
        assert d["seat_flag_revealed_arr"].shape == (N * 4,)
        assert d["turn"].shape == (N,)
        assert d["zobrist"].shape == (N,)
        assert d["move_counter"].shape == (N,)
        assert d["moves_since_last_combat"].shape == (N,)

    def test_dtypes(self, two_envs):
        d = _pack_state_arrays(two_envs)
        assert d["cell_piece_id_per_piece"].dtype == np.int16
        assert d["piece_seat_arr"].dtype == np.int8
        assert d["piece_type_arr"].dtype == np.int8
        assert d["alive"].dtype == bool
        assert d["pos_x"].dtype == np.int8
        assert d["pos_y"].dtype == np.int8
        assert d["zero_x"].dtype == np.int8
        assert d["zero_y"].dtype == np.int8
        assert d["move_count_arr"].dtype == np.int16
        assert d["active_eat_arr"].dtype == np.int16
        assert d["passive_surv_arr"].dtype == np.int16
        assert d["death_reason_arr"].dtype == np.int8
        assert d["death_step_arr"].dtype == np.int16
        assert d["death_loc_flat_arr"].dtype == np.int16
        assert d["cell_piece_id"].dtype == np.int16
        assert d["seat_dead_arr"].dtype == bool
        assert d["seat_flag_revealed_arr"].dtype == bool
        assert d["turn"].dtype == np.int8
        assert d["zobrist"].dtype == np.int64
        assert d["move_counter"].dtype == np.int32
        assert d["moves_since_last_combat"].dtype == np.int32

    def test_cell_piece_id_per_piece_alive_valid_range(self, two_envs):
        """Alive pieces must have flat index in [0, 288]."""
        N = len(two_envs)
        d = _pack_state_arrays(two_envs)
        cpip = d["cell_piece_id_per_piece"].reshape(N, 120)
        alv  = d["alive"].reshape(N, 120)
        alive_flats = cpip[alv]
        assert np.all(alive_flats >= 0), "alive piece has negative flat cell"
        assert np.all(alive_flats < NUM_CELLS), \
            f"alive piece flat >= NUM_CELLS={NUM_CELLS}"

    def test_cell_piece_id_per_piece_dead_is_neg1(self, two_envs):
        """Dead pieces must have flat index == -1."""
        N = len(two_envs)
        d = _pack_state_arrays(two_envs)
        cpip = d["cell_piece_id_per_piece"].reshape(N, 120)
        alv  = d["alive"].reshape(N, 120)
        dead_flats = cpip[~alv]
        assert np.all(dead_flats == -1), \
            f"dead piece has non-(-1) flat: {np.unique(dead_flats)}"

    def test_cell_piece_id_per_piece_matches_pos(self, two_envs):
        """Alive piece flat must equal pos_y * 17 + pos_x."""
        N = len(two_envs)
        d = _pack_state_arrays(two_envs)
        cpip = d["cell_piece_id_per_piece"].reshape(N, 120)
        alv  = d["alive"].reshape(N, 120)
        px   = d["pos_x"].reshape(N, 120).astype(np.int16)
        py   = d["pos_y"].reshape(N, 120).astype(np.int16)
        expected = py * np.int16(BOARD_SIZE) + px
        np.testing.assert_array_equal(
            cpip[alv], expected[alv],
            err_msg="cpip != pos_y*17+pos_x for alive pieces"
        )

    def test_c_contiguous(self, two_envs):
        d = _pack_state_arrays(two_envs)
        for k, v in d.items():
            assert v.flags['C_CONTIGUOUS'], f"{k} is not C-contiguous"

    def test_single_env(self, single_env):
        """_pack_state_arrays should work for N=1."""
        d = _pack_state_arrays([single_env])
        assert d["cell_piece_id_per_piece"].shape == (120,)
        assert d["cell_piece_id"].shape == (289,)
