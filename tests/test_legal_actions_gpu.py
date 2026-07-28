"""tests/test_legal_actions_gpu.py — GPU legal-action kernel parity test.

Validates that ``junqi_cuda.legal_action_ids_batch`` produces a set of legal
action IDs that is bit-identical (as a **sorted set**) to the Python reference
implementation ``generate_legal_action_ids_batch`` for the same game states.

Success criterion (PHASE_1_GPU_TODO.md §2 / M2):
  "GPU results set-equivalent with CPU on 50 random games @ N=32"

Test structure
--------------
We create N=32 environments each advanced by a different number of random steps
(0–49) so that the set of states covers varied game phases.  For each env we
compare the GPU and CPU legal-action sets using sorted arrays and
``np.testing.assert_array_equal``.

Additionally:
* Counts: ``d_counts[i]`` must equal ``len(cpu_aids[i])`` for all i.
* Overflow guard: ``d_counts[i] < 512`` for all i (MAX_ACTIONS_PER_ENV).
* Action validity: every GPU action id must be in [0, 289*289).
* Dead/terminated envs: counts must be 0 when the active seat is dead.

Requires
--------
``junqi_cuda`` CUDA extension compiled and importable.  Tests are automatically
skipped when the extension or a CUDA-capable GPU is unavailable.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional import
# ---------------------------------------------------------------------------
try:
    import junqi_cuda as _cuda
    _CUDA_AVAILABLE = _cuda.get_gpu_count() > 0
except ImportError:
    _cuda = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="junqi_cuda extension not available or no CUDA-capable GPU",
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from junqi_core.move_gen import generate_legal_action_ids_batch
from junqi_core.rules import ALL_SEATS
from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import _pack_state_arrays

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUM_ENVS  = 32
_SEED_BASE = 0xDEADBEEF
_MAX_ACTIONS_PER_ENV = 512
_FLAT_ACTION_SPACE   = 289 * 289  # 83521


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _advance_env(env: JunqiEnv, steps: int, rng: random.Random) -> None:
    """Advance *env* by up to *steps* random legal moves."""
    for _ in range(steps):
        if env.state.terminated:
            break
        aids = env.legal_action_ids()
        if aids.size == 0:
            break
        action_id = int(rng.choice(aids))
        env._step_game_only(action_id)


def _make_envs(n: int, seed_base: int) -> list[JunqiEnv]:
    """Create N envs with random setups at various depths (0–49 steps)."""
    envs: list[JunqiEnv] = []
    for i in range(n):
        rng = random.Random(seed_base + i)
        env = JunqiEnv()
        env.reset(seed=seed_base + i)
        _advance_env(env, steps=i % 50, rng=rng)
        envs.append(env)
    return envs


def _cpu_legal_actions(envs: list[JunqiEnv]) -> list[np.ndarray]:
    """Return list of CPU legal-action id arrays, one per env (acting seat)."""
    result = []
    for env in envs:
        st = env.state
        if st.terminated:
            result.append(np.empty(0, dtype=np.int32))
            continue
        acting = st.turn
        if st.info[acting].dead:
            result.append(np.empty(0, dtype=np.int32))
            continue
        aids = generate_legal_action_ids_batch(
            st.cell_piece_id, st.piece_seat_arr, st.piece_type_arr,
            st.alive, st.pos_x, st.pos_y, acting.value,
        )
        result.append(aids.astype(np.int32))
    return result


def _gpu_legal_actions(
    envs: list[JunqiEnv],
) -> tuple[np.ndarray, np.ndarray]:
    """Run GPU legal-action kernel and return (action_ids, counts).

    action_ids : shape (N, 512) int32 — valid entries are [i, :counts[i]]
    counts     : shape (N,)     int32
    """
    N = len(envs)

    # Build acting_seats from each env's current turn.
    acting_seats = np.array(
        [e.state.turn.value for e in envs], dtype=np.int8
    )

    # Pack and upload state.
    state_dict  = _pack_state_arrays(envs)
    gpu_state   = _cuda.DeviceGameStateBatch(N)
    gpu_state.copy_from_host(state_dict)

    action_ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting_seats)
    return action_ids, counts


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _shared():
    """Compute CPU and GPU legal actions once for the whole module."""
    _cuda.init_tables()
    envs = _make_envs(_NUM_ENVS, _SEED_BASE)
    cpu_aids   = _cpu_legal_actions(envs)
    gpu_ids, gpu_counts = _gpu_legal_actions(envs)
    return envs, cpu_aids, gpu_ids, gpu_counts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLegalActionsGpuParity:
    """GPU legal-action kernel parity with Python reference."""

    def test_shapes(self, _shared) -> None:
        """Output arrays have the expected shapes."""
        _, _, gpu_ids, gpu_counts = _shared
        assert gpu_ids.shape   == (_NUM_ENVS, _MAX_ACTIONS_PER_ENV), \
            f"action_ids shape mismatch: {gpu_ids.shape}"
        assert gpu_counts.shape == (_NUM_ENVS,), \
            f"counts shape mismatch: {gpu_counts.shape}"

    def test_counts_match_cpu(self, _shared) -> None:
        """GPU counts must equal CPU action-set sizes for all envs."""
        _, cpu_aids, _, gpu_counts = _shared
        for i, (cpu, cnt) in enumerate(zip(cpu_aids, gpu_counts)):
            assert int(cnt) == len(cpu), (
                f"env {i}: GPU count={cnt}, CPU count={len(cpu)}"
            )

    def test_no_overflow(self, _shared) -> None:
        """No env should hit the MAX_ACTIONS_PER_ENV cap."""
        _, _, _, gpu_counts = _shared
        assert int(gpu_counts.max()) < _MAX_ACTIONS_PER_ENV, (
            f"MAX_ACTIONS_PER_ENV overflow: max count = {gpu_counts.max()}"
        )

    def test_action_ids_in_range(self, _shared) -> None:
        """All GPU action IDs (in valid slots) must be within [0, 289*289)."""
        _, _, gpu_ids, gpu_counts = _shared
        for i, cnt in enumerate(gpu_counts):
            if cnt == 0:
                continue
            ids = gpu_ids[i, :cnt]
            assert np.all(ids >= 0) and np.all(ids < _FLAT_ACTION_SPACE), (
                f"env {i}: action IDs out of range [0, {_FLAT_ACTION_SPACE})"
            )

    def test_action_sets_equal(self, _shared) -> None:
        """GPU action set must be identical to CPU action set (sorted comparison)."""
        _, cpu_aids, gpu_ids, gpu_counts = _shared
        for i, (cpu, cnt) in enumerate(zip(cpu_aids, gpu_counts)):
            gpu_slice = np.sort(gpu_ids[i, :cnt])
            cpu_sorted = np.sort(cpu.astype(np.int32))
            np.testing.assert_array_equal(
                gpu_slice, cpu_sorted,
                err_msg=(
                    f"env {i}: GPU action set differs from CPU reference "
                    f"(GPU={cnt}, CPU={len(cpu)})\n"
                    f"  GPU only: {np.setdiff1d(gpu_slice, cpu_sorted)}\n"
                    f"  CPU only: {np.setdiff1d(cpu_sorted, gpu_slice)}"
                ),
            )

    def test_dead_terminated_envs_zero_count(self, _shared) -> None:
        """Terminated or dead-seat envs should have GPU count == 0."""
        envs, _, _, gpu_counts = _shared
        for i, (env, cnt) in enumerate(zip(envs, gpu_counts)):
            st = env.state
            if st.terminated or st.info[st.turn].dead:
                assert cnt == 0, (
                    f"env {i}: expected 0 actions (terminated/dead) but GPU count={cnt}"
                )


class TestLegalActionsGpuEdgeCases:
    """Edge-case and regression checks for the GPU legal-action kernel."""

    @pytest.fixture(autouse=True)
    def _init(self) -> None:
        _cuda.init_tables()

    def test_fresh_game(self) -> None:
        """A freshly reset game should produce legal actions on the GPU."""
        env = JunqiEnv()
        env.reset(seed=42)

        cpu_aids = env.legal_action_ids()
        state_dict = _pack_state_arrays([env])
        gpu_state  = _cuda.DeviceGameStateBatch(1)
        gpu_state.copy_from_host(state_dict)

        acting = np.array([env.state.turn.value], dtype=np.int8)
        ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)

        assert int(counts[0]) == len(cpu_aids), (
            f"fresh game: GPU count={counts[0]}, CPU count={len(cpu_aids)}"
        )
        gpu_sorted = np.sort(ids[0, :counts[0]])
        cpu_sorted = np.sort(cpu_aids.astype(np.int32))
        np.testing.assert_array_equal(gpu_sorted, cpu_sorted,
                                      err_msg="fresh game action-set mismatch")

    def test_single_env_all_seats(self) -> None:
        """Query legal actions for every seat in a single env (4 queries)."""
        env = JunqiEnv()
        env.reset(seed=123)
        # Advance a few steps to make the board less trivial.
        rng = random.Random(123)
        _advance_env(env, 10, rng)

        state_dict = _pack_state_arrays([env])
        gpu_state  = _cuda.DeviceGameStateBatch(1)
        gpu_state.copy_from_host(state_dict)

        from junqi_core.rules import Seat
        for seat in ALL_SEATS:
            cpu_aids = env.state.legal_action_ids(seat)
            acting   = np.array([seat.value], dtype=np.int8)
            ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)

            assert int(counts[0]) == len(cpu_aids), (
                f"seat {seat}: GPU count={counts[0]}, CPU count={len(cpu_aids)}"
            )
            gpu_sorted = np.sort(ids[0, :counts[0]])
            cpu_sorted = np.sort(cpu_aids.astype(np.int32))
            np.testing.assert_array_equal(
                gpu_sorted, cpu_sorted,
                err_msg=f"seat {seat}: action-set mismatch"
            )

    def test_n_equals_1_batch(self) -> None:
        """Minimal batch size (N=1) works correctly."""
        env = JunqiEnv()
        env.reset(seed=7)

        cpu_aids = env.legal_action_ids()
        state_dict = _pack_state_arrays([env])
        gpu_state  = _cuda.DeviceGameStateBatch(1)
        gpu_state.copy_from_host(state_dict)

        acting = np.array([env.state.turn.value], dtype=np.int8)
        ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)
        assert ids.shape   == (1, _MAX_ACTIONS_PER_ENV)
        assert counts.shape == (1,)
        assert int(counts[0]) == len(cpu_aids)

    def test_large_batch(self) -> None:
        """Larger batch (N=64) runs without error and counts sum to a positive number."""
        N = 64
        _cuda.init_tables()
        envs = _make_envs(N, seed_base=0xABCDEF)
        state_dict = _pack_state_arrays(envs)
        gpu_state  = _cuda.DeviceGameStateBatch(N)
        gpu_state.copy_from_host(state_dict)

        acting = np.array([e.state.turn.value for e in envs], dtype=np.int8)
        ids, counts = _cuda.legal_action_ids_batch(gpu_state, acting)

        assert ids.shape    == (N, _MAX_ACTIONS_PER_ENV)
        assert counts.shape == (N,)
        assert int(counts.sum()) > 0, "Expected at least some legal actions in N=64 batch"
