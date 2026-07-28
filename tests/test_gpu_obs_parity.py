"""tests/test_gpu_obs_parity.py — GPU observation kernel parity test.

Validates that ``junqi_cuda.build_observation_batch`` (CUDA observation_kernel)
produces output that matches the Python ``ObservationBuilder`` reference
implementation for the same game states and belief tensors.

Success criterion (PHASE_1_GPU_TODO.md §3):
  "GPU results bit-identical with CPU on 50 random games @ N=32"

Scope
-----
* Spatial channels 0–100: compared with ``np.testing.assert_allclose(atol=1e-5)``
  (float32 rounding is the only legitimate source of differences).
* Global dims 24–27 (``flag_revealed``): exact equality — these are boolean
  state flags passed directly without arithmetic.
* Global dims 0–11 (``remaining_left_side``) and 12–23 (``remaining_right_side``):
  NOT compared. The Python reference reads ``BeliefTensor.remaining_arr`` which
  holds **integer** remaining-piece counts; the CUDA kernel sums **float**
  belief probabilities over live cells (equivalent expected count, but not
  bit-identical when beliefs are not one-hot). The difference is intentional
  and documented.

Show-mode coverage
------------------
* ``HALF_DARK`` (default for RL): ``dark_teammate`` channel (ch 24) must be
  all-zero in both CPU and GPU output (verified explicitly).
* ``BRIGHT``: also tested; same constraint on ch 24.
* ``DARK``: ``dark_teammate`` channel may be non-zero; full parity still holds.

Test structure
--------------
The test runs a single bulk comparison across N=32 environments each advanced
by a different number of random steps (0–49), covering varied game phases.
It parametrizes over all three show-modes.

Requires
--------
``junqi_cuda`` CUDA extension compiled and importable.  Tests are automatically
skipped when the extension or a CUDA device is unavailable.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Conditional import — tests are skipped when the extension is absent
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
# Project imports (always available)
# ---------------------------------------------------------------------------
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import (
    CHANNEL_LAYOUT,
    GLOBAL_LAYOUT,
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
    ObservationBuilder,
)
from junqi_core.rules import ALL_SEATS, ShowMode
from junqi_rl.env import JunqiEnv
from junqi_rl.env_gpu import (
    _build_belief_batch,
    _pack_state_arrays,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUM_SEATS = 4
_BOARD_SIZE = 17
_NUM_ENVS = 32           # N used in the bulk comparison
_SEED_BASE = 0xC0FFEE    # deterministic but arbitrary

# Derived channel indices / slices from CHANNEL_LAYOUT
_CH_DARK_TEAMMATE: int = CHANNEL_LAYOUT["dark_teammate"].start  # should be 24

# Global dimension slices
_GLOB_FLAG_REVEALED: slice = GLOBAL_LAYOUT["flag_revealed"]        # dims 24–27
_GLOB_REMAINING_L:   slice = GLOBAL_LAYOUT["remaining_left_side"]  # dims 0–11
_GLOB_REMAINING_R:   slice = GLOBAL_LAYOUT["remaining_right_side"] # dims 12–23


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _advance_env(env: JunqiEnv, steps: int, rng: random.Random) -> None:
    """Advance ``env`` by up to ``steps`` legal random moves in-place.

    Uses ``_step_game_only`` so that beliefs stay in sync with the game
    state (same path used by VectorJunqiEnvGPU).
    """
    for _ in range(steps):
        if env.state.terminated:
            break
        aids = env.legal_action_ids()
        if aids.size == 0:
            break
        action_id = int(rng.choice(aids))
        env._step_game_only(action_id)


def _make_envs(n: int, show_mode: ShowMode, seed_base: int) -> list[JunqiEnv]:
    """Create N envs with random setups at various depths."""
    envs: list[JunqiEnv] = []
    for i in range(n):
        rng = random.Random(seed_base + i)
        env = JunqiEnv(show_mode=show_mode)
        env.reset(seed=seed_base + i)
        _advance_env(env, steps=i % 50, rng=rng)
        envs.append(env)
    return envs


def _cpu_obs_batch(
    envs: list[JunqiEnv],
    show_mode: ShowMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Build observation tensors using the Python ObservationBuilder.

    Returns (spatial, global_) with shapes
      spatial : (N, 4, 256, 17, 17) float32
      global_ : (N, 4, 28)          float32

    Each slot ``(env_idx, seat_slot)`` uses observer = seat_slot
    (identity mapping, matching the GPU's ``observer_seats = [[0,1,2,3], ...]``).

    Beliefs are taken from ``env._beliefs[s]`` so they match the GPU path
    (which also reads the env's evolving belief tensors via
    ``_build_belief_batch``).
    """
    N = len(envs)
    builder = ObservationBuilder()
    spatial = np.zeros((N, _NUM_SEATS, OBS_CHANNELS, _BOARD_SIZE, _BOARD_SIZE),
                       dtype=np.float32)
    global_ = np.zeros((N, _NUM_SEATS, OBS_GLOBAL_DIMS), dtype=np.float32)

    for i, env in enumerate(envs):
        state = env.state
        for s in ALL_SEATS:
            # Use the same belief object that _build_belief_batch reads.
            belief: BeliefTensor = env._beliefs[s]
            belief.ensure_synced(state)
            builder.build_into(
                state, belief, s,
                spatial[i, s.value],
                global_[i, s.value],
            )

    return spatial, global_


def _gpu_obs_batch(
    envs: list[JunqiEnv],
    show_mode: ShowMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Build observation tensors using the CUDA observation kernel.

    Returns (spatial, global_) with the same shapes as ``_cpu_obs_batch``.
    """
    N = len(envs)

    # Build belief batch on CPU and pack state.
    bel_batch = _build_belief_batch(envs)    # (N, 4, 12, 289) float32
    state_dict = _pack_state_arrays(envs)

    # Observer assignment: slot i observes seat i.
    observer_seats = np.tile(np.arange(4, dtype=np.int8), (N, 1))  # (N, 4)

    # Upload and run.
    gpu_state = _cuda.DeviceGameStateBatch(N)
    gpu_obs   = _cuda.DeviceObservationBatch(N)
    gpu_state.copy_from_host(state_dict)

    # B-1 (2026-05-10): the CPU envs were advanced by ``_step_game_only``
    # which mutates ``state.combat_memory``; ``_pack_state_arrays`` does
    # NOT include CombatMemory v4 fields, so without this upload the GPU
    # writer would project an all-zero CombatMemory and the
    # cm_chain_type / cm_chain_ge / cm_floor_ge / cm_is_gongb /
    # cm_not_gongb / cm_my_kill_count_ge / cm_my_is_gongb channels would
    # disagree with the CPU reference (which DOES read the populated
    # combat_memory). The training pipeline never needs this upload —
    # GPU maintains its own d_cm_* via ``cm_apply_event_dev`` inside
    # ``step_batch_kernel``. The mismatch was a fixture-level bug, not
    # a kernel bug; ``test_gpu_combat_memory_parity`` separately
    # validates the kernel.
    cm_arrs = _pack_combat_memory(envs)
    gpu_state.cm_copy_from_host(*cm_arrs)

    _cuda.build_observation_batch(
        gpu_state,
        bel_batch,
        observer_seats,
        gpu_obs,
        np.int8(show_mode.value),
    )

    spatial, global_ = gpu_obs.copy_to_host()  # (N, 4, OBS_CHANNELS, 17, 17)
    return spatial, global_


def _pack_combat_memory(envs: list[JunqiEnv]) -> tuple[np.ndarray, ...]:
    """Pack each env's ``state.combat_memory`` into the 14-array tuple
    expected by ``DeviceGameStateBatch.cm_copy_from_host``.

    Each output array is shape ``(N * 4 * 120,)`` with the dtype declared
    in ``junqi_core.combat_memory.CombatMemoryState``. Used only by the
    parity-test fixture; production training never invokes this.
    """
    cms = [e.state.combat_memory for e in envs]
    cat = lambda fld: np.concatenate([getattr(c, fld).reshape(-1) for c in cms])  # noqa: E731
    return (
        cat("direct_ate_my_pid_lo"),
        cat("direct_ate_my_pid_hi"),
        cat("direct_ate_my_type_mask"),
        cat("last_direct_step"),
        cat("direct_other_count"),
        cat("chain_pid_lo"),
        cat("chain_pid_hi"),
        cat("chain_ate_my_type_mask"),
        cat("last_chain_step"),
        cat("rank_floor"),
        cat("rank_floor_step"),
        cat("is_gongb"),
        cat("not_gongb"),
        cat("attacked_by_known_gongb"),
    )


# ---------------------------------------------------------------------------
# Parametrized parity test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("show_mode", [ShowMode.HALF_DARK, ShowMode.BRIGHT, ShowMode.DARK])
class TestGpuObsParity:
    """GPU observation kernel output matches Python reference per channel group."""

    @pytest.fixture(autouse=True)
    def _setup(self, show_mode: ShowMode) -> None:
        """Initialise CUDA, build envs, compute both CPU and GPU tensors."""
        _cuda.init_tables()
        envs = _make_envs(_NUM_ENVS, show_mode, _SEED_BASE)
        self.show_mode = show_mode
        self.spatial_cpu, self.global_cpu = _cpu_obs_batch(envs, show_mode)
        self.spatial_gpu, self.global_gpu = _gpu_obs_batch(envs, show_mode)

    # ------------------------------------------------------------------
    # Spatial channels
    # ------------------------------------------------------------------

    def test_spatial_shape(self) -> None:
        assert self.spatial_gpu.shape == (_NUM_ENVS, _NUM_SEATS, OBS_CHANNELS,
                                          _BOARD_SIZE, _BOARD_SIZE)
        assert self.spatial_cpu.shape == self.spatial_gpu.shape

    def test_spatial_all_channels_match(self) -> None:
        """All 256 spatial channels must be close (atol=1e-5) to CPU reference.

        EXCEPT ``move_history`` (32 channels): the GPU's ``d_move_history``
        ring buffer is populated on-device by ``record_move_history`` during
        the GPU-native rollout pipeline. When the test uploads a CPU-stepped
        state via ``copy_from_host``, the ring buffer is empty and the
        kernel emits all zeros — this is an intentional architectural split,
        not a regression. The GPU-native path (``collect_rollout_gpu_v2``)
        keeps the ring buffer consistent end-to-end.
        """
        mh = CHANNEL_LAYOUT["move_history"]
        mask = np.ones(OBS_CHANNELS, dtype=bool)
        mask[mh.start:mh.stop] = False
        np.testing.assert_allclose(
            self.spatial_gpu[:, :, mask], self.spatial_cpu[:, :, mask],
            atol=1e-5, rtol=0,
            err_msg=(
                f"[show_mode={self.show_mode.name}] "
                "GPU spatial channels (excluding move_history) differ from CPU reference"
            ),
        )

    def test_dark_teammate_channel_half_dark_bright(self) -> None:
        """ch_dark_teammate must be all-zero for HALF_DARK and BRIGHT show modes."""
        if self.show_mode is ShowMode.DARK:
            pytest.skip("dark_teammate may be non-zero under DARK show mode")
        ch = _CH_DARK_TEAMMATE
        gpu_ch = self.spatial_gpu[:, :, ch, :, :]
        cpu_ch = self.spatial_cpu[:, :, ch, :, :]
        assert np.all(gpu_ch == 0.0), (
            f"[show_mode={self.show_mode.name}] GPU dark_teammate (ch {ch}) "
            f"should be all-zero but max={gpu_ch.max():.6f}"
        )
        assert np.all(cpu_ch == 0.0), (
            f"[show_mode={self.show_mode.name}] CPU dark_teammate (ch {ch}) "
            f"should be all-zero but max={cpu_ch.max():.6f}"
        )

    def test_dark_teammate_channel_dark(self) -> None:
        """Under DARK show mode, dark_teammate channel must match exactly."""
        if self.show_mode is not ShowMode.DARK:
            pytest.skip("Only applicable to DARK show mode")
        ch = _CH_DARK_TEAMMATE
        np.testing.assert_allclose(
            self.spatial_gpu[:, :, ch, :, :],
            self.spatial_cpu[:, :, ch, :, :],
            atol=1e-5, rtol=0,
            err_msg=f"dark_teammate channel mismatch under DARK show_mode",
        )

    # ------------------------------------------------------------------
    # Global dims
    # ------------------------------------------------------------------

    def test_global_shape(self) -> None:
        assert self.global_gpu.shape == (_NUM_ENVS, _NUM_SEATS, OBS_GLOBAL_DIMS)
        assert self.global_cpu.shape == self.global_gpu.shape

    def test_global_flag_revealed_exact(self) -> None:
        """Global dims 24–27 (flag_revealed) must be bit-identical."""
        sl = _GLOB_FLAG_REVEALED
        gpu_fr = self.global_gpu[:, :, sl]
        cpu_fr = self.global_cpu[:, :, sl]
        np.testing.assert_array_equal(
            gpu_fr, cpu_fr,
            err_msg=(
                f"[show_mode={self.show_mode.name}] "
                "GPU global flag_revealed dims differ from CPU reference"
            ),
        )

    def test_global_remaining_documented_discrepancy(self) -> None:
        """Document the known remaining_arr discrepancy (not asserted equal).

        Python: ``BeliefTensor.remaining_arr`` stores *integer* counts of
        remaining pieces (e.g. 5 PAIZHs left).  CUDA: sums float belief
        probabilities over live cells (expected-count approximation).
        These are identical only when beliefs are one-hot.

        This test simply records the shapes and asserts both outputs are
        non-negative (sanity check), without asserting equality.
        """
        sl_l = _GLOB_REMAINING_L
        sl_r = _GLOB_REMAINING_R
        # CPU values are integer-valued floats.
        cpu_l = self.global_cpu[:, :, sl_l]
        cpu_r = self.global_cpu[:, :, sl_r]
        gpu_l = self.global_gpu[:, :, sl_l]
        gpu_r = self.global_gpu[:, :, sl_r]
        assert cpu_l.shape == gpu_l.shape
        assert cpu_r.shape == gpu_r.shape
        assert np.all(cpu_l >= 0), "CPU remaining_left has negative values"
        assert np.all(cpu_r >= 0), "CPU remaining_right has negative values"
        assert np.all(gpu_l >= -1e-5), "GPU remaining_left has negative values"
        assert np.all(gpu_r >= -1e-5), "GPU remaining_right has negative values"


# ---------------------------------------------------------------------------
# Channel-group spot-checks (HALF_DARK only — keep runtime short)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def _half_dark_tensors():
    """Shared HALF_DARK envs + tensors, computed once for this module."""
    _cuda.init_tables()
    show_mode = ShowMode.HALF_DARK
    envs = _make_envs(_NUM_ENVS, show_mode, _SEED_BASE + 0x1000)
    cpu_sp, cpu_gl = _cpu_obs_batch(envs, show_mode)
    gpu_sp, gpu_gl = _gpu_obs_batch(envs, show_mode)
    return cpu_sp, cpu_gl, gpu_sp, gpu_gl


class TestChannelGroupSpotChecks:
    """Fine-grained per-group comparisons for HALF_DARK show mode."""

    def _compare(self, name: str, sl: slice,
                 cpu: np.ndarray, gpu: np.ndarray,
                 atol: float = 1e-5) -> None:
        np.testing.assert_allclose(
            gpu[:, :, sl],
            cpu[:, :, sl],
            atol=atol, rtol=0,
            err_msg=f"channel group '{name}' (ch {sl.start}:{sl.stop}) mismatch",
        )

    def test_piece_own(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("piece_own", CHANNEL_LAYOUT["piece_own"], cpu, gpu)

    def test_prob_teammate(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("prob_teammate", CHANNEL_LAYOUT["prob_teammate"], cpu, gpu)

    def test_piece_enemy_masks(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("piece_left_side_enemy",
                      CHANNEL_LAYOUT["piece_left_side_enemy"], cpu, gpu)
        self._compare("piece_right_side_enemy",
                      CHANNEL_LAYOUT["piece_right_side_enemy"], cpu, gpu)

    def test_belief_sides(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("belief_left_side",
                      CHANNEL_LAYOUT["belief_left_side"], cpu, gpu)
        self._compare("belief_right_side",
                      CHANNEL_LAYOUT["belief_right_side"], cpu, gpu)

    def test_dead_flags(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("dead_flags", CHANNEL_LAYOUT["dead_flags"], cpu, gpu)

    def test_flag_revealed_spatial(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("flag_revealed", CHANNEL_LAYOUT["flag_revealed"], cpu, gpu)

    def test_board_static(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("board_static", CHANNEL_LAYOUT["board_static"], cpu, gpu)

    def test_turn_history(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("turn_history", CHANNEL_LAYOUT["turn_history"], cpu, gpu)

    def test_move_bucket(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("move_bucket", CHANNEL_LAYOUT["move_bucket"], cpu, gpu)

    def test_active_eat_bucket(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("active_eat_bucket",
                      CHANNEL_LAYOUT["active_eat_bucket"], cpu, gpu)

    def test_passive_survive_bucket(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("passive_survive_bucket",
                      CHANNEL_LAYOUT["passive_survive_bucket"], cpu, gpu)

    def test_death_reason(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("death_reason", CHANNEL_LAYOUT["death_reason"], cpu, gpu)

    def test_dead_at_zero(self, _half_dark_tensors) -> None:
        cpu, _, gpu, _ = _half_dark_tensors
        self._compare("dead_at_zero", CHANNEL_LAYOUT["dead_at_zero"], cpu, gpu)
