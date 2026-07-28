"""junqi_rl.env_gpu — GPU-accelerated VectorJunqiEnv using the CUDA backend.

``VectorJunqiEnvGPU`` presents the **same interface** as ``VectorJunqiEnv``
(reset / step / obs_spatial / obs_global / legal_action_ids / current_seats)
but offloads observation building to the GPU via :mod:`junqi_cuda`.

Design
------
The CPU side (via ``JunqiEnv._step_game_only``) still handles all game logic
(rules, move validation, state transitions) because the Python ``GameState``
is the authoritative source of truth.  Only observation construction is
accelerated.  This is "Phase 1" of the GPU migration; game-step offload is
planned for Phase 2.

Belief tensor layout for CUDA
------------------------------
``observation_kernel`` expects ``d_beliefs`` of shape
``[N, 4, NUM_TRACKED_TYPES=12, NUM_CELLS=289] float32`` in flat C order.

The Python ``BeliefTensor.probs_arr`` is shaped ``(num_pids=120, 12)`` and
indexed by piece_id.  We scatter it into cell-space using
``state.cell_piece_id: ndarray[289] int16``:

    For each env ``e``, seat ``s`` (0..3):
        bel_out[e, s, :, flat] = belief.probs_arr[cell_piece_id[flat], :]
        (for flat where cell_piece_id[flat] >= 0)

This scatter is done via vectorised NumPy so it's fast even without the GPU.

Observer assignment
-------------------
``observer_seats[e, slot]`` — for the default case (each slot observes the
matching seat) this is just ``np.tile([0,1,2,3], (N,1))``.

See Also
--------
``junqi_rl.env.VectorJunqiEnv`` — CPU reference implementation.
``junqi_cuda`` — compiled CUDA extension (must be importable).
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from junqi_core.board import BOARD_SIZE, NUM_CELLS
from junqi_core.info_model import BeliefTensor, NUM_TRACKED_TYPES
from junqi_core.rules import ALL_SEATS, Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState

from .env import JunqiEnv, JunqiStepInfo

# ---------------------------------------------------------------------------
# Import the CUDA extension (optional — allows import even without GPU)
# ---------------------------------------------------------------------------
try:
    import junqi_cuda as _cuda  # type: ignore[import]
    _CUDA_AVAILABLE = True
except ImportError:
    _cuda = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False

_NUM_SEATS = 4


# ---------------------------------------------------------------------------
# Helper: build the belief tensor for one env in [4, 12, 289] layout
# ---------------------------------------------------------------------------

def _build_belief_tensor_one(
    envobj: JunqiEnv,
) -> np.ndarray:
    """Return float32 array [4, NUM_TRACKED_TYPES, NUM_CELLS] for one env.

    Uses ``BeliefTensor.probs_arr`` (piece-indexed, shape [120, 12]) and
    scatters into cell-indexed layout via ``state.cell_piece_id`` (shape [289]).
    """
    state = envobj.state
    cell_pid: np.ndarray = state.cell_piece_id   # [289] int16, -1 = empty

    bel = np.zeros((_NUM_SEATS, NUM_TRACKED_TYPES, NUM_CELLS), dtype=np.float32)

    for s in ALL_SEATS:
        belief: BeliefTensor = envobj._beliefs[s]
        # Ensure tensor mirror is up to date.
        belief.ensure_synced(state)
        probs_arr = belief.probs_arr  # [120, 12] float32

        # Vectorised scatter: for each cell with a live piece, copy its
        # belief vector into the output.
        live_cells = np.where(cell_pid >= 0)[0]   # flat cell indices
        if live_cells.size == 0:
            continue
        pids = cell_pid[live_cells].astype(np.intp)   # piece ids
        # Guard: pids must be within probs_arr bounds
        valid = (pids >= 0) & (pids < probs_arr.shape[0])
        live_cells = live_cells[valid]
        pids = pids[valid]
        if live_cells.size == 0:
            continue
        # bel[s, :, cell] = probs_arr[pid, :].
        # NumPy advanced-index rule: bel[scalar, :, fancy] → shape (K, 12)
        # where K = len(live_cells).  probs_arr[pids] is also (K, 12).
        bel[s.value, :, live_cells] = probs_arr[pids]   # (K, 12)

    return bel


def _build_belief_batch(envs: list[JunqiEnv]) -> np.ndarray:
    """Build [N, 4, 12, 289] belief batch from a list of JunqiEnv."""
    N = len(envs)
    out = np.zeros((N, _NUM_SEATS, NUM_TRACKED_TYPES, NUM_CELLS), dtype=np.float32)
    for i, env in enumerate(envs):
        out[i] = _build_belief_tensor_one(env)
    return out


# ---------------------------------------------------------------------------
# Helper: pack one env's SoA state into the 120-length arrays expected by
# DeviceGameStateBatch.copy_from_host.
# ---------------------------------------------------------------------------



def _pack_state_arrays_lite(
    envs: list[JunqiEnv],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack only the 6 fields the GPU legal_action kernel reads.

    Returns a tuple of C-contiguous flat arrays:
        (piece_seat_arr, piece_type_arr, alive, pos_x, pos_y, cell_piece_id)

    This is the companion of :py:meth:`DeviceGameStateBatch.copy_from_host_legal_lite`
    and is ~3× faster than :func:`_pack_state_arrays` because it skips 14 of
    the 20 SoA fields that the legal-action kernel does not consume.
    """
    states = [e.state for e in envs]
    psa = np.concatenate([s.piece_seat_arr for s in states])  # int8 (N*120,)
    pta = np.concatenate([s.piece_type_arr for s in states])  # int8
    alv = np.concatenate([s.alive          for s in states])  # bool
    px  = np.concatenate([s.pos_x          for s in states])  # int8
    py_ = np.concatenate([s.pos_y          for s in states])  # int8
    cpi = np.concatenate([s.cell_piece_id  for s in states])  # int16 (N*289,)
    return psa, pta, alv, px, py_, cpi


def _pack_state_arrays(
    envs: list[JunqiEnv],
) -> dict[str, np.ndarray]:
    """Pack all N envs' SoA state into a dict of (N * 120) or (N * K) arrays.

    Returns the dict expected by ``DeviceGameStateBatch.copy_from_host``.
    The arrays must be C-contiguous with the correct dtype.

    Implementation note — performance
    ---------------------------------
    Each state array in ``junqi_core.state.GameState`` is already the exact
    expected size with the correct dtype, so we concatenate lists of per-env
    arrays.  Measurements at N=1024 (conda numpy 2.x):

      * ``np.stack(list).reshape`` ≈ 0.73 ms   (previous impl)
      * ``np.concatenate(list)``   ≈ 0.24 ms   (this impl — 3× faster)

    ``np.concatenate`` directly produces the flat (N*K,) output we need; the
    reshape is only for computing the per-piece cell index, where we need the
    2-D view.  Total pack time for N=1024 drops from ~15 ms → ~5 ms.
    """
    N = len(envs)
    states = [e.state for e in envs]

    # ---- Concatenate all (120,) piece-indexed arrays into flat (N*120,) ----
    alv   = np.concatenate([s.alive              for s in states])  # bool
    psa   = np.concatenate([s.piece_seat_arr     for s in states])  # int8
    pta   = np.concatenate([s.piece_type_arr     for s in states])  # int8
    px    = np.concatenate([s.pos_x              for s in states])  # int8
    py_   = np.concatenate([s.pos_y              for s in states])  # int8
    zx    = np.concatenate([s.zero_x             for s in states])  # int8
    zy    = np.concatenate([s.zero_y             for s in states])  # int8
    mca   = np.concatenate([s.move_count_arr     for s in states])  # int16
    aea   = np.concatenate([s.active_eat_arr     for s in states])  # int16
    psa2  = np.concatenate([s.passive_surv_arr   for s in states])  # int16
    dra   = np.concatenate([s.death_reason_arr   for s in states])  # int8
    dsa   = np.concatenate([s.death_step_arr     for s in states])  # int16
    dlfa  = np.concatenate([s.death_loc_flat_arr for s in states])  # int16

    # ---- cell_piece_id_per_piece[i*120 + pid] ----
    # alive pieces: flat = pos_y * 17 + pos_x,  dead: -1.
    # Compute vectorised over the whole flat (N*120,) arrays.
    cpip = np.where(
        alv,
        py_.astype(np.int16) * np.int16(17) + px.astype(np.int16),
        np.int16(-1),
    ).astype(np.int16, copy=False)

    # ---- (N*289,) cell-indexed ----
    cpi  = np.concatenate([s.cell_piece_id       for s in states])  # int16

    # ---- (N*4,) seat-indexed ----
    sda  = np.concatenate([s.seat_dead_arr          for s in states])  # bool
    sfra = np.concatenate([s.seat_flag_revealed_arr for s in states])  # bool

    # ---- (N,) scalars ----
    # np.fromiter is faster than np.array(list(...)) for generators.
    turn = np.fromiter(
        (s.turn.value for s in states), dtype=np.int8, count=N,
    )
    zob  = np.fromiter(
        (getattr(s, "zobrist", 0) for s in states), dtype=np.int64, count=N,
    )
    mc   = np.fromiter(
        (s.move_counter for s in states), dtype=np.int32, count=N,
    )
    mslc = np.fromiter(
        (s.moves_since_last_combat for s in states), dtype=np.int32, count=N,
    )

    return {
        "cell_piece_id_per_piece":   cpip,
        "piece_seat_arr":            psa,
        "piece_type_arr":            pta,
        "alive":                     alv,
        "pos_x":                     px,
        "pos_y":                     py_,
        "zero_x":                    zx,
        "zero_y":                    zy,
        "move_count_arr":            mca,
        "active_eat_arr":            aea,
        "passive_surv_arr":          psa2,
        "death_reason_arr":          dra,
        "death_step_arr":            dsa,
        "death_loc_flat_arr":        dlfa,
        "cell_piece_id":             cpi,
        "seat_dead_arr":             sda,
        "seat_flag_revealed_arr":    sfra,
        "turn":                      turn,
        "zobrist":                   zob,
        "move_counter":              mc,
        "moves_since_last_combat":   mslc,
    }


# ---------------------------------------------------------------------------
# VectorJunqiEnvGPU
# ---------------------------------------------------------------------------


class VectorJunqiEnvGPU:
    """GPU-accelerated vectorised JunQi environment.

    Drop-in replacement for :class:`~junqi_rl.env.VectorJunqiEnv` that
    offloads observation building to the CUDA backend.  Game logic (rules,
    move generation, belief updates) still runs on CPU.

    Parameters
    ----------
    num_envs
        Number of parallel game environments.
    show_mode
        Visibility mode (default ``HALF_DARK`` for RL).
    max_num_moves
        Optional per-episode step limit.
    num_workers
        Thread-pool size for CPU game stepping.  Defaults to ``1`` (no pool).
    device_id
        CUDA device index (default 0).

    Raises
    ------
    ImportError
        If ``junqi_cuda`` extension is not compiled / importable.
    RuntimeError
        If the requested CUDA device is unavailable.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        show_mode: ShowMode = ShowMode.HALF_DARK,
        max_num_moves: int | None = None,
        num_workers: int = 1,
        device_id: int = 0,
    ) -> None:
        if not _CUDA_AVAILABLE:
            raise ImportError(
                "junqi_cuda extension is not available. "
                "Build it with: cmake -B build && cmake --build build -j8 "
                "from JunQi/src/env/cuda/"
            )
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if _cuda.get_gpu_count() == 0:
            raise RuntimeError("No CUDA-capable GPU found")
        _cuda.set_device(device_id)
        _cuda.init_tables()  # Upload board topology + Zobrist tables to GPU constant memory

        self.num_envs = num_envs
        self.show_mode = show_mode
        self._device_id = device_id

        # CPU-side envs for game logic.
        self._envs: list[JunqiEnv] = [
            JunqiEnv(show_mode=show_mode, max_num_moves=max_num_moves)
            for _ in range(num_envs)
        ]

        # GPU state batch (allocated once, reused each step).
        self._gpu_state = _cuda.DeviceGameStateBatch(num_envs)

        # GPU observation batch (allocated once, reused each step).
        self._gpu_obs = _cuda.DeviceObservationBatch(num_envs)

        # Observer assignment: slot i observes seat i (identity mapping).
        # Shape (N, 4) int8, C-contiguous.
        self._observer_seats = np.tile(
            np.arange(4, dtype=np.int8), (num_envs, 1)
        )  # (N, 4)

        # Host-side observation slabs (filled from GPU after each obs build).
        from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
        self.obs_spatial = np.zeros(
            (num_envs, 4, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        self.obs_global = np.zeros(
            (num_envs, 4, OBS_GLOBAL_DIMS), dtype=np.float32,
        )

        # Per-env done flag.
        self._done = np.zeros(num_envs, dtype=bool)

        # Thread pool for parallel CPU game stepping.
        _workers = num_workers if num_workers > 1 else 0
        self._pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=_workers) if _workers > 1 else None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self, seed_base: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset all N envs and return observation slabs.

        Returns
        -------
        (obs_spatial, obs_global)
            Arrays of shape ``(N, 4, C, 17, 17)`` and ``(N, 4, G)``.
        """
        for i, env in enumerate(self._envs):
            sd = (seed_base + i) if seed_base is not None else None
            env.reset(seed=sd)
        self._done.fill(False)
        self._fill_all_obs()
        return self.obs_spatial, self.obs_global

    def step(
        self, action_ids: np.ndarray,
    ) -> tuple[
        np.ndarray, np.ndarray,   # (obs_spatial, obs_global)
        np.ndarray,               # rewards (N, 4) int32
        np.ndarray,               # done (N,) bool
        list[JunqiStepInfo | None],
    ]:
        """Step all active envs, then rebuild observations on GPU.

        Parameters
        ----------
        action_ids
            World-frame flat action ids, shape ``(N,)`` int32/int64.
        """
        if action_ids.shape != (self.num_envs,):
            raise ValueError(
                f"action_ids shape mismatch: expected ({self.num_envs},), "
                f"got {action_ids.shape}"
            )
        rewards = np.zeros((self.num_envs, 4), dtype=np.int32)
        infos: list[JunqiStepInfo | None] = [None] * self.num_envs

        if self._pool is not None:
            futures = {}
            for i, env in enumerate(self._envs):
                if not self._done[i]:
                    futures[i] = self._pool.submit(
                        env._step_game_only, int(action_ids[i])
                    )
            for i, fut in futures.items():
                rwd, done, info = fut.result()
                rewards[i] = rwd
                self._done[i] = done
                infos[i] = info
        else:
            for i, env in enumerate(self._envs):
                if self._done[i]:
                    continue
                rwd, done, info = env._step_game_only(int(action_ids[i]))
                rewards[i] = rwd
                self._done[i] = done
                infos[i] = info

        self._fill_all_obs()
        return (
            self.obs_spatial, self.obs_global,
            rewards, self._done.copy(), infos,
        )

    # ------------------------------------------------------------------
    # Observation building (GPU path)
    # ------------------------------------------------------------------

    def _fill_all_obs(self) -> None:
        """Upload state + beliefs to GPU and run the observation kernel.

        Writes into the pre-allocated host slabs ``obs_spatial`` /
        ``obs_global`` by:
          1. Build belief tensor batch on CPU  [N, 4, 12, 289] float32
          2. Pack SoA state arrays             various shapes
          3. Upload both to GPU
          4. Launch ``observation_kernel`` (all N×4 threads)
          5. Download result into host slabs
        """
        N = self.num_envs

        # Step 1: build belief batch [N, 4, 12, 289]
        bel_batch = _build_belief_batch(self._envs)   # (N, 4, 12, 289) float32

        # Step 2: pack state
        state_dict = _pack_state_arrays(self._envs)

        # Step 3: upload state and call the kernel
        self._gpu_state.copy_from_host(state_dict)
        _cuda.build_observation_batch(
            self._gpu_state,
            bel_batch,                          # (N*4*12*289,) after reshape
            self._observer_seats,               # (N, 4) int8
            self._gpu_obs,
            np.int8(self.show_mode.value),      # 0=BRIGHT,1=DARK,2=HALF_DARK
        )

        # Step 4: download
        sp, gl = self._gpu_obs.copy_to_host()   # (N, 4, C, 17, 17), (N, 4, G)
        np.copyto(self.obs_spatial, sp)
        np.copyto(self.obs_global,  gl)

    # ------------------------------------------------------------------
    # Introspection — mirror VectorJunqiEnv interface
    # ------------------------------------------------------------------

    def current_seats(self) -> list[Seat]:
        """Per-env current seat, length N."""
        return [
            e.current_seat() if not d else ALL_SEATS[0]
            for e, d in zip(self._envs, self._done)
        ]

    def legal_action_ids(self, env_idx: int, seat: Seat | None = None) -> np.ndarray:
        """World-frame legal action ids for env ``env_idx``."""
        return self._envs[env_idx].legal_action_ids(seat)

    @property
    def envs(self) -> list[JunqiEnv]:
        """Underlying CPU envs."""
        return list(self._envs)

    @property
    def done(self) -> np.ndarray:
        """Per-env done flag, shape ``(num_envs,)`` bool."""
        return self._done

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
