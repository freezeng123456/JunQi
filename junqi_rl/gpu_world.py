"""junqi_rl/gpu_world.py — GPU-resident batched Junqi world (Phase A scaffolding).

``GpuWorld`` is the long-term home for the GPU-native Junqi environment,
designed around the principle that **state lives on GPU and is never packed
back to CPU unless explicitly requested**.

Phase A scope (this module)
---------------------------
The current implementation is a **thin Python facade** over the existing
``junqi_cuda`` C++ classes.  It fixes three structural problems of the old
``VectorJunqiEnvGPU``:

1. **State is owned by GpuWorld, not mirrored from a list[JunqiEnv]**.
   Users push CPU-built state into the GPU once per episode boundary; all
   intermediate steps touch only device memory.

2. **All scratch buffers (belief, observer_seats, acting_seats, action
   outputs) are persistent** via ``junqi_cuda.gpu_scratch_reset()`` — no
   per-call cudaMalloc/Free.

3. **Three output formats for legal actions** selected by the caller:

   * ``dense``  — ``(N, 512)`` int32 + ``(N,)`` counts  (back-compat)
   * ``csr``    — ``(N+1,)`` offsets + ``(Σ,)`` values (bandwidth-optimal)
   * ``mask``   — ``(N, 120, 32)`` bool per-piece slot mask (network-friendly)

Phase B (not in this module) will implement ``step_batch`` on GPU, at which
point ``GpuWorld.push_state`` only runs once per episode (reset), and the
CPU game-loop goes away entirely.

Usage
-----
::

    world = GpuWorld(num_envs=1024)

    # Seed from a list of CPU JunqiEnv (one-time at reset)
    world.push_state_from_envs(envs)

    # Per-step: everything stays on GPU
    acting_seats = np.array([e.state.turn.value for e in envs], dtype=np.int8)
    offsets, values = world.legal_actions_csr(acting_seats)
    # ... or ...
    mask = world.legal_actions_mask(acting_seats)

    # After M3 ships:
    #   world.step(action_ids)  → rewards, dones, new legal mask, new obs
    #   without ever touching CPU state.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import junqi_cuda as _cuda  # type: ignore[import]
    _CUDA_AVAILABLE = True
except ImportError:
    _cuda = None  # type: ignore[assignment]
    _CUDA_AVAILABLE = False

if TYPE_CHECKING:
    from .env import JunqiEnv


def _require_cuda() -> None:
    if not _CUDA_AVAILABLE:
        raise ImportError(
            "junqi_cuda extension is not available. "
            "Build it with: python3 build_cuda.py"
        )


def _upload_zobrist_tables() -> None:
    """Push the CPU-seeded zobrist tables into GPU device memory so the
    GPU ``step_batch`` kernel produces bit-identical hashes with the CPU
    ``BatchedGameState``.  Safe to call multiple times — the upload is
    idempotent.  Must follow ``_cuda.init_tables()``.
    """
    from junqi_core import _zobrist as z  # lazy import (avoids cyclic)
    _cuda.upload_zobrist_tables(
        z.ZOB_PIECE.astype(np.int64, copy=False).ravel(),
        z.ZOB_TURN.astype(np.int64, copy=False).ravel(),
        z.ZOB_MOVE_COUNTER.astype(np.int64, copy=False).ravel(),
        z.ZOB_MOVES_SINCE_COMBAT.astype(np.int64, copy=False).ravel(),
        z.ZOB_WINNER.astype(np.int64, copy=False).ravel(),
        int(z.ZOB_TERMINATED),
        int(z.ZOB_DRAW),
        z.ZOB_SEAT_DEAD.astype(np.int64, copy=False).ravel(),
        z.ZOB_SEAT_FLAG_REVEALED.astype(np.int64, copy=False).ravel(),
    )


class GpuWorld:
    """GPU-resident Junqi world (Phase A facade).

    Parameters
    ----------
    num_envs
        Number of parallel game environments.
    device_id
        CUDA device index.
    init_tables
        Whether to upload board topology and Zobrist tables (default True).
        Set False if you call ``junqi_cuda.init_tables()`` externally.

    Attributes
    ----------
    num_envs : int
    state    : junqi_cuda.DeviceGameStateBatch
        Underlying device state.  Access directly for advanced use.
    obs      : junqi_cuda.DeviceObservationBatch
        Underlying device observation tensors.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        device_id: int = 0,
        init_tables: bool = True,
    ) -> None:
        _require_cuda()
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if _cuda.get_gpu_count() == 0:
            raise RuntimeError("No CUDA-capable GPU found")

        _cuda.set_device(device_id)
        if init_tables:
            _cuda.init_tables()
            # Phase 1b: upload CPU-seeded zobrist tables so GPU step_batch
            # produces bit-identical hashes with CPU BatchedGameState.
            _upload_zobrist_tables()

        self.num_envs = num_envs
        self._device_id = device_id
        self.state = _cuda.DeviceGameStateBatch(num_envs)
        self.obs = _cuda.DeviceObservationBatch(num_envs)

    # ------------------------------------------------------------------
    # State upload (Phase A: still from CPU envs; Phase B: GPU-native reset)
    # ------------------------------------------------------------------

    def push_state_from_envs(self, envs: "list[JunqiEnv]") -> None:
        """Pack state from N CPU envs and upload to GPU (full 20-field path)."""
        if len(envs) != self.num_envs:
            raise ValueError(
                f"push_state_from_envs: got {len(envs)} envs, expected {self.num_envs}"
            )
        # Lazy import to avoid circular import at module top.
        from .env_gpu import _pack_state_arrays
        sd = _pack_state_arrays(envs)
        self.state.copy_from_host(sd)

    def push_state_lite(self, envs: "list[JunqiEnv]") -> None:
        """Lite upload — only the 6 fields legal_action_ids_batch reads.

        Use this between episodes when you know the next call will be
        legal_actions only (no step, no observation). ~2.5× faster than
        push_state_from_envs.
        """
        if len(envs) != self.num_envs:
            raise ValueError(
                f"push_state_lite: got {len(envs)} envs, expected {self.num_envs}"
            )
        from .env_gpu import _pack_state_arrays_lite
        psa, pta, alv, px, py_, cpi = _pack_state_arrays_lite(envs)
        self.state.copy_from_host_legal_lite(psa, pta, alv, px, py_, cpi)

    # ------------------------------------------------------------------
    # Legal action queries — three output formats
    # ------------------------------------------------------------------

    def legal_actions_dense(
        self, acting_seats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dense ``(N, 512)`` int32 action IDs + ``(N,)`` counts.

        This is the back-compat format.  Prefer ``legal_actions_csr`` for
        training loops (bandwidth-optimal) or ``legal_actions_mask`` for
        policy networks with fixed action heads.
        """
        self._validate_acting_seats(acting_seats)
        return _cuda.legal_action_ids_batch(self.state, acting_seats)

    def legal_actions_csr(
        self, acting_seats: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """CSR ``(offsets, values)`` legal-action output.

        Returns
        -------
        offsets : int32 (N+1,)
            env i's actions live at values[offsets[i]:offsets[i+1]]
        values : int32 (sum,)
            Concatenated flat action IDs.

        At N=1024 with avg 27 actions/env, the D2H cost is 19× smaller than
        the dense path.
        """
        self._validate_acting_seats(acting_seats)
        return _cuda.legal_action_ids_batch_csr(self.state, acting_seats)

    def legal_actions_mask(self, acting_seats: np.ndarray) -> np.ndarray:
        """Per-piece ``(N, 120, 32)`` bool legal-action mask.

        Action space:  120 pieces × 32 slots/piece = 3840 compact actions.
        Slot layout documented in ``junqi_cuda.legal_action_mask_batch``
        docstring.  The mask is ready to multiply directly into a policy
        network's action logits.
        """
        self._validate_acting_seats(acting_seats)
        return _cuda.legal_action_mask_batch(self.state, acting_seats)

    # ------------------------------------------------------------------
    # Step (Phase 1b — GPU-resident state mutation)
    # ------------------------------------------------------------------

    def step_batch(self, action_ids: np.ndarray) -> dict:
        """Advance every env by one action on the GPU.

        Parameters
        ----------
        action_ids
            Shape ``(N,)``, dtype ``int32``.  Each entry is
            ``src_flat * 289 + dst_flat``.  Terminated envs' actions are
            ignored.

        Returns
        -------
        dict with keys:
            * ``valid``          — bool (N,)
            * ``event``          — int8 (N,)   Event.value ∈ {0..4}
            * ``terminated``     — bool (N,)
            * ``winner_team``    — int8 (N,)   -1/0/1
            * ``draw``           — bool (N,)
            * ``flag_captured``  — bool (N,)

        The on-device state is mutated in place; subsequent ``step_batch``
        calls see the updated state without any H2D/D2H round-trip.
        """
        if action_ids.shape != (self.num_envs,):
            raise ValueError(
                f"action_ids: expected shape ({self.num_envs},), "
                f"got {action_ids.shape}"
            )
        if action_ids.dtype != np.int32:
            action_ids = action_ids.astype(np.int32, copy=False)
        return self.state.step_batch(action_ids)

    def push_termination_state(
        self,
        terminated: np.ndarray,
        winner_team: np.ndarray,
        draw: np.ndarray,
    ) -> None:
        """Upload per-env termination flags (needed before the first
        ``step_batch`` call on a batch seeded with ``push_state_from_envs``,
        which does not populate the three new scalar arrays).
        """
        self.state.copy_termination_from_host(terminated, winner_team, draw)

    def pull_termination_state(self) -> dict:
        """Read back terminated/winner_team/draw from the device."""
        return self.state.copy_termination_to_host()


    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def build_observation(
        self,
        beliefs: np.ndarray,            # (N, 4, 12, 289) float32
        observer_seats: np.ndarray,     # (N, 4) int8
        show_mode: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ``(N, 4, 256, 17, 17)`` spatial + ``(N, 4, 28)`` global obs.

        Returns host-side numpy arrays (copy from device).  For a zero-copy
        path to PyTorch, use ``build_observation_into_torch`` (Phase B).
        """
        _cuda.build_observation_batch(
            self.state, beliefs, observer_seats, self.obs, np.int8(show_mode),
        )
        return self.obs.copy_to_host()   # (spatial, global)

    # ------------------------------------------------------------------
    # Lifecycle & resource management
    # ------------------------------------------------------------------

    def release_scratch(self) -> None:
        """Free all persistent GPU scratch buffers.

        Useful for testing tight memory; next call will reallocate.
        """
        _cuda.gpu_scratch_reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_acting_seats(self, acting_seats: np.ndarray) -> None:
        if acting_seats.shape != (self.num_envs,):
            raise ValueError(
                f"acting_seats: expected shape ({self.num_envs},), "
                f"got {acting_seats.shape}"
            )
        if acting_seats.dtype != np.int8:
            raise ValueError(
                f"acting_seats: expected int8, got {acting_seats.dtype}"
            )

    def __repr__(self) -> str:
        return f"GpuWorld(num_envs={self.num_envs}, device_id={self._device_id})"
