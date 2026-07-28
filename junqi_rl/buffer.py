"""junqi_rl.buffer — fixed-capacity experience replay buffer.

Phase 0.4 M7.  Implements ADR-122's "fixed numpy slab + O(1) append"
experience buffer used by the PPO collector.  The buffer stores raw
observations in the canonical frame (per ``JunqiEnv.step``) alongside
the flat action ids, per-seat rewards, done flags, acting seat, and an
optional legal-action mask.

Layout (capacity ``C``, OBS_CHANNELS ``Cobs``, board ``H=W=17``, flat
action space ``F=83_521``):

    obs_spatial  : (C, Cobs, H, W)   float32
    obs_global   : (C, Gobs)         float32
    action_id    : (C,)              int32     world-frame flat id
    reward       : (C, 4)            float32   per-seat step reward
    done         : (C,)              bool
    seat         : (C,)              int8      Seat.value of actor
    value_target : (C,)              float32   placeholder, filled by trainer
    legal_mask   : (C, F)            bool      (dense mode only)
    legal_ids    : list[np.ndarray]  int32     (sparse mode only)

The buffer is a **ring**: once ``size == capacity``, subsequent
``append_step`` overwrites the oldest slot.  Trainers that need strict
FIFO iteration can call :meth:`iter_ordered`; for randomized sampling
:meth:`sample_indices` returns uniform indices over ``[0, size)``.

Dense vs sparse legal mask:

* ``legal_mask_mode="dense"`` — one ``bool[83_521]`` row per step.
  Cheapest for downstream indexing but ~84 KB / step.
* ``legal_mask_mode="sparse"`` — store the variable-length ``legal_ids``
  per step as a ``list[np.ndarray[int32]]``.  ~100 B / step in
  mid-game; useful for long-horizon rollouts.
* ``legal_mask_mode="none"`` — do not store any legal-action metadata
  (saves ~80 KB / step when the trainer regenerates on the fly).

All arrays are C-contiguous NumPy slabs preallocated at construction
time.  The only allocation during a steady-state rollout is the
Python-side list append for sparse masks; in "dense" and "none" modes
``append_step`` is strictly zero-allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal

import numpy as np

from junqi_core.observation import (
    OBS_CHANNELS,
    OBS_GLOBAL_DIMS,
)

if TYPE_CHECKING:  # pragma: no cover
    import torch


BOARD_SIZE = 17
FLAT_ACTION_DIM = 289 * 289  # 83 521


LegalMaskMode = Literal["dense", "sparse", "none"]


# ---------------------------------------------------------------------------
# ExperienceBuffer
# ---------------------------------------------------------------------------


class ExperienceBuffer:
    """Fixed-capacity ring buffer of RL experience tuples.

    Parameters
    ----------
    capacity
        Maximum number of steps stored.  Once full, oldest slots are
        overwritten.
    legal_mask_mode
        Storage mode for legal-action metadata; see module docstring.
    obs_channels, obs_global_dims, board_size
        Override defaults only for testing / future channel-layout
        changes; production code should use the junqi_core constants.
    """

    def __init__(
        self,
        capacity: int,
        *,
        legal_mask_mode: LegalMaskMode = "none",
        obs_channels: int = OBS_CHANNELS,
        obs_global_dims: int = OBS_GLOBAL_DIMS,
        board_size: int = BOARD_SIZE,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if legal_mask_mode not in ("dense", "sparse", "none"):
            raise ValueError(
                f"invalid legal_mask_mode {legal_mask_mode!r}; expect "
                f"'dense', 'sparse', or 'none'"
            )
        self.capacity = int(capacity)
        self.legal_mask_mode: LegalMaskMode = legal_mask_mode
        self._obs_c = int(obs_channels)
        self._obs_g = int(obs_global_dims)
        self._board = int(board_size)

        # Pre-allocated slabs.
        self.obs_spatial = np.zeros(
            (self.capacity, self._obs_c, self._board, self._board),
            dtype=np.float32,
        )
        self.obs_global = np.zeros(
            (self.capacity, self._obs_g), dtype=np.float32,
        )
        self.action_id   = np.zeros((self.capacity,), dtype=np.int32)
        self.reward      = np.zeros((self.capacity, 4), dtype=np.float32)
        self.done        = np.zeros((self.capacity,), dtype=bool)
        self.seat        = np.zeros((self.capacity,), dtype=np.int8)
        self.value_target = np.zeros((self.capacity,), dtype=np.float32)
        if legal_mask_mode == "dense":
            self.legal_mask: np.ndarray | None = np.zeros(
                (self.capacity, FLAT_ACTION_DIM), dtype=bool,
            )
            self.legal_ids: list[np.ndarray] | None = None
        elif legal_mask_mode == "sparse":
            self.legal_mask = None
            # Parallel list; slot i holds the int32 legal-action-id
            # array for the step currently living at ring index i.
            self.legal_ids = [
                np.zeros(0, dtype=np.int32) for _ in range(self.capacity)
            ]
        else:  # "none"
            self.legal_mask = None
            self.legal_ids = None

        self._write_idx: int = 0  # next slot to write
        self._size: int = 0       # number of valid entries (<= capacity)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of valid stored steps (<= capacity)."""
        return self._size

    @property
    def full(self) -> bool:
        """Whether the buffer has wrapped at least once."""
        return self._size >= self.capacity

    def reset(self) -> None:
        """Logical clear; leaves underlying slabs allocated.

        Slabs are NOT zeroed (that would defeat the O(1) contract);
        ``size`` is simply reset to 0 and the next append overwrites
        stale slots.  Callers that require freshly zeroed memory (e.g.,
        for deterministic sampling tests) can call
        :meth:`zero_` explicitly.
        """
        self._write_idx = 0
        self._size = 0

    def zero_(self) -> None:
        """Zero out every slab; O(capacity)."""
        self.obs_spatial.fill(0.0)
        self.obs_global.fill(0.0)
        self.action_id.fill(0)
        self.reward.fill(0.0)
        self.done.fill(False)
        self.seat.fill(0)
        self.value_target.fill(0.0)
        if self.legal_mask is not None:
            self.legal_mask.fill(False)
        if self.legal_ids is not None:
            for i in range(self.capacity):
                self.legal_ids[i] = np.zeros(0, dtype=np.int32)
        self._write_idx = 0
        self._size = 0

    def append_step(
        self,
        *,
        obs_spatial: np.ndarray,
        obs_global: np.ndarray,
        action_id: int,
        reward: np.ndarray | tuple[float, float, float, float],
        done: bool,
        seat: int,
        legal: np.ndarray | None = None,
        value_target: float = 0.0,
    ) -> int:
        """Append a single step; return the slab row it was written to.

        Copy semantics: observations and reward are **copied** into the
        preallocated slab (so callers can safely mutate their inputs
        after the call).  The row index returned is valid until the
        next wrap-around write to that slot.

        Performance note: the observation memcpy (~117 KiB per step for
        the default 101-channel, 17×17 obs) caps throughput at the CPU
        memory-bandwidth limit (~25-30 k/s on a single core, ~3.5 GB/s
        effective).  Trainers that want to bypass this should collect
        rollouts directly into the ``VectorJunqiEnv`` slab — which is
        already the ``(N, 4, C, H, W)`` layout PPO needs — and only
        use :class:`ExperienceBuffer` for *trajectory-level* data
        (actions, rewards, dones) that are small enough not to hit the
        memcpy ceiling.

        Parameters
        ----------
        obs_spatial
            ``float32[Cobs, H, W]``.  Must match the configured layout.
        obs_global
            ``float32[Gobs]``.
        action_id
            World-frame flat action id.
        reward
            4-tuple or ``float32[4]`` of per-seat rewards.
        done
            Whether this step ended the episode.
        seat
            Acting seat as ``Seat.value`` (0..3).
        legal
            Optional.  In ``"dense"`` mode, a ``bool[83521]`` mask.  In
            ``"sparse"`` mode, an ``int32[K]`` array of legal ids.  In
            ``"none"`` mode, must be ``None``.
        value_target
            Scalar value target (filled by trainer; defaults to 0).
        """
        i = self._write_idx
        self.obs_spatial[i] = obs_spatial
        self.obs_global[i]  = obs_global
        self.action_id[i] = action_id
        self.reward[i] = reward
        self.done[i] = done
        self.seat[i] = seat
        self.value_target[i] = value_target
        if self.legal_mask_mode == "dense":
            assert self.legal_mask is not None
            if legal is None:
                self.legal_mask[i] = False
            else:
                self.legal_mask[i] = legal
        elif self.legal_mask_mode == "sparse":
            assert self.legal_ids is not None
            if legal is None:
                self.legal_ids[i] = np.zeros(0, dtype=np.int32)
            else:
                # Keep a private copy so caller mutations don't bleed in.
                self.legal_ids[i] = np.asarray(legal, dtype=np.int32).copy()
        else:  # "none"
            if legal is not None:
                raise ValueError(
                    "legal was provided but legal_mask_mode='none'"
                )

        self._write_idx = (i + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1
        return i

    # ------------------------------------------------------------------
    # Sampling / iteration
    # ------------------------------------------------------------------

    def sample_indices(
        self, batch_size: int, rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Uniform sample ``batch_size`` indices from the live region.

        Raises ``ValueError`` if the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("buffer is empty; cannot sample")
        g = rng if rng is not None else np.random.default_rng()
        return g.integers(0, self._size, size=batch_size, dtype=np.int64)

    def iter_ordered(self) -> Iterator[int]:
        """Yield the live slot indices in insertion order (oldest first)."""
        if not self.full:
            yield from range(self._size)
            return
        start = self._write_idx
        for k in range(self.capacity):
            yield (start + k) % self.capacity

    # ------------------------------------------------------------------
    # Torch bridge
    # ------------------------------------------------------------------

    def as_torch(
        self,
        device: "str | torch.device" = "cpu",
        *,
        include_legal: bool = False,
    ) -> dict[str, "torch.Tensor"]:
        """Materialize the live region as a torch tensor dict.

        Uses ``torch.from_numpy`` (zero-copy on CPU) on fresh contiguous
        slices; when ``device != 'cpu'`` a single ``.to(device)`` copy
        ensues per tensor.

        Parameters
        ----------
        device
            Target device (string or ``torch.device``).
        include_legal
            If True and the buffer stores legal masks, include them in
            the output.  Sparse mode emits a ragged Python list rather
            than a tensor (documented via dict key ``legal_ids``).
        """
        import torch  # local import so the buffer is usable without torch

        n = self._size
        if n == 0:
            raise ValueError("buffer is empty")

        # Copy the contiguous prefix so that the next wraparound write
        # does not mutate torch views.  (Callers that want zero-copy can
        # reach into the public numpy slabs directly.)
        out: dict[str, torch.Tensor] = {
            "obs_spatial":  torch.from_numpy(self.obs_spatial[:n].copy()),
            "obs_global":   torch.from_numpy(self.obs_global[:n].copy()),
            "action_id":    torch.from_numpy(self.action_id[:n].copy()),
            "reward":       torch.from_numpy(self.reward[:n].copy()),
            "done":         torch.from_numpy(self.done[:n].copy()),
            "seat":         torch.from_numpy(self.seat[:n].copy()),
            "value_target": torch.from_numpy(self.value_target[:n].copy()),
        }

        if device != "cpu":
            for k, t in out.items():
                out[k] = t.to(device, non_blocking=True)

        if include_legal:
            if self.legal_mask_mode == "dense":
                assert self.legal_mask is not None
                lm = torch.from_numpy(self.legal_mask[:n].copy())
                out["legal_mask"] = (
                    lm.to(device, non_blocking=True)
                    if device != "cpu" else lm
                )
            elif self.legal_mask_mode == "sparse":
                assert self.legal_ids is not None
                # Emit the live prefix as a Python list of int32 tensors.
                out["legal_ids"] = [  # type: ignore[assignment]
                    torch.from_numpy(a.copy()) for a in self.legal_ids[:n]
                ]

        return out


__all__ = ["ExperienceBuffer", "LegalMaskMode", "FLAT_ACTION_DIM"]
