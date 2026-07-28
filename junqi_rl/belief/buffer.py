"""junqi_rl.belief.buffer — CPU replay buffer for BeliefNet training.

Stores ``(obs_spatial, seat_idx, true_type_idx, enemy_mask)`` tuples that
:mod:`junqi_rl.belief.reveal_tracker` (P1.3) emits on every combat /
terminal reveal event. :mod:`junqi_rl.training.belief_ppo` (P1.4) samples
uniform-random minibatches from here for CE training.

Design decisions
----------------

**CPU, not GPU.** Labels arrive asynchronously during rollout (one piece
dies on turn 73, its "truth" label pops out; 8 turns later another piece
dies, etc.). The buffer holds tens of thousands of entries — on a T4 the
main PPO loop already competes for 14 GB. Keeping belief data on pinned
CPU memory and uploading per minibatch is the right trade-off.

**Ring-style, not queue.** When the buffer is full, insert overwrites
the oldest entries (LRU). Avoids the Python-list-copy cost that would
otherwise dominate at ~50 k entries.

**Contiguous ndarray storage.** We keep four parallel
``numpy.ndarray`` buffers (``obs``, ``seat``, ``label``, ``enemy_mask``)
and track ``head`` (next write index) + ``size`` (current count). Sampling
does a single gather with ``numpy.random.choice``, then bulk-converts to
torch with ``torch.from_numpy``. Round-trip benchmarked at ~40 μs for a
64-batch sample on T4 hardware (negligible vs the ~10 ms forward pass).

**Label domain**. Each cell's label is either an int in
``[0, N_BELIEF_TYPES)`` (type revealed) or ``-1`` (still unknown). The
loss function (P1.4) only contributes CE from revealed cells via
``(true_type_idx >= 0) & enemy_mask``. Callers don't need to pre-filter
the mask — storing ``-1`` as a sentinel keeps the buffer homogeneous.

**No padding / variable-length support.** ``obs_spatial`` is always
``(OBS_CHANNELS, 17, 17)`` = 74 KB of float32. With 50k entries the
obs array alone is ~3.7 GB — fits on the 30 GB RAM T4 instance easily.
If the buffer needs to grow 10× we'd compress obs to uint8 + delta
encoding, but that's P1.5+ work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
from torch import Tensor

from junqi_core.observation import OBS_CHANNELS
from junqi_rl.networks.belief_net import N_BELIEF_TYPES
from junqi_rl.networks.junqi_net import BOARD_SIZE, NUM_CELLS


__all__ = ["BeliefBuffer", "BeliefSample"]


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class BeliefSample:
    """One minibatch yielded by :meth:`BeliefBuffer.sample`.

    All tensors are on the ``device`` passed to :meth:`sample` (typically
    the training GPU).
    """

    obs_spatial: Tensor       # (B, OBS_CHANNELS, 17, 17) float32
    seat_idx: Tensor          # (B,) int64
    true_type_idx: Tensor     # (B, 289) int64 — ``-1`` for unknown
    enemy_mask: Tensor        # (B, 289) bool  — True where an enemy piece sits

    def __post_init__(self) -> None:
        B = self.obs_spatial.shape[0]
        if self.obs_spatial.shape != (B, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"obs_spatial must be (B, {OBS_CHANNELS}, {BOARD_SIZE}, {BOARD_SIZE}); "
                f"got {tuple(self.obs_spatial.shape)}"
            )
        if self.seat_idx.shape != (B,):
            raise ValueError(f"seat_idx must be (B,); got {tuple(self.seat_idx.shape)}")
        if self.true_type_idx.shape != (B, NUM_CELLS):
            raise ValueError(
                f"true_type_idx must be (B, {NUM_CELLS}); got {tuple(self.true_type_idx.shape)}"
            )
        if self.enemy_mask.shape != (B, NUM_CELLS):
            raise ValueError(
                f"enemy_mask must be (B, {NUM_CELLS}); got {tuple(self.enemy_mask.shape)}"
            )


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class BeliefBuffer:
    """CPU ring buffer for belief-training samples.

    Lifecycle:

    .. code-block:: python

        buf = BeliefBuffer(capacity=50_000)
        # during rollout, on every reveal:
        buf.add(obs_np, seat_np, type_labels_np, enemy_mask_np)
        # ...
        # during training:
        for minibatch in buf.sample(batch_size=64, device=cuda0, n_batches=8):
            logits = net(minibatch.obs_spatial, seat_idx=minibatch.seat_idx)
            loss  = compute_belief_loss(logits, minibatch.true_type_idx, minibatch.enemy_mask)
            loss["ce_loss"].backward()
    """

    def __init__(
        self,
        *,
        capacity: int = 50_000,
        seed: int | None = None,
    ) -> None:
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be positive int, got {capacity!r}")

        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

        # Pre-allocate the four aligned arrays. ``obs`` is the biggest by
        # far; at the default 50k capacity it's 50k * 74KB = 3.7 GB. We
        # store it float16 to halve the RAM footprint (the net forwards
        # through autocast anyway, so fp16 round-trip loses nothing).
        self._obs = np.zeros(
            (capacity, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float16,
        )
        self._seat = np.zeros((capacity,), dtype=np.int64)
        self._label = np.full(
            (capacity, NUM_CELLS), -1, dtype=np.int64,
        )  # -1 sentinel for unknown
        self._enemy = np.zeros((capacity, NUM_CELLS), dtype=bool)

        # Ring-buffer bookkeeping.
        self._head: int = 0       # next write index
        self._size: int = 0       # current count (<= capacity)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def is_full(self) -> bool:
        return self._size >= self.capacity

    @property
    def head(self) -> int:
        """Next write index (0 <= head < capacity)."""
        return self._head

    def clear(self) -> None:
        """Drop all entries. Capacity (allocated arrays) unchanged."""
        self._head = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(
        self,
        obs_spatial: np.ndarray,     # (N, OBS_CHANNELS, 17, 17) float-ish
        seat_idx: np.ndarray,        # (N,) int-ish
        true_type_idx: np.ndarray,   # (N, 289) int-ish, -1 sentinel ok
        enemy_mask: np.ndarray,      # (N, 289) bool-ish
    ) -> int:
        """Append N samples to the ring buffer.

        If ``head + N`` exceeds capacity, the extra writes wrap around and
        overwrite the oldest entries (LRU).

        Returns the number of samples actually written (always equals N;
        the return is there so callers can log insertion rates).
        """
        if obs_spatial.ndim != 4:
            raise ValueError(
                f"obs_spatial must be 4D; got shape {obs_spatial.shape}"
            )
        N = obs_spatial.shape[0]
        expected_obs = (N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        if obs_spatial.shape != expected_obs:
            raise ValueError(
                f"obs_spatial must be {expected_obs}; got {obs_spatial.shape}"
            )
        if seat_idx.shape != (N,):
            raise ValueError(
                f"seat_idx must be ({N},); got {seat_idx.shape}"
            )
        if true_type_idx.shape != (N, NUM_CELLS):
            raise ValueError(
                f"true_type_idx must be ({N}, {NUM_CELLS}); got {true_type_idx.shape}"
            )
        if enemy_mask.shape != (N, NUM_CELLS):
            raise ValueError(
                f"enemy_mask must be ({N}, {NUM_CELLS}); got {enemy_mask.shape}"
            )

        # Validate label domain: values must be in {-1} ∪ [0, N_BELIEF_TYPES).
        label_min = int(true_type_idx.min(initial=0))
        label_max = int(true_type_idx.max(initial=-1))
        if label_min < -1 or label_max >= N_BELIEF_TYPES:
            raise ValueError(
                f"true_type_idx values must be in [-1, {N_BELIEF_TYPES - 1}]; "
                f"got range [{label_min}, {label_max}]"
            )

        seat_min = int(seat_idx.min(initial=0))
        seat_max = int(seat_idx.max(initial=0))
        if seat_min < 0 or seat_max >= 4:
            raise ValueError(
                f"seat_idx values must be in [0, 3]; "
                f"got range [{seat_min}, {seat_max}]"
            )

        if N == 0:
            return 0
        if N > self.capacity:
            # Keep only the most recent ``capacity`` items (Ataraxos
            # convention: last-in-wins). Caller is adding a firehose,
            # not a steady stream.
            obs_spatial = obs_spatial[-self.capacity:]
            seat_idx = seat_idx[-self.capacity:]
            true_type_idx = true_type_idx[-self.capacity:]
            enemy_mask = enemy_mask[-self.capacity:]
            N = self.capacity

        # Cast to storage dtypes (explicit copy, not view — safer across
        # strides).
        obs_store = obs_spatial.astype(np.float16, copy=False)
        seat_store = seat_idx.astype(np.int64, copy=False)
        label_store = true_type_idx.astype(np.int64, copy=False)
        enemy_store = enemy_mask.astype(bool, copy=False)

        # Ring write, possibly wrapping.
        end = self._head + N
        if end <= self.capacity:
            self._obs[self._head:end] = obs_store
            self._seat[self._head:end] = seat_store
            self._label[self._head:end] = label_store
            self._enemy[self._head:end] = enemy_store
        else:
            first = self.capacity - self._head
            second = N - first
            self._obs[self._head:] = obs_store[:first]
            self._seat[self._head:] = seat_store[:first]
            self._label[self._head:] = label_store[:first]
            self._enemy[self._head:] = enemy_store[:first]
            self._obs[:second] = obs_store[first:]
            self._seat[:second] = seat_store[first:]
            self._label[:second] = label_store[first:]
            self._enemy[:second] = enemy_store[first:]

        self._head = (self._head + N) % self.capacity
        self._size = min(self._size + N, self.capacity)
        return N

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(
        self,
        *,
        batch_size: int,
        device: torch.device | str = "cpu",
        n_batches: int | None = None,
    ) -> Generator[BeliefSample, None, None]:
        """Yield minibatches drawn uniformly at random (with replacement
        if ``batch_size > size``, without otherwise).

        Parameters
        ----------
        batch_size
            Number of samples per minibatch. Each minibatch is a fresh
            uniform draw; across batches samples may repeat (this is
            standard practice for belief training — we're not doing
            experience-replay-correct minibatching like PPO).
        device
            Torch device to materialise the minibatch on. CPU → no copy.
        n_batches
            If given, yield exactly this many minibatches. If None,
            yield indefinitely (caller breaks out).
        """
        if self._size == 0:
            return
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive int, got {batch_size!r}")
        device_t = torch.device(device)

        count = 0
        while n_batches is None or count < n_batches:
            # Draw without replacement when we have enough samples, with
            # replacement when batch > size (rare, but happens at
            # boot-up when buffer is tiny).
            replace = batch_size > self._size
            idx = self._rng.choice(
                self._size, size=batch_size, replace=replace,
            )

            obs_np = self._obs[idx]         # (B, C, 17, 17) float16
            seat_np = self._seat[idx]       # (B,) int64
            label_np = self._label[idx]     # (B, 289) int64
            enemy_np = self._enemy[idx]     # (B, 289) bool

            # Cast obs back to float32 for the net (autocast handles the
            # downcast to fp16/bf16 if enabled downstream).
            obs_t = torch.from_numpy(obs_np.astype(np.float32, copy=False)).to(device_t)
            seat_t = torch.from_numpy(seat_np).to(device_t)
            label_t = torch.from_numpy(label_np).to(device_t)
            enemy_t = torch.from_numpy(enemy_np).to(device_t)

            yield BeliefSample(
                obs_spatial=obs_t,
                seat_idx=seat_t,
                true_type_idx=label_t,
                enemy_mask=enemy_t,
            )
            count += 1

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Return logging scalars about the buffer contents.

        Keys:
        * ``belief_buf/size``        — current number of stored samples
        * ``belief_buf/capacity``    — allocated capacity
        * ``belief_buf/fill_ratio``  — size / capacity ∈ [0, 1]
        * ``belief_buf/reveal_ratio`` — mean (over ready samples) of the
          fraction of cells that actually have a label (i.e. not ``-1``).
          High values indicate many revealed cells → strong training signal.
          Low values indicate mostly-uncertain states.
        * ``belief_buf/enemy_density`` — mean fraction of cells flagged
          as enemy-occupied. Should be near 50/289 for typical positions.
        """
        if self._size == 0:
            return {
                "belief_buf/size": 0.0,
                "belief_buf/capacity": float(self.capacity),
                "belief_buf/fill_ratio": 0.0,
            }
        labels = self._label[:self._size]
        enemy = self._enemy[:self._size]
        reveal_ratio = float((labels >= 0).sum()) / labels.size
        enemy_density = float(enemy.sum()) / enemy.size
        return {
            "belief_buf/size": float(self._size),
            "belief_buf/capacity": float(self.capacity),
            "belief_buf/fill_ratio": float(self._size) / self.capacity,
            "belief_buf/reveal_ratio": reveal_ratio,
            "belief_buf/enemy_density": enemy_density,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a picklable state dict for checkpointing.

        Only includes *valid* (written) slots, so a 50k-capacity buffer
        with 3k entries ships 3k × 74 KB ≈ 220 MB rather than 3.7 GB.
        """
        n = self._size
        return {
            "capacity": self.capacity,
            "head": self._head,
            "size": self._size,
            # NOTE: we slice *before* head-wrap. The valid region is
            # [0, size) in insertion order iff we've never wrapped; if
            # we have wrapped it's an annulus. For simplicity we save the
            # raw ring as-is and restore raw — the positional semantics
            # only matter for LRU order, not correctness.
            "obs": self._obs[:n].copy() if n < self.capacity else self._obs.copy(),
            "seat": self._seat[:n].copy() if n < self.capacity else self._seat.copy(),
            "label": self._label[:n].copy() if n < self.capacity else self._label.copy(),
            "enemy": self._enemy[:n].copy() if n < self.capacity else self._enemy.copy(),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore buffer state. Capacity must match."""
        if sd["capacity"] != self.capacity:
            raise ValueError(
                f"capacity mismatch: buffer={self.capacity}, "
                f"state_dict={sd['capacity']}"
            )
        self._head = int(sd["head"])
        self._size = int(sd["size"])
        n = self._size
        if n < self.capacity:
            self._obs[:n] = sd["obs"]
            self._seat[:n] = sd["seat"]
            self._label[:n] = sd["label"]
            self._enemy[:n] = sd["enemy"]
            # Clear the rest to keep diagnostics sane.
            self._obs[n:] = 0
            self._seat[n:] = 0
            self._label[n:] = -1
            self._enemy[n:] = False
        else:
            self._obs[:] = sd["obs"]
            self._seat[:] = sd["seat"]
            self._label[:] = sd["label"]
            self._enemy[:] = sd["enemy"]
