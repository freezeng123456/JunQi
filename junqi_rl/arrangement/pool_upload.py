"""junqi_rl.arrangement.pool_upload — ArrangementNet ⟷ GPU setup pool glue.

The CUDA rollout picks each episode's starting setup from a pre-uploaded
pool of ``(pool_size, 120)`` int8 piece-type arrays (4 seats × 30 slots =
120 entries per setup). This module converts arrangement-net samples into
that layout and re-uploads the pool so subsequent
``reset_terminated_envs`` calls will seed new games from the policy's
current distribution.

Design
------
* Each pool entry is a *combined* setup: one arrangement per seat.
* Zip pairing (``pool_size<=0``): group samples into ``min_per_seat``
  combined entries, sample ``k`` of seat ``s`` becoming seat ``s`` of
  pool entry ``k``. This is the legacy mapping.
* Expanded pairing (``pool_size>0``): keep the zip prefix so every
  sampled arrangement appears at least once, then fill the remaining
  rows by independently sampling one lineup per seat. CUDA reset hashes
  each env onto one combined row, so a pool of size ``n_arr//4`` reuses
  the same 4-tuples across thousands of games. Independent pairing is
  how a per-seat setup net becomes a large unique-game pool without
  generating ``num_envs`` autoregressive samples.
* Pool entries with wrong seat distribution raise rather than silently
  producing an all-zero pool.

Reverse lookup
--------------
After a reset, each env's 120-entry piece-type array is a concrete
setup. To identify *which arrangement row* each env is using (for
``ArrangementBuffer.add_rewards``), we:
  1. Copy ``piece_type_arr`` D2H once after reset.
  2. Split it into 4 seats × 30 slots.
  3. Map PieceType → vocab_idx for each seat, hash via blake2b
     (same as the buffer's dedup hash), and look up in the buffer's
     ``_lookup_index`` dict.

For the *terminal-reward add* path we actually need the per-seat
arrangement and the seat that "owns" it. Since JunQi teams are 0 (S+N)
and 1 (W+E), and the terminal reward applies per-team, we credit the
arrangement of the acting seat (whose POV the reward is computed from)
to the buffer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from junqi_core.rules import SLOTS_PER_SEAT
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    N_SEATS,
    VOCAB_IDX_TO_PIECE_TYPE_VALUE,
)

assert ARRANGEMENT_SIZE == SLOTS_PER_SEAT == 30


# ---------------------------------------------------------------------------
# Forward: arrangements → GPU setup pool
# ---------------------------------------------------------------------------


def arrangements_to_pool(
    samples: Tensor,    # (n_arr, 30, 13) one-hot
    seat_idx: Tensor,   # (n_arr,) int64 in [0, 4)
    *,
    pool_size: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Convert arrangement-net samples to a ``(pool_size, 120)`` int8 pool.

    Zip pairing (``pool_size<=0``): slot ``k`` of seat ``s`` is the ``k``-th
    sample of that seat. Excess samples of any one seat are dropped, so the
    returned pool has ``min_per_seat`` rows.

    Expanded pairing (``pool_size>0``): keep that zip prefix so every
    sampled arrangement is used at least once, then fill remaining rows by
    independently sampling one lineup per seat. The CUDA reset kernel
    hashes each env onto a single combined row, so zip-only pools of size
    ``n_arr//4`` make thousands of parallel games reuse a few hundred
    openings.

    Parameters
    ----------
    samples : (n_arr, 30, 13) float
    seat_idx : (n_arr,) int64
    pool_size : int
        Target number of combined 4-seat boards. ``<=0`` keeps zip-only
        behaviour. Otherwise ``max(pool_size, min_per_seat)``.
    seed : int
        RNG seed for the independent-combination tail.

    Returns
    -------
    pool : (P, 120) int8 — piece-type values (0..13) flattened
        row-major as [seat0_slot0, seat0_slot1, ..., seat3_slot29].
    """
    if samples.ndim != 3 or samples.size(1) != ARRANGEMENT_SIZE or samples.size(2) != N_PIECE_TYPE_WITH_NONE:
        raise ValueError(
            f"samples must be (n_arr, {ARRANGEMENT_SIZE}, {N_PIECE_TYPE_WITH_NONE}), "
            f"got {tuple(samples.shape)}"
        )
    if seat_idx.shape != (samples.size(0),):
        raise ValueError(
            f"seat_idx shape {seat_idx.shape} must match samples.size(0)"
        )
    if int(pool_size) < 0:
        raise ValueError(f"pool_size must be >= 0, got {pool_size}")

    # Convert one-hot to vocab indices (n_arr, 30) → PieceType.value (n_arr, 30).
    vocab = samples.argmax(dim=-1).to("cpu").numpy().astype(np.int64)  # (n_arr, 30)
    lut = np.asarray(VOCAB_IDX_TO_PIECE_TYPE_VALUE, dtype=np.int8)
    piece_types = lut[vocab]  # (n_arr, 30) int8 piece-type values

    seats_np = seat_idx.cpu().numpy().astype(np.int64)

    # For each seat 0..3 collect the indices of samples belonging to that seat.
    per_seat_indices: list[np.ndarray] = [
        np.nonzero(seats_np == s)[0] for s in range(N_SEATS)
    ]
    min_per_seat = min(len(ids) for ids in per_seat_indices)
    if min_per_seat == 0:
        raise ValueError(
            "Cannot build pool: at least one seat has zero arrangement samples. "
            "Call generate_arrangements with seats covering all 4 indices."
        )

    zip_size = min_per_seat
    target = zip_size if int(pool_size) <= 0 else max(int(pool_size), zip_size)
    pool = np.zeros((target, 4 * SLOTS_PER_SEAT), dtype=np.int8)  # (P, 120)
    for s in range(N_SEATS):
        seat_idxs = per_seat_indices[s][:zip_size]
        pool[:zip_size, s * SLOTS_PER_SEAT:(s + 1) * SLOTS_PER_SEAT] = piece_types[seat_idxs]
    if target > zip_size:
        rng = np.random.default_rng(int(seed))
        extra = target - zip_size
        for s in range(N_SEATS):
            ids = per_seat_indices[s]
            pick = rng.integers(0, len(ids), size=extra)
            pool[zip_size:, s * SLOTS_PER_SEAT:(s + 1) * SLOTS_PER_SEAT] = piece_types[ids[pick]]
    return pool


def refresh_gpu_setup_pool(pool: np.ndarray) -> None:
    """Re-upload ``pool`` to CUDA via ``_cuda.upload_setup_pool``.

    Split out so tests can monkey-patch the CUDA call.
    """
    if pool.ndim != 2 or pool.shape[1] != 120:
        raise ValueError(f"pool must be (P, 120) int8, got {pool.shape}")
    if pool.dtype != np.int8:
        pool = pool.astype(np.int8, copy=False)

    import junqi_cuda as _cuda

    _cuda.upload_setup_pool(pool)


# ---------------------------------------------------------------------------
# Reverse: read-back each env's active arrangement (per seat)
# ---------------------------------------------------------------------------


def read_env_arrangements_from_state(rollout) -> np.ndarray:
    """Snapshot each env's (4 seats × 30 slots) piece-type array from GPU.

    Parameters
    ----------
    rollout
        A ``GpuRollout`` instance (we call its ``state.copy_to_host()``).

    Returns
    -------
    per_env_per_seat : (N, 4, 30) int64 — vocab indices (0..12).

    Notes
    -----
    This is a D2H copy, so callers should cache the result across all envs
    that terminate within one ``reset_terminated_envs`` cycle.
    """
    state_host = rollout.state.copy_to_host()
    piece_type_arr = state_host["piece_type_arr"]  # flat (N*120,) int8
    N = rollout.num_envs

    # Map PieceType.value → vocab_idx. PIECE_TYPE_VALUE_TO_VOCAB_IDX gives
    # the inverse; we need a (14,)-sized LUT for fast numpy indexing.
    from junqi_rl.networks.arrangement_net import PIECE_TYPE_VALUE_TO_VOCAB_IDX

    max_value = max(PIECE_TYPE_VALUE_TO_VOCAB_IDX.keys()) + 1
    inv_lut = np.zeros(max_value, dtype=np.int64)
    for val, idx in PIECE_TYPE_VALUE_TO_VOCAB_IDX.items():
        inv_lut[val] = idx
    # DARK and any unmapped value left at 0 (NONE) — these should never
    # actually appear on a fresh reset, so leaving as 0 is safe.

    per_env = piece_type_arr.reshape(N, N_SEATS, SLOTS_PER_SEAT).astype(np.int64)
    # Clamp out-of-range values (DARK etc.) to 0 before LUT indexing so we
    # can't trip an IndexError if the device state contains placeholder vals.
    per_env = np.clip(per_env, 0, max_value - 1)
    return inv_lut[per_env]


__all__ = [
    "arrangements_to_pool",
    "refresh_gpu_setup_pool",
    "read_env_arrangements_from_state",
]
