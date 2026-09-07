"""junqi_rl.arrangement.sampling — autoregressive arrangement generation.

Entry point: :func:`generate_arrangements`. Given an :class:`ArrangementNet`
and a batch size, it autoregressively emits ``N`` lineups for the specified
seat(s). For every generated arrangement we record:

* ``samples``    — one-hot lineup, shape ``(N, 30, 13)``
* ``values``     — per-prefix value prediction, shape ``(N, 30, N_VF_CAT|1)``
* ``ent_pred``   — per-prefix predicted future NLL, shape ``(N, 30)``
* ``log_probs``  — per-prefix log-softmax distribution used for sampling,
                   shape ``(N, 30, 13)``
* ``seat_idx``   — the seat this lineup was generated for, shape ``(N,)``

All returned tensors live on the model's device; callers can ``.cpu()`` at
a convenient granularity.

Implementation notes
--------------------
Unlike Ataraxos' Stratego, JunQi arrangements can *dead-end*: e.g. placing
all 3 ``DILEI`` on non-mine-eligible rows before reaching the back two
rows. The network is trained to avoid such traps, but during very early
training (random weights) some trajectories are infeasible. We handle this
via **resample-on-dead-end**: whenever the mask at slot t is entirely
False, we roll back to slot 0 and try again for that specific row. After
``max_retries`` failures we fall back to ``junqi_core.setup.generate_random_lineup``
and back-fill. This matches the pragmatic "always produce n_sample valid
arrangements" contract Ataraxos' buffer relies on.

The retries only kick in while the net is near-random; once training
stabilises they vanish. We log the retry rate for observability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.distributions import Categorical

from junqi_core.rules import PieceType
from junqi_core.setup import generate_random_lineup
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    N_SEATS,
    PIECE_TYPE_VALUE_TO_VOCAB_IDX,
    VOCAB_IDX_TO_PIECE_TYPE_VALUE,
    ArrangementNet,
)


_LEGAL_THRESHOLD: float = -1e30  # logits below this are masked-out sentinels


@dataclass
class GenerationResult:
    """Return bundle for :func:`generate_arrangements`."""

    samples: Tensor        # (N, 30, 13) one-hot, float
    values: Tensor         # (N, 30, N_VF_CAT) or (N, 30) depending on use_cat_vf
    ent_pred: Tensor       # (N, 30) float
    log_probs: Tensor      # (N, 30, 13) float  (log_softmax at each step)
    seat_idx: Tensor       # (N,) int64
    fallback_mask: Tensor  # (N,) bool; random-lineup fallback, not policy data
    stats: dict            # {'retry_rate': ..., 'fallback_rate': ...}


@torch.no_grad()
def generate_arrangements(
    n_sample: int,
    model: ArrangementNet,
    *,
    seats: Tensor | Iterable[int] | int | None = None,
    dtype: torch.dtype = torch.float32,
    max_resample: int = 8,
    rng_seed: int | None = None,
) -> GenerationResult:
    """Sample ``n_sample`` arrangements from ``model``.

    Parameters
    ----------
    n_sample
        Number of arrangements to draw.
    model
        :class:`ArrangementNet` in eval mode (we call ``.eval()`` internally).
    seats
        Which seat each sample belongs to. If ``None``, seats are cycled
        uniformly (i.e. ``torch.arange(n_sample) % 4``). If an int, all samples
        use that seat. Otherwise must be length-``n_sample`` int tensor/list.
    dtype
        Precision for the forward pass. Defaults to fp32 (safe); callers
        doing repeated generation from the main training loop may pass
        ``torch.bfloat16`` or ``torch.float16`` for a ~2× speedup.
    max_resample
        If sampling dead-ends, resample the single offending row up to this
        many times before falling back to ``generate_random_lineup``.
    rng_seed
        Optional seed for the ``generate_random_lineup`` fallback path so
        runs are reproducible in tests.

    Returns
    -------
    :class:`GenerationResult`
    """
    if n_sample <= 0:
        raise ValueError(f"n_sample must be positive, got {n_sample}")

    device = model.device
    model.eval()

    # Resolve seats to (N,) int64 on device.
    if seats is None:
        seat_idx = torch.arange(n_sample, device=device) % N_SEATS
    elif isinstance(seats, int):
        seat_idx = torch.full((n_sample,), int(seats), device=device, dtype=torch.long)
    elif isinstance(seats, Tensor):
        seat_idx = seats.to(device=device, dtype=torch.long)
        if seat_idx.shape != (n_sample,):
            raise ValueError(f"seats shape {seat_idx.shape} != ({n_sample},)")
    else:
        seat_idx = torch.tensor(list(seats), device=device, dtype=torch.long)
        if seat_idx.shape != (n_sample,):
            raise ValueError(f"seats length {seat_idx.numel()} != n_sample={n_sample}")
    if not ((seat_idx >= 0) & (seat_idx < N_SEATS)).all():
        raise ValueError("all seats must be in [0, 4)")

    use_cat_vf = model.cfg.use_cat_vf
    n_vf_cat = model.cfg.n_vf_cat if use_cat_vf else 1

    # Allocate outputs on device.
    samples = torch.zeros(n_sample, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE,
                          device=device, dtype=torch.float32)
    if use_cat_vf:
        values = torch.zeros(n_sample, ARRANGEMENT_SIZE, n_vf_cat,
                             device=device, dtype=torch.float32)
    else:
        values = torch.zeros(n_sample, ARRANGEMENT_SIZE,
                             device=device, dtype=torch.float32)
    ent_pred = torch.zeros(n_sample, ARRANGEMENT_SIZE,
                           device=device, dtype=torch.float32)
    log_probs = torch.zeros(n_sample, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE,
                            device=device, dtype=torch.float32)

    # For dead-end handling we need to roll a single row back to t=0; track
    # which rows have "finished" so we can resample only the failed ones.
    needs_work = torch.ones(n_sample, device=device, dtype=torch.bool)
    resample_counts = torch.zeros(n_sample, device=device, dtype=torch.long)
    fallback_flags = torch.zeros(n_sample, device=device, dtype=torch.bool)

    # For the fallback path we need a CPU RNG so repeated calls are deterministic.
    py_rng = random.Random(rng_seed) if rng_seed is not None else random.Random()

    # --- Main loop: iterate until every row has finished. ---
    # Most rows will complete on the first pass. A handful may need resampling.
    outer_iter = 0
    MAX_OUTER = max_resample + 1
    while bool(needs_work.any()) and outer_iter < MAX_OUTER:
        outer_iter += 1
        # Reset the rows we're about to (re)generate.
        active_mask = needs_work.clone()
        if outer_iter > 1:
            # For resampling: zero out prior state for active rows so we
            # start fresh.
            samples[active_mask] = 0.0
            values[active_mask] = 0.0
            ent_pred[active_mask] = 0.0
            log_probs[active_mask] = 0.0

        active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)  # (M,)
        if active_idx.numel() == 0:
            break
        M = int(active_idx.numel())
        seat_active = seat_idx[active_idx]

        # Slot-wise autoregressive loop for the active batch.
        died = torch.zeros(M, device=device, dtype=torch.bool)
        for t in range(ARRANGEMENT_SIZE):
            prefix = samples[active_idx, :t]                             # (M, t, V)
            with autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                out = model(prefix, seat_active)
            # out["logits"] shape: (M, t+1, V). Predict slot t from last row.
            logits_t = out["logits"][:, -1].to(torch.float32)            # (M, V)
            # Rows that have already died on an earlier slot: keep them
            # "alive" but mark to be re-tried. For simplicity we still
            # produce an output value (NONE) but flag `died[m]=True`.
            legal_any = (logits_t > _LEGAL_THRESHOLD).any(dim=-1)        # (M,)
            # Dead-end at slot t → mark and clamp logits so sampling works.
            dead_now = ~legal_any
            if bool(dead_now.any()):
                # Force legal_mask of NONE to avoid NaN in softmax; we discard
                # these samples in this outer pass anyway.
                logits_t[dead_now] = 0.0  # uniform over vocab — safe, won't be used
                died |= dead_now

            log_pi_t = F.log_softmax(logits_t, dim=-1)                   # (M, V)
            # Sample from the categorical distribution of legal choices.
            pick_t = Categorical(logits=log_pi_t).sample()               # (M,)

            # Write back.
            samples[active_idx, t] = F.one_hot(pick_t, num_classes=N_PIECE_TYPE_WITH_NONE).float()
            log_probs[active_idx, t] = log_pi_t
            v_t = out["value"][:, -1].to(torch.float32)
            if use_cat_vf:
                values[active_idx, t] = v_t                              # (M, C)
            else:
                values[active_idx, t] = v_t.squeeze(-1)                  # (M,)
            ent_pred[active_idx, t] = out["ent_pred"][:, -1].squeeze(-1).to(torch.float32)

        # After the inner loop: rows that `died` need a redo.
        global_died = torch.zeros(n_sample, device=device, dtype=torch.bool)
        global_died[active_idx] = died
        needs_work = global_died
        resample_counts[global_died] += 1

    # Any rows still in `needs_work` after max_resample passes → fallback.
    if bool(needs_work.any()):
        fallback_idx = torch.nonzero(needs_work, as_tuple=False).squeeze(-1).tolist()
        for m in fallback_idx:
            fallback_flags[m] = True
            lineup = generate_random_lineup(py_rng)
            vocab_idxs = torch.tensor(
                [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup],
                device=device, dtype=torch.long,
            )
            samples[m] = F.one_hot(vocab_idxs, num_classes=N_PIECE_TYPE_WITH_NONE).float()
            # Fallback-filled rows have no valid old-policy statistics.
            # Keep placeholder tensors for the fixed-size generation contract;
            # training callers must exclude `fallback_mask`.
            log_probs[m] = 0.0
            values[m] = 0.0
            ent_pred[m] = 0.0

    # Final invariant: rows sum to 1 in each slot (one-hot).
    assert torch.allclose(samples.sum(dim=-1), torch.ones_like(samples.sum(dim=-1))), (
        "samples are not one-hot after generation"
    )

    stats = {
        "retry_rate": float(resample_counts.float().mean().item()),
        "fallback_rate": float(fallback_flags.float().mean().item()),
        "outer_iterations": outer_iter,
    }

    return GenerationResult(
        samples=samples,
        values=values,
        ent_pred=ent_pred,
        log_probs=log_probs,
        seat_idx=seat_idx,
        fallback_mask=fallback_flags,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Conversions: sampled vocab-onehot  <->  junqi_core lineup
# ---------------------------------------------------------------------------


def samples_to_lineups(samples: Tensor) -> list[list[PieceType]]:
    """Convert ``(N, 30, 13)`` one-hot to a list of N lineups (PieceType)."""
    if samples.ndim != 3 or samples.size(1) != ARRANGEMENT_SIZE:
        raise ValueError(f"expected (N, {ARRANGEMENT_SIZE}, V), got {tuple(samples.shape)}")
    idxs = samples.argmax(dim=-1).cpu().tolist()  # (N, 30)
    return [
        [PieceType(VOCAB_IDX_TO_PIECE_TYPE_VALUE[vi]) for vi in row]
        for row in idxs
    ]


def lineup_to_sample(lineup: Iterable[PieceType]) -> Tensor:
    """Convert a PieceType lineup to ``(30, 13)`` one-hot float tensor."""
    lineup_list = list(lineup)
    if len(lineup_list) != ARRANGEMENT_SIZE:
        raise ValueError(f"lineup must have {ARRANGEMENT_SIZE} entries, got {len(lineup_list)}")
    vocab = torch.tensor(
        [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup_list],
        dtype=torch.long,
    )
    return F.one_hot(vocab, num_classes=N_PIECE_TYPE_WITH_NONE).float()


__all__ = [
    "GenerationResult",
    "generate_arrangements",
    "samples_to_lineups",
    "lineup_to_sample",
]
