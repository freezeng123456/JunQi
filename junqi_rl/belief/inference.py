"""junqi_rl.belief.inference — Upload BeliefNet output to GpuRollout.d_belief.

Sits between the main PPO step loop and the on-device belief buffer. Runs a
BeliefNet forward over the current per-seat observations and pushes the
resulting per-cell distributions into ``rollout.d_belief`` via
:meth:`GpuRollout.upload_beliefs`.

Integration contract
--------------------
The deductive rule engine in ``junqi_rl/env/cuda/src/belief.cu`` (R1, R4,
R5/R7, R6, R9, I5) runs first on every step via
:meth:`GpuRollout.update_beliefs_device`. After a main-PPO rollout
completes, the training script calls :func:`refresh_beliefs_neural` to
replace the device-resident belief tensor with the neural net's output.
The deductive rules then resume on subsequent steps, now starting from
the neural prior instead of the hand-coded prior table.

This interleaving (net → deductive → next step → net refresh) mirrors
Ataraxos's design choice (§D.5): the deductive updates are bit-exact for
deterministic events (flag captured → 1.0 prob JUNQI at that cell),
while the neural net interpolates for everything else. Keeping both
layers means the net doesn't have to learn what the rules already know.

Throttling
----------
BeliefNet forward on ``(N × 4, 256, 17, 17)`` is not free. For a T4 with
``N=128`` that's a 512-batch of ~4M-param model ≈ 60 ms per call. Calling
this every step would cut fps by ~20%. Sensible cadence: every rollout
(not every step), after the main-PPO update. For short training this is
enough signal; for long training you can bump to every-k-rollouts at the
cost of slightly staler beliefs.

Why no autocast?
----------------
BeliefNet output is float32 logits which we softmax-then-upload to a
float32 ``(N, 4, 12, 289)`` tensor. fp16 autocast here would add a
downcast + upcast sandwich for no throughput win (the forward is already
cheap) and risk NaN overflow in softmax at the tail of training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from junqi_rl.belief.reveal_tracker import _SEAT_CELLS, _build_enemy_mask
from junqi_rl.networks.belief_net import BeliefNet, N_BELIEF_TYPES

if TYPE_CHECKING:
    from junqi_rl.gpu_rollout import GpuRollout


__all__ = ["refresh_beliefs_neural", "belief_logits_to_upload_shape"]


BOARD_SIZE: int = 17
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE  # 289
N_SEATS: int = 4


def _build_enemy_mask_tensor(device: torch.device) -> torch.Tensor:
    """Return ``(4, 289) bool`` enemy masks on ``device``.

    Row ``s`` is the enemy mask from seat ``s``'s perspective (True on
    cells owned by opponents of seat ``s``). Matches
    :func:`junqi_rl.belief.reveal_tracker._build_enemy_mask`.

    We build this in python (fast) and cache per-device as needed.
    """
    masks = np.stack([_build_enemy_mask(s) for s in range(N_SEATS)], axis=0)
    return torch.from_numpy(masks).to(device=device, dtype=torch.bool)


# Per-device cached enemy-mask tensor. Invalidated on device change.
_ENEMY_MASK_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_enemy_mask(device: torch.device) -> torch.Tensor:
    """Cached accessor for the (4, 289) enemy-mask tensor."""
    if device not in _ENEMY_MASK_CACHE:
        _ENEMY_MASK_CACHE[device] = _build_enemy_mask_tensor(device)
    return _ENEMY_MASK_CACHE[device]


def belief_logits_to_upload_shape(
    probs: torch.Tensor,              # (N, 4, 289, 12) — softmax output per (env, seat)
    enemy_mask: torch.Tensor | None = None,   # (4, 289) bool; None → derive from cache
) -> torch.Tensor:
    """Reshape the per-cell belief distribution into GpuRollout's upload
    layout ``(N, 4, 12, 289) float32``.

    BUG-O fix (2026-05-11): we no longer zero out non-enemy-territory
    cells via ``_build_enemy_mask`` (the static seat-geometry mask).
    The static mask only describes pieces' INITIAL territories — but
    enemy pieces walk into the observer's own territory in the regular
    course of play (~23% of alive enemies after ~500 random plies). The
    legacy mask zeroed out those cells' BeliefNet outputs before upload,
    silently discarding the network's predictions for ~1/4 of the live
    enemy pieces.

    The GPU observation kernel separately reads ``d_belief[cell]`` only
    when the cell holds an enemy piece (per-cell piece_seat check in
    observation.cu), so leaving non-enemy cells' beliefs intact in the
    upload buffer is harmless: those cells' beliefs are never read.

    The GpuRollout ``_beliefs`` buffer is indexed ``[env, seat, type, cell]``,
    where "seat" is the OBSERVER and "type" is the 12-way belief over the
    piece type at that cell. We just transpose our net's
    ``(..., cell, type)`` layout.
    """
    if probs.dim() != 4 or probs.shape[-1] != N_BELIEF_TYPES or probs.shape[-2] != NUM_CELLS:
        raise ValueError(
            f"probs must be (N, 4, {NUM_CELLS}, {N_BELIEF_TYPES}); got {tuple(probs.shape)}"
        )
    if probs.size(1) != N_SEATS:
        raise ValueError(
            f"probs must have 4 seats in dim 1; got {probs.size(1)}"
        )

    if enemy_mask is not None:
        # Backwards compat: callers may pass a custom mask (e.g. tests).
        # In that case honour it. Production callers in train.py pass None.
        mask_expanded = enemy_mask.unsqueeze(0).unsqueeze(-1)
        masked = probs * mask_expanded.to(probs.dtype)
    else:
        masked = probs

    # Transpose (N, 4, 289, 12) → (N, 4, 12, 289).
    out = masked.transpose(-1, -2).contiguous()
    return out.to(torch.float32)


def refresh_beliefs_neural(
    rollout: "GpuRollout",
    belief_net: BeliefNet,
    *,
    apply_softmax: bool = True,
    autocast: bool = False,
    empty_cache: bool = True,
    chunk_size: int = 128,
) -> dict[str, float]:
    """Run BeliefNet on the current rollout state and push results to d_belief.

    Memory-disciplined: all temporaries are explicitly deleted before
    returning and ``torch.cuda.empty_cache()`` is called by default. This
    is important because BeliefNet forward on ``(4N, 256, 17, 17)`` at
    ``N=128`` allocates ~3 GB of activations that the caching allocator
    otherwise reserves across calls, starving the main PPO forward.

    Chunked forward
    ---------------
    The peak allocation inside :class:`BeliefNet.forward` is the FFN
    inner-dim activation: ``(B, 289, embed*ff_factor) fp32`` — at B=512,
    embed=256, ff_factor=4 that's ~0.58 GB per block, which OOMs on T4
    once the main PPO + arrangement buffer + belief buffer have filled
    the allocator (v22 crashed rollout 54, v23 crashed rollout 320 here).
    We slice the 4N-row batch into chunks of ``chunk_size`` rows (default
    128 — matches the PPO minibatch), run the forward sequentially, and
    concatenate logits on-device before softmax. Peak drops by
    ``4N / chunk_size`` (4× for N=128, chunk=128). FP throughput cost is
    trivial since BeliefNet is tiny relative to main PPO forward.

    Parameters
    ----------
    rollout
        The :class:`GpuRollout` whose device-resident belief buffer will be
        overwritten. Must have been constructed with ``enable_beliefs=True``.
    belief_net
        The inference-time belief net. Typically the EMA shadow copy from
        :class:`BeliefPPOTrainer` (not the trainable network).
    apply_softmax
        If True (default), softmax the logits before uploading. Pass False
        if ``belief_net`` already returns probabilities.
    autocast
        If True, run the forward in fp16 autocast. Off by default (see
        module docstring — no throughput win on T4).
    empty_cache
        If True (default), call ``torch.cuda.empty_cache()`` at the end to
        release allocator fragments. Set False in unit tests to avoid the
        ~50 ms per-call sync cost.
    chunk_size
        Per-forward batch size for BeliefNet. Peak FFN activation scales
        linearly with this value. Default 128 is safe on T4 with
        ``embed_dim=256, ff_factor=4``. Set higher for bigger GPUs to
        recover throughput; set 64 if still tight. Pass ``0`` or any
        value ``>= 4*num_envs`` to run a single unchunked forward.

    Returns
    -------
    dict of logging scalars:
        ``belief_infer/num_envs``, ``belief_infer/mean_entropy`` (average
        over enemy cells), ``belief_infer/max_prob_mean`` (avg confidence
        in top-1 type — 1/12 = pure uniform, 1.0 = fully certain).
    """
    N = rollout.num_envs

    belief_net.eval()
    # Pull obs for all 4 observer seats. Returns (N, 4, C, 17, 17) fp32 view.
    obs_sp_all, _obs_gl_all = rollout.build_all_seat_observations_torch()
    # Flatten to a single batch of (N*4) observations. reshape on the
    # CUDA-array-interface view is a view-op (no copy). No .clone() needed
    # since we consume the tensor synchronously in the belief_net forward
    # before the next build_*_observations_torch call.
    # (If a caller ever interleaves rollout stepping with inference, we'd
    # need .clone(); that isn't the current integration pattern.)
    C_in, H, W = obs_sp_all.shape[2:]
    obs_flat = obs_sp_all.reshape(N * N_SEATS, C_in, H, W)
    seat_flat = (
        torch.arange(N_SEATS, device=obs_flat.device, dtype=torch.long)
        .repeat(N)                                               # (4N,) seat per row
    )

    device_type = obs_flat.device.type
    amp_ctx = (
        torch.amp.autocast(device_type, dtype=torch.float16)
        if autocast and device_type == "cuda"
        else _NullContext()
    )
    total_rows = N * N_SEATS
    # Normalize chunk_size: 0 or negative means "no chunking".
    if chunk_size is None or chunk_size <= 0 or chunk_size >= total_rows:
        chunk_size = total_rows

    logit_chunks: list[torch.Tensor] = []
    with torch.no_grad(), amp_ctx:
        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)
            obs_chunk = obs_flat[start:end]
            seat_chunk = seat_flat[start:end]
            out = belief_net(obs_chunk, seat_idx=seat_chunk)
            # .float() upcasts fp16→fp32 under autocast; no-op otherwise.
            # .contiguous() so torch.cat downstream doesn't hold a non-
            # contig view of the last FFN output.
            logit_chunks.append(out["logits"].float().contiguous())
            del out, obs_chunk, seat_chunk
    logits = (
        logit_chunks[0] if len(logit_chunks) == 1 else torch.cat(logit_chunks, dim=0)
    )                                                # (4N, 289, 12)
    del logit_chunks

    # Free the intermediate forward-pass state before we allocate more.
    del obs_flat, seat_flat

    if apply_softmax:
        probs = F.softmax(logits, dim=-1)
        del logits
    else:
        probs = logits
    probs_5d = probs.reshape(N, N_SEATS, NUM_CELLS, N_BELIEF_TYPES)

    # Enemy-mask + transpose + upload.
    upload = belief_logits_to_upload_shape(probs_5d)    # (N, 4, 12, 289) fp32
    upload_cpu = upload.detach().cpu().numpy()
    rollout.upload_beliefs(upload_cpu)
    del upload

    # --- Diagnostics: compute over ENEMY cells only (others would be zero
    # if we gathered the masked upload, but we compute on raw probs_5d so
    # the entropy reflects the net's actual output). ---
    enemy_mask = _get_enemy_mask(probs_5d.device)       # (4, 289) bool
    em_expand = enemy_mask.unsqueeze(0).expand(N, -1, -1).reshape(-1)
    flat_probs = probs_5d.reshape(-1, N_BELIEF_TYPES)
    enemy_probs = flat_probs[em_expand]
    if enemy_probs.numel() == 0:
        metrics = {"belief_infer/num_envs": float(N)}
    else:
        eps = 1e-12
        entropy = -(enemy_probs * (enemy_probs + eps).log()).sum(dim=-1).mean().item()
        max_prob = enemy_probs.max(dim=-1).values.mean().item()
        metrics = {
            "belief_infer/num_envs": float(N),
            "belief_infer/mean_entropy": float(entropy),
            "belief_infer/max_prob_mean": float(max_prob),
        }

    # Release remaining CUDA tensors + fragments so the main PPO forward
    # gets a clean allocator pool.
    del probs, probs_5d, enemy_probs, flat_probs, em_expand, upload_cpu
    if empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


class _NullContext:
    """Fallback no-op context manager for when autocast is disabled.

    ``torch.amp.autocast`` only supports ``enabled=False`` on some torch
    versions; this avoids that compatibility wart.
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
