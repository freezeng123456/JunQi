"""junqi_rl.networks.belief_net — Neural belief distribution over enemy pieces.

Replaces (actually: augments, sits in parallel with) the hand-coded deductive
belief rules in ``src/env/cuda/src/belief.cu`` by producing a per-cell
distribution over the 12 tracked piece types for every enemy-occupied cell.

Architecture (P1 first cut — "BeliefTransformer" variant from Ataraxos,
see docs/P1_BELIEF_NET_PLAN.md)
---------------------------------------------------------------------------
* CNN stem (reused from :mod:`.junqi_net`) consumes the 256-ch observation.
* Board features flatten to 289 tokens.
* Per-cell learned positional embedding.
* ``n_encoder_layer`` non-causal self-attention layers (no look-ahead mask:
  this is a position-classification problem, not autoregressive).
* Per-cell linear head → ``(B, 289, 12)`` logits.
* Constraint masking (placement rules + per-seat remaining inventory)
  applied in :func:`compute_belief_loss` and callers (not inside forward,
  so the raw logits tensor remains differentiable).

Key choices relative to Ataraxos
--------------------------------
* **Non-autoregressive decoder**: P1 drops the causal AR decoder over
  pieces and decodes per-cell in one shot. Ataraxos's ablation claims
  ~+5% belief-accuracy from AR; we skip to save ~50 LOC and a sampling
  loop. If win-rate gains stall we promote in P1.5.
* **No temporal attention**: Ataraxos interleaves spatial + temporal
  attention (their ``TemporalBeliefTransformer`` variant). JunQi's
  observation already encodes 32-step move history + 12-ch death reasons,
  so we get history via features (the same trick Ataraxos's stateless
  variant uses). Promotion path is documented in the plan.
* **Seat-shared**: one network handles all 4 seats via a seat-id
  embedding, matching :class:`ArrangementNet`.

Forward contract
----------------
Input
~~~~~
* ``obs_spatial : (B, 256, 17, 17)``   — same tensor fed to JunqiNet.
* ``obs_global : (B, 28)``             — ignored by P1.1 (fed for API
                                         consistency; may be used by a
                                         future global-feature head).
* ``seat_idx : (B,) int64``            — observer seat for each sample.

Output
~~~~~~
* ``belief_logits : (B, 289, 12)``    — per-cell categorical logits over
                                         the 12 tracked piece types.
* ``belief_log_prob : (B, 289, 12)``   — ``log_softmax`` over the final
                                         axis (provided for loss-side
                                         convenience; callers should NOT
                                         re-log-softmax).

Masking contract
----------------
The forward pass does **not** apply placement-rule or inventory masks to
``belief_logits``. Callers (loss functions, CUDA uploaders) must:

1. Multiply/set unreachable (type, seat) pairs to ``-inf`` based on
   ``SLOT_TYPE_ALLOWED`` (placement rule) — shape ``(30, 12)`` per seat,
   tiled to ``(129, 12)`` after taking only on-board slots.
2. Zero out cells without enemy pieces via ``enemy_mask``.
3. Mask types whose per-seat remaining inventory is zero (dynamic).

Separating logits from mask keeps the forward differentiable regardless
of which cells happen to be occupied in a given batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from junqi_core.info_model import NUM_TRACKED_TYPES
from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.networks.junqi_net import CNNStem, BOARD_SIZE, NUM_CELLS


__all__ = [
    "BeliefNetConfig",
    "BeliefNet",
    "N_BELIEF_TYPES",
    "belief_net_config_from_dict",
]


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

N_BELIEF_TYPES: int = NUM_TRACKED_TYPES       # 12
assert N_BELIEF_TYPES == 12, (
    "BeliefNet expects NUM_TRACKED_TYPES == 12. If the tracked-type set "
    "ever grows (e.g. by splitting ZHADAN into front/back bombs), update "
    "the model's output head and all callers."
)

_N_SEATS: int = 4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BeliefNetConfig:
    """Hyper-parameters for :class:`BeliefNet`.

    Defaults target a T4 (16 GB) with ``batch_size=128``. On a bigger
    GPU you'd bump ``embed_dim`` to 512 and ``n_encoder_layer`` to 6 to
    match Ataraxos's Table 25.
    """

    n_encoder_layer: int = 4
    """Transformer encoder layers (Ataraxos: 6). Smaller for T4."""

    n_head: int = 8
    """Attention heads. ``embed_dim`` must be divisible by this."""

    embed_dim: int = 256
    """Token dimension. Ataraxos uses 512; 256 halves memory per layer."""

    ff_factor: int = 4
    """Feed-forward inner dim = ``embed_dim * ff_factor``."""

    dropout: float = 0.0
    """Dropout inside attention + FFN. Ataraxos belief uses 0.2 — we keep
    0 for P1 (no dropout at inference; can re-add for training if
    belief CE starts overfitting)."""

    pos_emb_std: float = 0.1
    """Std for truncated-normal initialisation of the learned positional
    embedding. Matches ArrangementNet."""

    cnn_channels: int = 128
    """CNN-stem output channels before token projection."""

    cnn_layers: int = 3
    """CNN-stem residual blocks."""

    use_seat_embedding: bool = True
    """If True, add a seat-id embedding to every token so one network
    produces beliefs for all 4 seats (matches the ArrangementNet trick)."""

    def __post_init__(self) -> None:
        if self.embed_dim % self.n_head != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by n_head "
                f"({self.n_head})."
            )


def belief_net_config_from_dict(d: dict) -> "BeliefNetConfig":
    """Build a :class:`BeliefNetConfig` from a YAML/JSON dict, accepting the
    legacy ``depth`` alias for ``n_encoder_layer``.

    Older configs (anything checked in before 2026-05-10) used the field
    name ``depth: N`` for the encoder layer count. The dataclass field is
    actually called ``n_encoder_layer``. The mismatch caused a hard
    ``TypeError`` on first launch of v35 (see
    ``exps/v35_ddp_combat_memory/launch.log.v35_buggy``). This helper
    transparently re-maps ``depth`` and emits a one-time deprecation
    warning so old cfgs keep working.
    """
    import warnings as _w

    d = dict(d)  # shallow copy
    if "depth" in d and "n_encoder_layer" not in d:
        _w.warn(
            "BeliefNetConfig: 'depth' is deprecated, use 'n_encoder_layer' "
            "instead. (Auto-mapping for backwards compat.)",
            DeprecationWarning,
            stacklevel=2,
        )
        d["n_encoder_layer"] = d.pop("depth")
    elif "depth" in d and "n_encoder_layer" in d:
        # Both supplied — prefer the new name, drop the legacy alias.
        d.pop("depth")
    return BeliefNetConfig(**d)


# ---------------------------------------------------------------------------
# Transformer block (non-causal, reused pattern from ArrangementNet)
# ---------------------------------------------------------------------------


class _SelfAttentionBlock(nn.Module):
    """Pre-LN encoder block: MHA (non-causal) + FFN, both residual.

    Uses ``F.scaled_dot_product_attention`` which picks flash-attention on
    CUDA when the hardware supports it.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_head: int,
        ff_factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.ln2 = nn.LayerNorm(d_model)
        ff_dim = d_model * ff_factor
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)

        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        H = self.n_head
        hd = self.head_dim

        y = self.ln1(x)
        q = self.q_proj(y).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(y).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(y).view(B, T, H, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn)

        z = self.ln2(x)
        z = self.ff2(F.gelu(self.ff1(z)))
        x = x + z
        return x


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class BeliefNet(nn.Module):
    """Per-cell belief over the 12 tracked piece types for enemy pieces.

    See the module docstring for the forward contract and the "masking is
    a caller's responsibility" rule.
    """

    def __init__(self, cfg: BeliefNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or BeliefNetConfig()
        cfg = self.cfg
        D = cfg.embed_dim

        # -- CNN stem: 256-ch obs → D-ch feature map ---------------------
        self.cnn = CNNStem(OBS_CHANNELS, cfg.cnn_channels, cfg.cnn_layers)

        # Project CNN output channels to embed_dim if mismatched.
        self.patch_proj: nn.Module
        if cfg.cnn_channels != D:
            self.patch_proj = nn.Linear(cfg.cnn_channels, D)
        else:
            self.patch_proj = nn.Identity()

        # -- Positional embedding ---------------------------------------
        # One learned vector per board cell (289), truncated-normal init.
        self.pos_emb = nn.Parameter(torch.zeros(1, NUM_CELLS, D))
        nn.init.trunc_normal_(self.pos_emb, std=cfg.pos_emb_std)

        # -- Seat embedding (shared-weights across all 4 seats) ----------
        self.seat_emb: nn.Module
        if cfg.use_seat_embedding:
            self.seat_emb = nn.Embedding(_N_SEATS, D)
            nn.init.trunc_normal_(self.seat_emb.weight, std=cfg.pos_emb_std)
        else:
            self.seat_emb = nn.Identity()

        # -- Transformer encoder ----------------------------------------
        self.blocks = nn.ModuleList([
            _SelfAttentionBlock(
                d_model=D,
                n_head=cfg.n_head,
                ff_factor=cfg.ff_factor,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_encoder_layer)
        ])
        self.ln_final = nn.LayerNorm(D)

        # -- Per-cell belief head: (B, 289, D) → (B, 289, 12) -----------
        self.head = nn.Linear(D, N_BELIEF_TYPES)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_spatial: Tensor,          # (B, OBS_CHANNELS, 17, 17)
        seat_idx: Tensor | None = None,  # (B,) int64 — required iff use_seat_embedding
        obs_global: Tensor | None = None,  # (B, OBS_GLOBAL_DIMS) — ignored by P1.1
    ) -> dict[str, Tensor]:
        """Compute per-cell belief logits for enemy pieces.

        Parameters
        ----------
        obs_spatial
            Float tensor of shape ``(B, 256, 17, 17)``. The same tensor
            you feed to :class:`JunqiNet.forward`.
        seat_idx
            Long tensor of shape ``(B,)`` with values in ``{0, 1, 2, 3}``,
            one observer seat per sample. Required if ``use_seat_embedding``.
        obs_global
            Optional global scalars (ignored by P1.1; present to match
            the JunqiNet signature so callers can pass the same batch).

        Returns
        -------
        dict with keys:
        * ``logits`` — ``(B, 289, 12)`` raw per-cell logits.
        * ``log_probs`` — ``(B, 289, 12)`` log-softmax over the last axis.
        """
        del obs_global  # Reserved; unused in P1.1

        if obs_spatial.ndim != 4:
            raise ValueError(
                f"obs_spatial must be 4D (B, C, H, W); got shape {tuple(obs_spatial.shape)}"
            )
        B = obs_spatial.size(0)
        if obs_spatial.shape[1:] != (OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"obs_spatial shape mismatch: expected (B, {OBS_CHANNELS}, "
                f"{BOARD_SIZE}, {BOARD_SIZE}); got {tuple(obs_spatial.shape)}"
            )

        # --- CNN stem ---------------------------------------------------
        x = self.cnn(obs_spatial)                       # (B, C, 17, 17)
        # Flatten spatial → (B, 289, C_cnn).
        x = x.flatten(start_dim=2).transpose(1, 2)

        # --- Project to embed_dim --------------------------------------
        x = self.patch_proj(x)                           # (B, 289, D)

        # --- Add positional + seat embeddings --------------------------
        x = x + self.pos_emb                             # broadcast (1, 289, D)
        if self.cfg.use_seat_embedding:
            if seat_idx is None:
                raise ValueError(
                    "BeliefNet was built with use_seat_embedding=True; "
                    "pass seat_idx=(B,) int64 to forward()."
                )
            if seat_idx.shape != (B,):
                raise ValueError(
                    f"seat_idx must have shape ({B},); got {tuple(seat_idx.shape)}"
                )
            if seat_idx.dtype != torch.long:
                raise ValueError(
                    f"seat_idx must be int64; got {seat_idx.dtype}"
                )
            s = self.seat_emb(seat_idx).unsqueeze(1)     # (B, 1, D)
            x = x + s

        # --- Transformer encoder ---------------------------------------
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_final(x)                             # (B, 289, D)

        # --- Belief head -----------------------------------------------
        logits = self.head(x)                            # (B, 289, 12)
        log_probs = F.log_softmax(logits, dim=-1)

        return {
            "logits": logits,
            "log_probs": log_probs,
        }

    # ------------------------------------------------------------------
    # Parameter count — useful for sanity / logging
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
