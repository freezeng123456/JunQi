"""junqi_rl.networks.junqi_net — Policy + value network for 四国军棋.

Architecture overview
---------------------
The network maps a canonical-frame observation to a policy distribution over
the flat 83,521-action space and a scalar (or categorical) value estimate.

Input
~~~~~
* ``obs_spatial``  : float32 tensor ``(B, OBS_CHANNELS, 17, 17)``  — 256 channels
* ``obs_global``   : float32 tensor ``(B, OBS_GLOBAL_DIMS)``        — 28 scalars
* ``legal_mask``   : bool tensor    ``(B, FLAT_ACTION_DIM)``         — 83,521 bits

Pipeline
~~~~~~~~
1. **CNN stem** — 3 conv layers compress (256, 17, 17) → (D, 17, 17),
   keeping spatial resolution to preserve board topology.

2. **Positional patch embedding** — reshape (D, 17, 17) → (289, D), add
   learnable positional embeddings (one per board cell).

3. **Global token injection** — project ``obs_global`` (28,) → (1, D) and
   prepend as a CLS token; sequence length becomes 290.

4. **Transformer trunk** — L layers of pre-norm self-attention + FFN
   (default L=6, D=256, 8 heads).

5. **Heads**:
   - *Policy head*: For each of the 289 cell tokens, project to D/8 keys and
     queries; build (289, 289) cross-attention map → reshape to (83,521)
     logits; apply legal mask.
   - *Value head*: Read from CLS token; project to 1 scalar or N_VF_CAT
     categorical bins (default: scalar).

Action space
~~~~~~~~~~~~
``action_id = src_flat * 289 + dst_flat``  where ``src_flat = sy * 17 + sx``.
The model outputs logits in **canonical** frame.  Callers must convert to
world frame with ``unrotate_action_id`` before passing to ``JunqiEnv.step``.

See also
--------
ADR-119 (flat action encoding), ADR-102 (canonical rotation), ADR-106 (obs layout).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_core.board import (
    NUM_ON_BOARD_CELLS,
    COMPACT_ACTION_DIM,
    COMPACT_TO_FLAT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE: int = 17
NUM_CELLS: int = BOARD_SIZE * BOARD_SIZE        # 289
# Compact action space: only on-board cells (129 × 129 = 16,641)
FLAT_ACTION_DIM: int = COMPACT_ACTION_DIM       # 16,641 (was 83,521)
N_VF_CAT: int = 3                               # categorical value bins (win/draw/lose)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JunqiNetConfig:
    """All hyper-parameters for :class:`JunqiNet`.

    Defaults mirror Ataraxos ``MoveTransformerConfig`` scaled to the
    larger 4-player board (17×17 vs 10×10) and 83,521-action space.
    """

    # CNN stem
    cnn_channels: int = 128
    """Number of channels after the CNN stem (= transformer embed dim D)."""

    cnn_layers: int = 3
    """Number of convolutional layers in the stem."""

    # Transformer trunk
    depth: int = 6
    """Number of transformer layers."""

    embed_dim: int = 256
    """Transformer embedding dimension (must be divisible by n_head)."""

    n_head: int = 8
    """Number of self-attention heads."""

    ff_factor: int = 4
    """Feed-forward hidden dim multiplier (ff_dim = embed_dim * ff_factor)."""

    dropout: float = 0.0
    """Dropout probability in transformer layers."""

    pos_emb_std: float = 0.02
    """Std for truncated-normal positional embedding initialisation."""

    # Value head
    use_cat_vf: bool = False
    """If True, value head predicts N_VF_CAT bins (categorical RL).
    If False, predicts a single scalar (standard actor-critic)."""

    # Action head
    action_key_dim: int = 64
    """Dimension of query/key projections for the bilinear action logits."""


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre-norm residual transformer layer.

    ``x → LN → MHA → Add → LN → FFN → Add``

    Parameters
    ----------
    embed_dim
        Token dimension.
    n_head
        Number of attention heads.
    ff_factor
        Feed-forward hidden dim = embed_dim × ff_factor.
    dropout
        Dropout rate (applied inside MHA and FFN).
    """

    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        ff_factor: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = embed_dim * ff_factor
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D)
        # Self-attention (pre-norm)
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2, need_weights=False)
        x = x + x2
        # FFN (pre-norm)
        x = x + self.ff(self.norm2(x))
        return x


class CNNStem(nn.Module):
    """Convolutional stem: (B, C_in, 17, 17) → (B, D, 17, 17).

    Uses residual blocks with GELU activations.  The spatial resolution is
    preserved throughout (padding='same') so that the output can be
    directly unrolled into 289 board-cell tokens.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for i in range(num_layers):
            c_out = out_channels
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.GELU(),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        # 1×1 projection to match residual if in_channels != out_channels
        self.proj: nn.Module
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # (B, C_in, 17, 17)
        return self.net(x) + self.proj(x)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Main network
# ---------------------------------------------------------------------------


class JunqiNet(nn.Module):
    """Policy + value network for 4-player 四国军棋.

    Parameters
    ----------
    cfg
        :class:`JunqiNetConfig` hyper-parameters.

    Forward
    -------
    Call :meth:`forward` to get the full output dict (for training).
    Call :meth:`act` for inference-only (no gradient, greedy / sampled action).

    Output dict (``forward``)
    -------------------------
    ``action``         : int32 tensor  (B,)          canonical-frame action id
    ``action_log_prob``: float tensor  (B,)          log-prob of chosen action
    ``log_probs``      : float tensor  (B, 83521)    full log-prob distribution
    ``value``          : float tensor  (B,) or (B, N_VF_CAT)   value estimate
    """

    def __init__(self, cfg: JunqiNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or JunqiNetConfig()
        cfg = self.cfg

        # Counter for in-forward NaN/Inf guard hits. Incremented by forward()
        # each time a row of logits was non-finite and had to be replaced
        # with uniform-over-legal. Persists across calls; trainers can read
        # this to surface fp16 instability frequency.
        self._nan_fwd_count: int = 0

        D = cfg.embed_dim

        # -- CNN stem ----------------------------------------------------------
        self.cnn = CNNStem(OBS_CHANNELS, cfg.cnn_channels, cfg.cnn_layers)

        # Project CNN output channels to embed_dim (may be a no-op if equal)
        if cfg.cnn_channels != D:
            self.patch_proj: nn.Module = nn.Linear(cfg.cnn_channels, D)
        else:
            self.patch_proj = nn.Identity()

        # -- On-board cell index (129 of 289) for compact Transformer input ---
        # Register as buffer so it moves with .to(device) and is not a parameter.
        self.register_buffer(
            "on_board_idx",
            torch.from_numpy(COMPACT_TO_FLAT.astype("int64")),  # (129,) long
            persistent=False,
        )

        # -- Positional embedding: one vector per on-board cell (129) + CLS ---
        self.pos_emb = nn.Parameter(
            torch.empty(1, NUM_ON_BOARD_CELLS + 1, D)  # (1, 130, D)
        )
        nn.init.trunc_normal_(self.pos_emb, std=cfg.pos_emb_std)

        # -- Global (CLS) token projection ------------------------------------
        self.global_proj = nn.Linear(OBS_GLOBAL_DIMS, D)

        # -- Transformer trunk ------------------------------------------------
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(D, cfg.n_head, cfg.ff_factor, cfg.dropout)
                for _ in range(cfg.depth)
            ]
        )
        self.norm_out = nn.LayerNorm(D)

        # -- Policy head -------------------------------------------------------
        # Bilinear action logits: for each (src, dst) pair the score is
        #   score[b, src, dst] = q[b, src] · k[b, dst]^T / sqrt(key_dim)
        # which is then flattened to (B, 289*289) = (B, 83521).
        Kd = cfg.action_key_dim
        self.q_proj = nn.Linear(D, Kd, bias=False)
        self.k_proj = nn.Linear(D, Kd, bias=False)

        # -- Value head --------------------------------------------------------
        if cfg.use_cat_vf:
            self.value_head: nn.Module = nn.Linear(D, N_VF_CAT)
        else:
            self.value_head = nn.Linear(D, 1)

        self._init_weights()

    # -------------------------------------------------------------------------
    # Weight init
    # -------------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    # -------------------------------------------------------------------------
    # Forward helpers
    # -------------------------------------------------------------------------

    def _encode(
        self,
        obs_spatial: Tensor,   # (B, OBS_CHANNELS, 17, 17)
        obs_global: Tensor,    # (B, OBS_GLOBAL_DIMS)
    ) -> tuple[Tensor, Tensor]:
        """Encode inputs → (cls_token, cell_tokens).

        The CNN runs on the full 17×17 grid, but we only extract the 129
        on-board cell patches for the Transformer (5x attention savings).

        Returns
        -------
        cls : (B, D)   — global context token
        cells : (B, 129, D) — per-on-board-cell tokens
        """
        B = obs_spatial.size(0)
        D = self.cfg.embed_dim

        # CNN stem → (B, cnn_channels, 17, 17)
        feat = self.cnn(obs_spatial)

        # Flatten spatial → (B, 289, cnn_channels)
        feat = feat.permute(0, 2, 3, 1).reshape(B, NUM_CELLS, -1)

        # Extract only 129 on-board cells → (B, 129, cnn_channels)
        feat = feat[:, self.on_board_idx]

        # Project to embed_dim → (B, 129, D)
        cell_tokens = self.patch_proj(feat)  # type: ignore[operator]

        # CLS token from global vector → (B, 1, D)
        cls_token = self.global_proj(obs_global).unsqueeze(1)

        # Concatenate → (B, 130, D)
        tokens = torch.cat([cls_token, cell_tokens], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_emb

        # Transformer trunk
        for layer in self.transformer:
            tokens = layer(tokens)
        tokens = self.norm_out(tokens)

        cls_out = tokens[:, 0, :]          # (B, D)
        cells_out = tokens[:, 1:, :]       # (B, 129, D)
        return cls_out, cells_out

    def _policy_logits(
        self,
        cells: Tensor,          # (B, 129, D)
        legal_mask: Tensor,     # (B, 16641) bool
    ) -> Tensor:
        """Compute action logits from on-board cell embeddings.

        Score for (src → dst) = q[src] · k[dst]^T / sqrt(Kd).
        Both src and dst range over 129 on-board cells, producing a
        (B, 129, 129) attention map flattened to (B, 16641).
        Illegal actions are masked to a very negative finite sentinel before
        returning.  The finite form is important for the compiled full-
        distribution entropy/KL computation; action legality is checked
        explicitly by PPO before an update.
        """
        Kd = self.cfg.action_key_dim
        q = self.q_proj(cells)   # (B, 129, Kd)
        k = self.k_proj(cells)   # (B, 129, Kd)
        # Cast to fp32 for the bilinear attention to avoid fp16 overflow
        # when q·k^T accumulates over Kd dimensions.
        q_f = q.float()
        k_f = k.float()
        attn = torch.bmm(q_f, k_f.transpose(1, 2)) / math.sqrt(Kd)
        # Flatten to (B, 16641)
        logits = attn.reshape(-1, FLAT_ACTION_DIM)
        # Retain a finite sentinel here.  Under torch.compile, exact -inf in
        # the normal action distribution makes entropy/KL terms hit 0 * -inf
        # and can poison every PPO gradient.  A direct selected-action mask
        # assertion in PPO makes the sentinel unable to conceal a legality
        # reconstruction error.
        _NEG_INF = torch.tensor(-1e9, dtype=torch.float32, device=logits.device)
        logits = torch.where(legal_mask, logits, _NEG_INF)
        return logits

    def _value(self, cls: Tensor) -> Tensor:
        """Compute value from CLS token.

        Returns
        -------
        (B,) scalar value  OR  (B, N_VF_CAT) log-probs (categorical).
        """
        # Run value head in fp32 to prevent fp16 overflow in the final
        # Linear / log_softmax. Under fp16 autocast a transformer trunk
        # can produce CLS tokens with peak magnitudes that exceed fp16's
        # ±65504 dynamic range when multiplied by the value-head weight,
        # producing +inf in `v` and then NaN in `log_softmax`. The cast
        # is essentially free (cls is (B, D), B≤2048, D≤512).
        v = self.value_head(cls.float())
        if self.cfg.use_cat_vf:
            return v.log_softmax(dim=-1)   # (B, N_VF_CAT)
        return v.squeeze(-1)               # (B,)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def forward(
        self,
        obs_spatial: Tensor,    # (B, OBS_CHANNELS, 17, 17)  float32
        obs_global: Tensor,     # (B, OBS_GLOBAL_DIMS)        float32
        legal_mask: Tensor,     # (B, FLAT_ACTION_DIM)        bool
        *,
        actions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Full forward pass used during training.

        Returns a dict with keys:
        ``action``, ``action_log_prob``, ``log_probs``, ``value``.

        ``actions`` is an optional training-only fast path.  When supplied,
        the method evaluates those actions instead of drawing a throwaway
        sample.  The returned keys, shapes and checkpoint structure are
        unchanged.  Existing callers that pass the original three inputs keep
        the exact sampling behaviour.

        NaN-safe: if any row of ``logits_f`` contains NaN/Inf (observed
        on T4 with fp16 autocast under slow-lr-decay regimes — see v29
        R51 crash), we replace that row's logits with a uniform-over-
        legal fallback so ``Categorical.sample()`` doesn't blow up.
        The PPO trainer's outer NaN guard will still catch and skip the
        bad minibatch via the returned log_probs, so the EMA/optimiser
        stay consistent. Counter ``_nan_fwd_count`` is incremented so
        callers can surface the frequency.
        """
        cls, cells = self._encode(obs_spatial, obs_global)
        logits = self._policy_logits(cells, legal_mask)
        # Cast to fp32 for numerically stable log_softmax / sampling
        # (fp16 logits with large masked regions can overflow).
        logits_f = logits.float()

        # --- NaN / Inf guard (pre-sample) ----------------------------
        # Any non-finite value in a row poisons Categorical.sample() with
        # "probability tensor contains either `inf`, `nan` or element <
        # 0". Replace whole rows that are non-finite with a uniform-
        # over-legal fallback (0.0 logit for legal, -inf for illegal).
        # Per-element replacement (just setting NaN to 0 while keeping
        # the rest) doesn't help — a single -inf from the legal mask
        # mixed with a finite NaN still makes the row all-zero after
        # softmax, which sample() rejects.
        # The guard and its diagnostic counter are only needed before random
        # sampling: Categorical rejects a poisoned probability tensor. During
        # PPO action evaluation no sampler is constructed, and the trainer has
        # one batched NaN/Inf check that can skip the update. Avoiding these
        # ``bool(cuda_tensor)`` checks removes two forced stream
        # synchronisations from every training forward.
        if actions is None:
            finite_mask = torch.isfinite(logits_f).all(dim=-1)  # (B,) bool
            if not bool(finite_mask.all()):
                # Count occurrences for downstream logging.
                self._nan_fwd_count += int((~finite_mask).sum().item())
                # Build fallback logits: 0 where legal, -inf where illegal.
                fallback = torch.where(
                    legal_mask,
                    torch.zeros_like(logits_f),
                    torch.full_like(logits_f, float("-inf")),
                )
                # Replace only the bad rows.
                # unsqueeze(-1) for broadcasting the row-mask across actions.
                bad_rows = (~finite_mask).unsqueeze(-1)
                logits_f = torch.where(bad_rows, fallback, logits_f)

        # Dead envs (terminated transitions stored as dummy slots) carry a
        # legal_mask with zero True entries. Their logits row is then
        # all-(-inf), and ``log_softmax`` on such a row returns all-NaN
        # (log(0/0)). Downstream the PPO trainer's
        # ``torch.isfinite(...).all()`` guard sees one NaN and
        # short-circuits the ENTIRE minibatch -> 96/96 minibatches per
        # rollout were being silently dropped (root cause of the v17-v32
        # "0.80 ceiling = no actual training" observation on T4 fp16).
        # Replace any row with no legal action by uniform-over-all
        # logits so log_softmax stays finite. Their advantage is 0 in
        # vs-random mode and these transitions contribute zero gradient,
        # so the artificial uniform prior is harmless.
        no_legal = ~legal_mask.any(dim=-1)                  # (B,) bool
        if actions is None:
            if bool(no_legal.any()):
                uniform = torch.zeros_like(logits_f)        # all-zero -> uniform
                logits_f = torch.where(no_legal.unsqueeze(-1), uniform, logits_f)
        else:
            # This branch is entirely device-side: it is a no-op for normal
            # legal rows and keeps legacy dummy/dead rows finite without a
            # host synchronisation.
            logits_f = torch.where(
                no_legal.unsqueeze(-1),
                torch.zeros_like(logits_f),
                logits_f,
            )

        log_probs = logits_f.log_softmax(dim=-1)

        if actions is None:
            dist = Categorical(logits=logits_f)
            chosen_actions = dist.sample()  # (B,)
        else:
            if actions.ndim != 1 or actions.shape[0] != logits_f.shape[0]:
                raise ValueError(
                    "actions must have shape (B,), matching the observation "
                    f"batch; got {tuple(actions.shape)} for B={logits_f.shape[0]}"
                )
            chosen_actions = actions.to(device=logits_f.device, dtype=torch.long)

        value = self._value(cls)

        return {
            "action": chosen_actions.int(),
            "action_log_prob": log_probs.gather(
                1, chosen_actions.unsqueeze(1)
            ).squeeze(1),
            "log_probs": log_probs,          # full distribution
            "value": value,
        }

    @torch.no_grad()
    def act(
        self,
        obs_spatial: Tensor,    # (B, OBS_CHANNELS, 17, 17)
        obs_global: Tensor,     # (B, OBS_GLOBAL_DIMS)
        legal_mask: Tensor,     # (B, FLAT_ACTION_DIM)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Inference-only forward using Gumbel-max sampling.

        Uses the Gumbel-max trick for fast sampling without building the full
        probability distribution.  Illegal actions have logits = -inf, so
        logits + Gumbel = -inf and they never win the argmax.

        Returns
        -------
        actions     : int32  (B,)
        log_probs   : float  (B,)   log-prob of chosen action
        values      : float  (B,) or (B, N_VF_CAT)
        """
        cls, cells = self._encode(obs_spatial, obs_global)
        logits = self._policy_logits(cells, legal_mask)  # (B, 83521), illegal=-inf

        logits_f = logits.float()
        u = torch.rand_like(logits_f).clamp_(1e-10, 1.0)
        gumbel = -torch.log(-torch.log(u))
        actions = (logits_f + gumbel).argmax(dim=-1)
        lse = logits_f.logsumexp(dim=-1)
        log_probs = (
            logits_f.gather(1, actions.unsqueeze(1)).squeeze(1)
            - lse
        )

        # Mean entropy over legal actions, for collect-wide H (not the
        # advantage-filtered training batch). Illegal logits are -inf.
        log_p = logits_f - lse.unsqueeze(-1)
        safe_p = torch.where(legal_mask, log_p.exp(), torch.zeros_like(log_p))
        safe_log_p = torch.where(legal_mask, log_p, torch.zeros_like(log_p))
        self._last_entropy_t = -(safe_p * safe_log_p).sum(dim=-1)

        values = self._value(cls)
        return actions.int(), log_probs, values

    @torch.no_grad()
    def act_greedy(
        self,
        obs_spatial: Tensor,
        obs_global: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Greedy (argmax) action selection — useful for evaluation."""
        _, cells = self._encode(obs_spatial, obs_global)
        logits = self._policy_logits(cells, legal_mask)
        return logits.argmax(dim=-1).int()

    def num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
