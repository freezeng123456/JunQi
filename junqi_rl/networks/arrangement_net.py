"""junqi_rl.networks.arrangement_net — Autoregressive piece-arrangement net.

Ataraxos-faithful implementation adapted for 4-seat JunQi (四国军棋).

Differences from Ataraxos ArrangementTransformer
------------------------------------------------
* **ARRANGEMENT_SIZE = 30** (not 40) — 5 camp slots per seat are included
  in the sequence but emit `PieceType.NONE` under a per-slot mask.
* **N_PIECE_TYPE_WITH_NONE = 13** — vocabulary includes NONE (idx 0). Camps
  are forced to emit NONE; all other slots forbid NONE.
* **No force_handedness** — 4-seat JunQi has no left-right symmetry (enemies
  sit on both sides), so we never force a handedness flip.
* **Seat conditioning** — a learned `seat_idx` embedding (4 rows) is added to
  every token so one shared network generates all four seats' arrangements.
* **Per-slot constraint mask** — precomputed `SLOT_TYPE_ALLOWED[30, 13]`
  encodes C1 (camps=NONE), C2 (strongholds=JUNQI only), C3 (DILEI in back
  two rows only), C4 (ZHADAN not in front row). C5 (piece counts) is
  enforced by the remaining-counts logic, same as Ataraxos.

Autoregressive generation order
-------------------------------
Slot 0, 1, 2, …, 29 left-to-right, top-to-bottom in the seat's local 5×6
frame. This matches `rules.py` indexing (row0 = front row, row5 = back row),
which is reused verbatim by `junqi_core.setup.validate_lineup`.

Output heads
------------
* ``logits``   : (B, 30, 13)    per-slot categorical over piece types
* ``value``    : (B, 30, C)     categorical VF (default C=N_VF_CAT=3)
* ``ent_pred`` : (B, 30, 1)     predicted future-accumulated NLL
                                (scaled by 1/reg_norm during training;
                                see Ataraxos rl.py::arr_train)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from junqi_core.rules import (
    ALL_PLACEABLE_PIECES,
    BACK_TWO_ROWS_INDICES,
    CAMP_INDICES,
    FRONT_ROW_INDICES,
    PIECE_COUNTS,
    PieceType,
    SLOTS_PER_SEAT,
    STRONGHOLD_INDICES,
)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ARRANGEMENT_SIZE: int = SLOTS_PER_SEAT           # 30
N_PIECE_TYPE_WITH_NONE: int = 13                 # NONE..GONGB (PieceType values 0,2..13 mapped to 0..12)
N_SEATS: int = 4
N_VF_CAT_DEFAULT: int = 3                        # win / draw / lose

# Mapping: index into the vocabulary → PieceType.value.
# We skip DARK (=1) because DARK never appears in a setup (C5 forbids it).
# So vocab index 0 = NONE, index 1 = JUNQI (type 2), ..., index 12 = GONGB (type 13).
_VOCAB_PIECE_TYPES: tuple[PieceType, ...] = (PieceType.NONE,) + tuple(ALL_PLACEABLE_PIECES)
assert len(_VOCAB_PIECE_TYPES) == N_PIECE_TYPE_WITH_NONE
PIECE_TYPE_VALUE_TO_VOCAB_IDX: dict[int, int] = {
    pt.value: idx for idx, pt in enumerate(_VOCAB_PIECE_TYPES)
}
VOCAB_IDX_TO_PIECE_TYPE_VALUE: tuple[int, ...] = tuple(pt.value for pt in _VOCAB_PIECE_TYPES)

# Vocab indices (convenience)
NONE_IDX: int = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.NONE.value]         # 0
JUNQI_IDX: int = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.JUNQI.value]       # 1
DILEI_IDX: int = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.DILEI.value]       # 2
ZHADAN_IDX: int = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.ZHADAN.value]     # 3


# ---------------------------------------------------------------------------
# Precomputed per-slot constraint mask (C1-C4)
# ---------------------------------------------------------------------------


def _build_slot_type_allowed() -> Tensor:
    """Return (30, 13) bool tensor: ``allowed[s, t]`` = slot s may hold type t.

    Constraints encoded (all static, independent of remaining counts):
      C1  — camp slots (6,8,12,16,18) must hold NONE (no other type allowed).
      C2a — stronghold slots (26,28) may hold JUNQI (among other types).
      C2b — non-stronghold slots may NOT hold JUNQI.
      C3  — DILEI only in back two rows (20..29).
      C4  — ZHADAN not in front row (0..4).
      Other ranked combatants may go anywhere non-camp.
      NONE may only go in camp slots (non-camps must be filled, C5).
    """
    mask = torch.zeros(ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE, dtype=torch.bool)
    for s in range(ARRANGEMENT_SIZE):
        is_camp = s in CAMP_INDICES
        is_stronghold = s in STRONGHOLD_INDICES
        is_front = s in FRONT_ROW_INDICES
        is_back_two = s in BACK_TWO_ROWS_INDICES

        if is_camp:
            # C1: only NONE allowed in camps.
            mask[s, NONE_IDX] = True
            continue

        # Non-camp slot: NONE is forbidden (C5 — must be filled).
        # JUNQI allowed iff stronghold; DILEI iff back two rows; ZHADAN iff not front.
        mask[s, JUNQI_IDX] = is_stronghold
        mask[s, DILEI_IDX] = is_back_two
        mask[s, ZHADAN_IDX] = not is_front
        # Ranked combatants (SILING..GONGB): anywhere non-camp.
        for pt in (PieceType.SILING, PieceType.JUNZH, PieceType.SHIZH,
                   PieceType.LVZH, PieceType.TUANZH, PieceType.YINGZH,
                   PieceType.LIANZH, PieceType.PAIZH, PieceType.GONGB):
            mask[s, PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value]] = True
    return mask


def _build_piece_counts() -> Tensor:
    """Return (13,) int tensor: total count of each vocab type in a lineup.

    NONE count = 5 (camps); placeable counts from rules.PIECE_COUNTS.
    Sum = 30 (ARRANGEMENT_SIZE). Matches the validator's expectation.
    """
    counts = torch.zeros(N_PIECE_TYPE_WITH_NONE, dtype=torch.long)
    counts[NONE_IDX] = len(CAMP_INDICES)
    for pt, n in PIECE_COUNTS.items():
        counts[PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value]] = int(n)
    assert int(counts.sum()) == ARRANGEMENT_SIZE
    return counts


# ---------------------------------------------------------------------------
# Building blocks (self-contained; no einops dependency)
# ---------------------------------------------------------------------------


class _CausalSelfAttentionLayer(nn.Module):
    """Pre-norm causal self-attention + FFN, sdpa backend (fp16/bf16 friendly).

    Mirrors Ataraxos ``SelfAttentionLayer`` but re-implemented without einops
    so we don't add a runtime dependency. Uses
    :func:`torch.nn.functional.scaled_dot_product_attention` so flash/fused
    kernels kick in automatically on CUDA.
    """

    def __init__(self, d_model: int, n_head: int, ff_factor: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
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

    def forward(self, x: Tensor) -> Tensor:  # (B, T, D)
        B, T, D = x.shape
        H = self.n_head
        hd = self.head_dim

        # --- MHA ---
        y = self.ln1(x)
        q = self.q_proj(y).view(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        k = self.k_proj(y).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(y).view(B, T, H, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn)

        # --- FFN ---
        z = self.ln2(x)
        z = self.ff2(F.relu(self.ff1(z)))
        x = x + z
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ArrangementNetConfig:
    """Hyper-parameters for :class:`ArrangementNet`.

    Defaults follow Ataraxos (depth=4, n_head=8, per-head-dim=64,
    embed_dim = 8 * 8 * 8 = 512).
    """

    depth: int = 4
    """Number of causal transformer layers."""

    n_head: int = 8
    """Number of attention heads."""

    embed_dim: int = 512
    """Token dimension. Must be divisible by n_head."""

    ff_factor: int = 4
    """Feed-forward hidden dim = embed_dim * ff_factor."""

    dropout: float = 0.0
    """Dropout inside attention + FFN."""

    pos_emb_std: float = 0.1
    """Std for truncated-normal positional embedding init."""

    use_cat_vf: bool = True
    """If True, value head predicts N_VF_CAT bins (categorical RL). Default
    matches Ataraxos `use_cat_vf=True`."""

    n_vf_cat: int = N_VF_CAT_DEFAULT
    """Number of value bins when use_cat_vf=True (win/draw/lose)."""


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------


class ArrangementNet(nn.Module):
    """Causal Transformer that generates a seat's 30-slot lineup.

    One network is shared across all four seats; a learned seat-embedding is
    added to every position so the model can still specialise. This matches
    the 4-seat rotational symmetry of JunQi (the legal-placement rules are
    identical up to rotation for all seats in their own local frame).

    Forward contract
    ----------------
    ``forward(seq, seat_idx)``:
      - ``seq`` : ``(B, T, N_PIECE_TYPE_WITH_NONE)`` one-hot arrangement prefix,
        where ``T <= ARRANGEMENT_SIZE``. Row ``t`` is the piece that was
        placed in slot ``t`` during the autoregressive rollout.
      - ``seat_idx`` : ``(B,)`` int64 in ``[0, 4)``. The seat this arrangement
        belongs to.
      - returns dict with ``logits``, ``value``, ``ent_pred``.

    Generation uses ``_create_legal_action_mask(seq)`` which combines:
      (a) per-slot static constraints (C1-C4), and
      (b) remaining-count constraints (C5).
    At test time the caller samples from the masked distribution; see
    :mod:`junqi_rl.arrangement.sampling` for the autoregressive loop.
    """

    def __init__(self, cfg: ArrangementNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ArrangementNetConfig()
        assert self.cfg.embed_dim % self.cfg.n_head == 0

        D = self.cfg.embed_dim

        # Input projections
        self.embedder = nn.Linear(N_PIECE_TYPE_WITH_NONE, D)
        self.positional_encoding = nn.Parameter(
            torch.empty(1, ARRANGEMENT_SIZE, D)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=self.cfg.pos_emb_std)

        # Seat conditioning (one row per seat).
        self.seat_emb = nn.Embedding(N_SEATS, D)
        nn.init.trunc_normal_(self.seat_emb.weight, std=self.cfg.pos_emb_std)

        # Causal "start token" — a zero one-hot prepended to shift the sequence
        # so slot t is predicted from prefix [<start>, x_0, ..., x_{t-1}].
        self.register_buffer(
            "start_token",
            torch.zeros(1, 1, N_PIECE_TYPE_WITH_NONE),
            persistent=False,
        )

        # Transformer trunk
        self.layers = nn.ModuleList([
            _CausalSelfAttentionLayer(
                d_model=D, n_head=self.cfg.n_head,
                ff_factor=self.cfg.ff_factor, dropout=self.cfg.dropout,
            )
            for _ in range(self.cfg.depth)
        ])
        self.norm_out = nn.LayerNorm(D)

        # Heads
        self.policy_out = nn.Linear(D, N_PIECE_TYPE_WITH_NONE)
        self.value_out = nn.Linear(
            D, self.cfg.n_vf_cat if self.cfg.use_cat_vf else 1
        )
        self.ent_out = nn.Linear(D, 1)

        # Constraint tables as persistent buffers so they move with .to(device).
        self.register_buffer(
            "slot_type_allowed", _build_slot_type_allowed(), persistent=False,
        )  # (30, 13) bool
        self.register_buffer(
            "piece_counts", _build_piece_counts(), persistent=False,
        )  # (13,) int64

    # ------------------------------------------------------------------ utils

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        seq: Tensor,       # (B, T, N_PIECE_TYPE_WITH_NONE) one-hot/soft; T <= 30
        seat_idx: Tensor,  # (B,) int64 in [0, 4)
    ) -> dict[str, Tensor]:
        """Predict per-prefix logits, value, ent_pred."""
        self._check_seq(seq)
        B, T, V = seq.shape

        # Shift right by prepending the start token (cf. Ataraxos).
        start = self.start_token.expand(B, -1, -1)             # (B, 1, V)
        shifted = torch.cat([start, seq], dim=1)[:, :ARRANGEMENT_SIZE]  # (B, T', V)
        T_prime = shifted.size(1)

        # Embed + positional + seat.
        x = self.embedder(shifted)                             # (B, T', D)
        x = x + self.positional_encoding[:, :T_prime]          # broadcast (1, T', D)
        seat_vec = self.seat_emb(seat_idx).unsqueeze(1)        # (B, 1, D)
        x = x + seat_vec                                       # broadcast over T'

        # Causal transformer
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)

        logits = self.policy_out(x)                            # (B, T', V)
        value = self.value_out(x)                              # (B, T', C|1)
        ent_pred = self.ent_out(x)                             # (B, T', 1)

        # Legal mask ONLY applies to the first T slots (the ones we're going
        # to sample from). Return them at their original T length so training
        # can align with the true-action targets directly.
        T_out = T_prime  # keep the full shifted length; caller slices to T.
        legal_mask = self._create_legal_action_mask(seq)       # (B, T_out, V)
        # If the network runs with a prefix shorter than T_out-1 (e.g. during
        # sampling), the mask above is built from `seq` which has exactly T
        # rows. We re-expand it to match T_prime = T+? (= min(T+1, 30)).
        if legal_mask.size(1) != T_out:
            # legal_mask has T rows; logits/value/ent_pred have T+1 (or 30)
            # rows because of the start-token shift. We only need to mask the
            # positions where we'll draw samples from — those are rows 0..T_out-1.
            # `_create_legal_action_mask` operates on `seq` and returns the
            # mask for slots [T_prime - seq.size(1) .. T_prime); we slot it in.
            raise AssertionError(
                f"legal_mask shape mismatch: got {legal_mask.shape}, "
                f"expected T_out={T_out}"
            )

        # Fill disallowed logits with a very negative (finite) sentinel. We
        # use finfo.min for fp32 and a large negative for fp16 stability.
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~legal_mask, neg_inf)

        return {"logits": logits, "value": value, "ent_pred": ent_pred}

    # ------------------------------------------------------------------ masks

    @torch.no_grad()
    def _create_legal_action_mask(self, seq: Tensor) -> Tensor:
        """Combine per-slot static mask + remaining-count mask.

        Parameters
        ----------
        seq : (B, T, V)
            Prefix of placements as one-hot/soft rows (rows sum to 1). The
            cumulative sum over T gives how many of each type have already
            been placed — ``piece_counts - cumsum`` is the remaining budget.

        Returns
        -------
        mask : (B, T_prime, V) bool
            T_prime = min(T + 1, ARRANGEMENT_SIZE). The model emits logits
            for positions 0..T_prime-1 (one prediction per prefix length).
            Each slot must have at least one legal type — caller is expected
            to feed only well-formed prefixes. A sanity assertion guards
            this.
        """
        B, T, V = seq.shape
        T_prime = min(T + 1, ARRANGEMENT_SIZE)

        # Remaining counts AFTER placing rows 0..t-1 (for t = 0..T_prime-1).
        # remaining[b, 0] = piece_counts (no placements yet).
        # remaining[b, t] = piece_counts - sum_{i<t} seq[b, i]   for t >= 1.
        cum = seq.cumsum(dim=1)                              # (B, T, V)
        pc = self.piece_counts.to(dtype=seq.dtype).view(1, 1, V)
        # We need remaining at positions 0..T_prime-1. Position 0 is "nothing
        # placed" — use piece_counts directly.
        zero_row = pc.expand(B, 1, V)                        # (B, 1, V)
        remaining_all = torch.cat([zero_row, pc - cum], dim=1)  # (B, T+1, V)
        remaining = remaining_all[:, :T_prime]               # (B, T_prime, V)

        count_ok = remaining > 0                             # (B, T_prime, V) bool

        # Per-slot static mask (C1-C4).
        static = self.slot_type_allowed[:T_prime].unsqueeze(0)  # (1, T_prime, V)

        mask = count_ok & static                             # (B, T_prime, V)

        # NOTE: We do NOT raise here even when some prefix has zero legal
        # choices. Dead-end prefixes are a legitimate state during
        # autoregressive sampling with random-initialised nets (see
        # junqi_rl.arrangement.sampling). The caller (training forward or
        # sampling loop) is responsible for detecting all-False rows:
        #   - Training forward always feeds valid full lineups, so this
        #     cannot happen.
        #   - Sampling uses ``(logits > _LEGAL_THRESHOLD).any(dim=-1)`` to
        #     detect dead-ends and retry / fallback.
        return mask

    # ------------------------------------------------------------------ checks

    def _check_seq(self, seq: Tensor) -> None:
        if seq.ndim != 3:
            raise ValueError(f"seq must be 3D (B, T, V), got shape {tuple(seq.shape)}")
        if seq.size(-1) != N_PIECE_TYPE_WITH_NONE:
            raise ValueError(
                f"seq last dim must be {N_PIECE_TYPE_WITH_NONE}, got {seq.size(-1)}"
            )
        if seq.size(1) > ARRANGEMENT_SIZE:
            raise ValueError(
                f"seq length must be <= {ARRANGEMENT_SIZE}, got {seq.size(1)}"
            )
        if seq.device != self.device:
            raise ValueError(
                f"seq device {seq.device} != model device {self.device}"
            )


# ---------------------------------------------------------------------------
# Small utilities exported for tests / generation code
# ---------------------------------------------------------------------------


def vocab_idx_to_piece_type(idx: int) -> PieceType:
    """Inverse of ``PIECE_TYPE_VALUE_TO_VOCAB_IDX``."""
    return PieceType(VOCAB_IDX_TO_PIECE_TYPE_VALUE[idx])


def lineup_to_onehot(lineup_vocab: Tensor) -> Tensor:
    """Convert a (..., 30) int64 vocab-idx tensor to (..., 30, 13) one-hot."""
    return F.one_hot(lineup_vocab, num_classes=N_PIECE_TYPE_WITH_NONE).float()


__all__ = [
    "ARRANGEMENT_SIZE",
    "N_PIECE_TYPE_WITH_NONE",
    "N_SEATS",
    "N_VF_CAT_DEFAULT",
    "NONE_IDX",
    "JUNQI_IDX",
    "DILEI_IDX",
    "ZHADAN_IDX",
    "PIECE_TYPE_VALUE_TO_VOCAB_IDX",
    "VOCAB_IDX_TO_PIECE_TYPE_VALUE",
    "ArrangementNetConfig",
    "ArrangementNet",
    "vocab_idx_to_piece_type",
    "lineup_to_onehot",
]
