"""Tests for junqi_rl.networks.arrangement_net."""

from __future__ import annotations

import pytest
import torch

from junqi_core.rules import (
    BACK_TWO_ROWS_INDICES,
    CAMP_INDICES,
    FRONT_ROW_INDICES,
    PIECE_COUNTS,
    PieceType,
    STRONGHOLD_INDICES,
)
from junqi_core.setup import validate_lineup
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    N_SEATS,
    N_VF_CAT_DEFAULT,
    NONE_IDX,
    PIECE_TYPE_VALUE_TO_VOCAB_IDX,
    VOCAB_IDX_TO_PIECE_TYPE_VALUE,
    ArrangementNet,
    ArrangementNetConfig,
    _build_piece_counts,
    _build_slot_type_allowed,
    lineup_to_onehot,
)


# ---------------------------------------------------------------------------
# Constant / table tests
# ---------------------------------------------------------------------------


def test_vocab_size_and_bijection():
    assert N_PIECE_TYPE_WITH_NONE == 13
    assert ARRANGEMENT_SIZE == 30
    assert len(VOCAB_IDX_TO_PIECE_TYPE_VALUE) == N_PIECE_TYPE_WITH_NONE
    # Round-trip
    for vtype in VOCAB_IDX_TO_PIECE_TYPE_VALUE:
        idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[vtype]
        assert VOCAB_IDX_TO_PIECE_TYPE_VALUE[idx] == vtype
    # DARK must NOT be in the vocab.
    assert PieceType.DARK.value not in PIECE_TYPE_VALUE_TO_VOCAB_IDX


def test_piece_counts_sum():
    pc = _build_piece_counts()
    assert pc.dtype == torch.long
    assert pc.shape == (N_PIECE_TYPE_WITH_NONE,)
    assert int(pc.sum()) == ARRANGEMENT_SIZE
    assert int(pc[NONE_IDX]) == len(CAMP_INDICES)  # 5
    for pt, n in PIECE_COUNTS.items():
        assert int(pc[PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value]]) == n


def test_slot_type_allowed_encodes_c1_c4():
    mask = _build_slot_type_allowed()
    assert mask.shape == (ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE)
    assert mask.dtype == torch.bool

    # C1 — camp slots: only NONE allowed.
    for s in CAMP_INDICES:
        assert bool(mask[s, NONE_IDX])
        assert int(mask[s].sum()) == 1

    # C2 — JUNQI only in strongholds.
    junqi_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.JUNQI.value]
    for s in range(ARRANGEMENT_SIZE):
        if s in STRONGHOLD_INDICES:
            assert bool(mask[s, junqi_idx]) is True
        else:
            assert bool(mask[s, junqi_idx]) is False

    # C3 — DILEI only in back two rows.
    dilei_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.DILEI.value]
    for s in range(ARRANGEMENT_SIZE):
        if s in CAMP_INDICES:
            continue
        if s in BACK_TWO_ROWS_INDICES:
            assert bool(mask[s, dilei_idx]) is True
        else:
            assert bool(mask[s, dilei_idx]) is False

    # C4 — ZHADAN not in front row (but allowed everywhere else non-camp).
    zhadan_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.ZHADAN.value]
    for s in range(ARRANGEMENT_SIZE):
        if s in CAMP_INDICES:
            continue
        if s in FRONT_ROW_INDICES:
            assert bool(mask[s, zhadan_idx]) is False
        else:
            assert bool(mask[s, zhadan_idx]) is True

    # NONE only in camps.
    for s in range(ARRANGEMENT_SIZE):
        if s in CAMP_INDICES:
            assert bool(mask[s, NONE_IDX])
        else:
            assert not bool(mask[s, NONE_IDX])


# ---------------------------------------------------------------------------
# Module forward / shape tests
# ---------------------------------------------------------------------------


def _tiny_cfg() -> ArrangementNetConfig:
    # Small network for fast tests.
    return ArrangementNetConfig(depth=2, n_head=4, embed_dim=64, ff_factor=2)


def test_forward_shapes_empty_prefix():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    B = 3
    seq = torch.zeros(B, 0, N_PIECE_TYPE_WITH_NONE)
    seat = torch.tensor([0, 1, 2])
    out = net(seq, seat)
    # Start-token shift → T_prime = 1 when T=0.
    assert out["logits"].shape == (B, 1, N_PIECE_TYPE_WITH_NONE)
    assert out["value"].shape == (B, 1, N_VF_CAT_DEFAULT)
    assert out["ent_pred"].shape == (B, 1, 1)


def test_forward_shapes_full_prefix():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    B = 2
    # Feed a valid full lineup so legal-mask construction passes.
    seed_lineup = _sample_valid_lineup_indices()
    seq = lineup_to_onehot(torch.tensor([seed_lineup, seed_lineup], dtype=torch.long))
    seat = torch.tensor([0, 3])
    out = net(seq, seat)
    # T = 30; T_prime = min(30+1, 30) = 30.
    assert out["logits"].shape == (B, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE)
    assert out["value"].shape == (B, ARRANGEMENT_SIZE, N_VF_CAT_DEFAULT)
    assert out["ent_pred"].shape == (B, ARRANGEMENT_SIZE, 1)


def test_grad_flows_to_params():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    B = 2
    seq = torch.zeros(B, 0, N_PIECE_TYPE_WITH_NONE)
    seat = torch.tensor([0, 2])
    out = net(seq, seat)
    loss = out["logits"].sum() + out["value"].sum() + out["ent_pred"].sum()
    loss.backward()
    n_with_grad = sum(1 for p in net.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in net.parameters())
    # Most params should see gradient; start_token is a buffer (not param).
    assert n_with_grad >= int(n_total * 0.6), f"only {n_with_grad}/{n_total} params had nonzero grad"


def test_disallowed_logits_masked():
    """Camp slot 0-th emission (slot 0 is a front-row slot, not a camp) —
    check that NONE (idx 0) is disallowed there."""
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg()).eval()
    B = 1
    seq = torch.zeros(B, 0, N_PIECE_TYPE_WITH_NONE)
    seat = torch.tensor([0])
    with torch.no_grad():
        logits = net(seq, seat)["logits"][0, 0]   # (V,)
    # Slot 0 is in FRONT_ROW; NONE, JUNQI, DILEI, ZHADAN all illegal there.
    assert torch.isneginf(logits[NONE_IDX]) or logits[NONE_IDX].item() < -1e30
    junqi_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.JUNQI.value]
    dilei_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.DILEI.value]
    zhadan_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.ZHADAN.value]
    for bad_idx in (NONE_IDX, junqi_idx, dilei_idx, zhadan_idx):
        assert logits[bad_idx].item() < -1e30, f"idx {bad_idx} was not masked"
    # A ranked combatant (PAIZH) is legal in front row → finite logit.
    paizh_idx = PIECE_TYPE_VALUE_TO_VOCAB_IDX[PieceType.PAIZH.value]
    assert torch.isfinite(logits[paizh_idx])


def test_camp_slot_only_none_allowed():
    """At slot 6 (camp), only NONE should be a legal emission."""
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg()).eval()
    # Build a prefix of length 6 using valid types so legal-mask is well-formed.
    seed = _sample_valid_lineup_indices()
    prefix = torch.tensor([seed[:6]], dtype=torch.long)
    seq = lineup_to_onehot(prefix)  # (1, 6, V)
    seat = torch.tensor([0])
    with torch.no_grad():
        out = net(seq, seat)
        logits_at_slot6 = out["logits"][0, 6]   # T_prime = 7; row 6 is slot 6's prediction.
    # Only NONE should be finite.
    for i in range(N_PIECE_TYPE_WITH_NONE):
        if i == NONE_IDX:
            assert torch.isfinite(logits_at_slot6[i])
        else:
            assert logits_at_slot6[i].item() < -1e30, f"slot 6 allowed idx {i}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_valid_lineup_indices() -> list[int]:
    """Use the existing generator to obtain one valid lineup as vocab indices."""
    from junqi_core.setup import generate_random_lineup
    import random

    rng = random.Random(0)
    lineup = generate_random_lineup(rng)
    assert len(lineup) == ARRANGEMENT_SIZE
    return [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup]


# ---------------------------------------------------------------------------
# End-to-end: feeding REAL valid lineups through the net — at every prefix
# length the true next-piece's vocab idx must be among the legal choices.
# (Uniform-over-legal sampling WITHOUT look-ahead can dead-end because of
#  the C5 × slot-type constraints — that's a known property of Ataraxos-style
#  masking and is handled at sample-time by retry/backtrack, not by the mask.)
# ---------------------------------------------------------------------------


def test_mask_accepts_every_valid_lineup():
    """For each of 200 `generate_random_lineup`-produced lineups, check that
    the model's legal mask at every prefix length t ∈ [0, 29] keeps the true
    next-piece's vocab index among its allowed choices.
    """
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg()).eval()
    LEGAL_THRESHOLD = -1e30

    import random
    from junqi_core.setup import generate_random_lineup

    rng = random.Random(42)
    n_trials = 50
    for trial in range(n_trials):
        for seat_value in range(N_SEATS):
            lineup = generate_random_lineup(rng)
            lineup_idxs = [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup]
            # Feed prefixes of length 0, 1, ..., 29; verify the true next
            # action at position t is still legal.
            for t in range(ARRANGEMENT_SIZE):
                if t == 0:
                    prefix = torch.zeros(1, 0, N_PIECE_TYPE_WITH_NONE)
                else:
                    prefix = lineup_to_onehot(
                        torch.tensor([lineup_idxs[:t]], dtype=torch.long)
                    )
                seat = torch.tensor([seat_value])
                with torch.no_grad():
                    out = net(prefix, seat)
                logits_at_t = out["logits"][0, -1]  # last-row = prediction for slot t
                legal = logits_at_t > LEGAL_THRESHOLD
                true_idx = lineup_idxs[t]
                assert bool(legal[true_idx]), (
                    f"trial {trial} seat {seat_value} slot {t}: "
                    f"true piece idx {true_idx} was masked out "
                    f"(legal={legal.tolist()})"
                )


# ---------------------------------------------------------------------------
# Remaining-count invariants through a full roll
# ---------------------------------------------------------------------------


def test_remaining_counts_exhausted_after_full_lineup():
    """After the full 30-placement sequence, remaining = 0 for every type."""
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg()).eval()
    import random
    from junqi_core.setup import generate_random_lineup

    lineup = generate_random_lineup(random.Random(7))
    lineup_idxs = [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lineup]
    seq = lineup_to_onehot(torch.tensor([lineup_idxs], dtype=torch.long))
    cum = seq.cumsum(dim=1)
    after_full = cum[0, -1]  # (V,)
    pc = _build_piece_counts().float()
    assert torch.allclose(after_full, pc), (
        f"cumulative placements {after_full.tolist()} != piece_counts {pc.tolist()}"
    )


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward_runs():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg()).cuda()
    B = 4
    seq = torch.zeros(B, 0, N_PIECE_TYPE_WITH_NONE, device="cuda")
    seat = torch.randint(0, N_SEATS, (B,), device="cuda")
    out = net(seq, seat)
    assert out["logits"].device.type == "cuda"
    assert torch.isfinite(out["logits"]).any()
