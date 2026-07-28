"""Tests for junqi_rl.arrangement.sampling."""

from __future__ import annotations

import pytest
import torch

from junqi_core.setup import validate_lineup
from junqi_rl.arrangement.sampling import (
    GenerationResult,
    generate_arrangements,
    lineup_to_sample,
    samples_to_lineups,
)
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    N_SEATS,
    ArrangementNet,
    ArrangementNetConfig,
)


def _tiny_cfg() -> ArrangementNetConfig:
    return ArrangementNetConfig(depth=2, n_head=4, embed_dim=64, ff_factor=2)


# ---------------------------------------------------------------------------
# Return shapes / types
# ---------------------------------------------------------------------------


def test_return_shapes():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    res = generate_arrangements(n_sample=8, model=net, rng_seed=0)
    assert isinstance(res, GenerationResult)
    assert res.samples.shape == (8, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE)
    assert res.values.shape == (8, ARRANGEMENT_SIZE, 3)  # N_VF_CAT=3 default
    assert res.ent_pred.shape == (8, ARRANGEMENT_SIZE)
    assert res.log_probs.shape == (8, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE)
    assert res.seat_idx.shape == (8,)
    assert "retry_rate" in res.stats
    assert "fallback_rate" in res.stats


def test_samples_are_onehot():
    torch.manual_seed(1)
    net = ArrangementNet(_tiny_cfg())
    res = generate_arrangements(n_sample=5, model=net, rng_seed=1)
    # Every row should sum to exactly 1 and have exactly one non-zero.
    assert torch.allclose(res.samples.sum(dim=-1), torch.ones(5, ARRANGEMENT_SIZE))
    assert ((res.samples == 0.0) | (res.samples == 1.0)).all()


# ---------------------------------------------------------------------------
# Validity — every generated lineup must pass junqi_core.setup.validate_lineup.
# ---------------------------------------------------------------------------


def test_all_generated_lineups_are_valid():
    torch.manual_seed(2)
    net = ArrangementNet(_tiny_cfg())
    # Random net weights → many dead-ends. We rely on retry + fallback to
    # always produce valid outputs.
    res = generate_arrangements(n_sample=64, model=net, rng_seed=2)
    lineups = samples_to_lineups(res.samples)
    assert len(lineups) == 64
    for i, lu in enumerate(lineups):
        r = validate_lineup(lu)
        assert r.ok, f"sample {i}: {r.violations}"


def test_fallback_rate_bounded():
    """Even with a random-weight net, fallbacks should be rare (<50%).

    This is a smoke test — not a tight bound, just a sanity check that our
    retry loop actually succeeds most of the time.
    """
    torch.manual_seed(3)
    net = ArrangementNet(_tiny_cfg())
    res = generate_arrangements(n_sample=32, model=net, rng_seed=3)
    # Even a completely random policy should be able to produce a valid
    # lineup within `max_resample` attempts most of the time.
    assert res.stats["fallback_rate"] <= 0.5


# ---------------------------------------------------------------------------
# Seat handling
# ---------------------------------------------------------------------------


def test_seats_default_cyclic():
    torch.manual_seed(4)
    net = ArrangementNet(_tiny_cfg())
    res = generate_arrangements(n_sample=8, model=net, rng_seed=4)
    expected = torch.arange(8) % N_SEATS
    assert torch.equal(res.seat_idx.cpu(), expected)


def test_seats_int_broadcast():
    torch.manual_seed(5)
    net = ArrangementNet(_tiny_cfg())
    res = generate_arrangements(n_sample=6, model=net, seats=2, rng_seed=5)
    assert (res.seat_idx == 2).all()


def test_seats_tensor():
    torch.manual_seed(6)
    net = ArrangementNet(_tiny_cfg())
    seats = torch.tensor([0, 3, 1, 2, 0, 2])
    res = generate_arrangements(n_sample=6, model=net, seats=seats, rng_seed=6)
    assert torch.equal(res.seat_idx.cpu(), seats)


def test_invalid_seat_rejected():
    torch.manual_seed(7)
    net = ArrangementNet(_tiny_cfg())
    with pytest.raises(ValueError):
        generate_arrangements(n_sample=4, model=net, seats=torch.tensor([0, 4, 1, 2]))


# ---------------------------------------------------------------------------
# Round-trip: lineup_to_sample → samples_to_lineups should preserve order.
# ---------------------------------------------------------------------------


def test_lineup_roundtrip():
    import random
    from junqi_core.setup import generate_random_lineup

    rng = random.Random(42)
    lineup = generate_random_lineup(rng)
    sample = lineup_to_sample(lineup).unsqueeze(0)  # (1, 30, 13)
    back = samples_to_lineups(sample)[0]
    assert back == list(lineup)


# ---------------------------------------------------------------------------
# Log-probs consistency
# ---------------------------------------------------------------------------


def test_log_probs_sum_to_one():
    """log_probs[t] should be a valid log-distribution (exp-sum ≈ 1)."""
    torch.manual_seed(8)
    net = ArrangementNet(_tiny_cfg())
    res = generate_arrangements(n_sample=4, model=net, rng_seed=8)
    # For any row that was NOT filled by the fallback path, exp(log_probs).sum()
    # should be ~1.0. Fallback rows have log_probs=0 everywhere (a degenerate
    # uniform-ish placeholder) — so we skip them.
    fallback = res.stats["fallback_rate"] > 0.0
    if fallback:
        pytest.skip("random weights triggered fallback; log-prob row is placeholder")
    probs_sum = res.log_probs.exp().sum(dim=-1)  # (N, 30)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_reject_zero_samples():
    torch.manual_seed(9)
    net = ArrangementNet(_tiny_cfg())
    with pytest.raises(ValueError):
        generate_arrangements(n_sample=0, model=net)


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_generation():
    torch.manual_seed(10)
    net = ArrangementNet(_tiny_cfg()).cuda()
    res = generate_arrangements(n_sample=16, model=net, rng_seed=10)
    assert res.samples.device.type == "cuda"
    lineups = samples_to_lineups(res.samples)
    for lu in lineups:
        assert validate_lineup(lu).ok
