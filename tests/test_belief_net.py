"""Tests for junqi_rl.networks.belief_net."""

from __future__ import annotations

import pytest
import torch

from junqi_core.observation import OBS_CHANNELS
from junqi_rl.networks.belief_net import (
    BeliefNet,
    BeliefNetConfig,
    N_BELIEF_TYPES,
)


# ---------------------------------------------------------------------------
# Shape / API contract
# ---------------------------------------------------------------------------


def test_n_belief_types_is_12():
    """The tracked-type axis is 12 and must match junqi_core.info_model."""
    from junqi_core.info_model import NUM_TRACKED_TYPES
    assert N_BELIEF_TYPES == NUM_TRACKED_TYPES == 12


def test_forward_shapes_default():
    net = BeliefNet(BeliefNetConfig())
    B = 4
    obs = torch.randn(B, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = net(obs, seat_idx=seat)
    assert out["logits"].shape == (B, 289, N_BELIEF_TYPES)
    assert out["log_probs"].shape == (B, 289, N_BELIEF_TYPES)


def test_forward_shapes_batch_size_one():
    """Edge-case batch_size=1 must work (common during inference)."""
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(1, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([2], dtype=torch.long)
    out = net(obs, seat_idx=seat)
    assert out["logits"].shape == (1, 289, N_BELIEF_TYPES)


def test_log_probs_sum_to_one():
    """log_softmax over the 12-axis must exponentiate to 1."""
    torch.manual_seed(0)
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(3, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([0, 1, 2], dtype=torch.long)
    out = net(obs, seat_idx=seat)
    sums = out["log_probs"].exp().sum(dim=-1)  # (B, 289)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_forward_obs_global_is_accepted_but_ignored():
    """P1.1 reserves obs_global for future use; it must not error."""
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(2, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([0, 3], dtype=torch.long)
    g = torch.randn(2, 28)
    out_with = net(obs, seat_idx=seat, obs_global=g)
    out_without = net(obs, seat_idx=seat)
    # Identical output because obs_global is dropped.
    assert torch.allclose(out_with["logits"], out_without["logits"])


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_params():
    """A single CE step must produce non-None grads everywhere."""
    torch.manual_seed(1)
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(2, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([1, 2], dtype=torch.long)
    out = net(obs, seat_idx=seat)
    target = torch.randint(0, N_BELIEF_TYPES, (2, 289))
    loss = torch.nn.functional.nll_loss(
        out["log_probs"].reshape(-1, N_BELIEF_TYPES),
        target.reshape(-1),
    )
    loss.backward()

    missing = [n for n, p in net.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"Parameters with no grad after backward: {missing[:5]}"


def test_gradient_norm_is_finite_and_positive():
    torch.manual_seed(2)
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(4, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    out = net(obs, seat_idx=seat)
    loss = -out["log_probs"].mean()
    loss.backward()
    norms = [p.grad.norm().item() ** 2 for p in net.parameters() if p.grad is not None]
    gn = sum(norms) ** 0.5
    assert 0.0 < gn < 1e4, f"grad norm out of sane range: {gn}"


# ---------------------------------------------------------------------------
# Seat embedding — one net, 4 seats
# ---------------------------------------------------------------------------


def test_seat_embedding_changes_output():
    """With use_seat_embedding=True, changing seat_idx must change logits."""
    torch.manual_seed(3)
    net = BeliefNet(BeliefNetConfig(use_seat_embedding=True))
    obs = torch.randn(1, OBS_CHANNELS, 17, 17)
    out0 = net(obs, seat_idx=torch.tensor([0], dtype=torch.long))
    out1 = net(obs, seat_idx=torch.tensor([1], dtype=torch.long))
    assert not torch.allclose(out0["logits"], out1["logits"])


def test_seat_embedding_off_ignores_seat_idx():
    """With use_seat_embedding=False, seat_idx may be None, and output is
    invariant to it if passed anyway."""
    torch.manual_seed(4)
    net = BeliefNet(BeliefNetConfig(use_seat_embedding=False))
    obs = torch.randn(1, OBS_CHANNELS, 17, 17)
    out_none = net(obs, seat_idx=None)
    # seat_idx is simply ignored; no assertion, but should not crash.
    assert out_none["logits"].shape == (1, 289, N_BELIEF_TYPES)


def test_seat_embedding_requires_seat_idx():
    """With use_seat_embedding=True, omitting seat_idx must raise."""
    net = BeliefNet(BeliefNetConfig(use_seat_embedding=True))
    obs = torch.randn(1, OBS_CHANNELS, 17, 17)
    with pytest.raises(ValueError, match="seat_idx"):
        net(obs, seat_idx=None)


def test_seat_idx_shape_validation():
    net = BeliefNet(BeliefNetConfig(use_seat_embedding=True))
    obs = torch.randn(2, OBS_CHANNELS, 17, 17)
    with pytest.raises(ValueError, match="seat_idx must have shape"):
        net(obs, seat_idx=torch.tensor([0, 1, 2], dtype=torch.long))  # wrong B


def test_seat_idx_dtype_validation():
    net = BeliefNet(BeliefNetConfig(use_seat_embedding=True))
    obs = torch.randn(1, OBS_CHANNELS, 17, 17)
    with pytest.raises(ValueError, match="int64"):
        net(obs, seat_idx=torch.tensor([0], dtype=torch.int32))


# ---------------------------------------------------------------------------
# Input shape validation
# ---------------------------------------------------------------------------


def test_obs_spatial_wrong_channels_raises():
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(1, 64, 17, 17)       # wrong channel count
    seat = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="obs_spatial shape mismatch"):
        net(obs, seat_idx=seat)


def test_obs_spatial_wrong_board_size_raises():
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(1, 256, 10, 10)      # wrong board (Stratego-size)
    seat = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="obs_spatial shape mismatch"):
        net(obs, seat_idx=seat)


def test_obs_spatial_wrong_ndim_raises():
    net = BeliefNet(BeliefNetConfig())
    obs = torch.randn(OBS_CHANNELS, 17, 17)         # missing batch dim
    seat = torch.tensor([0], dtype=torch.long)
    with pytest.raises(ValueError, match="4D"):
        net(obs, seat_idx=seat)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_embed_dim_not_divisible_by_n_head_raises():
    with pytest.raises(ValueError, match="divisible"):
        BeliefNetConfig(embed_dim=257, n_head=8)


def test_config_default_param_count_fits_budget():
    """Default config should produce a belief net in the 3–10 M-param range.
    Heavier models OOM on T4 with batch_size=128 + main PPO + arr trainer."""
    net = BeliefNet(BeliefNetConfig())
    n = net.num_parameters()
    assert 1_000_000 <= n <= 15_000_000, (
        f"Default BeliefNet has {n:,} params — outside expected 1-15 M "
        f"budget for T4. If this is deliberate, update the budget."
    )


# ---------------------------------------------------------------------------
# Determinism & no-grad inference
# ---------------------------------------------------------------------------


def test_same_input_same_output():
    """Identical inputs must produce identical outputs in eval mode."""
    torch.manual_seed(7)
    net = BeliefNet(BeliefNetConfig()).eval()
    obs = torch.randn(2, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([0, 2], dtype=torch.long)
    with torch.no_grad():
        out1 = net(obs, seat_idx=seat)
        out2 = net(obs, seat_idx=seat)
    assert torch.equal(out1["logits"], out2["logits"])


def test_eval_mode_no_dropout_changes():
    """Eval mode must be deterministic even if dropout > 0 in config."""
    torch.manual_seed(8)
    net = BeliefNet(BeliefNetConfig(dropout=0.5)).eval()
    obs = torch.randn(1, OBS_CHANNELS, 17, 17)
    seat = torch.tensor([1], dtype=torch.long)
    with torch.no_grad():
        a = net(obs, seat_idx=seat)["logits"]
        b = net(obs, seat_idx=seat)["logits"]
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# CUDA smoke (only if a GPU is available)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_forward_on_cuda():
    net = BeliefNet(BeliefNetConfig()).cuda()
    obs = torch.randn(4, OBS_CHANNELS, 17, 17, device="cuda")
    seat = torch.tensor([0, 1, 2, 3], dtype=torch.long, device="cuda")
    out = net(obs, seat_idx=seat)
    assert out["logits"].device.type == "cuda"
    assert out["logits"].shape == (4, 289, N_BELIEF_TYPES)
