"""Test the NaN/Inf guard inside JunqiNet.forward.

v29 crashed at R51 with `probability tensor contains either inf, nan or
element < 0` inside `Categorical.sample()`, which runs mid-forward. The
PPO trainer's outer NaN guard can't save us because the crash happens
before forward returns.

These tests verify:
  1. Non-finite logits in a row are replaced with a uniform-over-legal
     fallback that Categorical.sample() accepts.
  2. The counter self._nan_fwd_count increments correctly.
  3. Good rows are not touched.
  4. Downstream log_probs are finite after the guard fires.
"""
from __future__ import annotations

import pytest
import torch

from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_core.observation import OBS_CHANNELS


def _tiny_net():
    """Minimal JunqiNet for fast CPU tests."""
    return JunqiNet(JunqiNetConfig(
        depth=1, n_head=2, embed_dim=32, ff_factor=2,
        cnn_channels=32, cnn_layers=1,
    ))


def test_nan_guard_counter_starts_zero():
    net = _tiny_net()
    assert net._nan_fwd_count == 0


def test_forward_on_clean_inputs_does_not_increment_counter():
    """Baseline: clean inputs leave the counter at 0."""
    torch.manual_seed(0)
    net = _tiny_net()
    net.eval()
    B = 4
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17)
    obs_gl = torch.randn(B, 28)
    legal = torch.ones(B, 16641, dtype=torch.bool)
    legal[:, 0] = False
    with torch.no_grad():
        out = net(obs_sp, obs_gl, legal)
    assert net._nan_fwd_count == 0
    # log_probs can be -inf for illegal actions (that's log_softmax
    # masking); what must NOT happen is NaN.
    assert not torch.isnan(out["log_probs"]).any()


def _legal_log_probs_finite(lp_row, legal_row):
    """log_probs must be finite at LEGAL positions; illegals can be -inf."""
    return torch.isfinite(lp_row[legal_row]).all() and not torch.isnan(lp_row).any()


def test_forward_guards_nan_logits_by_monkeypatching_policy_head(monkeypatch):
    """Force the policy-logit output to contain NaN/Inf in a specific row
    and verify the guard catches it."""
    torch.manual_seed(1)
    net = _tiny_net()
    net.eval()
    B = 4
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17)
    obs_gl = torch.randn(B, 28)
    legal = torch.ones(B, 16641, dtype=torch.bool)
    legal[:, 0] = False

    # Wrap _policy_logits so row 1 gets a NaN and row 2 gets an Inf.
    orig = net._policy_logits
    def _poisoned(cells, legal_mask):
        out = orig(cells, legal_mask)
        # Column 1 is legal for all rows; use it as the poisoning target.
        out[1, 1] = float("nan")
        out[2, 1] = float("inf")
        return out
    monkeypatch.setattr(net, "_policy_logits", _poisoned)

    with torch.no_grad():
        out = net(obs_sp, obs_gl, legal)

    # Two rows poisoned -> counter should be 2.
    assert net._nan_fwd_count == 2, f"expected 2, got {net._nan_fwd_count}"
    # Sampling did not crash. No row should contain NaN after the guard.
    assert not torch.isnan(out["log_probs"]).any()
    # Every row's legal positions must have finite log_probs.
    for b in range(B):
        assert _legal_log_probs_finite(out["log_probs"][b], legal[b]), (
            f"row {b} has non-finite legal log_probs"
        )


def test_forward_uniform_fallback_for_poisoned_row(monkeypatch):
    """When a row is replaced with the uniform-over-legal fallback, its
    log_probs should be uniform across legal actions and -inf for illegals."""
    torch.manual_seed(2)
    net = _tiny_net()
    net.eval()
    B = 2
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17)
    obs_gl = torch.randn(B, 28)
    # Custom legal mask: only actions 0, 1, 2, 3, 4 are legal for row 0
    # (so we can easily see uniform = log(1/5) ≈ -1.609).
    legal = torch.zeros(B, 16641, dtype=torch.bool)
    legal[0, :5] = True
    legal[1, :100] = True   # control row, not poisoned

    orig = net._policy_logits
    def _poisoned(cells, legal_mask):
        out = orig(cells, legal_mask)
        # Poison row 0 only.
        out[0, 0] = float("nan")
        return out
    monkeypatch.setattr(net, "_policy_logits", _poisoned)

    with torch.no_grad():
        out = net(obs_sp, obs_gl, legal)

    assert net._nan_fwd_count == 1
    lp0 = out["log_probs"][0]
    # Legal positions should be uniform log(1/5) = -log(5)
    expected = -torch.tensor(5.0).log()
    for i in range(5):
        assert torch.isclose(lp0[i], expected, atol=1e-5), (
            f"action {i}: got {lp0[i].item()}, expected {expected.item()}"
        )
    # Illegal positions should be -inf
    assert torch.isinf(lp0[5:]).all() and (lp0[5:] < 0).all()


def test_all_rows_poisoned_still_does_not_crash(monkeypatch):
    """Worst case: every row has NaN. Must not crash Categorical.sample()."""
    torch.manual_seed(3)
    net = _tiny_net()
    net.eval()
    B = 3
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17)
    obs_gl = torch.randn(B, 28)
    legal = torch.ones(B, 16641, dtype=torch.bool)

    orig = net._policy_logits
    def _poisoned(cells, legal_mask):
        out = orig(cells, legal_mask)
        out[:, 0] = float("nan")
        return out
    monkeypatch.setattr(net, "_policy_logits", _poisoned)

    with torch.no_grad():
        out = net(obs_sp, obs_gl, legal)

    assert net._nan_fwd_count == 3
    assert out["action"].shape == (B,)
    # No NaN allowed anywhere after the guard.
    assert not torch.isnan(out["log_probs"]).any()
    # Every legal position must be finite.
    for b in range(B):
        assert torch.isfinite(out["log_probs"][b][legal[b]]).all()
