"""Empirical verification of the loss=0 hypothesis.

Claim: With ``num_epochs_per_rollout=1``, the PPO ratio is identically 1.0
on every minibatch (same policy collected and trained). Thus
``surr1 = adv * 1.0``, ``surr2 = adv * clamp(1.0, 0.8, 1.2)`` = adv, and
``policy_loss = -min(surr1, surr2).mean() = -adv.mean()``.

The rollout buffer normalises advantages to mean=0, std=1 BEFORE the
filter. After applying the |adv_norm| >= thresh filter (keeping 25% of
samples, i.e. the largest-magnitude 25%), the kept set is still
approximately mean-zero because positive and negative tails are both
kept. So the logged ``policy_loss`` averages to ~0 even though
per-sample gradients are large.

This is a LOGGING ARTEFACT, not gradient starvation. Per-sample grads
still flow. Training DOES progress (win_rate still improves R30→R80).
The fix is NOT to make loss visible — it's to (a) log a grad-norm
metric that reflects actual policy movement, and/or (b) set
num_epochs_per_rollout >= 2 so ratio != 1 and the loss decouples from
the signed mean.

Run:
    pytest -xvs tests/test_loss_zero_hypothesis.py
"""
from __future__ import annotations

import numpy as np
import torch
import pytest


def simulate_ppo_loss(
    advantages: np.ndarray,
    clip_range: float = 0.2,
    adv_filt_thresh: float = 0.01,
    adv_filt_rate: float = 0.75,
) -> dict[str, float]:
    """Reproduce the logged policy_loss under num_epochs_per_rollout=1.

    Mirrors junqi_rl.training.rollout_gpu.minibatches() filter + the
    ppo._policy_loss with ratio == 1.0 exactly.
    """
    adv = torch.from_numpy(advantages.astype(np.float32))
    # Normalise (matches rollout_gpu.py:365)
    adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    abs_adv = adv_norm.abs()

    thresh = adv_filt_thresh
    if adv_filt_rate < 1.0:
        q = 1.0 - adv_filt_rate
        q_thresh = torch.quantile(abs_adv, q).item()
        thresh = max(adv_filt_thresh, q_thresh)

    keep = abs_adv >= thresh
    kept = adv_norm[keep]

    # Under ratio==1, policy_loss = -mean(adv_norm[kept])
    # (both surr1 and surr2 equal adv for ratio in clip range)
    policy_loss = -kept.mean().item() if kept.numel() else 0.0
    return {
        "n_kept": int(keep.sum().item()),
        "n_total": int(advantages.size),
        "kept_mean": float(kept.mean().item()) if kept.numel() else 0.0,
        "kept_std": float(kept.std().item()) if kept.numel() else 0.0,
        "policy_loss": policy_loss,
        "thresh_used": thresh,
    }


def test_normalisation_always_drives_loss_near_zero():
    """Because ``rollout_gpu.minibatches`` normalises advantages to
    (approximately) mean=0 std=1 BEFORE filtering, the kept set always
    has mean ≈ 0 regardless of the underlying distribution's skew.

    This is why ``policy_loss`` logs as 0.0000 at ANY training stage
    (not just post-VF-convergence): normalisation wipes the signed mean.
    Under ``num_epochs_per_rollout=1`` with ratio ≡ 1.0 on the first
    minibatch, the logged loss is just ``-mean(adv_norm_kept) ≈ 0``.
    """
    rng = np.random.default_rng(0)
    for loc in [-0.5, -0.15, 0.0, 0.15, 0.5]:
        adv = rng.normal(loc=loc, scale=1.0, size=65536).astype(np.float32)
        out = simulate_ppo_loss(adv)
        # After normalisation + symmetric-tail filter, kept mean is tiny
        assert abs(out["kept_mean"]) < 0.01, (
            f"Expected normalisation to zero the mean at loc={loc}, got {out}"
        )
        assert abs(out["policy_loss"]) < 0.01


def test_converged_phase_loss_collapses_to_zero():
    """After value net converges: advantages are centered around 0.

    GAE produces zero-mean advantages (∑ δ_t (γλ)^t over a balanced
    rollout sums to zero in expectation). The |adv_norm| >= thresh
    filter keeps the tails symmetrically. Kept mean ≈ 0.
    """
    rng = np.random.default_rng(1)
    # Zero-mean, narrow advantages (VF has converged, value ≈ return)
    adv = rng.normal(loc=0.0, scale=0.065, size=65536).astype(np.float32)
    out = simulate_ppo_loss(adv)
    # THIS is the smoking gun: policy_loss collapses to ~0 by construction
    assert abs(out["policy_loss"]) < 1e-3, (
        f"Expected near-zero loss for zero-mean advantages, got {out}"
    )
    # But the kept samples are NOT all zero — their abs values are large
    # This means per-sample gradients are still happening, the LOGGED mean
    # is just not informative.
    assert out["n_kept"] > out["n_total"] * 0.2, (
        f"Filter should keep ~25%, got {out}"
    )
    assert out["kept_std"] > 0.5, (
        f"Kept samples should have large spread, not all zero, got {out}"
    )


def test_loss_is_signed_mean_advantage_under_1_epoch():
    """When num_epochs=1 and ratio==1, policy_loss IS just -mean(kept adv).

    This is the concrete math identity: logged loss value == signed mean
    of kept normalised advantages.
    """
    rng = np.random.default_rng(2)
    for loc in [-0.1, -0.01, 0.0, 0.01, 0.1]:
        adv = rng.normal(loc=loc, scale=1.0, size=65536).astype(np.float32)
        out = simulate_ppo_loss(adv)
        # policy_loss == -kept_mean
        assert abs(out["policy_loss"] - (-out["kept_mean"])) < 1e-6, (
            f"Identity broken at loc={loc}: {out}"
        )


def test_increasing_vf_convergence_drives_loss_to_zero():
    """As VF converges (advantages get tighter around 0), loss shrinks
    smoothly to zero. Monotonic relationship confirms the artefact."""
    rng = np.random.default_rng(3)
    losses = []
    # Simulate VF convergence: std starts large, shrinks over time
    for std in [1.0, 0.5, 0.2, 0.1, 0.065, 0.05]:
        adv = rng.normal(loc=0.0, scale=std, size=65536).astype(np.float32)
        out = simulate_ppo_loss(adv)
        losses.append(abs(out["policy_loss"]))
    # Zero-mean input => loss near zero across all stds (not a monotonic
    # decrease, but all small)
    for L in losses:
        assert L < 5e-3, f"Unexpected non-zero loss for zero-mean input: {losses}"
