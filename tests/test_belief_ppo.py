"""Tests for junqi_rl.training.belief_ppo."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from junqi_core.observation import OBS_CHANNELS
from junqi_rl.belief.buffer import BeliefBuffer
from junqi_rl.networks.belief_net import (
    BeliefNet,
    BeliefNetConfig,
    N_BELIEF_TYPES,
)
from junqi_rl.training.belief_ppo import (
    BeliefPPOConfig,
    BeliefPPOTrainer,
    compute_belief_loss,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _fill_buffer(
    buf: BeliefBuffer,
    *,
    n: int = 128,
    seed: int = 0,
    reveal_ratio: float = 0.3,
) -> None:
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, OBS_CHANNELS, 17, 17)).astype(np.float32)
    seat = rng.integers(0, 4, size=(n,), dtype=np.int64)
    label = np.full((n, 289), -1, dtype=np.int64)
    n_reveal = int(reveal_ratio * 289)
    for i in range(n):
        cells = rng.choice(289, size=n_reveal, replace=False)
        types = rng.integers(0, N_BELIEF_TYPES, size=n_reveal)
        label[i, cells] = types
    enemy = (label >= 0)
    # Add a few extra enemy cells that aren't revealed.
    for i in range(n):
        unrevealed = np.where(label[i] < 0)[0]
        if len(unrevealed) >= 10:
            cells = rng.choice(unrevealed, size=10, replace=False)
            enemy[i, cells] = True
    buf.add(obs, seat, label, enemy)


# ---------------------------------------------------------------------------
# compute_belief_loss
# ---------------------------------------------------------------------------


def test_compute_loss_shapes():
    B = 4
    logits = torch.randn(B, 289, N_BELIEF_TYPES)
    label = torch.full((B, 289), -1, dtype=torch.long)
    enemy = torch.zeros(B, 289, dtype=torch.bool)
    out = compute_belief_loss(logits, label, enemy)
    assert "ce_loss" in out
    assert "uniform_ce" in out
    assert "uniform_kl" in out
    assert "accuracy" in out
    assert out["ce_loss"].ndim == 0


def test_compute_loss_no_revealed_returns_zero():
    """If no cell has a label, ce_loss should be 0 (masked sum / max(1,0))."""
    B = 2
    logits = torch.randn(B, 289, N_BELIEF_TYPES, requires_grad=True)
    label = torch.full((B, 289), -1, dtype=torch.long)
    enemy = torch.zeros(B, 289, dtype=torch.bool)
    out = compute_belief_loss(logits, label, enemy)
    assert out["ce_loss"].item() == 0.0
    assert out["n_revealed"].item() == 1.0  # clamp(min=1)


def test_compute_loss_uniform_kl_near_zero_for_random_net():
    """A random (untrained) net should give CE ≈ log(12) ≈ 2.48,
    matching uniform baseline → uniform_kl near 0."""
    torch.manual_seed(0)
    B = 8
    logits = torch.randn(B, 289, N_BELIEF_TYPES) * 0.01   # nearly uniform
    label = torch.randint(0, N_BELIEF_TYPES, (B, 289))
    enemy = torch.ones(B, 289, dtype=torch.bool)
    out = compute_belief_loss(logits, label, enemy)
    assert abs(out["uniform_kl"].item()) < 0.05, (
        f"near-uniform logits should give uniform_kl ≈ 0; got {out['uniform_kl'].item()}"
    )


def test_compute_loss_gradient_flows():
    B = 4
    logits = torch.randn(B, 289, N_BELIEF_TYPES, requires_grad=True)
    label = torch.randint(0, N_BELIEF_TYPES, (B, 289))
    enemy = torch.ones(B, 289, dtype=torch.bool)
    out = compute_belief_loss(logits, label, enemy)
    out["ce_loss"].backward()
    assert logits.grad is not None
    # Grad should be non-zero.
    assert logits.grad.abs().sum().item() > 0


def test_compute_loss_mask_zeroes_unrevealed_cells():
    """Unrevealed cells must not contribute to the gradient."""
    B = 1
    logits = torch.randn(B, 289, N_BELIEF_TYPES, requires_grad=True)
    label = torch.full((B, 289), -1, dtype=torch.long)
    label[0, 0] = 3   # only one revealed cell
    enemy = torch.zeros(B, 289, dtype=torch.bool)
    enemy[0, 0] = True
    out = compute_belief_loss(logits, label, enemy)
    out["ce_loss"].backward()
    # Only cell 0 should have non-zero grad.
    grad = logits.grad[0]
    # Cell 0 has grad (non-zero); other cells should be zero.
    assert grad[0].abs().sum().item() > 0
    assert grad[1:].abs().sum().item() == 0


def test_compute_loss_accuracy_perfect():
    """If logits perfectly predict the label, accuracy == 1."""
    B = 2
    logits = torch.full((B, 289, N_BELIEF_TYPES), -100.0)
    label = torch.full((B, 289), -1, dtype=torch.long)
    enemy = torch.zeros(B, 289, dtype=torch.bool)
    # Set 5 cells to known types and make logits overwhelmingly prefer those.
    for i in range(B):
        for c in range(5):
            t = (c + i) % N_BELIEF_TYPES
            label[i, c] = t
            enemy[i, c] = True
            logits[i, c, t] = 100.0
    out = compute_belief_loss(logits, label, enemy)
    assert out["accuracy"].item() == 1.0
    assert out["n_revealed"].item() == 10.0


def test_compute_loss_requires_label_in_range():
    """Labels must be in [-1, N_BELIEF_TYPES). Out-of-range should raise
    (indirect: gather with clamp on safe_label, but we don't validate).
    Ensure values of exactly -1 and N_BELIEF_TYPES-1 both work."""
    B = 1
    logits = torch.randn(B, 289, N_BELIEF_TYPES)
    label = torch.full((B, 289), -1, dtype=torch.long)
    label[0, 0] = N_BELIEF_TYPES - 1
    enemy = torch.zeros(B, 289, dtype=torch.bool)
    enemy[0, 0] = True
    out = compute_belief_loss(logits, label, enemy)
    assert torch.isfinite(out["ce_loss"])


def test_compute_loss_rejects_bad_logits_shape():
    with pytest.raises(ValueError, match="logits must be"):
        compute_belief_loss(
            torch.zeros(4, 289, 7),   # wrong type count
            torch.zeros(4, 289, dtype=torch.long),
            torch.ones(4, 289, dtype=torch.bool),
        )


# ---------------------------------------------------------------------------
# BeliefPPOTrainer
# ---------------------------------------------------------------------------


def test_trainer_construction_defaults():
    net = BeliefNet(BeliefNetConfig())
    trainer = BeliefPPOTrainer(net)
    assert trainer.cfg.lr == 5e-5
    assert trainer.cfg.max_grad_norm == 0.5
    assert trainer.num_train_step == 0


def test_trainer_construction_rejects_bad_autocast():
    with pytest.raises(ValueError, match="autocast_dtype"):
        BeliefPPOTrainer(
            BeliefNet(BeliefNetConfig()),
            BeliefPPOConfig(autocast_dtype="bogus"),
        )


def test_trainer_train_epoch_empty_buffer_returns_nan():
    """An empty buffer must not crash; returns nan metrics."""
    net = BeliefNet(BeliefNetConfig())
    trainer = BeliefPPOTrainer(net)
    buf = BeliefBuffer(capacity=100)
    metrics = trainer.train_epoch(buf)
    assert math.isnan(metrics["belief_train/ce_loss"])
    assert metrics["belief_train/num_updates"] == 0.0
    assert metrics["belief_train/buffer_size"] == 0.0


def test_trainer_loss_decreases_with_gradient_steps():
    """Classic 'train on static batch, loss goes down' sanity."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Use a smaller net config for speed.
    cfg_net = BeliefNetConfig(
        n_encoder_layer=2, embed_dim=128, cnn_channels=32, cnn_layers=2,
    )
    net = BeliefNet(cfg_net)
    cfg_tr = BeliefPPOConfig(
        lr=1e-3,                   # higher than default for quick test
        batch_size=16,
        epochs_per_rollout=1,
        autocast_dtype="float32",
    )
    trainer = BeliefPPOTrainer(net, cfg_tr)

    buf = BeliefBuffer(capacity=64, seed=42)
    _fill_buffer(buf, n=64, seed=42)

    initial_loss = None
    final_loss = None
    for step in range(15):
        metrics = trainer.train_epoch(buf)
        ce = metrics["belief_train/ce_loss"]
        if step == 0:
            initial_loss = ce
        final_loss = ce

    # Allow for some noise but demand clear improvement.
    assert initial_loss > final_loss, (
        f"loss should decrease; got {initial_loss:.4f} → {final_loss:.4f}"
    )


def test_trainer_ema_tracks_but_lags():
    """After a training step, EMA params should be close to but not equal
    to the trainable params (EMA lags with decay 0.999)."""
    torch.manual_seed(0)
    net = BeliefNet(BeliefNetConfig(embed_dim=128, cnn_channels=32, cnn_layers=2))
    trainer = BeliefPPOTrainer(net, BeliefPPOConfig(lr=1e-2, autocast_dtype="float32"))

    # Grab initial weights.
    init_ema_weight = trainer.ema.model.head.weight.clone()

    # Train a bit.
    buf = BeliefBuffer(capacity=64, seed=0)
    _fill_buffer(buf, n=64, seed=0)
    for _ in range(3):
        trainer.train_epoch(buf)

    # EMA should have moved.
    assert not torch.allclose(trainer.ema.model.head.weight, init_ema_weight)
    # But NOT all the way to the trainable net (decay 0.999).
    assert not torch.allclose(
        trainer.ema.model.head.weight, trainer.net.head.weight
    )


def test_trainer_state_dict_roundtrip():
    torch.manual_seed(0)
    net = BeliefNet(BeliefNetConfig(embed_dim=128, cnn_channels=32, cnn_layers=2))
    trainer1 = BeliefPPOTrainer(net, BeliefPPOConfig(autocast_dtype="float32"))
    buf = BeliefBuffer(capacity=64, seed=0)
    _fill_buffer(buf, n=64, seed=0)
    trainer1.train_epoch(buf)
    sd = trainer1.state_dict()

    net2 = BeliefNet(BeliefNetConfig(embed_dim=128, cnn_channels=32, cnn_layers=2))
    trainer2 = BeliefPPOTrainer(net2, BeliefPPOConfig(autocast_dtype="float32"))
    trainer2.load_state_dict(sd)
    assert trainer2.num_train_step == trainer1.num_train_step
    # Net weights match.
    for p1, p2 in zip(trainer1.net.parameters(), trainer2.net.parameters()):
        assert torch.allclose(p1, p2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_trainer_on_cuda():
    # This test needs ~1 GB of free GPU memory. If another job is
    # hogging the card (e.g. a concurrent training run) we skip rather
    # than fail — the CUDA code path is the same regardless of which
    # process is holding the allocator.
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 2 * 1024**3:   # need 2 GB headroom
        pytest.skip(
            f"insufficient free CUDA memory for test: "
            f"{free_bytes / 1024**3:.1f} GB free of {total_bytes / 1024**3:.1f} GB"
        )
    net = BeliefNet(BeliefNetConfig(embed_dim=128, cnn_channels=32, cnn_layers=2))
    trainer = BeliefPPOTrainer(
        net, BeliefPPOConfig(autocast_dtype="float32"), device="cuda",
    )
    buf = BeliefBuffer(capacity=64, seed=0)
    _fill_buffer(buf, n=64, seed=0)
    metrics = trainer.train_epoch(buf)
    assert math.isfinite(metrics["belief_train/ce_loss"])
    assert trainer.num_train_step >= 1
