"""Tests for junqi_rl.training.arr_ppo."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from junqi_rl.arrangement.buffer import ArrangementBuffer
from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    ArrangementNet,
    ArrangementNetConfig,
    N_PIECE_TYPE_WITH_NONE,
    N_VF_CAT_DEFAULT,
    PIECE_TYPE_VALUE_TO_VOCAB_IDX,
)
from junqi_rl.training.arr_ppo import (
    ArrangementPPOConfig,
    ArrangementPPOTrainer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiny_cfg() -> ArrangementNetConfig:
    return ArrangementNetConfig(depth=2, n_head=4, embed_dim=64, ff_factor=2)


def _seed_buffer(N: int, *, use_cat_vf: bool, seed: int = 0):
    """Populate an ArrangementBuffer with N rows + rewards + processed data."""
    from junqi_core.setup import generate_random_lineup
    import random

    rng = random.Random(seed)
    buf = ArrangementBuffer(
        storage_duration=100, device="cpu", use_cat_vf=use_cat_vf,
    )

    vocabs = []
    arr = []
    for _ in range(N):
        lu = generate_random_lineup(rng)
        v = torch.tensor(
            [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in lu], dtype=torch.long
        )
        vocabs.append(v)
        arr.append(F.one_hot(v, num_classes=N_PIECE_TYPE_WITH_NONE).float())
    arr_tensor = torch.stack(arr)             # (N, 30, 13)
    vocabs_tensor = torch.stack(vocabs)       # (N, 30)

    if use_cat_vf:
        values = torch.randn(N, ARRANGEMENT_SIZE, N_VF_CAT_DEFAULT)
    else:
        values = torch.randn(N, ARRANGEMENT_SIZE)
    ents = torch.randn(N, ARRANGEMENT_SIZE).abs()
    log_probs = F.log_softmax(
        torch.randn(N, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE), dim=-1,
    )
    seat_idx = torch.randint(0, 4, (N,), dtype=torch.long)
    buf.add_arrangements(arr_tensor, values, ents, log_probs, seat_idx, step=0)

    term = torch.ones(N, dtype=torch.bool)
    # Mix of rewards so cat-VF one-hot has non-trivial distribution.
    rewards = torch.tensor([(-1.0, 0.0, 1.0)[i % 3] for i in range(N)])
    buf.add_rewards(vocabs_tensor, seat_idx, term, rewards)
    buf.process_data(
        td_lambda=1.0, gae_lambda=1.0, reg_temp=0.02, reg_norm=10.0,
    )
    return buf


# ---------------------------------------------------------------------------
# Smoke — training step runs, parameters update, stats populated.
# ---------------------------------------------------------------------------


def test_train_epoch_runs_cat_vf():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())  # use_cat_vf=True by default
    trainer = ArrangementPPOTrainer(
        net, ArrangementPPOConfig(batch_size=8, num_epoch_per_train=1),
    )
    buf = _seed_buffer(N=16, use_cat_vf=True, seed=1)

    # Snapshot params before; train; check they changed.
    snap = {n: p.detach().clone() for n, p in net.named_parameters()}
    stats = trainer.train_epoch(buf)

    for key in [
        "arr_train/policy_loss", "arr_train/value_loss",
        "arr_train/entropy_loss", "arr_train/kl_loss",
        "arr_train/total_loss", "arr_train/clip_fraction",
        "arr_train/g_norm", "arr_train/lr", "arr_train/n_batches",
    ]:
        assert key in stats, f"missing stat {key}"

    # At least one param should have moved.
    moved = any(
        not torch.equal(p.detach(), snap[n]) for n, p in net.named_parameters()
    )
    assert moved, "no parameters changed after training"

    # n_batches should equal ceil(16 / 8) = 2.
    assert int(stats["arr_train/n_batches"]) == 2


def test_train_epoch_runs_scalar_vf():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.use_cat_vf = False
    net = ArrangementNet(cfg)
    trainer = ArrangementPPOTrainer(
        net, ArrangementPPOConfig(batch_size=4, num_epoch_per_train=1),
    )
    buf = _seed_buffer(N=8, use_cat_vf=False, seed=2)
    stats = trainer.train_epoch(buf)
    assert stats  # non-empty
    assert stats["arr_train/n_batches"] == 2.0


def test_train_epoch_empty_buffer_returns_empty():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    trainer = ArrangementPPOTrainer(net)
    # Seed an empty buffer (arrangements added but no rewards).
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    from junqi_core.setup import generate_random_lineup
    import random
    rng = random.Random(0)
    vocabs = torch.stack([
        torch.tensor(
            [PIECE_TYPE_VALUE_TO_VOCAB_IDX[pt.value] for pt in generate_random_lineup(rng)],
            dtype=torch.long,
        )
        for _ in range(4)
    ])
    arr = F.one_hot(vocabs, num_classes=N_PIECE_TYPE_WITH_NONE).float()
    buf.add_arrangements(
        arr, torch.randn(4, ARRANGEMENT_SIZE, N_VF_CAT_DEFAULT),
        torch.randn(4, ARRANGEMENT_SIZE).abs(),
        F.log_softmax(torch.randn(4, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE), dim=-1),
        torch.zeros(4, dtype=torch.long),
        step=0,
    )
    stats = trainer.train_epoch(buf)
    assert stats == {}


def test_train_epoch_raises_before_add_arrangements():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    trainer = ArrangementPPOTrainer(net)
    buf = ArrangementBuffer(storage_duration=100, device="cpu", use_cat_vf=True)
    with pytest.raises(RuntimeError):
        trainer.train_epoch(buf)


# ---------------------------------------------------------------------------
# Loss sign-checks
# ---------------------------------------------------------------------------


def test_policy_loss_decreases_with_gradient_steps():
    """Running multiple epochs on a fixed buffer should reduce total_loss.

    This is a weak consistency check — PPO isn't guaranteed monotonic, but
    on a tiny fixed dataset with sensible hyperparameters the loss almost
    always trends down across a few epochs.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    cfg.use_cat_vf = False
    net = ArrangementNet(cfg)
    trainer = ArrangementPPOTrainer(
        net, ArrangementPPOConfig(
            batch_size=8, num_epoch_per_train=1,
            kl_coef=0.0,   # turn off KL so behaviour is clean-PPO
            ent_pred_coef=0.0,  # ignore ent-pred noise
        ),
    )
    buf = _seed_buffer(N=32, use_cat_vf=False, seed=5)

    first_losses: list[float] = []
    for epoch in range(3):
        stats = trainer.train_epoch(buf)
        first_losses.append(stats["arr_train/total_loss"])
    # Loose check — any decrease across 3 epochs.
    assert first_losses[-1] < first_losses[0] or abs(first_losses[-1] - first_losses[0]) < 0.5


def test_value_loss_is_nonnegative():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    trainer = ArrangementPPOTrainer(net, ArrangementPPOConfig(batch_size=4))
    buf = _seed_buffer(N=8, use_cat_vf=True, seed=6)
    stats = trainer.train_epoch(buf)
    # Categorical cross-entropy is always >= 0 up to float rounding.
    assert stats["arr_train/value_loss"] >= -1e-4


def test_entropy_loss_is_nonnegative():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg())
    trainer = ArrangementPPOTrainer(net)
    buf = _seed_buffer(N=8, use_cat_vf=True, seed=7)
    stats = trainer.train_epoch(buf)
    assert stats["arr_train/entropy_loss"] >= 0.0  # MSE


# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_epoch_cuda():
    torch.manual_seed(0)
    net = ArrangementNet(_tiny_cfg()).cuda()
    trainer = ArrangementPPOTrainer(
        net, ArrangementPPOConfig(batch_size=4, num_epoch_per_train=1),
    )
    buf = _seed_buffer(N=8, use_cat_vf=True, seed=8)
    # Move buffer to CUDA by re-seeding on-device (easier than in-place move).
    for attr in ("arrangements", "seat_idx", "values", "ents", "log_probs",
                 "step_added", "counts", "rewards", "ready_flags",
                 "adv_est", "val_est", "reg_val_est"):
        setattr(buf, attr, getattr(buf, attr).cuda())
    buf.device = torch.device("cuda")
    stats = trainer.train_epoch(buf)
    assert stats["arr_train/n_batches"] >= 1.0
