from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import junqi_rl.training.ppo as ppo_module
from junqi_rl.training.gpu_collector import _categorical_value_to_scalar
from junqi_rl.training.ppo import PPOTrainer
from junqi_rl.training.rollout_gpu import (
    _csr_to_dense_selected,
    _dense_mask_to_csr,
    observation_storage_dtype,
)


class _GuardedRollout:
    """Fails if the trainer eagerly consumes batches before updating."""

    def __init__(self) -> None:
        self.consumed = 0

    def minibatches(self, *_args, **_kwargs):
        for index in range(3):
            assert self.consumed == index, "minibatches were materialized eagerly"
            yield index

    def stats(self) -> dict[str, float]:
        return {"rollout/test": 1.0}


class _FakeEMA:
    def __init__(self) -> None:
        self.updates = 0

    def update(self, _model) -> None:
        self.updates += 1


def test_single_process_ppo_streams_minibatches() -> None:
    trainer = object.__new__(PPOTrainer)
    trainer.cfg = SimpleNamespace(
        num_epochs_per_rollout=1,
        minibatch_size=2,
    )
    trainer.num_rollout = 0
    trainer._nan_skip_count = 0
    trainer._grad_nan_skip_count = 0
    trainer._policy_unwrapped = object()
    trainer.ema = _FakeEMA()
    rollout = _GuardedRollout()

    def update_step(batch_index: int) -> dict[str, object]:
        rollout.consumed += 1
        return {
            "train/test_loss": torch.tensor(float(batch_index + 1)),
            "train/batch_size": 2,
        }

    trainer._update_step = update_step
    metrics = trainer.train_epoch(rollout)

    assert rollout.consumed == 3
    assert trainer.ema.updates == 3
    assert trainer.num_rollout == 1
    assert metrics["train/test_loss"] == pytest.approx(2.0)
    assert metrics["train/num_updates"] == 3.0


def test_distributed_ppo_still_aligns_minibatch_count(monkeypatch) -> None:
    trainer = object.__new__(PPOTrainer)
    trainer.cfg = SimpleNamespace(
        num_epochs_per_rollout=1,
        minibatch_size=2,
    )
    trainer.device = torch.device("cpu")
    trainer.num_rollout = 0
    trainer._nan_skip_count = 0
    trainer._grad_nan_skip_count = 0
    trainer._policy_unwrapped = object()
    trainer.ema = _FakeEMA()
    consumed: list[int] = []

    class Rollout:
        def minibatches(self, *_args, **_kwargs):
            yield from range(3)

        def stats(self):
            return {}

    def update_step(batch_index: int) -> dict[str, object]:
        consumed.append(batch_index)
        return {
            "train/test_loss": torch.tensor(float(batch_index)),
            "train/batch_size": 2,
        }

    def fake_all_reduce(count, **_kwargs) -> None:
        count.fill_(2)

    trainer._update_step = update_step
    monkeypatch.setattr(ppo_module, "_is_distributed", lambda: True)
    monkeypatch.setattr(ppo_module.dist, "all_reduce", fake_all_reduce)

    metrics = trainer.train_epoch(Rollout())

    assert consumed == [0, 1]
    assert trainer.ema.updates == 2
    assert metrics["train/num_updates"] == 2.0


def test_compact_history_rejects_ddp_until_index_plans_are_shared(
    monkeypatch,
) -> None:
    trainer = object.__new__(PPOTrainer)
    trainer.cfg = SimpleNamespace(
        num_epochs_per_rollout=1,
        minibatch_size=2,
    )
    trainer._nan_skip_count = 0
    trainer._grad_nan_skip_count = 0
    rollout = SimpleNamespace(uses_compact_history=True)
    monkeypatch.setattr(ppo_module, "_is_distributed", lambda: True)

    with pytest.raises(RuntimeError, match="single-GPU"):
        trainer.train_epoch(rollout)


def test_h20_observation_storage_matches_bfloat16_compute() -> None:
    assert observation_storage_dtype(torch.bfloat16) == torch.bfloat16
    assert observation_storage_dtype(torch.float16) == torch.float16
    assert observation_storage_dtype(torch.float32) == torch.float16
    with pytest.raises(ValueError, match="unsupported"):
        observation_storage_dtype(torch.int8)


def test_categorical_value_fast_path_matches_expected_value() -> None:
    probabilities = torch.tensor(
        [
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1],
        ],
        dtype=torch.float32,
    )
    values = probabilities.log()

    actual = _categorical_value_to_scalar(values)
    expected = probabilities[:, 2] - probabilities[:, 0]

    assert torch.allclose(actual, expected)


def test_scalar_value_fast_path_is_unchanged() -> None:
    values = torch.tensor([-0.5, 0.0, 0.75])
    assert _categorical_value_to_scalar(values) is values


@pytest.mark.parametrize(
    "mask",
    [
        torch.zeros(3, 16, dtype=torch.bool),
        torch.tensor(
            [
                [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
            ],
            dtype=torch.bool,
        ),
    ],
)
def test_csr_roundtrip_without_synchronizing_branches(mask: torch.Tensor) -> None:
    ids = torch.zeros((3, 8), dtype=torch.int32)
    counts = torch.zeros(3, dtype=torch.int32)
    _dense_mask_to_csr(mask, ids, counts)

    restored = _csr_to_dense_selected(
        ids,
        counts,
        torch.arange(3),
        mask.shape[1],
    )

    assert torch.equal(restored, mask)
