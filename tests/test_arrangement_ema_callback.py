from __future__ import annotations

from types import SimpleNamespace

import torch

from junqi_rl.training.arr_ppo import ArrangementPPOTrainer


class _FakeBuffer:
    need_arrangements = False
    ready_flags = torch.tensor([True])

    def sample(self, _batch_size: int):
        yield "step"
        yield "skip"
        yield "step"


def test_arrangement_ema_callback_tracks_successful_optimizer_steps() -> None:
    trainer = object.__new__(ArrangementPPOTrainer)
    trainer.cfg = SimpleNamespace(num_epoch_per_train=1, batch_size=1)
    trainer.net = object()
    trainer._net_for_train = SimpleNamespace(train=lambda: None)
    trainer._grad_nan_skip_count = 0

    def fake_step(batch, stats):
        stats["arr_train/n_batches"].append(1.0)
        return batch == "step"

    trainer._step = fake_step
    callback_models: list[object] = []

    metrics = trainer.train_epoch(
        _FakeBuffer(),
        on_optimizer_step=callback_models.append,
    )

    assert callback_models == [trainer.net, trainer.net]
    assert metrics["arr_train/n_batches"] == 3.0
