"""junqi_rl.training.logger — Lightweight training logger.

Supports TensorBoard and optional Weights & Biases.  Provides a
:class:`MultiCounter` (mirrors Ataraxos ``utils.MultiCounter``) for
accumulating and flushing metric averages.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

import numpy as np


class MultiCounter:
    """Accumulate scalar metrics and compute means for logging.

    Usage::

        mc = MultiCounter(step=0)
        mc["train/loss"] += 0.5
        mc.inc("train/steps")
        summary = mc.summary()   # dict of means
        mc.reset()
    """

    def __init__(self, step: int = 0) -> None:
        self._counts: dict[str, list[float]] = defaultdict(list)
        self.step = step

    def __getitem__(self, key: str) -> "_Accumulator":
        return _Accumulator(self._counts[key])

    def inc(self, key: str, value: float = 1.0) -> None:
        self._counts[key].append(value)

    def update(self, d: dict[str, float]) -> None:
        for k, v in d.items():
            self._counts[k].append(v)

    def summary(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self._counts.items() if v}

    def reset(self) -> None:
        self._counts.clear()


class _Accumulator:
    """Internal helper for ``counter["key"] += value`` syntax."""

    def __init__(self, lst: list[float]) -> None:
        self._lst = lst

    def __iadd__(self, value: float) -> "_Accumulator":
        self._lst.append(value)
        return self


class TrainingLogger:
    """Unified logger for TensorBoard + optional Weights & Biases.

    Parameters
    ----------
    log_dir
        Directory for TensorBoard event files and run metadata.
    use_wandb
        Enable Weights & Biases logging (requires ``wandb`` installed).
    wandb_project
        W&B project name.
    wandb_run_name
        W&B run name (default: timestamp-based).
    config
        Dict of hyper-parameters to log to W&B.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        use_wandb: bool = False,
        wandb_project: str = "junqi-rl",
        wandb_run_name: str | None = None,
        config: dict[str, Any] | None = None,
        rank: int = 0,
    ) -> None:
        # Under DDP we instantiate one TrainingLogger per rank so that the
        # higher-level training loop doesn't have to branch on rank. Only
        # rank-0 writes anything; other ranks get a degenerate object that
        # short-circuits log() / log_text() / close(). This keeps a single
        # SummaryWriter / wandb run per experiment (avoiding 8 duplicate
        # tensorboard event streams in the same dir).
        self.log_dir = log_dir
        self._rank = int(rank)
        self._is_rank0 = self._rank == 0

        self._tb_writer: Any = None
        self._wb = None
        self._t_start = time.time()

        if not self._is_rank0:
            return  # silent on non-rank-0 ranks

        os.makedirs(log_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import]
            self._tb_writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            pass

        if use_wandb:
            try:
                import wandb  # type: ignore[import]
                self._wb = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or f"junqi-{int(time.time())}",
                    dir=log_dir,
                    config=config or {},
                )
            except Exception as e:
                print(f"[logger] wandb init failed: {e}")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Log a dict of scalar metrics at ``step`` (no-op on non-rank-0)."""
        if not self._is_rank0:
            return
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, global_step=step)
        if self._wb is not None:
            try:
                self._wb.log({**metrics, "step": step})
            except Exception:
                pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        if not self._is_rank0:
            return
        if self._tb_writer is not None:
            self._tb_writer.add_text(tag, text, global_step=step)

    def elapsed(self) -> float:
        return time.time() - self._t_start

    def close(self) -> None:
        if not self._is_rank0:
            return
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wb is not None:
            try:
                self._wb.finish()
            except Exception:
                pass

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
