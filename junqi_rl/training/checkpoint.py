"""Checkpoint persistence shared by training and evaluation tooling."""

from __future__ import annotations

import os
import shutil
from typing import Any

import torch

from junqi_rl.training.config import TrainConfig, _dataclass_to_dict


def update_checkpoint_alias(
    checkpoint_path: str,
    alias_path: str,
    *,
    copy_fallback: bool = False,
) -> None:
    """Point ``alias_path`` at a checkpoint, optionally copying if needed."""

    if os.path.islink(alias_path) or os.path.exists(alias_path):
        os.remove(alias_path)
    try:
        os.symlink(os.path.abspath(checkpoint_path), alias_path)
    except OSError:
        if copy_fallback:
            shutil.copy2(checkpoint_path, alias_path)


def save_checkpoint(
    trainer: Any,
    cfg: TrainConfig,
    rollout_idx: int,
    save_dir: str,
    *,
    arr_trainer: Any | None = None,
    arr_ema: Any | None = None,
    belief_trainer: Any | None = None,
) -> str:
    """Save move-policy and optional auxiliary-network training state."""

    checkpoint_path = os.path.join(save_dir, f"ckpt_{rollout_idx:06d}.pt")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    state = trainer.state_dict()
    state["train_cfg"] = _dataclass_to_dict(cfg)
    if arr_trainer is not None:
        state["arrangement"] = {
            "trainer": arr_trainer.state_dict(),
            "ema": arr_ema.state_dict() if arr_ema is not None else None,
        }
    if belief_trainer is not None:
        state["belief"] = belief_trainer.state_dict()

    torch.save(state, checkpoint_path)
    update_checkpoint_alias(
        checkpoint_path,
        os.path.join(save_dir, "ckpt_latest.pt"),
    )
    return checkpoint_path


__all__ = ["save_checkpoint", "update_checkpoint_alias"]
