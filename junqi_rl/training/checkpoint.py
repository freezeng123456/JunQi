"""Checkpoint persistence shared by training and evaluation tooling."""

from __future__ import annotations

import os
import shutil
from typing import Any

import torch

from junqi_rl.checkpoint_compat import validate_policy_checkpoint
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.config import TrainConfig, _dataclass_to_dict, _dict_to_dataclass


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


def load_evaluation_policy(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> JunqiNet:
    """Build a frozen eval policy from a current-format training checkpoint.

    Network hyperparameters come from ``checkpoint["train_cfg"]["net"]``.
    The trainer's pickled ``cfg`` is a ``PPOConfig`` and is not used here.
    """

    ckpt_path = os.fspath(path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"evaluation checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = state.get("train_cfg")
    if not isinstance(train_cfg, dict) or "net" not in train_cfg:
        raise ValueError(
            f"{ckpt_path} is missing train_cfg.net; "
            "refusing to guess a network config from trainer cfg"
        )
    net_cfg = _dict_to_dataclass(
        JunqiNetConfig,
        train_cfg["net"],
        path="train_cfg.net",
    )
    policy = JunqiNet(net_cfg).to(device)
    validate_policy_checkpoint(
        policy,
        state,
        source=f"eval baseline {ckpt_path}",
    )
    policy.load_state_dict(state["policy"])
    policy.eval()
    return policy


__all__ = [
    "load_evaluation_policy",
    "save_checkpoint",
    "update_checkpoint_alias",
]
