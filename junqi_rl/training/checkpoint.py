"""Checkpoint persistence shared by training and evaluation tooling."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

import torch

from junqi_rl.checkpoint_compat import current_checkpoint_metadata, validate_policy_checkpoint
from junqi_rl.feature_specs import resolve_policy_features
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training._checkpoint_io import atomic_alias, atomic_write
from junqi_rl.training.config import TrainConfig, _dataclass_to_dict, _dict_to_dataclass


def update_checkpoint_alias(
    checkpoint_path: str,
    alias_path: str,
    *,
    copy_fallback: bool = False,
) -> None:
    """Point ``alias_path`` at a checkpoint, optionally copying if needed."""

    atomic_alias(checkpoint_path, alias_path, copy_fallback=copy_fallback)


def save_checkpoint(
    trainer: Any,
    cfg: TrainConfig,
    rollout_idx: int,
    save_dir: str,
    *,
    arr_trainer: Any | None = None,
    belief_trainer: Any | None = None,
) -> str:
    """Save move-policy and optional auxiliary-network training state."""

    checkpoint_path = os.path.join(save_dir, f"ckpt_{rollout_idx:06d}.pt")
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    state = trainer.state_dict()
    state["train_cfg"] = _dataclass_to_dict(cfg)
    runtime_provider = getattr(trainer, "runtime_state_provider", None)
    if runtime_provider is not None:
        state["runtime_state"] = runtime_provider()
    if arr_trainer is not None:
        state["arrangement"] = {
            "trainer": arr_trainer.state_dict(),
        }
    if belief_trainer is not None:
        state["belief"] = belief_trainer.state_dict()
    opening_monitor = getattr(trainer, "opening_monitor", None)
    if opening_monitor is not None:
        state["opening_monitor"] = opening_monitor.state_dict()

    atomic_write(checkpoint_path, lambda handle: torch.save(state, handle))
    update_checkpoint_alias(
        checkpoint_path,
        os.path.join(save_dir, "ckpt_latest.pt"),
        copy_fallback=True,
    )
    return checkpoint_path


def load_evaluation_policy(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    net_config: JunqiNetConfig | dict[str, Any] | None = None,
) -> JunqiNet:
    """Build a frozen eval policy from a current-format training checkpoint.

    Network hyperparameters come from ``checkpoint["train_cfg"]["net"]``.
    The trainer's pickled ``cfg`` is a ``PPOConfig`` and is not used here.
    """

    ckpt_path = os.fspath(path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"evaluation checkpoint not found: {ckpt_path}")

    # Training checkpoints can contain large optimizer tensors: deserialize on
    # CPU and move only the final policy to the requested inference device.
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise ValueError(f"{ckpt_path} must contain a checkpoint mapping")
    supplied = _net_config(net_config) if net_config is not None else None
    train_cfg = state.get("train_cfg")
    saved_config = train_cfg.get("net") if isinstance(train_cfg, Mapping) else None
    if saved_config is not None:
        net_cfg = _net_config(saved_config)
        if supplied is not None and _effective_config(supplied) != _effective_config(net_cfg):
            raise ValueError(f"{ckpt_path}: supplied net_config conflicts with checkpoint config")
    elif supplied is not None:
        net_cfg = supplied
    else:
        raise ValueError(
            f"{ckpt_path} is missing train_cfg.net; provide an explicit net_config "
            "for a legacy weights-only checkpoint"
        )
    if "policy" not in state:
        if "net" in state and isinstance(state["net"], Mapping):
            state = {**state, "policy": state["net"]}
        elif state and all(isinstance(v, torch.Tensor) for v in state.values()):
            state = {"policy": state}
        else:
            raise ValueError(f"{ckpt_path} does not contain policy weights")
    policy = JunqiNet(net_cfg)
    validate_policy_checkpoint(
        policy,
        state,
        source=f"eval baseline {ckpt_path}",
    )
    policy.load_state_dict(state["policy"])
    policy.to(device).eval()
    return policy


def _net_config(raw: JunqiNetConfig | Mapping[str, Any]) -> JunqiNetConfig:
    if isinstance(raw, JunqiNetConfig):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("net configuration must be a mapping or JunqiNetConfig")
    return _dict_to_dataclass(JunqiNetConfig, dict(raw), path="train_cfg.net")


def _effective_config(cfg: JunqiNetConfig) -> dict[str, Any]:
    values = asdict(cfg)
    for flag in ("combat_outcome_features", "tactical_features", "relational_features"):
        values.pop(flag)
    values["policy_features"] = {s.name: s.version for s in resolve_policy_features(cfg)}
    return values


def export_evaluation_policy(policy: JunqiNet, path: str | os.PathLike[str]) -> str:
    """Atomically export a self-describing, CPU-portable inference checkpoint.

    Includes raw move weights only. Optimizer, arrangement and belief state
    remain in the original training checkpoint; this export cannot resume it.
    """
    target = os.fspath(path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    state = {
        "policy": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "train_cfg": {"net": asdict(policy.cfg)},
        "checkpoint_meta": current_checkpoint_metadata(policy),
        "capabilities": {"move_inference": True, "training_resume": False},
    }
    atomic_write(target, lambda handle: torch.save(state, handle))
    return target


def migrate_policy_without_global_summary(policy: JunqiNet) -> JunqiNet:
    """Warm-start a new spatial-only architecture; optimizer/counters are reset.

    All shared policy weights are retained. The learned CLS starts from the
    old projection bias, equal to the old encoder with global inputs set to
    zero. This deliberately changes the policy and is not an exact resume.
    """
    if not policy.uses_global_summary:
        raise ValueError("source policy already has no global summaries")
    cfg = replace(policy.cfg, use_global_summary=False,
                  policy_features=dict(policy.cfg.policy_features))
    migrated = JunqiNet(cfg)
    weights = {key: tensor.detach().cpu().clone() for key, tensor in policy.state_dict().items()
               if not key.startswith("global_proj.")}
    weights["cls_token"] = policy.global_proj.bias.detach().cpu().reshape(1, 1, -1).clone()
    migrated.load_state_dict(weights, strict=True)
    return migrated.eval()


__all__ = [
    "load_evaluation_policy",
    "export_evaluation_policy",
    "migrate_policy_without_global_summary",
    "save_checkpoint",
    "update_checkpoint_alias",
]
