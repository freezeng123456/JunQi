"""Checkpoint schema metadata and policy compatibility checks.

The project has had deliberate observation/action/network schema changes.  A
checkpoint may therefore be a valid PyTorch file but still be structurally
incompatible with the current policy.  Keep the check here dependency-light so
training, evaluation and PPO loading can all use the same guard.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.nn import Module

from junqi_core.board import COMPACT_ACTION_DIM, NUM_ON_BOARD_CELLS
from junqi_core.observation import OBS_CHANNELS


CHECKPOINT_FORMAT_VERSION = 2


def current_checkpoint_metadata(policy: Module | None = None) -> dict[str, Any]:
    """Return runtime schema metadata saved with new policy checkpoints."""

    metadata: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_channels": OBS_CHANNELS,
        "action_dim": COMPACT_ACTION_DIM,
        "num_on_board_cells": NUM_ON_BOARD_CELLS,
    }
    if policy is not None:
        metadata["policy_class"] = type(policy).__name__
        stem = getattr(policy, "stem", None)
        metadata["stem_class"] = type(stem).__name__ if stem is not None else None
    return metadata


def validate_policy_checkpoint(
    policy: Module,
    checkpoint: Mapping[str, Any],
    *,
    source: str = "checkpoint",
) -> None:
    """Fail early with a useful error when a policy checkpoint cannot load.

    Legacy checkpoints do not have ``checkpoint_meta``.  They are still
    accepted when their policy state-dict keys and tensor shapes match the
    current model exactly.  New checkpoints additionally carry explicit
    observation/action metadata so schema drift is diagnosed before
    ``load_state_dict`` mutates anything.
    """

    saved = checkpoint.get("policy")
    if not isinstance(saved, Mapping):
        raise ValueError(f"{source} does not contain a mapping-valued 'policy' state dict")

    expected = policy.state_dict()
    saved_keys = set(saved)
    expected_keys = set(expected)

    missing = sorted(expected_keys - saved_keys)
    unexpected = sorted(saved_keys - expected_keys)
    shape_mismatches: list[str] = []
    for key in sorted(expected_keys & saved_keys):
        expected_shape = getattr(expected[key], "shape", None)
        saved_shape = getattr(saved[key], "shape", None)
        if expected_shape != saved_shape:
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(saved_shape) if saved_shape is not None else saved_shape} "
                f"runtime={tuple(expected_shape) if expected_shape is not None else expected_shape}"
            )

    metadata_mismatches: list[str] = []
    saved_meta = checkpoint.get("checkpoint_meta")
    if isinstance(saved_meta, Mapping):
        current_meta = current_checkpoint_metadata(policy)
        for key in ("observation_channels", "action_dim", "num_on_board_cells"):
            if key in saved_meta and saved_meta[key] != current_meta[key]:
                metadata_mismatches.append(
                    f"{key}: checkpoint={saved_meta[key]!r} runtime={current_meta[key]!r}"
                )
        if (
            saved_meta.get("stem_class") is not None
            and saved_meta.get("stem_class") != current_meta.get("stem_class")
        ):
            metadata_mismatches.append(
                "stem_class: "
                f"checkpoint={saved_meta.get('stem_class')!r} "
                f"runtime={current_meta.get('stem_class')!r}"
            )

    if not (missing or unexpected or shape_mismatches or metadata_mismatches):
        return

    details: list[str] = []
    if metadata_mismatches:
        details.append("metadata mismatch: " + "; ".join(metadata_mismatches))
    if missing:
        details.append("missing keys: " + ", ".join(missing[:8]))
    if unexpected:
        details.append("unexpected keys: " + ", ".join(unexpected[:8]))
    if shape_mismatches:
        details.append("shape mismatch: " + "; ".join(shape_mismatches[:8]))

    raise ValueError(
        f"{source} is incompatible with the current Junqi policy schema. "
        + " | ".join(details)
        + ". This commonly happens across observation/network migrations "
          "(for example 412→317 channels or CNNStem→GraphStem). "
          "Use a checkpoint produced by the current schema or an explicit migration."
    )


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "current_checkpoint_metadata",
    "validate_policy_checkpoint",
]
