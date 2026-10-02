"""Checkpoint schema metadata and policy compatibility checks.

The project has had deliberate observation/action/network schema changes.  A
checkpoint may therefore be a valid PyTorch file but still be structurally
incompatible with the current policy.  Keep the check here dependency-light so
training, evaluation and PPO loading can all use the same guard.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from torch.nn import Module

from junqi_core.board import COMPACT_ACTION_DIM, COMPACT_TO_FLAT, NUM_ON_BOARD_CELLS
from junqi_core.observation import CHANNEL_LAYOUT, GLOBAL_LAYOUT, OBS_CHANNELS, OBS_GLOBAL_DIMS


CHECKPOINT_FORMAT_VERSION = 2
# v4: public commander-death flags identify the exact flag cell and exclude
# JUNQI from every other piece of that army, on CPU and native CUDA paths.
OBSERVATION_SEMANTICS_VERSION = 4
POLICY_ARCHITECTURE_ID = "junqi.graph317.v1"
SPATIAL_ONLY_ARCHITECTURE_ID = "junqi.graph317.cls.v1"


@lru_cache(maxsize=4)
def _input_layout_hash(use_global_summary: bool = True, identity_encoding: str = "legacy_probabilities") -> str:
    layout = {
        "spatial": {k: [v.start, v.stop, v.step] for k, v in CHANNEL_LAYOUT.items()},
        "global": {k: [v.start, v.stop, v.step] for k, v in GLOBAL_LAYOUT.items()} if use_global_summary else {},
        "compact_to_flat": COMPACT_TO_FLAT.tolist(),
        "frame": "observer_south",
    }
    if identity_encoding != "legacy_probabilities":
        layout["identity_encoding"] = identity_encoding
    return hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest()


def current_checkpoint_metadata(policy: Module | None = None) -> dict[str, Any]:
    """Return runtime schema metadata saved with new policy checkpoints."""

    use_global_summary = getattr(policy, "uses_global_summary", True)
    identity_encoding = getattr(policy, "identity_encoding", "legacy_probabilities")
    known_only = identity_encoding == "known_only"
    metadata: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "observation_semantics_version": 5 if known_only else OBSERVATION_SEMANTICS_VERSION,
        "identity_encoding": identity_encoding,
        "observation_channels": OBS_CHANNELS,
        "observation_global_dims": OBS_GLOBAL_DIMS if use_global_summary else 0,
        "global_summary_enabled": use_global_summary,
        "input_layout_sha256": _input_layout_hash(use_global_summary, identity_encoding),
        "architecture_id": ("junqi.graph317.facts.v1" if known_only else
                            POLICY_ARCHITECTURE_ID if use_global_summary else SPATIAL_ONLY_ARCHITECTURE_ID),
        "action_dim": COMPACT_ACTION_DIM,
        "num_on_board_cells": NUM_ON_BOARD_CELLS,
    }
    if policy is not None:
        versions = dict(policy._feature_versions)
        metadata["policy_features"] = versions
        for name in ("combat", "tactical", "relational"):
            metadata[f"{name}_feature_version"] = versions.get(name, 0)
        metadata["policy_class"] = type(policy).__name__
        stem = getattr(policy, "stem", None)
        metadata["stem_class"] = type(stem).__name__ if stem is not None else None
    return metadata


def _metadata_type_matches(value: Any, expected: Any) -> bool:
    """Check schema types without Python's ``True == 1`` coercion.

    Feature maps carry version numbers, so their keys and values also need
    validation before ordinary equality can compare the selected features.
    """
    if isinstance(expected, dict):
        return isinstance(value, Mapping) and all(
            type(name) is str and type(version) is int
            for name, version in value.items()
        )
    return type(value) is type(expected)


def validate_policy_checkpoint(
    policy: Module,
    checkpoint: Mapping[str, Any],
    *,
    source: str = "checkpoint",
) -> None:
    """Fail early with a useful error when a policy checkpoint cannot load.

    Historical probability policies may load legacy checkpoints without
    complete metadata when state-dict keys and tensor shapes match exactly.
    Facts policies require the complete metadata written by the current
    exporter: identical tensor shapes cannot establish identity semantics.
    All supplied metadata fields are type-checked before comparison, so schema
    drift is diagnosed before ``load_state_dict`` mutates anything.
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
    expected_encoding = getattr(policy, "identity_encoding", "legacy_probabilities")
    saved_encoding = (saved_meta.get("identity_encoding", "legacy_probabilities")
                      if isinstance(saved_meta, Mapping) else "legacy_probabilities")
    current_meta = current_checkpoint_metadata(policy)
    requires_complete_metadata = (
        expected_encoding == "known_only" or saved_encoding == "known_only"
    )
    if "checkpoint_meta" in checkpoint and not isinstance(saved_meta, Mapping):
        metadata_mismatches.append("checkpoint_meta must be a mapping")
    if requires_complete_metadata:
        for key in current_meta:
            if not isinstance(saved_meta, Mapping) or key not in saved_meta:
                metadata_mismatches.append(f"missing required metadata field {key!r}")
    if saved_encoding != expected_encoding:
        metadata_mismatches.append(f"identity_encoding: checkpoint={saved_encoding!r} runtime={expected_encoding!r}")
    if not requires_complete_metadata and (
        not isinstance(saved_meta, Mapping) or "observation_semantics_version" not in saved_meta
    ):
        warnings.warn(
            f"{source}: legacy observation semantics are unspecified. Hidden-information "
            "and live-inventory fixes changed inputs without changing tensor shapes; "
            "loading these weights does not reproduce the original training run.",
            UserWarning, stacklevel=2,
        )
    if isinstance(saved_meta, Mapping):
        for key, current_value in current_meta.items():
            if key not in saved_meta:
                continue
            saved_value = saved_meta[key]
            # Older exporters sometimes omitted the concrete stem name.
            if key == "stem_class" and saved_value is None and not requires_complete_metadata:
                continue
            if not _metadata_type_matches(saved_value, current_value):
                metadata_mismatches.append(
                    f"{key}: invalid metadata type {type(saved_value).__name__}; "
                    f"expected {type(current_value).__name__}"
                )
            elif saved_value != current_value:
                metadata_mismatches.append(
                    f"{key}: checkpoint={saved_value!r} runtime={current_value!r}"
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
