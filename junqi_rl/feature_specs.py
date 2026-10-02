"""Versioned, load-time registry for optional policy features on obs317.

This module has no tensor imports. Checkpoints select registered identifiers;
they never supply Python import paths. Execution plans are fixed at model init.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyFeatureSpec:
    name: str
    version: int
    attribute: str
    module: str
    factory: str
    needs_legal_mask: bool = False
    legacy_flag: str | None = None
    channels: tuple[str, ...] = ()
    supported_identity_encodings: tuple[str, ...] = ("legacy_probabilities",)

    def build(self) -> Any:
        return getattr(importlib.import_module(self.module), self.factory)()


_FEATURES: dict[str, PolicyFeatureSpec] = {}


def register_policy_feature(spec: PolicyFeatureSpec) -> None:
    """Register trusted application code before constructing any policies."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", spec.name):
        raise ValueError(f"invalid policy feature name: {spec.name!r}")
    if type(spec.version) is not int or spec.version < 1:
        raise ValueError("feature version must be a positive integer")
    if not spec.attribute.isidentifier() or spec.attribute.startswith("_"):
        raise ValueError(f"invalid feature module attribute: {spec.attribute!r}")
    encodings = spec.supported_identity_encodings
    if (not isinstance(encodings, tuple) or not encodings
            or any(not isinstance(value, str) or value not in
                   {"legacy_probabilities", "known_only"} for value in encodings)
            or len(set(encodings)) != len(encodings)):
        raise ValueError("feature supported_identity_encodings must be a nonempty tuple "
                         "of distinct supported identity encodings")
    if spec.name in _FEATURES:
        raise ValueError(f"policy feature already registered: {spec.name}")
    if any(s.attribute == spec.attribute for s in _FEATURES.values()):
        raise ValueError(f"feature module attribute already registered: {spec.attribute}")
    _FEATURES[spec.name] = spec


def resolve_policy_features(cfg: Any) -> tuple[PolicyFeatureSpec, ...]:
    """Validate requests once, preserving the historical head execution order."""
    requested = getattr(cfg, "policy_features", {})
    if not isinstance(requested, Mapping):
        raise ValueError("net.policy_features must map feature names to versions")
    versions = dict(requested)
    for spec in _FEATURES.values():
        if spec.legacy_flag is None:
            continue
        enabled = getattr(cfg, spec.legacy_flag, False)
        if type(enabled) is not bool:
            raise ValueError(f"net.{spec.legacy_flag} must be a boolean")
        if enabled:
            versions.setdefault(spec.name, spec.version)
    for name, version in versions.items():
        if name not in _FEATURES:
            raise ValueError(f"unsupported policy feature: {name!r}")
        expected = _FEATURES[name].version
        if type(version) is not int or version != expected:
            raise ValueError(
                f"unsupported {name} feature version: {version!r}; supported: {expected}"
            )
    encoding = getattr(cfg, "identity_encoding", "legacy_probabilities")
    for name in versions:
        if encoding not in _FEATURES[name].supported_identity_encodings:
            raise ValueError(f"policy feature {name!r} does not support "
                             f"identity_encoding={encoding!r}")
    return tuple(spec for name, spec in _FEATURES.items() if name in versions)


def plan_feature_channels(
    specs: tuple[PolicyFeatureSpec, ...], layout: Mapping[str, slice],
    on_board: list[int], board_area: int,
) -> tuple[dict[str, slice], list[int]]:
    """Build the static union once, in layout order, without allocating tensors."""
    needed = {name for spec in specs for name in spec.channels}
    unknown = needed - layout.keys()
    if unknown:
        raise ValueError(f"unknown feature observation channels: {sorted(unknown)}")
    slices: dict[str, slice] = {}
    indices: list[int] = []
    width = 0
    for name, span in layout.items():
        if name in needed:
            count = span.stop - span.start
            slices[name] = slice(width, width + count)
            width += count
            indices.extend(channel * board_area + cell
                           for channel in range(span.start, span.stop) for cell in on_board)
    return slices, indices


_PIECE_CHANNELS = (
    "piece_own", "piece_left_side_enemy", "piece_right_side_enemy",
    "belief_left_side", "belief_right_side",
)

for _spec in (
    PolicyFeatureSpec("combat", 1, "combat_head", "junqi_rl.networks.combat_features",
                      "CombatOutcomeHead", legacy_flag="combat_outcome_features", channels=_PIECE_CHANNELS),
    PolicyFeatureSpec("tactical", 2, "tactical_head", "junqi_rl.networks.tactical_features",
                      "TacticalFeatureHead", legacy_flag="tactical_features", channels=_PIECE_CHANNELS),
    PolicyFeatureSpec("relational", 1, "relational_head", "junqi_rl.networks.relational_features",
                      "RelationalFeatureHead", needs_legal_mask=True,
                      legacy_flag="relational_features",
                      channels=(*_PIECE_CHANNELS, "prob_teammate", "dark_teammate",
                                "cm_my_is_gongb")),
):
    register_policy_feature(_spec)
