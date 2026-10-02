"""Load-time contracts runnable with stdlib alone (no Torch installation)."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


# Import the dependency-free registry without the package's environment imports.
_spec = importlib.util.spec_from_file_location(
    "_tested_feature_specs", Path(__file__).parents[1] / "junqi_rl/feature_specs.py"
)
registry = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = registry
_spec.loader.exec_module(registry)


class FeatureSpecTests(unittest.TestCase):
    def test_disabled_and_resolved_features_never_import_or_build_heads(self):
        with patch.object(registry.importlib, "import_module", side_effect=AssertionError):
            self.assertEqual(registry.resolve_policy_features(SimpleNamespace()), ())
            specs = registry.resolve_policy_features(SimpleNamespace(tactical_features=True))
            self.assertEqual([(s.name, s.version) for s in specs], [("tactical", 2)])

    def test_legacy_flags_and_versioned_requests_have_identical_order(self):
        legacy = SimpleNamespace(combat_outcome_features=True, tactical_features=True,
                                 relational_features=True)
        explicit = SimpleNamespace(policy_features={"relational": 1, "tactical": 2, "combat": 1})
        self.assertEqual(registry.resolve_policy_features(legacy),
                         registry.resolve_policy_features(explicit))

    def test_unknown_or_old_feature_version_is_rejected(self):
        for request in ({"missing": 1}, {"tactical": 1}, {"tactical": True},
                        {"combat": 0}, {"combat": "1"}, None):
            with self.subTest(request=request), self.assertRaises(ValueError):
                registry.resolve_policy_features(SimpleNamespace(policy_features=request))

    def test_legacy_flag_cannot_hide_conflicting_explicit_version(self):
        with self.assertRaisesRegex(ValueError, "version"):
            registry.resolve_policy_features(SimpleNamespace(
                tactical_features=True, policy_features={"tactical": 1}))

    def test_string_boolean_is_not_silently_enabled(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            registry.resolve_policy_features(SimpleNamespace(tactical_features="false"))

    def test_extension_is_registered_without_changing_network_dispatch(self):
        spec = registry.PolicyFeatureSpec("test_probe", 3, "test_probe_head",
                                         "test_module", "Head")
        with patch.object(registry, "_FEATURES", dict(registry._FEATURES)):
            registry.register_policy_feature(spec)
            actual = registry.resolve_policy_features(SimpleNamespace(
                policy_features={"test_probe": 3}))
            self.assertEqual(actual, (spec,))
            with self.assertRaisesRegex(ValueError, "already registered"):
                registry.register_policy_feature(spec)

    def test_duplicate_module_attribute_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "attribute already registered"):
            registry.register_policy_feature(registry.PolicyFeatureSpec(
                "different", 1, "combat_head", "test_module", "Head"))

    def test_facts_extensions_opt_in_per_feature_without_importing(self):
        spec = registry.PolicyFeatureSpec(
            "learned_facts", 1, "learned_facts_head", "test_module", "Head",
            supported_identity_encodings=("known_only",))
        with patch.object(registry, "_FEATURES", dict(registry._FEATURES)), \
                patch.object(registry.importlib, "import_module", side_effect=AssertionError):
            registry.register_policy_feature(spec)
            self.assertEqual(registry.resolve_policy_features(SimpleNamespace(
                identity_encoding="known_only", policy_features={"learned_facts": 1})), (spec,))
            for name, version in (("combat", 1), ("tactical", 2), ("relational", 1)):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "known_only"):
                    registry.resolve_policy_features(SimpleNamespace(
                        identity_encoding="known_only", policy_features={name: version}))
            with self.assertRaisesRegex(ValueError, "legacy_probabilities"):
                registry.resolve_policy_features(SimpleNamespace(policy_features={"learned_facts": 1}))

    def test_invalid_encoding_declarations_fail_at_registration(self):
        for encodings in ((), ("unknown",), ("known_only", "known_only"),
                          "known_only", ["known_only"], (None,), ([],)):
            with self.subTest(encodings=encodings), self.assertRaisesRegex(ValueError, "encodings"):
                registry.register_policy_feature(registry.PolicyFeatureSpec(
                    "invalid", 1, "invalid_head", "test_module", "Head",
                    supported_identity_encodings=encodings))


    def test_channel_plan_deduplicates_dependencies_in_layout_order(self):
        a = registry.PolicyFeatureSpec("a", 1, "a_head", "test", "Head", channels=("last", "first"))
        b = registry.PolicyFeatureSpec("b", 1, "b_head", "test", "Head", channels=("first",))
        layout = {"first": slice(0, 2), "unused": slice(2, 6), "last": slice(6, 7)}
        slices, indices = registry.plan_feature_channels((a, b), layout, [0, 2], 9)
        self.assertEqual(slices, {"first": slice(0, 2), "last": slice(2, 3)})
        self.assertEqual(indices, [0, 2, 9, 11, 54, 56])

    def test_disabled_plan_allocates_no_indices(self):
        self.assertEqual(registry.plan_feature_channels((), {"first": slice(0, 2)}, [0, 2], 9),
                         ({}, []))

    def test_unknown_channel_is_rejected(self):
        spec = registry.PolicyFeatureSpec("a", 1, "a_head", "test", "Head", channels=("typo",))
        with self.assertRaisesRegex(ValueError, "unknown feature observation channels"):
            registry.plan_feature_channels((spec,), {}, [0, 2], 9)


if __name__ == "__main__":
    unittest.main()
