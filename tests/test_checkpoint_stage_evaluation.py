"""Provenance and paired-match acceptance checks, runnable without torch."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_checkpoint_stages import (
    checkpoint_provenance,
    parse_pairs,
    resolve_checkpoint_spec,
    validate_records,
)


class StageEvaluationTests(unittest.TestCase):
    def test_pair_selection_and_duplicate_rejection(self):
        checkpoints = {"660k": {}, "975k": {}, "1m": {"completion": "absent"}}
        self.assertEqual(parse_pairs(["975k:660k"], checkpoints), [("975k", "660k")])
        self.assertEqual(parse_pairs(["1m:660k,1m:975k"], checkpoints),
                         [("1m", "660k"), ("1m", "975k")])
        for pairs in [["660k:660k"], ["missing:660k"], ["975k:660k", "660k:975k"]]:
            with self.assertRaises(ValueError):
                parse_pairs(pairs, checkpoints)

    def test_completion_is_hash_bound_and_does_not_wait(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            weights = root / "checkpoint.pt"
            weights.write_bytes(b"test-weights")
            training = root / "training_result.json"
            training.write_text(json.dumps(dict(status="target_reached", completed_games=1000003,
                                                lineage_id="same-lineage")))
            completion = root / "completion.json"
            receipt = dict(status="COMPLETE", completed_games=1000003, target_completed_games=1000000,
                           final_checkpoint="checkpoint.pt",
                           sha256={"training_result.json": hashlib.sha256(training.read_bytes()).hexdigest()})
            completion.write_text(json.dumps(receipt))
            entry = dict(completion="completion.json", run_root=".", training_result="training_result.json")
            resolved = resolve_checkpoint_spec(entry, root)
            self.assertEqual(resolved["completed_games"], 1000003)
            self.assertEqual(resolved["path"], str(weights.resolve()))
            training.write_text(training.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_checkpoint_spec(entry, root)
            completion.unlink()
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint_spec(entry, root)

    def test_fixed_checkpoint_digest_must_match(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            path.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                resolve_checkpoint_spec(dict(path="checkpoint.pt", completed_games=12, sha256="bad"),
                                        Path(folder))

    def test_actual_checkpoint_lineage_and_games_are_checked(self):
        state = {"runtime_state": {"games": {"completed_games": 660043, "lineage_id": "main"}},
                 "train_cfg": {"ppo": {"advantage_normalization": "raw", "return_perspective": "same_seat"},
                               "net": {"identity_encoding": "known_only", "use_global_summary": False}},
                 "num_rollout": 10653}
        spec = {"completed_games": 660043}
        self.assertEqual(checkpoint_provenance(state, spec, "main", "commit")["rollout"], 10653)
        with self.assertRaisesRegex(ValueError, "lineage"):
            checkpoint_provenance(state, spec, "another-seed", "commit")
        with self.assertRaisesRegex(ValueError, "completed_games"):
            checkpoint_provenance(state, {"completed_games": 660000}, "main", "commit")
        state["train_cfg"]["ppo"]["return_perspective"] = "next_actor"
        with self.assertRaisesRegex(ValueError, "Expected raw"):
            checkpoint_provenance(state, spec, "main", "commit")

    def records(self):
        return [dict(game_id=i, scenario_pair_id=i, first_team=t, setup_seed=100+i,
                     random_seed=200+i, setup_source="uniform_seeded_game", opponent="policy",
                     backend="gpu", setup_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                     outcome="win" if t == 0 else "loss", termination_reason="team_victory", moves=500)
                for t in (0, 1) for i in (0, 1)]

    def check(self, records, **overrides):
        metrics = {"h2h/wins": 2, "h2h/losses": 2, "h2h/draws": 0, "h2h/ongoing": 0,
                   "h2h/num_games": 4, "h2h/requested_games": 4}
        metrics.update(overrides)
        return validate_records(records, metrics, games=4, setup_seed=100, game_seed=200, max_moves=4000)

    def test_complete_pair_records_and_counts(self):
        self.assertEqual(len(self.check(self.records())), 2)
        with self.assertRaisesRegex(ValueError, "metrics differ"):
            self.check(self.records(), **{"h2h/wins": 3})

    def test_unfinished_duplicate_and_changed_setup_rejected(self):
        changes = [("outcome", "ongoing"), ("random_seed", 999),
                   ("setup_sha256", "a"*64), ("termination_reason", "evaluation_step_cap")]
        for field, value in changes:
            rows = copy.deepcopy(self.records())
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(rows)
        rows = self.records()
        rows[2] = rows[0].copy()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self.check(rows)


if __name__ == "__main__":
    unittest.main()
