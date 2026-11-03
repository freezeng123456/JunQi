"""Standard-library tests: run with python -m unittest tests.test_archive_completed_run."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "archive_completed_run", Path(__file__).parents[1] / "scripts/archive_completed_run.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArchiveCompletedRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        subprocess.run(["git", "-C", str(self.root), "-c", "user.name=Archive Test",
                        "-c", "user.email=archive-test@example.invalid", "commit",
                        "--allow-empty", "-qm", "fixture"], check=True)
        self.commit = subprocess.check_output(
            ["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        self.exp = self.root / "exps/test"
        self.control = self.exp / "control"
        self.control.mkdir(parents=True)
        self.output = Path(self.temp.name) / "archive"
        acceptance = self.exp / "acceptance_control"
        acceptance.mkdir()
        (acceptance / "source_commit.txt").write_text(self.commit + "\n")
        self.checkpoint = self.control / "ckpt_001000.pt"
        self.checkpoint.write_bytes(b"checkpoint-byte-integrity-fixture")
        (self.control / "cfg.runtime.yaml").write_text("target_completed_games: 1000000\n")
        self.raw = b'{"game_id":1}\n' * 1_000_000
        (self.control / "training_games.jsonl").write_bytes(self.raw)
        (self.control / "game_progress.jsonl").write_text('{"completed_games":1000000}\n')
        (self.control / "checkpoint_games.jsonl").write_text('{"completed_games":1000000}\n')
        self.result = dict(status="target_reached", completed_games=1_000_000,
                           target_completed_games=1_000_000)
        (self.control / "training_result.json").write_text(json.dumps(self.result))
        self.checkpoint_name = "exps/test/control/ckpt_001000.pt"
        (self.control / "first_move_1000000.json").write_text(json.dumps(
            dict(completed_games=1_000_000, checkpoint=self.checkpoint_name)))
        self.completion = dict(status="COMPLETE", completed_games=1_000_000,
                               target_completed_games=1_000_000,
                               final_checkpoint=self.checkpoint_name,
                               sha256={p.name: MODULE.sha256(p) for p in self.control.glob("*.json*")})
        self.save_completion()

    def save_completion(self):
        (self.exp / "control.completion.json").write_text(json.dumps(self.completion))

    def run_archive(self, **kwargs):
        return MODULE.archive(self.root, "exps/test", self.output, **kwargs)

    def test_completed_archive_has_verified_payload_and_receipt(self):
        receipt = self.run_archive(expected_source_commit=self.commit)
        self.assertEqual(receipt["game_record_count"], 1_000_000)
        self.assertEqual(receipt["raw_games_sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(gzip.decompress((self.output / "training_games.jsonl.gz").read_bytes()), self.raw)
        self.assertEqual(receipt["checkpoint_sha256"], MODULE.sha256(self.checkpoint))
        for name, entry in receipt["files"].items():
            self.assertEqual(entry["sha256"], MODULE.sha256(self.output / name))
        self.assertFalse((self.output / "INCOMPLETE").exists())
        self.assertEqual(json.loads((self.output / "freeze-receipt.json").read_text()), receipt)

    def test_rejects_unfinished_training_without_output(self):
        self.result["status"] = "interrupted"
        (self.control / "training_result.json").write_text(json.dumps(self.result))
        with self.assertRaisesRegex(ValueError, "not been reached"):
            self.run_archive()
        self.assertFalse(self.output.exists())

    def test_rejects_short_target(self):
        self.result.update(completed_games=20_015, target_completed_games=20_000)
        (self.control / "training_result.json").write_text(json.dumps(self.result))
        with self.assertRaisesRegex(ValueError, "one million"):
            self.run_archive()

    def test_rejects_checkpoint_missing(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "checkpoint must exist"):
            self.run_archive()

    def test_rejects_source_mismatch(self):
        with self.assertRaisesRegex(ValueError, "expected source"):
            self.run_archive(expected_source_commit="0" * 40)

    def test_rejects_any_completion_checksum_mismatch(self):
        (self.control / "game_progress.jsonl").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch: game_progress"):
            self.run_archive()
        self.assertFalse(self.output.exists())

    def test_rejects_missing_checksum_coverage(self):
        del self.completion["sha256"]["first_move_1000000.json"]
        self.save_completion()
        with self.assertRaisesRegex(ValueError, "required artifact checksums"):
            self.run_archive()

    def test_rejects_record_count_even_with_matching_raw_hash(self):
        (self.control / "training_games.jsonl").write_bytes(b'{"game_id":1}\n')
        self.completion["sha256"]["training_games.jsonl"] = MODULE.sha256(
            self.control / "training_games.jsonl")
        self.save_completion()
        with self.assertRaisesRegex(ValueError, "record count mismatch"):
            self.run_archive()
        self.assertTrue((self.output / "INCOMPLETE").exists())
        self.assertFalse((self.output / "freeze-receipt.json").exists())

    def test_refuses_existing_output_without_touching_it(self):
        self.output.mkdir()
        marker = self.output / "keep"
        marker.write_text("untouched")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_archive()
        self.assertEqual(marker.read_text(), "untouched")


if __name__ == "__main__":
    unittest.main()
