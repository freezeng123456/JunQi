#!/usr/bin/env python3
"""Read-only local audit of a fully recovered facts BeliefNet experiment.

Use the project's PyTorch environment. This verifies frozen checkpoints,
selection/data lineage, every dataset file hash, the held-out test record, and
complete paired matches. It recomputes W/L/D and layout-bootstrap intervals;
it does not replay matches, repeat GPU execution, or recompute neural NLL.
Missing/incomplete artifacts fail closed. --output writes only a new audit.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, message):
    _require(actual == expected, message)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative_file(root, relative):
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts,
             "artifact reference must remain inside its recovered root")
    resolved = (root / path).resolve()
    _require(resolved.is_relative_to(root.resolve()), "artifact reference escapes its recovered root")
    return resolved


def _conditional_metrics(metrics, rows, *, label):
    _same(metrics.get("rows"), rows, f"{label}: observation row count mismatch")
    count = metrics.get("unknown_targets")
    _require(type(count) is int and 0 < count <= rows * 75, f"{label}: invalid unknown-target count")
    nll, accuracy = metrics.get("conditional_nll_per_unknown_piece"), metrics.get("teacher_forced_accuracy")
    _require(isinstance(nll, (int, float)) and math.isfinite(nll) and nll >= 0,
             f"{label}: NLL must be finite and nonnegative")
    _require(isinstance(accuracy, (int, float)) and math.isfinite(accuracy) and 0 <= accuracy <= 1,
             f"{label}: accuracy must lie in [0, 1]")


def audit(args):
    report = dict(schema="junqi.facts-belief-recovery-audit.v1", status="failed", checks=[],
                  artifacts=[], errors=[], limitations=[
                      "Checks recovered artifacts and recomputes match statistics; does not rerun GPU kernels or matches.",
                      "Checks held-out neural metrics and their denominator; does not recompute neural NLL or accuracy.",
                      "Gameplay compares frozen policy against the same policy plus neural beliefs and search."])
    phase = "required_artifacts"
    protected = set()
    hashes = {}

    def check_file(path, expected=None, *, role):
        from junqi_rl.belief.facts_dataset import file_sha256
        path = Path(path).resolve()
        protected.add(path)
        _require(path.is_file(), f"missing {role}: {path}")
        if path not in hashes:
            hashes[path] = file_sha256(path)
            report["artifacts"].append(dict(role=role, path=str(path), sha256=hashes[path], bytes=path.stat().st_size))
        if expected is not None:
            _same(hashes[path], expected.lower(), f"SHA-256 mismatch for {role}: {path}")
        return hashes[path]

    try:
        # Optional CUDA-extension import diagnostics belong on stderr: this
        # local CPU audit keeps stdout a single machine-readable JSON record.
        with redirect_stdout(sys.stderr):
            import torch
            from junqi_core.rules import MAX_NUM_MOVES
            from junqi_rl.analysis.facts_joint_search import SearchConfig
            from junqi_rl.belief.facts_dataset import iter_shards
            from junqi_rl.belief.facts_joint_checkpoint import load_facts_joint_belief
            from scripts.compare_facts_joint import paired_statistics, validated_games
            from scripts.train_facts_joint_belief import _validate_manifest, unknown_targets

        torch.set_num_threads(1)
        training, dataset, comparison = args.training_dir, args.dataset, args.comparison_dir
        joint_path, last_path = training / "joint-belief.pt", training / "last.pt"
        report_path, training_protocol_path = training / "report.json", training / "training-protocol.json"
        test_path = args.test_audit or training / "test-audit.json"
        parent_hash = check_file(args.policy, role="frozen_parent_policy")
        joint_hash = check_file(joint_path, role="validation_selected_belief")
        last_hash = check_file(last_path, role="last_training_checkpoint")
        for path, role in ((report_path, "training_report"), (training_protocol_path, "training_protocol"),
                           (test_path, "held_out_test_audit"),
                           (dataset / "dataset-manifest.json", "dataset_manifest"),
                           (dataset / "protocol.json", "dataset_protocol"),
                           (comparison / "protocol.json", "match_protocol"),
                           (comparison / "comparison.json", "match_aggregate")):
            check_file(path, role=role)
        report["checks"].append("required_artifacts_present")

        phase = "checkpoint_selection_and_data_lineage"
        joint = torch.load(joint_path, map_location="cpu", weights_only=False)
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        training_report, contract = _json(report_path), joint.get("training_contract")
        _require(isinstance(contract, dict), "selected checkpoint lacks a training contract")
        _same(joint.get("artifact_role"), "validation-selected-facts-joint-belief", "wrong selected artifact role")
        for payload in (joint, last):
            _same(payload.get("parent_policy_sha256"), parent_hash, "checkpoint parent policy mismatch")
            _same(payload.get("training_contract"), contract, "checkpoint training contracts disagree")
        _same(_json(training_protocol_path), contract, "training-protocol.json disagrees with checkpoints")
        _same(training_report.get("training_contract"), contract, "training report contract mismatch")
        _same(contract.get("parent_policy_sha256"), parent_hash, "training contract parent mismatch")
        _same(contract.get("model_config"), joint.get("config"), "training/model architecture mismatch")
        _require(contract.get("test_fold_used") is False and training_report.get("test_fold_used") is False,
                 "training/selection used the reserved test fold")
        _require(training_report.get("finished") is True, "training report is incomplete")
        epochs = training_report.get("target_epochs")
        _require(type(epochs) is int and epochs > 0, "invalid target epoch count")
        _same(training_report.get("epochs_completed"), epochs, "training stopped before its target epoch")
        _same(last.get("epochs_completed"), epochs, "last checkpoint epoch mismatch")
        _same(joint.get("epochs_completed"), epochs, "selected export came from a different training run")
        _same(training_report.get("last_checkpoint_sha256"), last_hash, "last checkpoint hash mismatch")
        _same(training_report.get("selected_checkpoint_sha256"), joint_hash, "selected checkpoint hash mismatch")
        _same(last.get("resume_schema"), 1, "last checkpoint lacks its resume contract")
        _require(last.get("optimizer") and last.get("rng"), "last checkpoint is missing optimizer/RNG state")
        manifest, manifest_hash = _validate_manifest(SimpleNamespace(
            dataset=str(dataset), games=contract["games"], positions_per_game=contract["positions_per_game"]), parent_hash)
        _same(contract.get("dataset_manifest_sha256"), manifest_hash, "selected checkpoint dataset mismatch")
        _same(contract.get("dataset_protocol_sha256"), manifest["protocol_sha256"], "dataset protocol hash mismatch")
        _same(_json(dataset / "protocol.json"), manifest["protocol"], "dataset protocol file/manifest disagreement")
        _require(manifest["protocol"].get("acceptance_only") is False,
                 "a bounded smoke dataset cannot serve as the final training experiment")
        for field in ("split_games", "split_rows"):
            _same(training_report.get(field), manifest[field], f"training report {field} mismatch")
        _same(training_report.get("completed_collection_games"), manifest["completed_games"],
              "training report completed-game count mismatch")
        metrics = joint.get("epoch_metrics", [])
        _same(len(metrics), epochs, "checkpoint lacks a metric record for every completed epoch")
        _same(last.get("epoch_metrics"), metrics, "last/selected epoch records disagree")
        _same(training_report.get("epoch_metrics"), metrics, "training report epoch records disagree")
        for index, item in enumerate(metrics, start=1):
            _same(item.get("epoch"), index, "epoch records are not ordered and contiguous")
            _conditional_metrics(item["validation"], manifest["split_rows"]["validation"], label=f"epoch {index}")
        selected = min(range(epochs), key=lambda index: metrics[index]["validation"]["conditional_nll_per_unknown_piece"]) + 1
        for payload, field in ((joint, "selected_epoch"), (joint, "best_epoch"), (last, "best_epoch"),
                               (training_report, "best_epoch")):
            _same(payload.get(field), selected, "selected epoch violates earliest minimum validation NLL")
        _same(training_report.get("best_validation"), metrics[selected - 1]["validation"], "best validation record mismatch")
        _same(training_report.get("final_validation"), metrics[-1]["validation"], "final validation record mismatch")
        _same(set(joint["net"]), set(last["best_model_state"]), "selected and saved best weight keys differ")
        for key, value in joint["net"].items():
            torch.testing.assert_close(value, last["best_model_state"][key], rtol=0, atol=0)
        # Uses the existing strict architecture, input-schema, finite-weight,
        # and parent-policy loader. No policy sampling or training occurs.
        load_facts_joint_belief(joint_path, policy_path=args.policy, device="cpu")
        report["checks"].append("strict_weights_parent_data_and_validation_selection")
        report["selection"] = dict(epochs_completed=epochs, selected_epoch=selected,
                                    validation=metrics[selected - 1]["validation"])

        phase = "dataset_artifact_hashes"
        for family in ("raw_games", "shards"):
            for item in manifest[family]:
                check_file(_relative_file(dataset, item["path"]), item["sha256"], role=family)
        report["checks"].append("all_raw_and_supervised_dataset_sha256")

        phase = "held_out_test_audit"
        test_audit = _json(test_path)
        test_protocol = test_audit.get("protocol", {})
        for key, value in dict(schema="junqi.facts-joint-final-test.v1", joint_checkpoint_sha256=joint_hash,
                               parent_policy_sha256=parent_hash, dataset_manifest_sha256=manifest_hash,
                               selected_epoch=selected, split="test", weights_updated=False,
                               selection_performed_on_test=False).items():
            _same(test_protocol.get(key), value, f"held-out test {key} mismatch")
        _require(test_audit.get("test_fold_used") is True, "test audit did not evaluate the held-out fold")
        _same(test_audit.get("test_completed_games"), manifest["split_games"]["test"], "test game denominator mismatch")
        _same(test_audit.get("validation_at_selection"), metrics[selected - 1]["validation"], "test audit selection mismatch")
        _conditional_metrics(test_audit["test"], manifest["split_rows"]["test"], label="held-out test")
        unknown = sum(int(unknown_targets(shard).sum()) for shard in iter_shards(dataset, "test"))
        _same(test_audit["test"]["unknown_targets"], unknown, "test unknown-target denominator mismatch")
        report["test"] = test_audit["test"]
        report["checks"].append("held_out_test_binding_and_denominator")

        phase = "complete_paired_match_statistics"
        protocol = _json(comparison / "protocol.json")
        for key, value in dict(schema="junqi.facts-joint-comparison.v1", policy_sha256=parent_hash,
                               joint_sha256=joint_hash, dataset_manifest_sha256=manifest_hash,
                               team_assignments=[0, 1], deterministic_algorithms=True,
                               max_moves=MAX_NUM_MOVES,
                               baseline="frozen facts policy",
                               candidate="same frozen policy + facts joint belief + search",
                               action_selection="sample distribution").items():
            _same(protocol.get(key), value, f"match protocol {key} mismatch")
        _require(type(protocol.get("pairs")) is int and protocol["pairs"] >= 2,
                 "at least two complete layout pairs are required for bootstrap intervals")
        search_config = asdict(SearchConfig(**protocol["search_config"]))
        _same(protocol["search_config"], search_config, "incomplete search config")
        check_file(args.comparison_source, protocol["comparison_script_sha256"], role="frozen_comparison_source")
        digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        games = validated_games(comparison, protocol, digest)
        stats = paired_statistics(games, seed=protocol["seed"])
        _same(stats["complete_pairs"], protocol["pairs"], "requested layout pairs are incomplete")
        _same(stats["completed_games"], protocol["pairs"] * 2, "requested match game count is incomplete")
        _same(stats["unpaired_completed_games"], 0, "unpaired games remain")
        forbidden = {row["initial_setup_sha256"] for row in manifest["raw_games"]}
        _same(protocol.get("forbidden_layout_count"), len(forbidden), "held-out match layout count mismatch")
        for game in games:
            _require(game["setup_sha256"] not in forbidden, "match opening overlaps the belief dataset")
            _require(type(game.get("plies")) is int and 0 < game["plies"] <= protocol["max_moves"], "invalid complete-game move count")
            search = game["search_metrics"]
            _same(search.get("config"), search_config, "per-game search config mismatch")
            for key in ("sample_budget_failures", "successful_decisions", "decisions",
                        "candidate_actions", "legal_actions"):
                _require(type(search.get(key)) is int and search[key] >= 0,
                         f"invalid per-game search metric: {key}")
            _same(search.get("sample_budget_failures"), 0, "completed game includes failed search decisions")
            _same(search.get("successful_decisions"), search.get("decisions"), "search silently failed or fell back")
            _same(search.get("candidate_actions"), search.get("legal_actions"), "search did not cover every legal action")
            _require(search.get("decisions", 0) > 0, "candidate never made a search decision")
            check_file(comparison / f"game-{game['pair']:06d}-{game['candidate_team']}.json", role="completed_match")
        aggregate = _json(comparison / "comparison.json")
        _same(aggregate.get("protocol"), protocol, "aggregate match protocol mismatch")
        _same(aggregate.get("protocol_sha256"), digest, "aggregate protocol digest mismatch")
        _require(aggregate.get("finished") is True, "aggregate comparison is not marked complete")
        _same(aggregate.get("games"), games, "aggregate games differ from durable per-game records")
        _same(aggregate.get("statistics"), stats, "reported W/L/D or paired-bootstrap intervals differ from recomputation")
        report["comparison"] = stats
        report["checks"].append("complete_held_out_pairs_and_recomputed_wld_bootstrap")

        phase = "external_sha256_manifests"
        for checksum_path in args.sha256_file:
            check_file(checksum_path, role="sha256sum_manifest")
            for line in checksum_path.read_text().splitlines():
                if not line.strip():
                    continue
                match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
                _require(match is not None, f"unsupported sha256sum line in {checksum_path}")
                digest, relative = match.groups()
                check_file(_relative_file(checksum_path.parent, relative), digest, role="sha256sum_artifact")
        if args.archive:
            check_file(args.archive, args.archive_sha256, role="recovered_archive_remote_digest")
        report["external_sha256_evidence_supplied"] = bool(args.sha256_file or args.archive)
        if args.sha256_file or args.archive:
            report["checks"].append("supplied_external_sha256_evidence")
        report["status"] = "passed"
    except Exception as error:
        report["failed_phase"] = phase
        report["errors"].append(f"{type(error).__name__}: {error}")
    report["artifacts_checked"] = len(hashes)
    return report, protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--test-audit", type=Path)
    parser.add_argument("--comparison-source", type=Path, default=_ROOT / "scripts/compare_facts_joint.py")
    parser.add_argument("--sha256-file", type=Path, action="append", default=[], help="Optional sha256sum-format evidence")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256", help="Previously observed remote SHA-256 of --archive")
    parser.add_argument("--output", type=Path, help="New local audit JSON; defaults to stdout only")
    args = parser.parse_args(argv)
    if bool(args.archive) != bool(args.archive_sha256):
        parser.error("--archive and --archive-sha256 must be supplied together")
    if args.archive_sha256 and not re.fullmatch("[0-9a-fA-F]{64}", args.archive_sha256):
        parser.error("--archive-sha256 must contain exactly 64 hexadecimal characters")
    if args.output and args.output.exists():
        parser.error("--output must be a new audit file; recovered evidence is read-only")
    report, protected = audit(args)
    if args.output:
        if args.output.resolve() in protected:
            parser.error("--output must not overwrite a recovered artifact")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
