#!/usr/bin/env python3
"""Freeze a completed million-game run; no polling, training, or overwrite.

Only freeze-receipt.json marks a successful archive. A failed archive keeps its
INCOMPLETE marker and files for diagnosis, and must not be reused as an output.
This verifies byte integrity and counts; model/playing-strength audits are separate.
"""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contained(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Expected a relative path without traversal: {relative}")
    path = (root / relative).resolve()
    require(root == path or root in path.parents, f"Path escapes run root: {relative}")
    return path


def archive(run_root, experiment, output, expected_source_commit=None):
    run_root, output = Path(run_root), Path(output)
    require(run_root.is_absolute() and output.is_absolute(),
            "--run-root and --output must be absolute")
    run_root, output = run_root.resolve(), output.resolve()
    require(not output.exists(), f"Refusing to overwrite output: {output}")
    exp = contained(run_root, experiment)
    control = exp / "control"
    require(output != exp and output != control and control not in output.parents,
            "Output must be separate from the training control directory")
    completion_path = exp / "control.completion.json"
    result_path = control / "training_result.json"
    completion = json.loads(completion_path.read_text())
    result = json.loads(result_path.read_text())
    require(completion.get("status") == "COMPLETE", "Completion status is not COMPLETE")
    require(result.get("status") == "target_reached", "Training target has not been reached")
    games = result.get("completed_games")
    target = result.get("target_completed_games")
    require(isinstance(games, int) and not isinstance(games, bool)
            and isinstance(target, int) and not isinstance(target, bool)
            and games >= target >= 1_000_000, "Expected at least one million completed games")
    require(completion.get("completed_games") == games
            and completion.get("target_completed_games") == target,
            "Completion and training result counts disagree")
    checkpoint = contained(run_root, completion["final_checkpoint"])
    require(checkpoint.is_file() and control.resolve() == checkpoint.parent,
            "Final checkpoint must exist in the training control directory")
    commit = subprocess.check_output(
        ["git", "-C", str(run_root), "rev-parse", "HEAD"], text=True).strip()
    if expected_source_commit is not None:
        require(commit == expected_source_commit, "Current source commit does not match expected source")
    accepted_commit = (exp / "acceptance_control" / "source_commit.txt").read_text().strip()
    require(accepted_commit == commit, "Current source differs from accepted training source")
    expected = completion.get("sha256")
    require(isinstance(expected, dict) and expected, "Completion has no file checksums")
    milestones = sorted(control.glob("first_move_*.json"))
    require(milestones, "No opening milestones exist")
    required = {"training_games.jsonl", "game_progress.jsonl", "checkpoint_games.jsonl",
                "training_result.json"} | {p.name for p in milestones}
    require(required <= expected.keys(), "Completion lacks required artifact checksums")
    for name, digest in expected.items():
        require(Path(name).name == name and name not in {".", ".."},
                f"Invalid completion filename: {name}")
        require(isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), f"Invalid SHA-256: {name}")
        require(sha256(control / name) == digest, f"Completion checksum mismatch: {name}")
    final_metric = json.loads((control / f"first_move_{target:06d}.json").read_text())
    require(final_metric.get("completed_games") == games
            and contained(run_root, final_metric["checkpoint"]) == checkpoint,
            "Final milestone does not match final checkpoint and game count")
    require((control / "cfg.runtime.yaml").is_file(), "Resolved runtime config is missing")

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / "INCOMPLETE"
    incomplete.write_text("Archive is not valid until freeze-receipt.json exists.\n")
    hashes = {}
    # The raw game record is stored only as gzip; all other completion files are copied.
    for name, digest in expected.items():
        if name == "training_games.jsonl":
            continue
        shutil.copy2(control / name, output / name)
        hashes[name] = sha256(output / name)
        require(hashes[name] == digest, f"Artifact changed while copying: {name}")
    for source, name in [(control / "cfg.runtime.yaml", "cfg.runtime.yaml"),
                         (completion_path, "control.completion.json")]:
        shutil.copy2(source, output / name)
        hashes[name] = sha256(output / name)
    (output / "source_commit.txt").write_text(commit + "\n")
    hashes["source_commit.txt"] = sha256(output / "source_commit.txt")
    raw_hash = hashlib.sha256()
    records = 0
    compressed = output / "training_games.jsonl.gz"
    with (control / "training_games.jsonl").open("rb") as source, compressed.open("wb") as dest:
        with gzip.GzipFile(filename="", mode="wb", fileobj=dest, compresslevel=1, mtime=0) as zipped:
            for line in source:
                require(line.endswith(b"\n") and line.strip(), "Invalid or truncated game-record line")
                zipped.write(line)
                raw_hash.update(line)
                records += 1
    require(records == games, f"Game record count mismatch: {records} != {games}")
    require(raw_hash.hexdigest() == expected["training_games.jsonl"],
            "Game records changed during compression")
    hashes[compressed.name] = sha256(compressed)
    checkpoint_hash = sha256(checkpoint)
    shutil.copy2(checkpoint, output / checkpoint.name)
    hashes[checkpoint.name] = sha256(output / checkpoint.name)
    require(hashes[checkpoint.name] == checkpoint_hash, "Checkpoint changed while copying")
    require(json.loads((output / "control.completion.json").read_text()) == completion,
            "Completion record changed during archiving")
    receipt = dict(
        status="COMPLETE", archived_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=commit, run_root=str(run_root), experiment=str(experiment),
        completed_games=games, target_completed_games=target, game_record_count=records,
        raw_games_sha256=raw_hash.hexdigest(), gzip_sha256=hashes[compressed.name],
        checkpoint=checkpoint.name, checkpoint_sha256=checkpoint_hash,
        checkpoint_bytes=(output / checkpoint.name).stat().st_size,
        verified_completion_sha256=expected, files={
            name: dict(bytes=(output / name).stat().st_size, sha256=digest)
            for name, digest in hashes.items()})
    (output / "freeze-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    incomplete.unlink()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args()
    receipt = archive(args.run_root, args.experiment, args.output, args.expected_source_commit)
    print(json.dumps({k: receipt[k] for k in ["status", "completed_games", "game_record_count",
                                             "checkpoint", "checkpoint_sha256"]}))


if __name__ == "__main__":
    main()
