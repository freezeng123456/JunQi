"""Paired GPU matches between frozen stages of one training lineage.

Measures move policies on common uniform-random setups, not ArrangementNet
benefits. A checkpoint entry has either ``path`` or ``completion``; completion
entries also supply ``run_root`` and ``training_result``. Only models selected
by --pairs are resolved, so a later final checkpoint can remain unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_pairs(values, checkpoints):
    pairs, seen = [], set()
    for value in values:
        for item in value.split(","):
            parts = item.split(":")
            if len(parts) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_-]+", p) for p in parts):
                raise ValueError(f"Invalid pair {item!r}; use first:second")
            first, second = parts
            if first == second or first not in checkpoints or second not in checkpoints:
                raise ValueError(f"Unknown checkpoint or self-match: {item}")
            key = frozenset(parts)
            if key in seen:
                raise ValueError(f"Repeated unordered pair: {item}")
            seen.add(key)
            pairs.append((first, second))
    if not pairs:
        raise ValueError("At least one pair is required")
    return pairs


def absolute_path(value, base):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_checkpoint_spec(entry, manifest_directory):
    """Resolve a completed job once; never wait for or poll a future artifact."""
    if ("path" in entry) == ("completion" in entry):
        raise ValueError("Each checkpoint needs exactly one of path or completion")
    result = dict(entry)
    if "completion" in entry:
        completion_path = absolute_path(entry["completion"], manifest_directory)
        completion = json.loads(completion_path.read_text())
        if completion.get("status") != "COMPLETE":
            raise ValueError("Final training completion is not COMPLETE")
        result_path = absolute_path(entry["training_result"], manifest_directory)
        training = json.loads(result_path.read_text())
        completed = completion.get("completed_games")
        if (type(completed) is not int or completed <= 0
                or training.get("status") != "target_reached"
                or training.get("completed_games") != completed
                or completed < completion.get("target_completed_games", completed)):
            raise ValueError("Final training completion and result disagree")
        expected_result_hash = completion.get("sha256", {}).get("training_result.json")
        if not expected_result_hash or sha256(result_path) != expected_result_hash:
            raise ValueError("Final training result hash mismatch or missing receipt")
        if "completed_games" in entry and entry["completed_games"] != completed:
            raise ValueError("Final completed games differ from manifest")
        root = absolute_path(entry["run_root"], manifest_directory)
        path = absolute_path(completion["final_checkpoint"], root)
        result.update(completed_games=completed, completion=str(completion_path),
                      completion_sha256=sha256(completion_path),
                      training_result=str(result_path), training_result_sha256=expected_result_hash,
                      expected_lineage_id=training["lineage_id"])
    else:
        path = absolute_path(entry["path"], manifest_directory)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is not available: {path}")
    if type(result.get("completed_games")) is not int or result["completed_games"] <= 0:
        raise ValueError("Checkpoint requires a positive expected completed_games count")
    result["path"] = str(path)
    result["sha256"] = sha256(path)
    if "sha256" in entry and result["sha256"] != entry["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    return result


def checkpoint_provenance(state, spec, lineage_id, source_commit):
    """Compare actual counters/config with declared checkpoint provenance."""
    games = state["runtime_state"]["games"]
    if games["completed_games"] != spec["completed_games"]:
        raise ValueError("Actual checkpoint completed_games differ from manifest/result")
    if (games["lineage_id"] != lineage_id
            or spec.get("expected_lineage_id", lineage_id) != lineage_id):
        raise ValueError("Checkpoints do not belong to the requested training lineage")
    cfg = state["train_cfg"]
    if (cfg["ppo"].get("advantage_normalization") != "raw"
            or cfg["ppo"].get("return_perspective") != "same_seat"
            or cfg["net"].get("identity_encoding") != "known_only"
            or cfg["net"].get("use_global_summary") is not False
            or cfg.get("belief", {}).get("enabled", False)):
        raise ValueError("Expected raw-advantage, same-seat, facts-only policy without BeliefNet")
    return dict(**spec, lineage_id=lineage_id, training_source_commit=source_commit,
                training_source_provenance="bound experiment manifest",
                rollout=state.get("num_rollout"), train_cfg=cfg,
                checkpoint_meta=state.get("checkpoint_meta"))


def validate_records(records, metrics, *, games, setup_seed, game_seed, max_moves):
    """Reject incomplete pairs, changed scenarios, unfinished games or wrong totals."""
    if len(records) != games or games <= 0 or games % 2:
        raise ValueError("Paired evaluation requires the exact even game budget")
    pairs, layout_ids = {}, set()
    for record in records:
        game, team = record["game_id"], record["first_team"]
        if (type(game) is not int or not 0 <= game < games // 2
                or type(team) is not int or team not in (0, 1)
                or team in pairs.setdefault(game, {})):
            raise ValueError("Invalid or duplicated paired game identity")
        if (record["scenario_pair_id"] != game
                or record["setup_seed"] != setup_seed + game
                or record["random_seed"] != game_seed + game
                or record["setup_source"] != "uniform_seeded_game"
                or record["opponent"] != "policy" or record["backend"] != "gpu"):
            raise ValueError("Game provenance differs from the bound protocol")
        if record["outcome"] not in ("win", "loss", "draw"):
            raise ValueError("Unfinished evaluation game cannot be counted")
        if (record["termination_reason"] not in ("team_victory", "total_move_limit", "no_combat_limit")
                or not 0 < record["moves"] <= max_moves):
            raise ValueError("Invalid termination reason or game length")
        if ((record["outcome"] == "draw")
                != (record["termination_reason"] in ("total_move_limit", "no_combat_limit"))):
            raise ValueError("Outcome and termination reason disagree")
        if not re.fullmatch(r"[0-9a-f]{64}", record["setup_sha256"]):
            raise ValueError("Invalid setup digest")
        pairs[game][team] = record
    for pair in pairs.values():
        if set(pair) != {0, 1} or pair[0]["setup_sha256"] != pair[1]["setup_sha256"]:
            raise ValueError("Paired games must share the same setup and exchange teams")
        layout_ids.add(pair[0]["setup_sha256"])
    if len(layout_ids) != games // 2:
        raise ValueError("Evaluation contains duplicate layouts")
    counts = Counter(record["outcome"] for record in records)
    expected = {"wins": counts["win"], "losses": counts["loss"], "draws": counts["draw"],
                "ongoing": 0, "num_games": games, "requested_games": games}
    if any(metrics.get("h2h/" + key) != value for key, value in expected.items()):
        raise ValueError("Aggregate evaluator metrics differ from actual records")
    return layout_ids


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=root)
    return dict(commit=commit, tracked_diff_sha256=hashlib.sha256(diff).hexdigest(),
                tracked_worktree_dirty=bool(diff), evaluator_sha256=sha256(__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--setup-seed", type=int, default=202611040)
    parser.add_argument("--game-seed", type=int, default=202611041)
    parser.add_argument("--bootstrap-seed", type=int, default=202611042)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    if args.games < 4 or args.games % 2 or args.num_envs <= 0 or args.bootstrap_samples < 100:
        parser.error("Need even games >=4, positive concurrency and >=100 bootstrap samples")
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {out}")
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    pairs = parse_pairs(args.pairs, manifest["checkpoints"])
    names = list(dict.fromkeys(name for pair in pairs for name in pair))
    specs = {name: resolve_checkpoint_spec(manifest["checkpoints"][name], manifest_path.parent)
             for name in names}
    # Must be set before importing torch / creating a CUDA context.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from junqi_rl.analysis.protocol import paired_score_interval, write_game_records
    from junqi_rl.analysis.random_eval import WEIGHTED_SAMPLING_VERSION, evaluate_paired_head_to_head
    from junqi_rl.training.checkpoint import load_evaluation_policy

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This protocol requires the assigned V100 CUDA device")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.game_seed)
    np.random.seed(args.game_seed)
    torch.manual_seed(args.game_seed)
    torch.cuda.manual_seed_all(args.game_seed)
    policies, resolved = {}, {}
    for name, spec in specs.items():
        state = torch.load(spec["path"], map_location="cpu", weights_only=False)
        resolved[name] = checkpoint_provenance(state, spec, manifest["lineage_id"],
                                               manifest["source_commit"])
        weights = state["policy"]
        if not weights or any(not torch.isfinite(tensor).all().item() for tensor in weights.values()):
            raise ValueError(f"Nonfinite policy tensors in {name}")
        resolved[name]["finite_policy_tensors"] = len(weights)
        del weights, state
        policies[name] = load_evaluation_policy(spec["path"], device="cuda")
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Checkpoint changed during loading: {name}")
    protocol = dict(schema="junqi.checkpoint-stage-matches.v1", manifest_sha256=sha256(manifest_path),
                    evaluator_source=source_provenance(), checkpoints=resolved, matchups=pairs,
                    games_per_matchup=args.games, num_envs=args.num_envs, max_moves=4000,
                    setup_seed=args.setup_seed, game_seed=args.game_seed,
                    bootstrap_seed=args.bootstrap_seed, bootstrap_samples=args.bootstrap_samples,
                    setup_source="uniform_seeded_game", team_assignments=[0, 1],
                    greedy=False, dtype="float32", deterministic_algorithms=True,
                    policy_sampling=WEIGHTED_SAMPLING_VERSION,
                    scope="move_policy_only_on_common_random_setups; excludes ArrangementNet benefits",
                    training_updated=False, torch_version=str(torch.__version__),
                    cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name())
    out.mkdir(parents=True)
    write_json(out / "protocol.json", protocol)
    protocol_hash = sha256(out / "protocol.json")
    start = time.monotonic()
    results, shared_layouts = [], None
    for first, second in pairs:
        label = f"{first}_vs_{second}"
        print("MATCH_START", label, args.games, flush=True)
        records = []
        pair_start = time.monotonic()
        with torch.inference_mode():
            metrics = evaluate_paired_head_to_head(
                policies[first], policies[second], num_games=args.games, num_envs=args.num_envs,
                device="cuda", seed=args.game_seed, setup_seed=args.setup_seed, max_moves=4000,
                autocast_dtype=torch.float32, greedy=False, game_records=records)
        # Preserve failed/incomplete evidence before validating it.
        for record in records:
            record.update(first_checkpoint=first, second_checkpoint=second,
                          first_checkpoint_sha256=specs[first]["sha256"],
                          second_checkpoint_sha256=specs[second]["sha256"],
                          protocol_sha256=protocol_hash)
        write_game_records(out / f"{label}.jsonl", records)
        layouts = validate_records(records, metrics, games=args.games, setup_seed=args.setup_seed,
                                   game_seed=args.game_seed, max_moves=4000)
        if any(r["policy_action_sampling"] != WEIGHTED_SAMPLING_VERSION
               or r["opponent_action_sampling"] != WEIGHTED_SAMPLING_VERSION for r in records):
            raise ValueError("Actual move sampling differs from protocol")
        if shared_layouts is not None and layouts != shared_layouts:
            raise ValueError("Matchups do not share the same fixed layouts")
        shared_layouts = layouts
        interval = paired_score_interval(records, seed=args.bootstrap_seed,
                                         samples=args.bootstrap_samples)
        counts = Counter(r["outcome"] for r in records)
        result = dict(matchup=label, first_checkpoint=first, second_checkpoint=second,
                      first_completed_games=resolved[first]["completed_games"],
                      second_completed_games=resolved[second]["completed_games"],
                      wins=counts["win"], losses=counts["loss"], draws=counts["draw"],
                      win_rate=counts["win"] / args.games, **interval,
                      termination_reasons=dict(Counter(r["termination_reason"] for r in records)),
                      elapsed_s=time.monotonic() - pair_start,
                      records_sha256=sha256(out / f"{label}.jsonl"), protocol_sha256=protocol_hash)
        write_json(out / f"{label}.json", result)
        results.append(result)
        print("MATCH_COMPLETE", json.dumps(result), flush=True)
    for name, spec in specs.items():
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError(f"Checkpoint changed during evaluation: {name}")
    report = dict(status="COMPLETE", finished_at=datetime.now(timezone.utc).isoformat(),
                  protocol_sha256=protocol_hash, scope=protocol["scope"],
                  completed_evaluation_games=sum(r["games"] for r in results),
                  checkpoints_unchanged=True, elapsed_s=time.monotonic() - start, matchups=results)
    write_json(out / "result.json", report)
    print("EVALUATION_COMPLETE", report["completed_evaluation_games"], flush=True)


if __name__ == "__main__":
    main()
