"""Complete-game facts datasets; simulator labels never enter model inputs.

Raw traces and labels are separate named arrays. Public histories use numeric
events and offsets, never Python object pickles. This module is CPU-only and
does not import Torch, CUDA, or the historical probability adapters.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.knowledge import ObserverKnowledge, PublicEvent
from junqi_core.observation import ObservationBuilder
from junqi_core.rules import Event, PieceType, Seat, ShowMode
from junqi_core.state import GameState
from junqi_rl.belief.joint import IDENTITY_INPUT_SCHEMA, IdentityConstraintInput, build_joint_input
from junqi_rl.env import JunqiEnv, unrotate_compact_action_id

DATASET_SCHEMA = "junqi.facts-belief-dataset.v1"
TRACE_SCHEMA = "junqi.facts-complete-game.v1"
ARRAY_FIELDS = tuple(field.name for field in fields(IdentityConstraintInput)
                     if field.name not in ("schema", "public_history"))
EVENT_WIDTH = 12
_TRACE_FIELDS = ("actions", "seats", "events", "terminated", "draw", "winner_team")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_sha256(protocol: dict) -> str:
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz(path, arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_npz(path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def split_for_setup(initial_setup_sha256: str) -> str:
    """Assign all games/observers sharing an opening to the same 80/10/10 split."""
    if len(initial_setup_sha256) != 64:
        raise ValueError("initial setup hash must be a SHA-256 digest")
    bucket = int(initial_setup_sha256, 16) % 10
    return "train" if bucket < 8 else "validation" if bucket == 8 else "test"


def prepare_protocol(root, protocol: dict) -> dict:
    root = Path(root)
    if protocol.get("schema") != DATASET_SCHEMA:
        raise ValueError("dataset protocol schema mismatch")
    path = root / "protocol.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != protocol:
            raise ValueError("dataset protocol/checkpoint hash conflicts with existing collection")
    else:
        write_json(path, protocol)
    return protocol


def selected_positions(length: int, count: int, *, seed: int, game_id: int) -> np.ndarray:
    if length <= 0 or count <= 0:
        raise ValueError("complete game length and positions per game must be positive")
    count = min(length, count)
    # One random ply from every temporal stratum, without duplicate positions.
    boundaries = np.linspace(0, length, count + 1, dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence([seed, game_id]))
    return np.asarray([rng.integers(start, end) for start, end in
                       zip(boundaries[:-1], boundaries[1:])], dtype=np.int64)


def pack_events(histories) -> tuple[np.ndarray, np.ndarray]:
    rows, offsets = [], [0]
    for history in histories:
        for event in history:
            rows.append((event.step, event.seat, event.src, event.dst,
                         event.attacker_pid, event.defender_pid, int(event.event),
                         event.flag_reveal_src, event.flag_reveal_dst, event.flag_captured,
                         event.engineer_revealed, sum(1 << seat for seat in event.seats_died)))
        offsets.append(len(rows))
    return (np.asarray(rows, dtype=np.int32).reshape(-1, EVENT_WIDTH),
            np.asarray(offsets, dtype=np.int64))


def _unpack_history(rows) -> tuple[PublicEvent, ...]:
    return tuple(PublicEvent(
        step=int(row[0]), seat=int(row[1]), src=int(row[2]), dst=int(row[3]),
        attacker_pid=int(row[4]), defender_pid=int(row[5]), event=Event(int(row[6])),
        flag_reveal_src=bool(row[7]), flag_reveal_dst=bool(row[8]), flag_captured=bool(row[9]),
        engineer_revealed=bool(row[10]), seats_died=tuple(s for s in range(4) if int(row[11]) & (1 << s)),
    ) for row in rows)


def batch_inputs(shard: dict[str, np.ndarray], indices) -> IdentityConstraintInput:
    """Return public facts only; targets and teacher masks are never read."""
    indices = np.asarray(indices)
    size = len(shard["observer_seats"])
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
        raise ValueError("batch indices must be a nonempty integer vector")
    if (indices < 0).any() or (indices >= size).any():
        raise IndexError("batch index outside shard")
    offsets = shard["event_offsets"]
    if offsets.shape != (size + 1,) or offsets[0] != 0 or offsets[-1] != len(shard["history_events"]):
        raise ValueError("invalid public history offsets")
    if (np.diff(offsets) < 0).any():
        raise ValueError("public history offsets must be monotone")
    arrays = {name: np.array(shard[name][indices], copy=True) for name in ARRAY_FIELDS}
    for array in arrays.values():
        array.setflags(write=False)
    histories = tuple(_unpack_history(shard["history_events"][offsets[i]:offsets[i + 1]]) for i in indices)
    return IdentityConstraintInput(schema=IDENTITY_INPUT_SCHEMA, **arrays, public_history=histories)


def validate_trace(trace: dict, metadata: dict) -> int:
    if metadata.get("schema") != TRACE_SCHEMA or metadata.get("complete") is not True:
        raise ValueError("only complete-game traces may enter the dataset")
    length = len(trace["actions"])
    if length <= 0 or length != metadata["moves"]:
        raise ValueError("trace length does not match the completed game")
    if any(np.asarray(trace[key]).shape != (length,) for key in _TRACE_FIELDS):
        raise ValueError("trace action/event/terminal columns have inconsistent lengths")
    ended = np.flatnonzero(trace["terminated"])
    if ended.tolist() != [length - 1]:
        raise ValueError("trace must end exactly at its first terminal event")
    types = np.asarray(trace["initial_piece_types"])
    if types.shape != (120,) or types.dtype != np.int8:
        raise ValueError("trace needs its initial 120-slot int8 simulator setup")
    if hashlib.sha256(types.tobytes()).hexdigest() != metadata["initial_setup_sha256"]:
        raise ValueError("initial setup hash mismatch")
    if (np.asarray(trace["seats"]) < 0).any() or (np.asarray(trace["seats"]) > 3).any():
        raise ValueError("invalid acting seat")
    return length


def persist_trace(root, trace: dict, metadata: dict, protocol: dict) -> dict:
    """Atomically persist a complete wave lane; an existing lane must match."""
    validate_trace(trace, metadata)
    root = Path(root)
    identity = f"game-{int(metadata['game_id']):07d}"
    path = root / "raw" / (identity + ".npz")
    meta_path = path.with_suffix(".json")
    completed = dict(metadata, protocol_sha256=protocol_sha256(protocol),
                     parent_checkpoint_sha256=protocol["parent_checkpoint_sha256"],
                     split=split_for_setup(metadata["initial_setup_sha256"]),
                     path=str(path.relative_to(root)))
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if any(old.get(key) != value for key, value in completed.items()):
            raise ValueError("attempted to replace a complete game with conflicting provenance")
        if file_sha256(path) != old["sha256"]:
            raise ValueError("persisted trace hash mismatch")
        return old
    write_npz(path, trace)
    completed["sha256"] = file_sha256(path)
    write_json(meta_path, completed)
    return completed


def replay_trace(trace: dict, metadata: dict, *, positions_per_game=8,
                 position_seed=20261003, cache_teacher_masks=True) -> dict[str, np.ndarray]:
    """Validate every GPU event/terminal against CPU rules and build public rows."""
    length = validate_trace(trace, metadata)
    types = trace["initial_piece_types"]
    setup = tuple(tuple(PieceType(int(x)) if x >= 0 else PieceType.NONE for x in army)
                  for army in types.reshape(4, 30))
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False, identity_encoding="known_only",
                   max_num_moves=int(metadata["max_moves"]), record_public_history=True)
    env._state = GameState.new_game(setup, show_mode=ShowMode.DARK)
    if not np.array_equal(env.state.piece_type_arr, types):
        raise ValueError("initial simulator types do not reconstruct stable piece ids")
    env._beliefs = {seat: None for seat in Seat}
    env._knowledge = {seat: ObserverKnowledge(env.state, seat) for seat in Seat}
    builder = ObservationBuilder(include_global=False, identity_encoding="known_only")
    positions = selected_positions(length, positions_per_game, seed=position_seed,
                                   game_id=int(metadata["game_id"]))
    selected = {int(ply): index for index, ply in enumerate(positions)}
    snapshots, spatial = [], []
    for ply in range(length):
        if ply in selected:
            observer = Seat((selected[ply] + int(metadata["game_id"])) % 4)
            snapshots.append(env.knowledge_snapshot(observer))
            spatial.append(builder.build(env.state, None, observer).spatial.copy())
        acting = Seat(int(trace["seats"][ply]))
        if env.current_seat() != acting:
            raise ValueError(f"CPU/GPU acting-seat mismatch at ply {ply}")
        world_action = unrotate_compact_action_id(int(trace["actions"][ply]), acting)
        _, done, info = env._step_game_only(world_action)
        expected = (int(info.move_result.event), bool(done), bool(info.draw),
                    -1 if info.winner_team is None else int(info.winner_team))
        recorded = (int(trace["events"][ply]), bool(trace["terminated"][ply]),
                    bool(trace["draw"][ply]), int(trace["winner_team"][ply]))
        if expected != recorded:
            raise ValueError(f"CPU/GPU event or outcome mismatch at ply {ply}: {expected} != {recorded}")
    inputs = build_joint_input(snapshots)
    # This is the only label construction. Truth arrays are never passed to
    # build_joint_input, batch_inputs, or the neural observation builder.
    truth = np.full(120, -1, dtype=np.int8)
    for pid in np.flatnonzero(types >= 0):
        truth[pid] = TYPE_TO_INDEX[PieceType(int(types[pid]))]
    targets = truth[inputs.piece_ids]
    arrays = {name: np.asarray(getattr(inputs, name)) for name in ARRAY_FIELDS}
    events, offsets = pack_events(inputs.public_history)
    arrays.update(spatial=np.stack(spatial).astype(np.float32), targets=targets,
                  history_events=events, event_offsets=offsets, ply=positions,
                  game_id=np.full(len(positions), metadata["game_id"], dtype=np.int64))
    if cache_teacher_masks:
        from junqi_rl.belief.constraints import compile_constraints
        arrays["teacher_masks"] = np.stack([
            compile_constraints(inputs, row).teacher_masks(targets[row]) for row in range(len(targets))])
    return arrays


def preprocess_game(root, game_id: int) -> dict:
    """CPU process-pool entry point; resume reuses only hash-verified shards."""
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    identity = f"game-{game_id:07d}"
    raw_meta = json.loads((root / "raw" / (identity + ".json")).read_text())
    if raw_meta["protocol_sha256"] != protocol_sha256(protocol):
        raise ValueError("raw trace belongs to a different dataset protocol")
    raw_path = root / raw_meta["path"]
    if file_sha256(raw_path) != raw_meta["sha256"]:
        raise ValueError("raw trace changed before preprocessing")
    path = root / "shards" / (identity + ".npz")
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if (saved["raw_sha256"] != raw_meta["sha256"] or
                saved["protocol_sha256"] != protocol_sha256(protocol) or
                file_sha256(path) != saved["sha256"]):
            raise ValueError("existing supervised shard provenance/hash mismatch")
        return saved
    arrays = replay_trace(read_npz(raw_path), raw_meta,
                          positions_per_game=protocol["positions_per_game"],
                          position_seed=protocol["position_seed"], cache_teacher_masks=True)
    write_npz(path, arrays)
    result = dict(schema=DATASET_SCHEMA, game_id=game_id, split=raw_meta["split"],
                  initial_setup_sha256=raw_meta["initial_setup_sha256"], rows=len(arrays["targets"]),
                  raw_sha256=raw_meta["sha256"], protocol_sha256=protocol_sha256(protocol),
                  parent_checkpoint_sha256=protocol["parent_checkpoint_sha256"],
                  cpu_replay_validated=True, teacher_masks_cached=True,
                  path=str(path.relative_to(root)), sha256=file_sha256(path))
    write_json(meta_path, result)
    return result


def rebuild_manifest(root) -> dict:
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    digest = protocol_sha256(protocol)
    raw, shards = [], []
    splits, rows, reasons = Counter(), Counter(), Counter()
    for path in sorted((root / "raw").glob("game-*.json")):
        item = json.loads(path.read_text())
        if item["protocol_sha256"] != digest or file_sha256(root / item["path"]) != item["sha256"]:
            raise ValueError("raw manifest contains a conflicting or corrupted game")
        raw.append(item)
        reasons[item["termination_reason"]] += 1
    raw_by_id = {item["game_id"]: item for item in raw}
    if len(raw_by_id) != len(raw) or any(i < 0 or i >= protocol["games"] for i in raw_by_id):
        raise ValueError("completed game IDs exceed or duplicate the frozen game budget")
    for path in sorted((root / "shards").glob("game-*.json")):
        item = json.loads(path.read_text())
        original = raw_by_id.get(item["game_id"])
        if (item["protocol_sha256"] != digest or original is None or
                item["raw_sha256"] != original["sha256"] or
                item["split"] != split_for_setup(original["initial_setup_sha256"]) or
                file_sha256(root / item["path"]) != item["sha256"]):
            raise ValueError("supervised manifest contains a conflicting or corrupted shard")
        shards.append(item)
        splits[item["split"]] += 1
        rows[item["split"]] += item["rows"]
    complete = len(raw) == len(shards) == protocol["games"]
    manifest = dict(schema=DATASET_SCHEMA, status="complete" if complete else "partial",
                    protocol=protocol, protocol_sha256=digest, completed_games=len(raw),
                    preprocessed_games=len(shards), observation_rows=sum(rows.values()),
                    split_games=dict(splits), split_rows=dict(rows), termination_reasons=dict(reasons),
                    raw_games=raw, shards=shards)
    write_json(root / "dataset-manifest.json", manifest)
    return manifest


def iter_shards(root, split: str) -> Iterator[dict[str, np.ndarray]]:
    """Load one complete, hash-verified supervised game at a time."""
    root = Path(root)
    if split == "val":
        split = "validation"
    if split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    manifest = json.loads((root / "dataset-manifest.json").read_text())
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("training requires a complete facts dataset manifest")
    for item in manifest["shards"]:
        if item["split"] != split:
            continue
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("supervised shard hash mismatch")
        yield read_npz(path)


def concatenate_shards(root, split: str) -> dict[str, np.ndarray]:
    """Convenience loader for small datasets, with correctly relocated offsets."""
    shards = list(iter_shards(root, split))
    if not shards:
        raise ValueError(f"dataset contains no {split} shards")
    keys = set(shards[0])
    if any(set(shard) != keys for shard in shards):
        raise ValueError("dataset shards have different array contracts")
    result = {key: np.concatenate([shard[key] for shard in shards], axis=0)
              for key in keys if key != "event_offsets"}
    offsets, base = [0], 0
    for shard in shards:
        offsets.extend((shard["event_offsets"][1:] + base).tolist())
        base += len(shard["history_events"])
    result["event_offsets"] = np.asarray(offsets, dtype=np.int64)
    for array in result.values():
        array.setflags(write=False)
    return result
