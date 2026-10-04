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


def validate_constraint_schema(protocol: dict) -> None:
    """Cached causal masks are valid only for their exact constraint semantics."""
    from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA
    if not isinstance(protocol, dict) or protocol.get("constraint_schema") != CONSTRAINT_SCHEMA:
        raise ValueError(f"dataset constraint schema mismatch; require {CONSTRAINT_SCHEMA!r}; "
                         "old cached teacher masks must be rebuilt")


def prepare_protocol(root, protocol: dict) -> dict:
    root = Path(root)
    if protocol.get("schema") != DATASET_SCHEMA:
        raise ValueError("dataset protocol schema mismatch")
    validate_constraint_schema(protocol)
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
    validate_constraint_schema(protocol)
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
    validate_constraint_schema(protocol)
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
    protocol = manifest.get("protocol")
    validate_constraint_schema(protocol)
    digest = protocol_sha256(protocol)
    if manifest.get("protocol_sha256") != digest:
        raise ValueError("dataset protocol hash mismatch")
    for item in manifest["shards"]:
        if item["split"] != split:
            continue
        if item.get("protocol_sha256") != digest:
            raise ValueError("supervised shard was cached under a different dataset protocol")
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("supervised shard hash mismatch")
        yield read_npz(path)


def concatenate_shards(root, split: str) -> dict[str, np.ndarray]:
    """Load a fold with one output allocation and at most one decoded shard.

    The planning pass reads only hash-verified NumPy headers; the copy pass
    decodes each shard once into its final slices. This avoids holding the
    complete decoded fold alongside another complete concatenated copy.
    """
    import ast
    import math
    import zipfile

    root = Path(root)
    manifest_path = root / "dataset-manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    selected_split = "validation" if split == "val" else split
    if selected_split not in ("train", "validation", "test"):
        raise ValueError("split must be train, validation, or test")
    if manifest.get("schema") != DATASET_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("training requires a complete facts dataset manifest")
    protocol = manifest.get("protocol")
    validate_constraint_schema(protocol)
    digest = protocol_sha256(protocol)
    if manifest.get("protocol_sha256") != digest:
        raise ValueError("dataset protocol hash mismatch")
    plans, keys, totals, dtypes = [], None, {}, {}
    offset_count = 1
    for item in manifest["shards"]:
        if item["split"] != selected_split:
            continue
        if item.get("protocol_sha256") != digest:
            raise ValueError("supervised shard was cached under a different dataset protocol")
        path = root / item["path"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("supervised shard hash mismatch")
        plan = {}
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if not member.filename.endswith(".npy"):
                    raise ValueError("dataset shard contains a non-array member")
                key = member.filename[:-4]
                if key in plan:
                    raise ValueError("dataset shard contains a duplicate array")
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    if version not in ((1, 0), (2, 0), (3, 0)):
                        raise ValueError("unsupported dataset NumPy header version")
                    width = 2 if version == (1, 0) else 4
                    length_bytes = handle.read(width)
                    if len(length_bytes) != width:
                        raise ValueError("truncated dataset NumPy header")
                    length = int.from_bytes(length_bytes, "little")
                    if length > 10000:
                        raise ValueError("dataset NumPy header exceeds the safe load limit")
                    raw = handle.read(length)
                    if len(raw) != length:
                        raise ValueError("truncated dataset NumPy header")
                    header = ast.literal_eval(raw.decode("utf8" if version == (3, 0) else "latin1"))
                    if set(header) != {"shape", "fortran_order", "descr"}:
                        raise ValueError("invalid dataset NumPy header")
                    shape = header["shape"]
                    dtype = np.lib.format.descr_to_dtype(header["descr"])
                    if (not isinstance(shape, tuple) or not shape
                            or any(type(size) is not int or size < 0 for size in shape)
                            or type(header["fortran_order"]) is not bool or dtype.hasobject):
                        raise ValueError("invalid dataset NumPy array contract")
                    if member.file_size - handle.tell() != math.prod(shape) * dtype.itemsize:
                        raise ValueError("dataset NumPy shape disagrees with payload size")
                    plan[key] = (shape, dtype)
        if keys is None:
            keys = set(plan)
        if set(plan) != keys:
            raise ValueError("dataset shards have different array contracts")
        if not {"history_events", "event_offsets"} <= keys:
            raise ValueError("dataset shard is missing public-history arrays")
        for key, (shape, dtype) in plan.items():
            if plans and shape[1:] != plans[0][key][0][1:]:
                raise ValueError("dataset shards have different array shapes")
            totals[key] = totals.get(key, 0) + shape[0]
            dtypes[key] = np.result_type(dtypes.get(key, dtype), dtype)
        shape, dtype = plan["event_offsets"]
        if len(shape) != 1 or not shape[0] or dtype.kind not in "iu":
            raise ValueError("dataset shard has invalid public-history offsets")
        offset_count += shape[0] - 1
        plans.append(plan)
    if not plans:
        raise ValueError(f"dataset contains no {split} shards")
    if file_sha256(manifest_path) != manifest_hash:
        raise ValueError("dataset manifest changed during fold shape planning")
    result = {key: np.empty((totals[key], *plans[0][key][0][1:]), dtype=dtypes[key])
              for key in keys if key != "event_offsets"}
    result["event_offsets"] = np.empty(offset_count, dtype=np.int64)
    result["event_offsets"][0] = 0
    positions = dict.fromkeys(result, 0)
    offset_position, history_base, loaded = 1, 0, 0
    for shard in iter_shards(root, split):
        plan = {key: (array.shape, array.dtype) for key, array in shard.items()}
        if loaded >= len(plans) or plan != plans[loaded]:
            raise ValueError("dataset shard array contract changed between fold passes")
        offsets = shard["event_offsets"]
        if (offsets[0] != 0 or offsets[-1] != len(shard["history_events"])
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("dataset shard has invalid public-history offsets")
        for key in keys - {"event_offsets"}:
            end = positions[key] + len(shard[key])
            np.copyto(result[key][positions[key]:end], shard[key], casting="same_kind")
            positions[key] = end
        end = offset_position + len(shard["event_offsets"]) - 1
        np.add(shard["event_offsets"][1:], history_base,
               out=result["event_offsets"][offset_position:end], casting="unsafe")
        offset_position = end
        history_base += len(shard["history_events"])
        loaded += 1
        del offsets, shard
    if loaded != len(plans) or file_sha256(manifest_path) != manifest_hash:
        raise ValueError("dataset manifest or shard coverage changed during fold loading")
    for array in result.values():
        array.setflags(write=False)
    return result
