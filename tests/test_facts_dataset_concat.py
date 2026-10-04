"""Fold loading preserves data while bounding live decoded shard storage."""
import json
import weakref

import numpy as np
import pytest

from junqi_rl.belief import facts_dataset as data
from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA


def dataset(root, games=3):
    protocol = {"schema": data.DATASET_SCHEMA, "constraint_schema": CONSTRAINT_SCHEMA}
    digest = data.protocol_sha256(protocol)
    shards = []
    for game in range(games):
        rows = 2 + game % 2
        events = game + 1
        arrays = {
            "spatial": np.arange(rows * 6, dtype=np.float32).reshape(rows, 3, 2) + game * 100,
            "targets": np.full((rows, 2), game, dtype=np.int8),
            "teacher_masks": np.ones((rows, 2, 3), dtype=np.bool_),
            "history_events": np.arange(events * 12, dtype=np.int32).reshape(events, 12),
            "event_offsets": np.arange(rows + 1, dtype=np.int64) * events // rows,
        }
        path = root / "shards" / f"game-{game:07d}.npz"
        data.write_npz(path, arrays)
        shards.append({"split": "train", "path": str(path.relative_to(root)),
                       "protocol_sha256": digest, "sha256": data.file_sha256(path)})
    # The reserved fold must not be opened by a train-fold load.
    shards.append({"split": "test", "path": "reserved-test.npz", "protocol_sha256": digest,
                   "sha256": "0" * 64})
    data.write_json(root / "dataset-manifest.json", {
        "schema": data.DATASET_SCHEMA, "status": "complete", "protocol": protocol,
        "protocol_sha256": digest, "shards": shards})
    return root


def legacy_concat(root):
    shards = list(data.iter_shards(root, "train"))
    result = {key: np.concatenate([shard[key] for shard in shards])
              for key in shards[0] if key != "event_offsets"}
    offsets, base = [0], 0
    for shard in shards:
        offsets.extend((shard["event_offsets"][1:] + base).tolist())
        base += len(shard["history_events"])
    result["event_offsets"] = np.asarray(offsets, dtype=np.int64)
    return result


def test_fold_bytes_dtypes_order_offsets_and_reserved_test(tmp_path):
    root = dataset(tmp_path)
    expected = legacy_concat(root)
    actual = data.concatenate_shards(root, "train")
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].shape == expected[key].shape
        assert actual[key].dtype == expected[key].dtype
        assert actual[key].tobytes() == expected[key].tobytes()
        assert not actual[key].flags.writeable


def test_decoded_source_shards_are_not_retained_as_a_fold(tmp_path, monkeypatch):
    root = dataset(tmp_path, games=24)
    original = data.read_npz
    references, peak = [], []

    def track(path):
        arrays = original(path)
        references.append(weakref.ref(arrays["spatial"]))
        peak.append(sum(reference() is not None for reference in references))
        return arrays

    monkeypatch.setattr(data, "read_npz", track)
    result = data.concatenate_shards(root, "train")
    assert result["spatial"].shape[0] == 60
    assert max(peak) == 1
    assert all(reference() is None for reference in references)


def test_hash_verification_is_preserved(tmp_path):
    root = dataset(tmp_path)
    path = root / "shards/game-0000001.npz"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        data.concatenate_shards(root, "train")


def test_manifest_cannot_change_between_passes(tmp_path, monkeypatch):
    root = dataset(tmp_path)
    original = data.read_npz
    calls = 0

    def change_after_planning(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            path = root / "dataset-manifest.json"
            manifest = json.loads(path.read_text())
            manifest["changed"] = True
            data.write_json(path, manifest)
        return original(*args)

    monkeypatch.setattr(data, "read_npz", change_after_planning)
    with pytest.raises(ValueError, match="manifest or shard coverage changed"):
        data.concatenate_shards(root, "train")
