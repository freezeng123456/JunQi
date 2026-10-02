"""Complete-game replay, public-only batching, and durable dataset contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest

from junqi_core.knowledge import PublicEvent
from junqi_core.rules import Event, Seat, ShowMode
from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA
from junqi_rl.belief.facts_dataset import (
    ARRAY_FIELDS, DATASET_SCHEMA, TRACE_SCHEMA, batch_inputs, concatenate_shards,
    iter_shards, pack_events, persist_trace, prepare_protocol, preprocess_game,
    protocol_sha256, read_npz, rebuild_manifest, replay_trace, selected_positions, split_for_setup,
    validate_trace, _unpack_history,
)
from junqi_rl.env import JunqiEnv, rotate_to_compact_action_id


def cpu_trace(seed=7, game_id=0, max_moves=96):
    """Real standard setups and rule-complete games, with a small draw limit."""
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False, identity_encoding="known_only",
                   max_num_moves=max_moves)
    env.reset(seed=seed)
    initial = env.state.piece_type_arr.copy()
    rng = random.Random(seed)
    columns = {name: [] for name in ("actions", "seats", "events", "terminated", "draw", "winner_team")}
    for _ in range(max_moves):
        seat = env.current_seat()
        legal = env.legal_action_ids()
        # A synthetic coverage fixture favors combats to exercise public
        # EAT/BOMB/KILLED information. The production collector uses π0 only.
        combat = [int(action) for action in legal if env.state.cell_piece_id[int(action) % 289] >= 0]
        action = rng.choice(combat or legal.tolist())
        canonical = rotate_to_compact_action_id(action, seat)
        _, done, info = env._step_game_only(action)
        values = (canonical, int(seat), int(info.move_result.event), done, info.draw,
                  -1 if info.winner_team is None else info.winner_team)
        for name, value in zip(columns, values):
            columns[name].append(value)
        if done:
            break
    trace = {key: np.asarray(value, dtype=bool if key in ("terminated", "draw") else np.int32)
             for key, value in columns.items()}
    trace["initial_piece_types"] = initial
    metadata = dict(schema=TRACE_SCHEMA, complete=True, game_id=game_id, moves=len(trace["actions"]),
                    initial_setup_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                    max_moves=max_moves, termination_reason="total_move_limit" if info.draw else "team_victory",
                    draw=bool(info.draw), winner_team=-1 if info.winner_team is None else int(info.winner_team))
    return trace, metadata


@pytest.fixture(scope="module")
def complete_trace():
    return cpu_trace()


def protocol(games=1):
    return dict(schema=DATASET_SCHEMA, parent_checkpoint_sha256="f" * 64, games=games,
                positions_per_game=2, position_seed=20261003, max_moves=96,
                constraint_schema=CONSTRAINT_SCHEMA)


def test_complete_cpu_replay_preserves_all_public_events_and_separate_targets(complete_trace):
    trace, metadata = complete_trace
    assert set(trace["events"]) == {int(event) for event in Event}
    shard = replay_trace(trace, metadata, cache_teacher_masks=False)
    assert shard["spatial"].shape == (8, 317, 17, 17)
    assert shard["targets"].shape == (8, 75)
    assert set(shard["observer_seats"].tolist()) == {0, 1, 2, 3}
    inputs = batch_inputs(shard, np.arange(8))
    assert [len(history) for history in inputs.public_history] == shard["ply"].tolist()
    assert np.isfinite(shard["spatial"]).all()
    assert (inputs.known_types < 0).any()
    # Batching remains possible after removing every truth-bearing array.
    public = {name: value for name, value in shard.items() if name != "targets"}
    selected = np.asarray([6, 1, 6])
    rebuilt = batch_inputs(public, selected)
    for name in ARRAY_FIELDS:
        np.testing.assert_array_equal(getattr(rebuilt, name), getattr(inputs, name)[selected])
    assert rebuilt.public_history == tuple(inputs.public_history[i] for i in selected)


@pytest.mark.parametrize("column", ["events", "seats", "winner_team"])
def test_cpu_replay_rejects_gpu_event_or_outcome_disagreement(complete_trace, column):
    trace, metadata = complete_trace
    broken = {name: value.copy() for name, value in trace.items()}
    index = 0
    broken[column][index] = (int(broken[column][index]) + 1) % (4 if column == "seats" else 5)
    with pytest.raises(ValueError, match="mismatch"):
        replay_trace(broken, metadata, cache_teacher_masks=False)


def test_unfinished_or_reused_terminal_games_are_never_counted(complete_trace):
    trace, metadata = complete_trace
    for terminal in (np.zeros(len(trace["actions"]), dtype=bool), np.ones(len(trace["actions"]), dtype=bool)):
        with pytest.raises(ValueError, match="first terminal"):
            validate_trace({**trace, "terminated": terminal}, metadata)
    with pytest.raises(ValueError, match="only complete"):
        validate_trace(trace, {**metadata, "complete": False})


def test_event_numeric_roundtrip_preserves_revelations_engineer_and_multiple_deaths():
    history = (
        PublicEvent(1, 0, 123, 124, 3, 35, Event.KILLED, True, False, False, False, (0, 2)),
        PublicEvent(2, 1, 120, 121, 31, 65, Event.EAT, False, True, True, True, (2,)),
        PublicEvent(3, 3, 19, 20, 97, 11, Event.BOMB, True, True, False, False, ()),
    )
    rows, offsets = pack_events(((), history, history[:1]))
    assert rows.dtype != object
    assert offsets.tolist() == [0, 0, 3, 4]
    for index, expected in enumerate(((), history, history[:1])):
        assert _unpack_history(rows[offsets[index]:offsets[index + 1]]) == expected


def test_durable_resume_split_isolation_masks_and_offset_concatenation(tmp_path, complete_trace):
    trace, metadata = complete_trace
    config = protocol(games=2)
    prepare_protocol(tmp_path, config)
    originals = []
    for game_id in range(2):
        game = {**metadata, "game_id": game_id}
        saved = persist_trace(tmp_path, trace, game, config)
        assert persist_trace(tmp_path, trace, game, config) == saved
        result = preprocess_game(tmp_path, game_id)
        assert preprocess_game(tmp_path, game_id) == result
        originals.append(read_npz(tmp_path / result["path"]))
    manifest = rebuild_manifest(tmp_path)
    assert manifest["status"] == "complete"
    assert manifest["completed_games"] == manifest["preprocessed_games"] == 2
    split = split_for_setup(metadata["initial_setup_sha256"])
    assert manifest["split_games"] == {split: 2}
    assert len(list(iter_shards(tmp_path, split))) == 2
    combined = concatenate_shards(tmp_path, split)
    restored = batch_inputs(combined, np.arange(4))
    expected = tuple(history for shard in originals
                     for history in batch_inputs(shard, np.arange(2)).public_history)
    assert restored.public_history == expected
    assert combined["event_offsets"][-1] == len(combined["history_events"])
    assert combined["teacher_masks"].shape == (4, 75, 12)
    assert combined["teacher_masks"][np.arange(4)[:, None], np.arange(75), combined["targets"]].all()
    with pytest.raises(ValueError, match="conflicts"):
        prepare_protocol(tmp_path, {**config, "parent_checkpoint_sha256": "e" * 64})
    shard_file = tmp_path / manifest["shards"][0]["path"]
    with shard_file.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        list(iter_shards(tmp_path, split))


def test_partial_manifest_cannot_start_training(tmp_path):
    prepare_protocol(tmp_path, protocol())
    assert rebuild_manifest(tmp_path)["status"] == "partial"
    with pytest.raises(ValueError, match="complete facts"):
        list(iter_shards(tmp_path, "train"))


@pytest.mark.parametrize("old_schema", [None, "junqi.identity-domain-inventory-combat.v1"])
def test_old_or_unspecified_constraint_masks_are_rejected_at_every_dataset_entry(tmp_path, old_schema):
    stale = protocol()
    if old_schema is None:
        stale.pop("constraint_schema")
    else:
        stale["constraint_schema"] = old_schema
    with pytest.raises(ValueError, match="constraint schema"):
        prepare_protocol(tmp_path, stale)
    assert not (tmp_path / "protocol.json").exists()
    # Direct preprocessing/loading must also reject pre-existing v1 caches,
    # before even trying to open the game or its supervised arrays.
    (tmp_path / "protocol.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="constraint schema"):
        preprocess_game(tmp_path, 0)
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(dict(
        schema=DATASET_SCHEMA, status="complete", protocol=stale,
        protocol_sha256=protocol_sha256(stale), shards=[])))
    with pytest.raises(ValueError, match="constraint schema"):
        list(iter_shards(tmp_path, "train"))


def test_relabeling_manifest_does_not_make_an_old_mask_cache_current(tmp_path):
    current = protocol()
    stale = {**current, "constraint_schema": "junqi.identity-domain-inventory-combat.v1"}
    manifest = dict(schema=DATASET_SCHEMA, status="complete", protocol=current,
                    protocol_sha256=protocol_sha256(current), shards=[dict(
                        split="train", protocol_sha256=protocol_sha256(stale), path="must-not-open.npz")])
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different dataset protocol"):
        list(iter_shards(tmp_path, "train"))


def test_positions_are_reproducible_distinct_and_cover_temporal_strata():
    positions = selected_positions(4000, 8, seed=81, game_id=19)
    np.testing.assert_array_equal(positions, selected_positions(4000, 8, seed=81, game_id=19))
    assert (positions // 500).tolist() == list(range(8))
    np.testing.assert_array_equal(selected_positions(3, 8, seed=0, game_id=0), [0, 1, 2])


def test_dataset_reference_import_does_not_load_torch_cuda_or_legacy():
    code = '''
import importlib.abc, sys
class Forbid(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in ('torch', 'junqi_cuda', 'junqi_core.info_model', 'junqi_rl.belief.legacy_joint'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Forbid())
from junqi_rl.belief.facts_dataset import batch_inputs, replay_trace
assert callable(batch_inputs) and callable(replay_trace)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
