"""Complete-game data, selected-model export and exact epoch-boundary resume."""
import hashlib
import json

import numpy as np
import pytest
import torch

from junqi_core.rules import ShowMode
from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA
from junqi_rl.belief.facts_dataset import (
    DATASET_SCHEMA, TRACE_SCHEMA, concatenate_shards, file_sha256, persist_trace,
    prepare_protocol, preprocess_game, protocol_sha256, rebuild_manifest, split_for_setup,
)
from junqi_rl.belief.facts_joint_checkpoint import load_facts_joint_belief
from junqi_rl.env import JunqiEnv, rotate_to_compact_action_id
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.checkpoint import export_evaluation_policy
from scripts.train_facts_joint_belief import _validate_manifest, build_parser, evaluate, train, unknown_targets


@pytest.fixture
def dataset(tmp_path):
    policy = JunqiNet(JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1,
        embed_dim=16, n_head=2, ff_factor=2, action_key_dim=8,
        identity_encoding='known_only', use_global_summary=False))
    policy_path = tmp_path / 'pi0.pt'
    export_evaluation_policy(policy, policy_path)
    root = tmp_path / 'data'
    protocol = {'schema': DATASET_SCHEMA, 'games': 3, 'positions_per_game': 2,
                'position_seed': 123, 'parent_checkpoint_sha256': file_sha256(policy_path),
                'constraint_schema': CONSTRAINT_SCHEMA,
                'test_only_collection_backend': 'CPU short complete games'}
    prepare_protocol(root, protocol)
    layouts = {}
    for seed in range(100):
        env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                       identity_encoding='known_only', max_num_moves=4)
        env.reset(seed=seed)
        types = env.state.piece_type_arr.copy()
        setup_hash = hashlib.sha256(types.tobytes()).hexdigest()
        split = split_for_setup(setup_hash)
        layouts.setdefault(split, (env, types, setup_hash))
        if len(layouts) == 3:
            break
    assert len(layouts) == 3
    for game_id, split in enumerate(('train', 'validation', 'test')):
        env, types, setup_hash = layouts[split]
        rows = []
        while not env.state.terminated:
            seat = env.current_seat()
            action = env.legal_action_ids()[0]
            canonical = rotate_to_compact_action_id(action, seat)
            _, done, info = env._step_game_only(action)
            rows.append((canonical, int(seat), int(info.move_result.event), done,
                         info.draw, -1 if info.winner_team is None else info.winner_team))
            if done:
                break
        columns = np.asarray(rows)
        trace = {name: columns[:, index].astype(dtype) for index, (name, dtype) in enumerate((
            ('actions', np.int32), ('seats', np.int8), ('events', np.int8),
            ('terminated', np.bool_), ('draw', np.bool_), ('winner_team', np.int8)))}
        trace['initial_piece_types'] = types.astype(np.int8)
        metadata = {'schema': TRACE_SCHEMA, 'complete': True, 'game_id': game_id,
                    'moves': len(rows), 'max_moves': 4, 'initial_setup_sha256': setup_hash,
                    'termination_reason': 'test_move_cap'}
        persist_trace(root, trace, metadata, protocol)
        preprocess_game(root, game_id)
    rebuild_manifest(root)
    return root, policy_path


def arguments(dataset, output, *extra):
    root, policy = dataset
    return build_parser().parse_args(['--policy', str(policy), '--dataset', str(root),
        '--output', str(output), '--games', '3', '--positions-per-game', '2',
        '--device', 'cpu', '--epochs', '2', '--batch-size', '2', '--num-threads', '1',
        '--embed-dim', '16', '--cnn-channels', '8', '--cnn-layers', '1',
        '--encoder-layers', '1', '--decoder-layers', '1', '--heads', '2',
        '--dropout', '0.1', *extra])


def test_training_exports_comparable_best_and_never_opens_test(dataset, tmp_path, monkeypatch):
    import junqi_rl.belief.facts_dataset as module
    original = module.read_npz
    reads = []
    manifest = json.loads((dataset[0] / 'dataset-manifest.json').read_text())
    reserved = {str(dataset[0] / item['path']) for item in manifest['shards'] if item['split'] == 'test'}

    def guarded(path):
        assert str(path) not in reserved, 'Training must not open final test rows'
        reads.append(str(path))
        return original(path)

    monkeypatch.setattr(module, 'read_npz', guarded)
    output = tmp_path / 'run'
    report = train(arguments(dataset, output))
    assert report['finished'] and report['epochs_completed'] == 2
    assert report['test_fold_used'] is False and len(reads) == 2
    assert report['completed_collection_games'] == 3
    assert report['initial_validation']['unknown_targets'] > 0
    restored = load_facts_joint_belief(output / 'joint-belief.pt', policy_path=dataset[1])
    assert restored.identity_encoding == 'known_only'
    last = torch.load(output / 'last.pt', weights_only=False)
    selected = torch.load(output / 'joint-belief.pt', weights_only=False)
    assert selected['selected_epoch'] == report['best_epoch']
    for key, value in selected['net'].items():
        torch.testing.assert_close(value, last['best_model_state'][key], rtol=0, atol=0)


def test_epoch_resume_reproduces_uninterrupted_weights_and_optimizer(dataset, tmp_path):
    full = tmp_path / 'full'
    partial = tmp_path / 'partial'
    resumed = tmp_path / 'resumed'
    train(arguments(dataset, full))
    train(arguments(dataset, partial, '--epochs', '1'))
    report = train(arguments(dataset, resumed, '--resume', str(partial / 'last.pt')))
    assert report['epochs_completed'] == 2
    expected = torch.load(full / 'last.pt', weights_only=False)
    actual = torch.load(resumed / 'last.pt', weights_only=False)
    for key, value in expected['net'].items():
        torch.testing.assert_close(value, actual['net'][key], rtol=0, atol=0)
    assert expected['updates'] == actual['updates']
    assert expected['best_epoch'] == actual['best_epoch']
    assert expected['best_validation_nll'] == actual['best_validation_nll']
    for index, state in expected['optimizer']['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, actual['optimizer']['state'][index][key], rtol=0, atol=0)


def test_resume_rejects_changed_training_contract_and_dataset(dataset, tmp_path):
    output = tmp_path / 'run'
    train(arguments(dataset, output, '--epochs', '1'))
    with pytest.raises(ValueError, match='contract'):
        train(arguments(dataset, tmp_path / 'changed-config', '--batch-size', '1',
                        '--resume', str(output / 'last.pt')))
    path = dataset[0] / 'dataset-manifest.json'
    manifest = json.loads(path.read_text())
    manifest['comment'] = 'changed dataset manifest'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='contract'):
        train(arguments(dataset, tmp_path / 'changed-data', '--resume', str(output / 'last.pt')))


@pytest.mark.parametrize('old_schema', [None, 'junqi.identity-domain-inventory-combat.v1'])
def test_training_rejects_old_or_missing_teacher_mask_semantics(dataset, tmp_path, old_schema):
    path = dataset[0] / 'dataset-manifest.json'
    manifest = json.loads(path.read_text())
    if old_schema is None:
        manifest['protocol'].pop('constraint_schema')
    else:
        manifest['protocol']['constraint_schema'] = old_schema
    # Recompute the digest so this tests semantic-version rejection, rather
    # than merely detecting an accidentally stale manifest hash.
    manifest['protocol_sha256'] = protocol_sha256(manifest['protocol'])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='constraint schema'):
        _validate_manifest(arguments(dataset, tmp_path / 'must-not-train'), file_sha256(dataset[1]))


def test_evaluation_denominator_excludes_known_and_singleton_facts(dataset):
    shard = concatenate_shards(dataset[0], 'validation')
    shard['known_types'] = shard['known_types'].copy()
    shard['type_domains'] = shard['type_domains'].copy()
    shard['known_types'][:, 0] = shard['targets'][:, 0]
    shard['type_domains'][:, 1] = False
    for row in range(len(shard['targets'])):
        shard['type_domains'][row, 1, shard['targets'][row, 1]] = True

    class Uniform:
        def eval(self):
            return self

        def teacher_logits(self, obs, **kwargs):
            return torch.zeros(len(obs), 75, 12)

    metrics = evaluate(Uniform(), shard, batch_size=2, device=torch.device('cpu'))
    assert metrics['unknown_targets'] == int(unknown_targets(shard).sum())
    assert not unknown_targets(shard)[:, :2].any()
    assert metrics['conditional_nll_per_unknown_piece'] == pytest.approx(np.log(12), abs=1e-6)


def test_cli_defaults_are_complete_game_and_validation_based():
    args = build_parser().parse_args(['--policy', 'p', '--dataset', 'd', '--output', 'o'])
    assert (args.games, args.positions_per_game, args.batch_size, args.epochs) == (2000, 8, 32, 10)


def test_final_test_audits_frozen_selected_weights_once(dataset, tmp_path):
    from scripts.evaluate_facts_joint_belief import main as audit
    output = tmp_path / 'run'
    train(arguments(dataset, output, '--epochs', '1'))
    checkpoint = output / 'joint-belief.pt'
    original_hash = file_sha256(checkpoint)
    argv = ['--joint', str(checkpoint), '--policy', str(dataset[1]), '--dataset', str(dataset[0]),
            '--output', str(output / 'test-audit.json'), '--device', 'cpu', '--num-threads', '1']
    report = audit(argv)
    assert report['test_fold_used'] is True
    assert report['test_completed_games'] == 1
    assert report['protocol']['selection_performed_on_test'] is False
    assert file_sha256(checkpoint) == original_hash
    assert audit(argv) == report
    argv[1] = str(output / 'last.pt')
    with pytest.raises(ValueError, match='validation-selected'):
        audit(argv)
