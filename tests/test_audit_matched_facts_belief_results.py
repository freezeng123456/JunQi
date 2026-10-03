"""Audit real training artifacts and deliberately synthetic paired records."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts.audit_matched_facts_belief_results import (
    MATCHUPS, audit, pooled_statistics, validate_search_metrics,
)
from junqi_rl.belief.facts_dataset import file_sha256, protocol_sha256, rebuild_manifest
from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
from scripts.compare_facts_joint import (
    excluded_layouts, paired_statistics, selected_belief_provenance, setup_sha256, write_json,
)
from scripts.evaluate_facts_joint_belief import main as evaluate_test
from scripts.train_facts_joint_belief import train
from tests.test_train_facts_joint_belief import arguments, dataset  # noqa: F401


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def search_metrics(side, config):
    return dict(decisions=2, successful_decisions=2, joint_refreshes=2,
        proposals=2, accepted_proposals=2, rejected_proposals=0,
        sample_budget_failures=0, constraint_budget_failures=0,
        public_replay_steps=4, rollout_world_steps=10, legal_actions=8, candidate_actions=8,
        unique_accepted_assignments=2, constraint_seconds=.1, sampling_seconds=.2,
        materialization_seconds=.1, simulation_seconds=.4,
        decision_seconds_total=.8, decision_seconds_p50=.4, decision_seconds_p95=.4,
        config=deepcopy(config), belief_kind=side['belief_kind'],
        blueprint_queries={'distribution': 2}, constraint_queries={'allowed_next': 150}, rejection_reasons={},
        acceptance_rate=1., action_coverage=1., proposal_semantics=side['distribution_semantics']
        + '; fresh samples conditioned by full replay and root observation')


@pytest.fixture
def recovered(dataset, tmp_path):
    from junqi_core.rules import MAX_NUM_MOVES
    data, policy = dataset
    # Fixture-only provenance; these short traces never establish game strength.
    data_protocol = json.loads((data / 'protocol.json').read_text())
    data_protocol['acceptance_only'] = False
    write_json(data / 'protocol.json', data_protocol)
    for family in ('raw', 'shards'):
        for path in (data / family).glob('game-*.json'):
            meta = json.loads(path.read_text())
            meta['protocol_sha256'] = protocol_sha256(data_protocol)
            write_json(path, meta)
    manifest = rebuild_manifest(data)
    root = tmp_path / 'experiment'
    root.mkdir()
    seeds = [41, 42]
    config = dict(particles=1, depth=1, beta=.5, proposal_batches=8, max_constraint_nodes=100000)
    protocol = dict(schema='junqi.facts-belief-ablation.v1', belief_seeds=seeds,
        pairs_per_seed=2, matchups=list(MATCHUPS), layout_seeds={'41': 300, '42': 400},
        parent_checkpoint_sha256=file_sha256(policy), dataset_manifest_sha256=file_sha256(data / 'dataset-manifest.json'),
        search_config=config, epochs=1, batch_size=2, learning_rate=1e-4, workers=1,
        source_commit='a' * 40, expected_complete_games=24)
    write_json(root / 'protocol.json', protocol)
    source = root / 'source/scripts/compare_facts_joint.py'
    source.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / 'scripts/compare_facts_joint.py', source)
    source_manifest = {'scripts/compare_facts_joint.py': file_sha256(source)}
    for relative in ('junqi_core/rules.py', 'junqi_rl/analysis/facts_joint_search.py'):
        destination = root / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).parents[1] / relative, destination)
        source_manifest[relative] = file_sha256(destination)
    inference_hash = digest({name: value for name, value in source_manifest.items()
                             if name.startswith(('junqi_core/', 'junqi_rl/'))})
    write_json(root / 'source-sha256.json', source_manifest)
    models = {}
    for seed in seeds:
        for kind in ('joint', 'marginal'):
            directory = root / 'training' / f'{kind}-s{seed}'
            train(arguments(dataset, directory, '--belief-kind', kind, '--seed', str(seed), '--epochs', '1'))
            selected = directory / f'{kind}-belief.pt'
            evaluate_test(['--belief', str(selected), '--policy', str(policy), '--dataset', str(data),
                           '--output', str(directory / 'test-audit.json'), '--device', 'cpu', '--num-threads', '1'])
            net = load_facts_belief(selected, policy_path=policy)
            side = selected_belief_provenance(selected, net, protocol['dataset_manifest_sha256'])
            side['policy_sha256'] = protocol['parent_checkpoint_sha256']
            models[(kind, seed)] = side
    forbidden, evidence = excluded_layouts(manifest, [])
    for seed in seeds:
        for matchup in MATCHUPS:
            directory = root / 'matches' / f's{seed}' / matchup
            directory.mkdir(parents=True)
            candidate, opponent = matchup.split('-')
            sides = dict(candidate=models[(candidate, seed)], opponent=(models[(opponent, seed)]
                if opponent != 'baseline' else dict(kind='direct_policy', policy_sha256=file_sha256(policy))))
            match = dict(schema='junqi.facts-belief-comparison.v2', policy_sha256=file_sha256(policy),
                joint_sha256=sides['candidate']['checkpoint_sha256'], dataset_manifest_sha256=protocol['dataset_manifest_sha256'],
                sides=sides, phase='formal', seed=protocol['layout_seeds'][str(seed)], pairs=2, max_moves=MAX_NUM_MOVES,
                search_config=config, workers=1, baseline=sides['opponent'], candidate=sides['candidate'],
                team_assignments=[0, 1], action_selection='sample distribution', deterministic_algorithms=True,
                comparison_script_sha256=file_sha256(source), exclusion_artifacts=evidence,
                inference_source_sha256=inference_hash,
                forbidden_layout_count=len(forbidden),
                forbidden_layout_sha256=hashlib.sha256(json.dumps(sorted(forbidden)).encode()).hexdigest())
            write_json(directory / 'protocol.json', match)
            games = []
            for pair in range(2):
                for team in (0, 1):
                    metrics = {role: (search_metrics(side, config) if side['kind'] == 'belief_search'
                                     else dict(decisions=2)) for role, side in sides.items()}
                    game = dict(pair=pair, candidate_team=team, outcome='win' if team == 0 else 'loss',
                        setup_seed=match['seed'] + pair, setup_sha256=setup_sha256(match['seed'] + pair),
                        protocol_sha256=digest(match), sides_sha256=digest(sides), side_metrics=metrics,
                        search_metrics=metrics['candidate'], plies=4, seconds=1.8,
                        action_history_sha256='f' * 64)
                    write_json(directory / f'game-{pair:06d}-{team}.json', game)
                    games.append(game)
            write_json(directory / 'comparison.json', dict(protocol=match, protocol_sha256=digest(match),
                statistics=paired_statistics(games, seed=match['seed']), finished=True, games=games))
    return SimpleNamespace(experiment_dir=root, dataset=data, policy=policy, source_dir=None,
                           comparison_source=None, exclude_layouts=[], archive=None, archive_sha256=None)


def test_full_audit_uses_actual_weights_once_and_recomputes_six_comparisons(recovered):
    result, protected = audit(recovered)
    assert result['status'] == 'passed', result
    assert len(result['training']) == 4 and len(result['comparisons']) == 6
    assert len(result['artifacts']) == len(protected) == result['artifacts_checked']
    assert result['dataset']['artifact_hash_passes'] == 1
    assert len([item for item in result['artifacts'] if item['role'] == 'dataset_shards']) == 3
    for metrics in result['pooled'].values():
        assert metrics['completed_games'] == 8
        assert metrics['win_rate'] == .5 and metrics['win_rate_ci95'] == [.5, .5]
        assert 'not a training-seed-population interval' in metrics['interval_scope']


@pytest.mark.parametrize('mutation,expected', [
    ('opponent-checkpoint', 'sides binding'),
    ('selected-epoch', 'held-out test selected_epoch'),
    ('aggregate-win-rate', 'bootstrap intervals'),
    ('resume-best-weights', 'selected_vs_best_weights'),
    ('inference-source-changed', 'inference_source_sha256 binding'),
    ('inference-source-missing', 'inference_source_sha256 binding'),
])
def test_malformed_provenance_or_report_is_rejected(recovered, mutation, expected):
    root = recovered.experiment_dir
    if mutation == 'opponent-checkpoint':
        path = root / 'matches/s41/joint-marginal/protocol.json'
        value = json.loads(path.read_text())
        value['sides']['opponent']['checkpoint_sha256'] = 'b' * 64
    elif mutation == 'selected-epoch':
        path = root / 'training/marginal-s41/test-audit.json'
        value = json.loads(path.read_text())
        value['protocol']['selected_epoch'] = 2
    elif mutation == 'aggregate-win-rate':
        path = root / 'matches/s41/marginal-baseline/comparison.json'
        value = json.loads(path.read_text())
        value['statistics']['win_rate'] = .99
    elif mutation.startswith('inference-source-'):
        path = root / 'matches/s42/joint-marginal/protocol.json'
        value = json.loads(path.read_text())
        if mutation == 'inference-source-changed':
            value['inference_source_sha256'] = 'b' * 64
        else:
            del value['inference_source_sha256']
    else:
        import torch
        path = root / 'training/joint-s41/last.pt'
        value = torch.load(path, weights_only=False)
        key = next(key for key, tensor in value['best_model_state'].items() if tensor.dtype.is_floating_point)
        value['best_model_state'][key] = value['best_model_state'][key] + .5
        torch.save(value, path)
        report_path = path.parent / 'report.json'
        report = json.loads(report_path.read_text())
        report['last_checkpoint_sha256'] = file_sha256(path)
        write_json(report_path, report)
        result, _ = audit(recovered)
        assert result['status'] == 'failed'
        assert expected in result['errors'][0]
        return
    write_json(path, value)
    result, _ = audit(recovered)
    assert result['status'] == 'failed'
    assert expected in result['errors'][0], result


def test_opponent_search_rejects_missing_coverage_and_impossible_counters():
    side = dict(kind='belief_search', belief_kind='marginal', distribution_semantics='test static marginals')
    config = dict(particles=1)
    metrics = search_metrics(side, config)
    validate_search_metrics(metrics, side, config, label='opponent')
    for field, value, message in [('candidate_actions', 7, 'coverage'),
                                  ('proposals', -1, 'integer'),
                                  ('accepted_proposals', 1, 'budget')]:
        wrong = deepcopy(metrics)
        wrong[field] = value
        with pytest.raises(ValueError, match=message):
            validate_search_metrics(wrong, side, config, label='opponent')


def test_pooled_interval_preserves_whole_layout_pairs_within_fixed_seeds():
    games = {seed: [dict(pair=pair, candidate_team=team, outcome='win' if team == 0 else 'loss',
                         setup_sha256=f'{pair:064d}') for pair in range(3) for team in (0, 1)]
             for seed in (41, 42)}
    result = pooled_statistics(games)
    assert result['win_rate_ci95'] == result['score_ci95'] == [.5, .5]
    assert result['complete_pairs'] == 6
