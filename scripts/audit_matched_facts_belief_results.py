#!/usr/bin/env python3
"""Read-only recovery audit of the frozen marginal/joint ablation experiment.

Checks actual checkpoint/data/source hashes once per file, all four training
lineages, both sides of every paired match, and recomputes layout-bootstrap
statistics. It does not replay GPU kernels, matches, or neural NLL. The pooled
interval resamples layouts within each fixed belief seed; it is conditional
on those trained seeds, not an interval over the training-seed population.
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

from scripts.audit_facts_belief_results import _json, _relative_file, _require


MATCHUPS = ('marginal-baseline', 'joint-baseline', 'joint-marginal')


def _same(actual, expected, message):
    """JSON booleans must not impersonate integer counters or schema fields."""
    _require(type(actual) is type(expected), message)
    if isinstance(expected, dict):
        _require(actual.keys() == expected.keys(), message)
        for key, value in expected.items():
            _same(actual[key], value, message)
    elif isinstance(expected, (list, tuple)):
        _require(len(actual) == len(expected), message)
        for left, right in zip(actual, expected):
            _same(left, right, message)
    else:
        _require(actual == expected, message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _number(value, *, label, minimum=0):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
             f'{label}: expected a finite number >= {minimum}')


def _integer(value, *, label, minimum=0):
    _require(type(value) is int and value >= minimum, f'{label}: invalid integer')


def _metrics(value, rows, unknown, kind, label):
    from scripts.train_facts_joint_belief import metric_names
    nll_key, accuracy_key = metric_names(kind)
    _same(value.get('rows'), rows, f'{label}: observation denominator mismatch')
    _same(value.get('unknown_targets'), unknown, f'{label}: unknown-target denominator mismatch')
    _number(value.get(nll_key), label=f'{label} NLL')
    _number(value.get(accuracy_key), label=f'{label} accuracy')
    _require(value[accuracy_key] <= 1, f'{label}: accuracy exceeds one')


def validate_experiment_protocol(protocol):
    from junqi_rl.analysis.facts_joint_search import SearchConfig
    _same(protocol.get('schema'), 'junqi.facts-belief-ablation.v1', 'wrong experiment schema')
    seeds = protocol.get('belief_seeds')
    _require(isinstance(seeds, list) and len(seeds) == 2
             and all(type(seed) is int and seed >= 0 for seed in seeds)
             and len(set(seeds)) == len(seeds), 'experiment requires two distinct fixed belief seeds')
    _same(protocol.get('matchups'), list(MATCHUPS), 'frozen experiment must specify all three matchups')
    _same(set(protocol.get('layout_seeds', {})), {str(seed) for seed in seeds}, 'layout seed mapping mismatch')
    for value in protocol['layout_seeds'].values():
        _integer(value, label='layout seed')
    _integer(protocol.get('pairs_per_seed'), label='pairs per seed', minimum=2)
    _integer(protocol.get('epochs'), label='epochs', minimum=1)
    _integer(protocol.get('batch_size'), label='batch size', minimum=1)
    _integer(protocol.get('workers'), label='workers', minimum=1)
    _number(protocol.get('learning_rate'), label='learning rate')
    _require(protocol['learning_rate'] > 0, 'learning rate must be positive')
    _same(protocol.get('search_config'), asdict(SearchConfig(**protocol.get('search_config', {}))),
          'experiment search config must contain every field')
    for field in ('parent_checkpoint_sha256', 'dataset_manifest_sha256'):
        _require(isinstance(protocol.get(field), str)
                 and re.fullmatch('[0-9a-f]{64}', protocol[field]), f'invalid experiment {field}')
    _require(isinstance(protocol.get('source_commit'), str)
             and re.fullmatch('[0-9a-f]{40}', protocol['source_commit']), 'missing frozen source commit')
    if 'expected_complete_games' in protocol:
        _same(protocol['expected_complete_games'], len(seeds) * len(MATCHUPS) * 2 * protocol['pairs_per_seed'],
              'experiment complete-game budget is inconsistent')
    return seeds


def validate_search_metrics(metrics, side, config, *, label):
    """Reject negative/ambiguous counters and silent partial search on either side."""
    _integer(metrics.get('decisions'), label=f'{label} decisions')
    if side['kind'] == 'direct_policy':
        _same(set(metrics), {'decisions'}, f'{label}: direct policy contains unexpected search metrics')
        return
    _same(metrics.get('config'), config, f'{label}: search config mismatch')
    _same(metrics.get('belief_kind'), side['belief_kind'], f'{label}: belief kind mismatch')
    integer_fields = ('successful_decisions', 'joint_refreshes', 'proposals', 'accepted_proposals',
                      'rejected_proposals', 'sample_budget_failures', 'constraint_budget_failures',
                      'public_replay_steps', 'rollout_world_steps', 'legal_actions', 'candidate_actions',
                      'unique_accepted_assignments')
    for field in integer_fields:
        _integer(metrics.get(field), label=f'{label} {field}')
    for field in ('sample_budget_failures', 'constraint_budget_failures'):
        _same(metrics[field], 0, f'{label}: search failure or fallback in completed game')
    _same(metrics['successful_decisions'], metrics['decisions'], f'{label}: incomplete search decisions')
    _same(metrics['joint_refreshes'], metrics['decisions'], f'{label}: hidden samples were not refreshed')
    _same(metrics['candidate_actions'], metrics['legal_actions'], f'{label}: incomplete root action coverage')
    _require(metrics['legal_actions'] >= metrics['decisions'], f'{label}: missing legal actions')
    _same(metrics['accepted_proposals'], metrics['decisions'] * config['particles'],
          f'{label}: accepted particles do not match the frozen budget')
    _same(metrics['accepted_proposals'] + metrics['rejected_proposals'], metrics['proposals'],
          f'{label}: proposal counters disagree')
    _require(metrics['decisions'] <= metrics['unique_accepted_assignments'] <= metrics['accepted_proposals'],
             f'{label}: invalid unique assignment count')
    for field in ('constraint_seconds', 'sampling_seconds', 'materialization_seconds',
                  'simulation_seconds', 'decision_seconds_total', 'decision_seconds_p50', 'decision_seconds_p95'):
        _number(metrics.get(field), label=f'{label} {field}')
    _same(metrics.get('acceptance_rate'), metrics['accepted_proposals'] / max(1, metrics['proposals']),
          f'{label}: acceptance rate mismatch')
    _same(metrics.get('action_coverage'), metrics['candidate_actions'] / max(1, metrics['legal_actions']),
          f'{label}: action coverage rate mismatch')
    _require(metrics['constraint_seconds'] <= metrics['sampling_seconds'] + 1e-6,
             f'{label}: constraint time must be included in sampling time')
    for field in ('blueprint_queries', 'constraint_queries', 'rejection_reasons'):
        _require(isinstance(metrics.get(field), dict), f'{label}: missing {field}')
        for key, value in metrics[field].items():
            _integer(value, label=f'{label} {field}.{key}')
    _same(sum(metrics['rejection_reasons'].values()), metrics['rejected_proposals'],
          f'{label}: rejection reasons do not account for rejected proposals')
    _same(metrics.get('proposal_semantics'), side['distribution_semantics']
          + '; fresh samples conditioned by full replay and root observation',
          f'{label}: proposal distribution semantics mismatch')


def summarize_search(games):
    """Sum counters/times, without averaging per-game latency quantiles."""
    totals = {}
    scalar_fields = ('decisions', 'successful_decisions', 'joint_refreshes', 'proposals', 'accepted_proposals',
                     'rejected_proposals', 'sample_budget_failures', 'constraint_budget_failures',
                     'public_replay_steps', 'rollout_world_steps', 'legal_actions', 'candidate_actions',
                     'unique_accepted_assignments', 'constraint_seconds', 'sampling_seconds',
                     'materialization_seconds', 'simulation_seconds', 'decision_seconds_total')
    for role in ('candidate', 'opponent'):
        total = {field: sum(game['side_metrics'][role].get(field, 0) for game in games)
                 for field in scalar_fields}
        for field in ('blueprint_queries', 'constraint_queries', 'rejection_reasons'):
            counts = {}
            for game in games:
                for key, value in game['side_metrics'][role].get(field, {}).items():
                    counts[key] = counts.get(key, 0) + value
            total[field] = counts
        if total['successful_decisions']:
            total['mean_search_decision_seconds'] = total['decision_seconds_total'] / total['decisions']
            total['acceptance_rate'] = total['accepted_proposals'] / total['proposals']
            total['action_coverage'] = total['candidate_actions'] / total['legal_actions']
        totals[role] = total
    return totals


def pooled_statistics(games_by_seed, *, seed=20261004, repeats=10000):
    """Resample whole layouts within each observed training seed, never games."""
    import numpy as np
    from scripts.compare_facts_joint import paired_statistics
    all_games = [game for games in games_by_seed.values() for game in games]
    wins = sum(game['outcome'] == 'win' for game in all_games)
    losses = sum(game['outcome'] == 'loss' for game in all_games)
    draws = len(all_games) - wins - losses
    rng = np.random.default_rng(seed)
    boot_win, boot_score = np.zeros(repeats), np.zeros(repeats)
    pairs = 0
    for training_seed in sorted(games_by_seed):
        games = games_by_seed[training_seed]
        checked = paired_statistics(games, repeats=1)
        _same(checked['unpaired_completed_games'], 0, 'pooled input contains unpaired games')
        by_pair = {}
        for game in games:
            by_pair.setdefault(game['pair'], []).append(game)
        values = list(by_pair.values())
        _require(len(values) >= 2, 'pooled interval needs at least two layouts per fixed belief seed')
        win = np.asarray([sum(game['outcome'] == 'win' for game in pair) / 2 for pair in values])
        score = np.asarray([sum({'win': 1, 'draw': .5, 'loss': 0}[game['outcome']] for game in pair) / 2
                            for pair in values])
        choices = rng.integers(0, len(values), (repeats, len(values)))
        boot_win += win[choices].sum(1)
        boot_score += score[choices].sum(1)
        pairs += len(values)
    _require(len(all_games) == 2 * pairs and pairs > 0, 'pooled complete-pair count mismatch')
    return dict(completed_games=len(all_games), complete_pairs=pairs, wins=wins, losses=losses, draws=draws,
                win_rate=wins / len(all_games), score=(wins + .5 * draws) / len(all_games),
                win_rate_ci95=np.quantile(boot_win / pairs, [.025, .975]).tolist(),
                score_ci95=np.quantile(boot_score / pairs, [.025, .975]).tolist(),
                training_seeds=sorted(games_by_seed), bootstrap_repeats=repeats, bootstrap_seed=seed,
                interval_method='percentile bootstrap of layout pairs stratified by fixed trained belief seed',
                interval_scope='conditional on these trained belief seeds; not a training-seed-population interval')


class RecoveryAudit:
    def __init__(self, args):
        self.args, self.root = args, args.experiment_dir.resolve()
        self.protected, self.hashes = set(), {}
        self.report = dict(schema='junqi.facts-belief-ablation-recovery-audit.v1', status='failed',
                           checks=[], artifacts=[], errors=[], training={}, comparisons={}, pooled={},
                           limitations=[
                               'Verifies recovered artifacts; does not replay GPU kernels, matches, or neural NLL.',
                               'Joint conditional NLL and public-only marginal NLL have different conditioning.',
                               'Pooled intervals condition on two trained belief seeds, not the training population.'])
        self.phase = 'required_artifacts'

    def file(self, path, expected=None, *, role):
        from junqi_rl.belief.facts_dataset import file_sha256
        path = Path(path).resolve()
        self.protected.add(path)
        _require(path.is_file(), f'missing {role}: {path}')
        if path not in self.hashes:
            self.hashes[path] = file_sha256(path)
            self.report['artifacts'].append(dict(role=role, path=str(path), sha256=self.hashes[path],
                                                bytes=path.stat().st_size))
        if expected is not None:
            _same(self.hashes[path], expected, f'SHA-256 mismatch for {role}')
        return self.hashes[path]

    def dataset(self):
        import numpy as np
        from scripts.train_facts_joint_belief import _validate_manifest
        path = self.args.dataset.resolve()
        self.manifest_hash = self.file(path / 'dataset-manifest.json', self.protocol['dataset_manifest_sha256'],
                                       role='dataset_manifest')
        self.file(path / 'protocol.json', role='dataset_protocol')
        raw = _json(path / 'dataset-manifest.json')
        self.manifest, _ = _validate_manifest(SimpleNamespace(dataset=str(path),
            games=raw['protocol']['games'], positions_per_game=raw['protocol']['positions_per_game']), self.parent_hash)
        _same(_json(path / 'protocol.json'), self.manifest['protocol'], 'dataset protocol differs from manifest')
        _require(self.manifest['protocol'].get('acceptance_only') is False, 'formal training cannot use a smoke dataset')
        self.unknown_counts = dict(train=0, validation=0, test=0)
        for family in ('raw_games', 'shards'):
            for item in self.manifest[family]:
                artifact = _relative_file(path, item['path'])
                self.file(artifact, item['sha256'], role=f'dataset_{family}')
                if family == 'shards':
                    # NPZ selective access avoids decompressing spatial tensors or
                    # teacher masks just to establish supervision denominators.
                    with np.load(artifact, allow_pickle=False) as arrays:
                        known, domains = arrays['known_types'], arrays['type_domains']
                        _same(known.shape, (item['rows'], 75), 'dataset known identity shape mismatch')
                        _same(domains.shape, (item['rows'], 75, 12), 'dataset public domain shape mismatch')
                        self.unknown_counts[item['split']] += int(((known < 0) & (domains.sum(-1) > 1)).sum())
        _require(all(self.unknown_counts.values()), 'dataset fold lacks unknown identity targets')
        self.report['dataset'] = dict(manifest_sha256=self.manifest_hash,
            completed_games=self.manifest['completed_games'], split_games=self.manifest['split_games'],
            split_rows=self.manifest['split_rows'], unknown_targets=self.unknown_counts,
            artifact_hash_passes=1)
        self.report['checks'].append('immutable_dataset_hashes_and_all_fold_denominators_once')

    def source(self):
        source = (self.args.source_dir or self.root / 'source').resolve()
        manifest_path = self.root / 'source-sha256.json'
        manifest_hash = self.file(manifest_path, self.protocol.get('source_manifest_sha256'), role='source_sha256_manifest')
        manifest = _json(manifest_path)
        _require(isinstance(manifest, dict) and manifest, 'frozen source manifest is empty')
        self.source_manifest = manifest
        for relative, digest in manifest.items():
            _require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'invalid source hash')
            self.file(_relative_file(source, relative), digest, role='frozen_source_file')
        # Match the producer's inference_source_sha256() exactly, using only
        # the source manifest whose individual files were verified above.
        # Checking the entry script alone does not bind imported rules/models.
        inference_manifest = {relative: digest for relative, digest in manifest.items()
                              if relative.startswith(('junqi_core/', 'junqi_rl/'))
                              and relative.endswith('.py')}
        for package in ('junqi_core/', 'junqi_rl/'):
            _require(any(relative.startswith(package) for relative in inference_manifest),
                     f'frozen source manifest omits inference package {package}')
        self.inference_source_hash = _digest(inference_manifest)
        _require('scripts/compare_facts_joint.py' in manifest, 'comparison source omitted from frozen source manifest')
        self.comparison_source = self.args.comparison_source or source / 'scripts/compare_facts_joint.py'
        self.file(self.comparison_source, manifest['scripts/compare_facts_joint.py'], role='verified_comparison_source')
        self.report['source'] = dict(commit=self.protocol['source_commit'], files=len(manifest),
            manifest_sha256=manifest_hash, inference_source_sha256=self.inference_source_hash,
            bound_to_experiment_manifest='source_manifest_sha256' in self.protocol)
        self.report['checks'].append('all_frozen_source_file_hashes')

    def training(self, kind, seed):
        import torch
        from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
        from scripts.accept_facts_belief_training import _assert_finite, assert_exact
        from scripts.compare_facts_joint import selected_belief_provenance
        from scripts.train_facts_joint_belief import metric_names
        nll_key, _ = metric_names(kind)
        name = f'{kind}-s{seed}'
        root = self.root / 'training' / name
        selected_path, last_path = root / f'{kind}-belief.pt', root / 'last.pt'
        selected_hash = self.file(selected_path, role=f'{name}_selected')
        last_hash = self.file(last_path, role=f'{name}_resume')
        for filename in ('report.json', 'training-protocol.json', 'test-audit.json'):
            self.file(root / filename, role=f'{name}_{filename}')
        selected = torch.load(selected_path, map_location='cpu', weights_only=False)
        last = torch.load(last_path, map_location='cpu', weights_only=False)
        report, contract = _json(root / 'report.json'), selected.get('training_contract')
        _require(isinstance(contract, dict), f'{name}: missing training contract')
        _same(selected.get('artifact_role'), f'validation-selected-facts-{kind}-belief', f'{name}: wrong artifact role')
        for payload in (last, selected):
            _same(payload.get('parent_policy_sha256'), self.parent_hash, f'{name}: parent binding mismatch')
            _same(payload.get('training_contract'), contract, f'{name}: checkpoint contract mismatch')
        _same(_json(root / 'training-protocol.json'), contract, f'{name}: training protocol mismatch')
        _same(report.get('training_contract'), contract, f'{name}: report contract mismatch')
        expected = dict(schema=f'junqi.facts-{kind}-training.v1', seed=seed,
            dataset_manifest_sha256=self.manifest_hash, dataset_protocol_sha256=self.manifest['protocol_sha256'],
            parent_policy_sha256=self.parent_hash, model_config=selected['config'],
            games=self.manifest['protocol']['games'], positions_per_game=self.manifest['protocol']['positions_per_game'],
            batch_size=self.protocol['batch_size'], learning_rate=self.protocol['learning_rate'],
            optimizer='Adam', adam_betas=[.9, .999], adam_eps=1e-8, gradient_clip=.5,
            deterministic_algorithms=True, test_fold_used=False)
        for key, value in expected.items():
            _same(contract.get(key), value, f'{name}: frozen training {key} mismatch')
        _same(contract.get('loss'), ('unknown-only joint conditional cross entropy' if kind == 'joint'
                                    else 'unknown-only public-domain marginal cross entropy'), f'{name}: objective mismatch')
        _same(contract.get('selection'), f'minimum validation {"conditional" if kind == "joint" else "marginal"} NLL; earliest epoch wins ties',
              f'{name}: selection rule mismatch')
        if kind == 'marginal':
            _same(contract.get('training_support'), 'public type_domains only; no teacher-prefix masks',
                  f'{name}: marginal objective leaks teacher-prefix support')
        _require(report.get('finished') is True and report.get('test_fold_used') is False,
                 f'{name}: incomplete training or used test fold')
        epochs = self.protocol['epochs']
        for payload in (report, selected, last):
            _same(payload.get('epochs_completed'), epochs, f'{name}: completed epochs mismatch')
        _same(report.get('target_epochs'), epochs, f'{name}: target epochs mismatch')
        _same(report.get('last_checkpoint_sha256'), last_hash, f'{name}: resume hash mismatch')
        _same(report.get('selected_checkpoint_sha256'), selected_hash, f'{name}: selected hash mismatch')
        _same(last.get('resume_schema'), 1, f'{name}: missing resume schema')
        _require(last.get('optimizer') and last.get('rng'), f'{name}: missing optimizer/RNG state')
        _same(set(last['rng']), {'python', 'numpy', 'minibatch', 'torch_cpu', 'torch_cuda'}, f'{name}: incomplete RNG state')
        _assert_finite(last['optimizer'], f'{name}.optimizer')
        for field in ('split_games', 'split_rows'):
            _same(report.get(field), self.manifest[field], f'{name}: dataset {field} mismatch')
        _same(report.get('completed_collection_games'), self.manifest['completed_games'], f'{name}: collection budget mismatch')
        metrics = selected.get('epoch_metrics')
        _require(isinstance(metrics, list) and len(metrics) == epochs, f'{name}: incomplete epoch history')
        _same(last.get('epoch_metrics'), metrics, f'{name}: selected/resume epoch records disagree')
        _same(report.get('epoch_metrics'), metrics, f'{name}: report epoch records disagree')
        updates = 0
        for index, item in enumerate(metrics, start=1):
            _same(item.get('epoch'), index, f'{name}: unordered epochs')
            _integer(item.get('skipped_fully_known_batches'), label=f'{name} skipped batches')
            updates += math.ceil(self.manifest['split_rows']['train'] / contract['batch_size']) - item['skipped_fully_known_batches']
            _same(item.get('updates'), updates, f'{name}: optimizer update count mismatch')
            _same(item.get('train_unknown_targets'), self.unknown_counts['train'], f'{name}: train denominator mismatch')
            _number(item.get(f'train_{nll_key}'), label=f'{name} training NLL')
            _metrics(item['validation'], self.manifest['split_rows']['validation'], self.unknown_counts['validation'], kind, f'{name} epoch {index}')
        for payload in (report, last, selected):
            _same(payload.get('updates'), updates, f'{name}: final optimizer update count mismatch')
        epoch = min(range(epochs), key=lambda index: metrics[index]['validation'][nll_key]) + 1
        for payload in (report, last, selected):
            _same(payload.get('best_epoch'), epoch, f'{name}: selected epoch is not earliest minimum validation NLL')
        _same(selected.get('selected_epoch'), epoch, f'{name}: selected epoch mismatch')
        for payload in (last, selected):
            _same(payload.get('best_validation_nll'), metrics[epoch - 1]['validation'][nll_key], f'{name}: best NLL mismatch')
        _same(report.get('best_validation'), metrics[epoch - 1]['validation'], f'{name}: best validation mismatch')
        _same(report.get('final_validation'), metrics[-1]['validation'], f'{name}: final validation mismatch')
        _same(last.get('initial_validation'), selected.get('initial_validation'), f'{name}: initial validation mismatch')
        _same(report.get('initial_validation'), selected.get('initial_validation'), f'{name}: initial report mismatch')
        _metrics(selected['initial_validation'], self.manifest['split_rows']['validation'], self.unknown_counts['validation'], kind, f'{name} initial')
        try:
            assert_exact(selected['net'], last['best_model_state'], f'{name}.selected_vs_best_weights')
        except AssertionError as error:
            raise ValueError(f'{name}: selected_vs_best_weights differ') from error
        net = load_facts_belief(selected_path, policy_path=self.args.policy, expected_kind=kind)
        resume_net = load_facts_belief(last_path, policy_path=self.args.policy, expected_kind=kind)
        _same(resume_net.cfg, net.cfg, f'{name}: selected/resume architecture differs')
        _same(report.get('parameters'), net.num_parameters(), f'{name}: parameter count mismatch')
        side = selected_belief_provenance(selected_path, net, self.manifest_hash)
        side['policy_sha256'] = self.parent_hash
        if kind == 'marginal':
            _same(contract.get('sampling_distribution'), net.distribution_semantics, f'{name}: marginal distribution mismatch')
        _require(len(last['optimizer'].get('param_groups', [])) == 1, f'{name}: unexpected optimizer groups')
        group = last['optimizer']['param_groups'][0]
        _same(group.get('lr'), contract['learning_rate'], f'{name}: optimizer learning rate mismatch')
        _same(list(group.get('betas', ())), contract['adam_betas'], f'{name}: optimizer betas mismatch')
        _same(group.get('eps'), contract['adam_eps'], f'{name}: optimizer epsilon mismatch')
        parameters = list(resume_net.parameters())
        ids = group.get('params', [])
        _require(len(ids) == len(parameters) and len(set(ids)) == len(ids), f'{name}: optimizer parameter IDs mismatch')
        _same(set(last['optimizer']['state']), set(ids), f'{name}: missing Adam state for a trainable parameter')
        for index, parameter in zip(ids, parameters):
            state = last['optimizer']['state'][index]
            _require({'step', 'exp_avg', 'exp_avg_sq'} <= set(state), f'{name}: incomplete Adam resume state')
            _same(float(state['step']), float(updates), f'{name}: Adam step counter mismatch')
            for field in ('exp_avg', 'exp_avg_sq'):
                _same(state[field].shape, parameter.shape, f'{name}: Adam moment shape mismatch')
            _require(bool((state['exp_avg_sq'] >= 0).all()), f'{name}: negative Adam second moment')
        test = _json(root / 'test-audit.json')
        test_protocol = test.get('protocol', {})
        expected_test = dict(schema=f'junqi.facts-{kind}-final-test.v1', parent_policy_sha256=self.parent_hash,
            dataset_manifest_sha256=self.manifest_hash, selected_epoch=epoch, split='test',
            weights_updated=False, selection_performed_on_test=False)
        expected_test['joint_checkpoint_sha256' if kind == 'joint' else 'belief_checkpoint_sha256'] = selected_hash
        if kind == 'marginal':
            expected_test.update(belief_kind=kind, architecture_id=net.architecture_id)
        for key, value in expected_test.items():
            _same(test_protocol.get(key), value, f'{name}: held-out test {key} mismatch')
        _require(test.get('test_fold_used') is True, f'{name}: test fold not evaluated')
        _same(test.get('test_completed_games'), self.manifest['split_games']['test'], f'{name}: test game count mismatch')
        _same(test.get('validation_at_selection'), metrics[epoch - 1]['validation'], f'{name}: test selection binding mismatch')
        _metrics(test['test'], self.manifest['split_rows']['test'], self.unknown_counts['test'], kind, f'{name} test')
        self.report['training'][name] = dict(kind=kind, seed=seed, selected_epoch=epoch, epochs=epochs,
            parameters=net.num_parameters(), selected_sha256=selected_hash, resume_sha256=last_hash,
            validation=metrics[epoch - 1]['validation'], test=test['test'], updates=updates, side_provenance=side)
        return side, contract

    def comparison(self, seed, matchup, models, forbidden, evidence):
        from junqi_core.rules import MAX_NUM_MOVES
        from scripts.compare_facts_joint import paired_statistics, validated_games
        root = self.root / 'matches' / f's{seed}' / matchup
        for filename in ('protocol.json', 'comparison.json'):
            self.file(root / filename, role=f'{seed}_{matchup}_{filename}')
        protocol, aggregate = _json(root / 'protocol.json'), _json(root / 'comparison.json')
        candidate, opponent = matchup.split('-')
        sides = {'candidate': models[(candidate, seed)],
                 'opponent': (models[(opponent, seed)] if opponent != 'baseline'
                              else dict(kind='direct_policy', policy_sha256=self.parent_hash))}
        expected = dict(schema='junqi.facts-belief-comparison.v2', policy_sha256=self.parent_hash,
            joint_sha256=sides['candidate']['checkpoint_sha256'], dataset_manifest_sha256=self.manifest_hash,
            sides=sides, phase='formal', seed=self.protocol['layout_seeds'][str(seed)],
            pairs=self.protocol['pairs_per_seed'], max_moves=MAX_NUM_MOVES,
            search_config=self.protocol['search_config'], workers=self.protocol['workers'],
            baseline=sides['opponent'], candidate=sides['candidate'], team_assignments=[0, 1],
            action_selection='sample distribution', deterministic_algorithms=True,
            inference_source_sha256=self.inference_source_hash,
            exclusion_artifacts=evidence, forbidden_layout_count=len(forbidden),
            forbidden_layout_sha256=hashlib.sha256(json.dumps(sorted(forbidden)).encode()).hexdigest())
        for key, value in expected.items():
            _same(protocol.get(key), value, f'{seed}/{matchup}: match {key} binding mismatch')
        _require(isinstance(protocol.get('comparison_script_sha256'), str)
                 and re.fullmatch('[0-9a-f]{64}', protocol['comparison_script_sha256']), 'missing comparison source digest')
        self.file(self.comparison_source, protocol['comparison_script_sha256'], role='frozen_comparison_source')
        digest = _digest(protocol)
        games = validated_games(root, protocol, digest)
        stats = paired_statistics(games, seed=protocol['seed'])
        _same(stats['complete_pairs'], protocol['pairs'], f'{seed}/{matchup}: incomplete pairs')
        _same(stats['completed_games'], protocol['pairs'] * 2, f'{seed}/{matchup}: incomplete game budget')
        _same(stats['unpaired_completed_games'], 0, f'{seed}/{matchup}: unpaired games')
        for game in games:
            self.file(root / f"game-{game['pair']:06d}-{game['candidate_team']}.json", role='durable_completed_match')
            _require(game['setup_sha256'] not in forbidden, f'{seed}/{matchup}: evaluation layout overlap')
            _integer(game.get('plies'), label='complete-game plies', minimum=1)
            _require(game['plies'] <= MAX_NUM_MOVES, 'complete game exceeds rule move limit')
            _number(game.get('seconds'), label='complete-game seconds')
            _require(isinstance(game.get('action_history_sha256'), str)
                     and re.fullmatch('[0-9a-f]{64}', game['action_history_sha256']), 'missing durable action history digest')
            _same(game.get('search_metrics'), game['side_metrics']['candidate'], 'candidate legacy search metric alias differs')
            for role in ('candidate', 'opponent'):
                validate_search_metrics(game['side_metrics'][role], sides[role], protocol['search_config'],
                                        label=f'{seed}/{matchup}/{game["pair"]}/{role}')
        _same(aggregate.get('protocol'), protocol, 'aggregate protocol mismatch')
        _same(aggregate.get('protocol_sha256'), digest, 'aggregate protocol digest mismatch')
        _require(aggregate.get('finished') is True, 'match aggregate is unfinished')
        _same(aggregate.get('games'), games, 'aggregate game records differ from durable files')
        _same(aggregate.get('statistics'), stats, 'reported W/L/D or bootstrap intervals differ from actual games')
        self.report['comparisons'][f'{seed}/{matchup}'] = dict(statistics=stats, sides=sides,
            search=summarize_search(games), mean_game_seconds=sum(game['seconds'] for game in games) / len(games))
        return games

    def run(self):
        from scripts.compare_facts_joint import excluded_layouts
        self.file(self.root / 'protocol.json', role='frozen_experiment_protocol')
        self.protocol = _json(self.root / 'protocol.json')
        seeds = validate_experiment_protocol(self.protocol)
        self.report['protocol'] = self.protocol
        self.parent_hash = self.file(self.args.policy, self.protocol['parent_checkpoint_sha256'], role='frozen_parent_policy')
        self.phase = 'frozen_source_hashes'
        self.source()
        self.phase = 'immutable_dataset'
        self.dataset()
        self.phase = 'frozen_layout_exclusions'
        exclusions = []
        for filename in self.protocol.get('exclusion_files', []):
            path = _relative_file(self.root, filename)
            expected = self.protocol.get('exclusion_sha256', {}).get(filename)
            self.file(path, expected, role='protocol_bound_layout_exclusion')
            exclusions.append(path)
        exclusions += list(self.args.exclude_layouts)
        exclusions = sorted({path.resolve() for path in exclusions})
        for path in exclusions:
            self.file(path / 'protocol.json' if path.is_dir() else path, role='layout_exclusion_evidence')
        forbidden, evidence = excluded_layouts(self.manifest, exclusions)
        self.phase = 'strict_training_checkpoint_and_test_lineage'
        models, contracts = {}, {}
        for seed in seeds:
            for kind in ('joint', 'marginal'):
                models[(kind, seed)], contracts[(kind, seed)] = self.training(kind, seed)
        paired_fields = ('model_config', 'batch_size', 'learning_rate', 'adam_betas', 'adam_eps',
                         'gradient_clip', 'optimizer', 'device_type', 'deterministic_algorithms', 'cpu_threads')
        reference = contracts[('joint', seeds[0])]
        for contract in contracts.values():
            for field in paired_fields:
                _same(contract[field], reference[field], f'matched experiment differs in training {field}')
        self.report['checks'].append('all_selected_resume_weights_optimizer_validation_selection_and_test_bindings')
        self.phase = 'all_six_comparisons'
        games = {matchup: {} for matchup in MATCHUPS}
        for seed in seeds:
            for matchup in MATCHUPS:
                games[matchup][seed] = self.comparison(seed, matchup, models, forbidden, evidence)
        from scripts.compare_facts_joint import setup_sha256
        seed_layouts = [{setup_sha256(self.protocol['layout_seeds'][str(seed)] + pair)
                        for pair in range(self.protocol['pairs_per_seed'])} for seed in seeds]
        _require(seed_layouts[0].isdisjoint(seed_layouts[1]), 'formal layouts overlap across belief seeds')
        for matchup in MATCHUPS:
            self.report['pooled'][matchup] = pooled_statistics(games[matchup])
        self.report['checks'].append('both_side_models_search_stats_completed_pairs_and_recomputed_bootstrap')
        self.phase = 'audit_runtime_source_binding'
        # A later checkout must not silently substitute new game-layout or
        # statistical validation logic for the frozen experiment's functions.
        for module in tuple(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if not filename:
                continue
            path = Path(filename).resolve()
            if path.is_relative_to(_ROOT):
                relative = path.relative_to(_ROOT).as_posix()
                if relative in self.source_manifest:
                    self.file(path, self.source_manifest[relative], role='audit_runtime_frozen_dependency')
        self.file(Path(__file__), role='independent_recovery_auditor_source')
        self.report['checks'].append('loaded_project_validation_dependencies_match_frozen_source')
        self.phase = 'recovered_archive'
        if self.args.archive:
            self.file(self.args.archive, self.args.archive_sha256.lower(), role='archive_remote_digest')
            self.report['checks'].append('archive_matches_observed_remote_sha256')
        self.report['status'] = 'passed'


def audit(args):
    session = RecoveryAudit(args)
    try:
        # CUDA extension diagnostics must not corrupt the JSON on stdout.
        with redirect_stdout(sys.stderr):
            import torch
            torch.set_num_threads(1)
            session.run()
    except Exception as error:
        session.report['failed_phase'] = session.phase
        session.report['errors'].append(f'{type(error).__name__}: {error}')
    session.report['artifacts_checked'] = len(session.hashes)
    return session.report, session.protected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, help='External immutable frozen source; defaults to experiment/source')
    parser.add_argument('--comparison-source', type=Path, help='Defaults to the manifest-verified frozen comparison script')
    parser.add_argument('--exclude-layouts', action='append', type=Path, default=[])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if bool(args.archive) != bool(args.archive_sha256):
        parser.error('--archive and --archive-sha256 must be supplied together')
    if args.archive_sha256 and not re.fullmatch('[0-9a-fA-F]{64}', args.archive_sha256):
        parser.error('--archive-sha256 must be a SHA-256 digest')
    if args.output and args.output.exists():
        parser.error('--output must be a new file; recovered evidence is read-only')
    report, protected = audit(args)
    if args.output:
        if args.output.resolve() in protected:
            parser.error('--output must not overwrite recovered evidence')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
