#!/usr/bin/env python3
"""Validation-only Q diagnostics with shared particles and separate real timing.

The default protocol has 32 distinct games, four matched trained models,
1/2/4 particles, and four repetitions: 1,536 Q estimates. Each repetition
evaluates one four-particle pool and derives its prefixes offline. Real timing
invokes the uninstrumented production sampler independently for every budget.
Neither measurement is a new complete-game experiment or a win-rate estimate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from junqi_rl.analysis.facts_joint_search import FactsBlueprint, FactsBeliefSearch, SearchConfig
from junqi_rl.belief.facts_dataset import read_npz, validate_trace
from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA
from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
from junqi_rl.training.checkpoint import load_evaluation_policy
from scripts.benchmark_facts_belief import restore_public_position, synchronize
from scripts.compare_facts_joint import (
    bind_protocol, inference_source_sha256, selected_belief_provenance, sha256, write_json,
)

SELECTION_SCHEMA = 'junqi.facts-belief-selected-validation.v1'
DIAGNOSTIC_SCHEMA = 'junqi.facts-belief-search-diagnostics.v1'
STAGES = ('early', 'middle', 'late')


def game_stage(ply, moves):
    if type(ply) is not int or type(moves) is not int or not 0 <= ply < moves:
        raise ValueError('Selected ply must be inside a complete game')
    return STAGES[min(2, 3 * ply // moves)]


def _dataset_manifest(dataset):
    path = Path(dataset) / 'dataset-manifest.json'
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'junqi.facts-belief-dataset.v1' or manifest.get('status') != 'complete'
            or manifest.get('protocol', {}).get('constraint_schema') != CONSTRAINT_SCHEMA):
        raise ValueError('Diagnostics require a complete facts dataset')
    return manifest, sha256(path)


def _read_validation_game(dataset, item, original):
    if (item['split'] != 'validation' or original['split'] != 'validation'
            or original['game_id'] != item['game_id']
            or original['initial_setup_sha256'] != item['initial_setup_sha256']):
        raise ValueError('Selection must preserve the raw validation game grouping')
    shard_path, trace_path = Path(dataset) / item['path'], Path(dataset) / original['path']
    if sha256(shard_path) != item['sha256'] or sha256(trace_path) != original['sha256']:
        raise ValueError('Selected validation shard or trace hash changed')
    shard, trace = read_npz(shard_path), read_npz(trace_path)
    validate_trace(trace, original)
    plies = shard['ply']
    if (plies.dtype.kind not in 'iu' or (plies < 0).any() or (plies >= original['moves']).any()):
        raise ValueError('Validation shard ply outside the complete trace')
    return shard, trace


def selection_metadata(item, original, shard, trace, row):
    """Portable row identity; consumers reconstruct truth only offline."""
    ply, moves = int(shard['ply'][row]), int(original['moves'])
    observer = int(shard['observer_seats'][row])
    if observer != int(trace['seats'][ply]):
        raise ValueError('Selected observation must belong to the acting player')
    return dict(game_id=int(item['game_id']), row=int(row), ply=ply, moves=moves,
                relative_ply=ply / moves, stage=game_stage(ply, moves), observer_seat=observer,
                setup_sha256=item['initial_setup_sha256'], shard_sha256=item['sha256'],
                trace_sha256=original['sha256'], shard_path=item['path'], trace_path=original['path'])


def stratified_validation_states(dataset, manifest, count, seed):
    """Deterministic quota matching: one acting row from each distinct game.

    A bipartite matching prevents an early quota from consuming the only games
    eligible for a later quota. No outcome, truth label, Q, or test fold is used.
    Large shard arrays are discarded during candidate discovery and read back
    only for the selected games.
    """
    if type(count) is not int or count < 1 or seed < 0:
        raise ValueError('Positive state count and nonnegative seed required')
    candidates = [item for item in manifest['shards'] if item['split'] == 'validation']
    if len({item['game_id'] for item in candidates}) != len(candidates):
        raise ValueError('Duplicate validation game IDs')
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    raw = {item['game_id']: item for item in manifest['raw_games']}
    options = {}
    for item in candidates:
        original = raw[item['game_id']]
        shard, trace = _read_validation_game(dataset, item, original)
        rows = np.flatnonzero(shard['observer_seats'] == trace['seats'][shard['ply']])
        per_stage = {stage: [] for stage in STAGES}
        for row in rows:
            per_stage[game_stage(int(shard['ply'][row]), int(original['moves']))].append(int(row))
        for rows_in_stage in per_stage.values():
            rng.shuffle(rows_in_stage)
        options[item['game_id']] = per_stage
    slots = [STAGES[index % 3] for index in range(count)]
    matched_games, matched_slots = {}, {}

    def augment(slot, visited):
        for item in candidates:
            game = item['game_id']
            if game in visited or not options[game][slots[slot]]:
                continue
            visited.add(game)
            previous = matched_games.get(game)
            if previous is None or augment(previous, visited):
                matched_games[game], matched_slots[slot] = slot, game
                return True
        return False

    for slot in range(count):
        if not augment(slot, set()):
            raise ValueError('Insufficient distinct acting-observer validation games for stage quotas')
    by_id = {item['game_id']: item for item in candidates}
    selected = []
    for slot, stage in enumerate(slots):
        game = matched_slots[slot]
        item, original = by_id[game], raw[game]
        shard, trace = _read_validation_game(dataset, item, original)
        row = options[game][stage][0]
        selected.append((item, original, shard, trace, row))
    return selected


def create_selection(dataset, destination, count=32, seed=20261006):
    manifest, manifest_hash = _dataset_manifest(dataset)
    if Path(destination).exists():
        saved = json.loads(Path(destination).read_text())
        if saved.get('count') != count or saved.get('seed') != seed:
            raise ValueError('Existing selection has a different state count or seed')
        return load_selected_validation_states(dataset, destination)
    selected = stratified_validation_states(dataset, manifest, count, seed)
    record = dict(schema=SELECTION_SCHEMA, phase='development', split='validation',
                  dataset_manifest_sha256=manifest_hash, count=count, seed=seed,
                  selection_rule='seeded-game-order; exact-distinct-game-stage-quota-matching; '
                      'acting-observer-row; ply/full-complete-game-length thirds',
                  stage_quotas={stage: sum(index % 3 == i for index in range(count))
                                for i, stage in enumerate(STAGES)},
                  states=[selection_metadata(*state) for state in selected],
                  test_fold_used=False, outcome_used_for_selection=False,
                  targets_used_for_selection=False)
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(destination), record)
    return selected


def load_selected_validation_states(dataset, selection_path):
    """Shared loader for search, proposal scoring, and value calibration CLIs."""
    manifest, manifest_hash = _dataset_manifest(dataset)
    selected = json.loads(Path(selection_path).read_text())
    if (selected.get('schema') != SELECTION_SCHEMA or selected.get('split') != 'validation'
            or selected.get('dataset_manifest_sha256') != manifest_hash
            or selected.get('test_fold_used') is not False
            or selected.get('targets_used_for_selection') is not False
            or selected.get('outcome_used_for_selection') is not False):
        raise ValueError('Selected-state manifest provenance mismatch')
    shards = {item['game_id']: item for item in manifest['shards']}
    raw = {item['game_id']: item for item in manifest['raw_games']}
    states, seen = [], set()
    for row_record in selected['states']:
        game = row_record['game_id']
        if game in seen:
            raise ValueError('Selected states must use distinct games')
        seen.add(game)
        item, original = shards[game], raw[game]
        shard, trace = _read_validation_game(dataset, item, original)
        row = row_record['row']
        if type(row) is not int or not 0 <= row < len(shard['ply']):
            raise ValueError('Selected row outside shard')
        state = item, original, shard, trace, row
        if selection_metadata(*state) != row_record:
            raise ValueError('Selected-state metadata differs from hash-bound replay')
        states.append(state)
    if len(states) != selected['count'] or len(states) < 1:
        raise ValueError('Selected-state count mismatch')
    quotas = {stage: sum(row['stage'] == stage for row in selected['states']) for stage in STAGES}
    if quotas != selected['stage_quotas']:
        raise ValueError('Selected-state stage quotas mismatch')
    return states


def seed_for(base, state_index, model_index, repeat, stream, particles=0):
    packed = json.dumps([base, state_index, model_index, repeat, stream, particles]).encode()
    return int.from_bytes(hashlib.sha256(packed).digest()[:8], 'little') % (2 ** 63 - 1)


def rng_digest(generator):
    return hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest()


def improved_probabilities(log_pi0, q, beta):
    log_pi0, q = np.asarray(log_pi0, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if (log_pi0.ndim != 1 or log_pi0.shape != q.shape or not len(q)
            or not np.isfinite(log_pi0).all() or not np.isfinite(q).all()
            or not np.isfinite(beta) or beta <= 0):
        raise ValueError('Legal prior/Q and beta must be finite')
    logits = log_pi0 + q / beta
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def probability_shift(pi0, improved):
    pi0, improved = np.asarray(pi0, dtype=np.float64), np.asarray(improved, dtype=np.float64)
    pi0, improved = pi0 / pi0.sum(), improved / improved.sum()
    tiny = np.finfo(np.float64).tiny
    kl = lambda p, q: float(np.sum(p * (np.log(np.maximum(p, tiny)) - np.log(np.maximum(q, tiny)))))
    return dict(kl_search_to_pi0=kl(improved, pi0), kl_pi0_to_search=kl(pi0, improved),
                total_variation=float(np.abs(improved - pi0).sum() / 2),
                pi0_argmax_index=int(pi0.argmax()), search_argmax_index=int(improved.argmax()),
                argmax_changed=bool(pi0.argmax() != improved.argmax()))


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    begin = 0
    while begin < len(values):
        end = begin + 1
        while end < len(values) and values[order[end]] == values[order[begin]]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2
        begin = end
    return ranks


def rank_correlation(left, right):
    a, b = average_ranks(left), average_ranks(right)
    a, b = a - a.mean(), b - b.mean()
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    # Constant scores carry no ranking evidence, even if their tied arrays match.
    return float(np.dot(a, b) / norm) if norm else None


def prefix_estimates(decision, budgets, betas):
    returns = np.asarray(decision['particle_returns'], dtype=np.float64)
    if returns.ndim != 2 or not np.isfinite(returns).all():
        raise ValueError('Every legal action requires finite per-particle returns')
    estimates = []
    for particles in budgets:
        if particles < 1 or particles > returns.shape[1]:
            raise ValueError('Prefix budget exceeds the shared particle pool')
        q = returns[:, :particles].mean(1)
        estimates.append(dict(particles=particles, measurement='shared-particle-prefix-Q', q=q.tolist(),
            q_range=float(q.max() - q.min()), particle_ids=list(range(particles)),
            beta_sensitivity=[dict(beta=beta,
                probabilities=(p := improved_probabilities(decision['pi0_log_probabilities'], q, beta)).tolist(),
                **probability_shift(decision['pi0_probabilities'], p)) for beta in betas]))
    return estimates


def repeated_q_summary(records, budgets, betas):
    summaries = []
    for particles in budgets:
        estimates = [next(item for item in record['prefix_estimates'] if item['particles'] == particles)
                     for record in records]
        q = np.asarray([item['q'] for item in estimates])
        variance = q.var(0, ddof=1) if len(q) > 1 else None
        correlations = [rank_correlation(q[a], q[b]) for a, b in itertools.combinations(range(len(q)), 2)]
        valid = [value for value in correlations if value is not None]
        beta_summaries = []
        for beta in betas:
            row = [next(item for item in estimate['beta_sensitivity'] if item['beta'] == beta)
                   for estimate in estimates]
            modes = [item['search_argmax_index'] for item in row]
            agreement = [a == b for a, b in itertools.combinations(modes, 2)]
            beta_summaries.append(dict(beta=beta,
                mean_kl_search_to_pi0=float(np.mean([item['kl_search_to_pi0'] for item in row])),
                mean_kl_pi0_to_search=float(np.mean([item['kl_pi0_to_search'] for item in row])),
                mean_total_variation=float(np.mean([item['total_variation'] for item in row])),
                argmax_change_fraction=float(np.mean([item['argmax_changed'] for item in row])),
                repeated_argmax_agreement=float(np.mean(agreement)) if agreement else None))
        summaries.append(dict(particles=particles, repeats=len(records),
            q_variance_per_legal_action=variance.tolist() if variance is not None else None,
            mean_q_repeat_variance=float(variance.mean()) if variance is not None else None,
            max_q_repeat_variance=float(variance.max()) if variance is not None else None,
            mean_q_range=float(np.mean([item['q_range'] for item in estimates])),
            mean_pairwise_q_spearman=float(np.mean(valid)) if valid else None,
            informative_ranking_pairs=len(valid), total_ranking_pairs=len(correlations),
            beta_sensitivity=beta_summaries))
    return summaries


def _memory(device):
    # ru_maxrss is a process-lifetime peak, not an incremental per-decision peak.
    scale = 1 if sys.platform == 'darwin' else 1024
    answer = dict(process_lifetime_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale))
    if device.type == 'cuda':
        answer.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return answer


def _run_shared(env, history, blueprint, model, model_index, state_index, args, provenance):
    config = SearchConfig(particles=max(args.particles), depth=1, beta=args.beta,
                          proposal_batches=args.proposal_batches, max_constraint_nodes=args.max_constraint_nodes)
    records = []
    for repeat in range(args.repeats):
        seed = seed_for(args.seed, state_index, model_index, repeat, 'shared-pool')
        search = FactsBeliefSearch(blueprint, model, config, seed=seed, capture_diagnostics=True,
            inventory_backend=args.inventory_backend,
            overlap_constraints=args.overlap_constraints and model.belief_kind == 'joint')
        before = rng_digest(search.generator)
        synchronize(blueprint.device)
        began = time.perf_counter()
        action = search.action(env, history)
        synchronize(blueprint.device)
        decision = search.last_decision
        records.append(dict(repeat=repeat, rng_seed=seed, rng_before_sha256=before,
            rng_after_sha256=rng_digest(search.generator), sampled_world_action=action,
            diagnostic_wall_seconds=time.perf_counter() - began,
            decision=decision, report=search.report(),
            prefix_estimates=prefix_estimates(decision, args.particles, args.betas)))
    return dict(model=provenance, config=asdict(config), measurement='shared-particle-pool',
                timing_scope='instrumented full-pool work; not deployed prefix-budget latency',
                repetitions=records, repeated_summary=repeated_q_summary(records, args.particles, args.betas))


def _run_timing(env, history, blueprint, models, state_index, args, provenances):
    tasks = list(itertools.product(range(len(models)), args.particles, range(args.timing_repeats)))
    rng = np.random.default_rng(seed_for(args.seed, state_index, 0, 0, 'timing-order'))
    rng.shuffle(tasks)
    records = []
    for ordinal, (model_index, particles, repeat) in enumerate(tasks):
        seed = seed_for(args.seed, state_index, model_index, repeat, 'actual-runtime', particles)
        config = SearchConfig(particles=int(particles), depth=1, beta=args.beta,
                              proposal_batches=args.proposal_batches, max_constraint_nodes=args.max_constraint_nodes)
        model = models[model_index][0]
        search = FactsBeliefSearch(blueprint, model, config, seed=seed,
            inventory_backend=args.inventory_backend,
            overlap_constraints=args.overlap_constraints and model.belief_kind == 'joint')
        before = rng_digest(search.generator)
        synchronize(blueprint.device)
        if blueprint.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(blueprint.device)
        began = time.perf_counter()
        action = search.action(env, history)
        synchronize(blueprint.device)
        elapsed = time.perf_counter() - began
        records.append(dict(measurement='independent-full-production-search', ordinal=ordinal,
            model=provenances[model_index], particles=int(particles), repeat=int(repeat),
            rng_seed=seed, rng_before_sha256=before, rng_after_sha256=rng_digest(search.generator),
            selected_world_action=action, wall_seconds=elapsed, memory=_memory(blueprint.device), report=search.report()))
    return records


def _mean_defined(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def validate_state_record(record, protocol, kind):
    """Fail closed on missing/duplicated cells and inconsistent derived Q."""
    index = record.get('state_index')
    if (record.get('schema') != DIAGNOSTIC_SCHEMA or record.get('complete') is not True
            or record.get('measurement') != kind or type(index) is not int
            or not 0 <= index < protocol['states']
            or record.get('state') != protocol['selected_states'][index]
            or record.get('public_history_steps') != record['state']['ply']):
        raise ValueError('Diagnostic state record metadata mismatch')
    models = {model['checkpoint_sha256']: model for model in protocol['models']}
    if kind == 'shared':
        identities = [model['model']['checkpoint_sha256'] for model in record['models']]
        if len(identities) != len(models) or set(identities) != set(models):
            raise ValueError('Shared diagnostic model coverage mismatch')
        for model in record['models']:
            identity = model['model']['checkpoint_sha256']
            if model['model'] != models[identity]:
                raise ValueError('Shared diagnostic model provenance mismatch')
            repetitions = model['repetitions']
            if sorted(row['repeat'] for row in repetitions) != list(range(protocol['repeats'])):
                raise ValueError('Shared diagnostic repetition coverage mismatch')
            for repetition in repetitions:
                decision = repetition['decision']
                actions = decision['legal_compact_action_ids']
                count = len(actions)
                returns = np.asarray(decision['particle_returns'], dtype=np.float64)
                particles = decision['particles']
                if (not count or len(set(actions)) != count
                        or len(decision['legal_world_action_ids']) != count
                        or returns.shape != (count, max(protocol['particles']))
                        or not np.isfinite(returns).all()
                        or np.asarray(particles['assignments']).shape != (max(protocol['particles']), 75)
                        or len(particles['piece_ids']) != 75
                        or not all(particles.get(key) is True for key in (
                            'support_validated', 'public_replay_validated',
                            'root_observation_validated', 'root_legal_actions_validated'))):
                    raise ValueError('Shared diagnostic legal action/particle coverage mismatch')
                for probabilities in ('pi0_probabilities', 'search_probabilities'):
                    p = np.asarray(decision[probabilities], dtype=np.float64)
                    if p.shape != (count,) or not np.isfinite(p).all() or (p < 0).any() or abs(p.sum() - 1) > 2e-6:
                        raise ValueError('Shared diagnostic probabilities malformed')
                q = np.asarray(decision['q'])
                if q.shape != (count,) or not np.allclose(q, returns.mean(1), rtol=0, atol=1e-6):
                    raise ValueError('Shared diagnostic Q differs from per-particle returns')
                expected = prefix_estimates(decision, protocol['particles'], protocol['betas'])
                if repetition['prefix_estimates'] != expected:
                    raise ValueError('Shared prefix Q/beta diagnostics differ from independent recomputation')
            if model['repeated_summary'] != repeated_q_summary(repetitions, protocol['particles'], protocol['betas']):
                raise ValueError('Repeated Q summary differs from independent recomputation')
    else:
        expected = set(itertools.product(models, protocol['particles'], range(protocol['timing_repeats'])))
        cells = []
        for search in record['searches']:
            identity = search['model']['checkpoint_sha256']
            cell = identity, search['particles'], search['repeat']
            cells.append(cell)
            config, report = search['report']['config'], search['report']
            if (identity not in models or search['model'] != models[identity]
                    or search['measurement'] != 'independent-full-production-search'
                    or config['particles'] != search['particles'] or config['depth'] != 1
                    or config['beta'] != protocol['beta'] or report['successful_decisions'] != 1
                    or report['candidate_actions'] != report['legal_actions'] or not report['legal_actions']
                    or not np.isfinite(search['wall_seconds']) or search['wall_seconds'] <= 0):
                raise ValueError('True-runtime diagnostic search provenance or coverage mismatch')
        if len(cells) != len(expected) or set(cells) != expected:
            raise ValueError('True-runtime diagnostic cell coverage mismatch')


def aggregate_output(output, protocol, scope='all'):
    selected_count, model_count = protocol['states'], len(protocol['models'])
    expected_indices = {'shared': set(range(selected_count)),
                        'timing': set(protocol.get('timing_state_indices', range(selected_count)))}
    records = {'shared': [], 'timing': []}
    for kind in records:
        for path in sorted(Path(output).glob(f'state-*-{kind}.json')):
            record = json.loads(path.read_text())
            if record.get('protocol_sha256') != hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest():
                raise ValueError('Diagnostic result protocol mismatch')
            validate_state_record(record, protocol, kind)
            records[kind].append(record)
        ids = [record['state_index'] for record in records[kind]]
        if len(set(ids)) != len(ids) or any(index not in expected_indices[kind] for index in ids):
            raise ValueError('Duplicate or out-of-range diagnostic states')
    required = ('shared', 'timing') if scope == 'all' else (scope,)
    complete = all({record['state_index'] for record in records[kind]} == expected_indices[kind] for kind in required)
    summary = dict(schema=DIAGNOSTIC_SCHEMA, complete=complete, scope=scope, protocol=protocol,
                   completed_games_added=0, states_completed={kind: len(rows) for kind, rows in records.items()},
                   shared_q_estimates=0, shared_sampler_searches=0, true_runtime_searches=0,
                   stage_and_model_summaries=[], timing_summaries=[])
    groups, timing_groups = {}, {}
    for record in records['shared']:
        if len(record['models']) != model_count:
            raise ValueError('Diagnostic state model coverage mismatch')
        for model in record['models']:
            if len(model['repetitions']) != protocol['repeats']:
                raise ValueError('Diagnostic repetition coverage mismatch')
            summary['shared_sampler_searches'] += len(model['repetitions'])
            summary['shared_q_estimates'] += len(model['repetitions']) * len(protocol['particles'])
            identity = model['model']['checkpoint_sha256']
            for stage in ('all', record['state']['stage']):
                for row in model['repeated_summary']:
                    groups.setdefault((identity, stage, row['particles']), []).append(row)
    for (identity, stage, particles), rows in sorted(groups.items()):
        summary['stage_and_model_summaries'].append(dict(checkpoint_sha256=identity, stage=stage,
            particles=particles, independent_games=len(rows),
            mean_q_repeat_variance=_mean_defined([row['mean_q_repeat_variance'] for row in rows]),
            mean_q_range=_mean_defined([row['mean_q_range'] for row in rows]),
            mean_pairwise_q_spearman=_mean_defined([row['mean_pairwise_q_spearman'] for row in rows]),
            beta_sensitivity=[dict(beta=beta, **{key: _mean_defined([
                next(item for item in row['beta_sensitivity'] if item['beta'] == beta)[key]
                for row in rows]) for key in ('mean_kl_search_to_pi0', 'mean_kl_pi0_to_search',
                    'mean_total_variation', 'argmax_change_fraction', 'repeated_argmax_agreement')})
                for beta in protocol['betas']]))
    for record in records['timing']:
        expected = model_count * len(protocol['particles']) * protocol['timing_repeats']
        if len(record['searches']) != expected:
            raise ValueError('True-runtime state coverage mismatch')
        summary['true_runtime_searches'] += len(record['searches'])
        for search in record['searches']:
            for stage in ('all', record['state']['stage']):
                key = search['model']['checkpoint_sha256'], stage, search['particles']
                timing_groups.setdefault(key, []).append(search['wall_seconds'])
    for (identity, stage, particles), values in sorted(timing_groups.items()):
        summary['timing_summaries'].append(dict(checkpoint_sha256=identity, stage=stage, particles=particles,
            searches=len(values), seconds_mean=float(np.mean(values)), seconds_p50=float(np.median(values)),
            seconds_p95=float(np.quantile(values, .95))))
    summary['measurement_scope'] = ('Shared particle prefixes describe coupled Q estimates; latency uses independent '
        'production samplers. Stage/model aggregates weight each selected game equally. No win rates are estimated.')
    write_json(Path(output) / 'diagnostic-summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--belief', action='append', default=[])
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('select', 'shared', 'timing', 'all', 'aggregate'), default='all')
    parser.add_argument('--aggregate-scope', choices=('shared', 'timing', 'all'), default='all')
    parser.add_argument('--require-complete', action='store_true',
        help='Fail an aggregate acceptance gate when its frozen state budget is incomplete')
    parser.add_argument('--states', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261006)
    parser.add_argument('--particles', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--timing-repeats', type=int, default=4)
    parser.add_argument('--state-indices', nargs='+', type=int,
        help='Frozen timing subset indices only; shared-Q always uses all selected states. Pass identically in all modes.')
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--betas', nargs='+', type=float, default=[.25, .5, 1., 2.])
    parser.add_argument('--proposal-batches', type=int, default=8)
    parser.add_argument('--max-constraint-nodes', type=int, default=100000)
    parser.add_argument('--inventory-backend', choices=('scan', 'incremental'), default='scan')
    parser.add_argument('--overlap-constraints', action='store_true', help='Opt-in joint-only GPU/CPU overlap')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Bound to <=2 states and one repeat; permit unmatched models')
    args = parser.parse_args(argv)
    if args.smoke:
        args.states = min(args.states, 2)
        args.repeats = args.timing_repeats = 1
    if (min(args.states, args.repeats, args.timing_repeats, args.workers, args.proposal_batches,
            args.max_constraint_nodes) < 1 or args.seed < 0 or not 0 <= args.worker_index < args.workers
            or not args.particles or min(args.particles) < 1 or len(set(args.particles)) != len(args.particles)
            or (args.state_indices is not None and (len(set(args.state_indices)) != len(args.state_indices)
                or any(index not in range(args.states) for index in args.state_indices)))
            or not args.betas or any(not np.isfinite(beta) or beta <= 0 for beta in [args.beta, *args.betas])):
        parser.error('Invalid diagnostic count, worker, seed, particle, or beta budget')
    if args.mode == 'select':
        selected = create_selection(args.dataset, args.selection, args.states, args.seed)
        print(json.dumps(dict(selection=str(args.selection), states=len(selected), sha256=sha256(args.selection))), flush=True)
        return selected
    if args.output is None:
        parser.error('--output is required for search or aggregation')
    if args.mode == 'aggregate':
        protocol = json.loads((args.output / 'protocol.json').read_text())
        summary = aggregate_output(args.output, protocol, args.aggregate_scope)
        print(json.dumps(dict(complete=summary['complete'], shared_q_estimates=summary['shared_q_estimates'],
                             true_runtime_searches=summary['true_runtime_searches'])), flush=True)
        if args.require_complete and not summary['complete']:
            raise ValueError('Diagnostic aggregate has not completed its frozen state budget')
        return summary
    if not args.policy or not args.belief:
        parser.error('--policy and --belief are required for search')
    if args.mode in ('timing', 'all') and args.workers != 1:
        parser.error('True-runtime latency must run in one process; use parallel --mode shared separately')
    args.output.mkdir(parents=True, exist_ok=True)
    chosen = load_selected_validation_states(args.dataset, args.selection)
    if len(chosen) != args.states:
        raise ValueError('CLI states differ from the frozen selected-state manifest')
    manifest, manifest_hash = _dataset_manifest(args.dataset)
    if manifest['protocol']['parent_checkpoint_sha256'] != sha256(args.policy):
        raise ValueError('Dataset parent differs from frozen diagnostic policy')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    blueprint = FactsBlueprint(load_evaluation_policy(args.policy, device=device), device)
    models = [(load_facts_belief(path, policy_path=args.policy, device=device), Path(path)) for path in args.belief]
    provenances = [selected_belief_provenance(path, model, manifest_hash) for model, path in models]
    matched = {(item['belief_kind'], item['training_seed']) for item in provenances}
    seeds = {item['training_seed'] for item in provenances}
    if not args.smoke and (len(models) != 4 or len(seeds) != 2
            or matched != set(itertools.product(('joint', 'marginal'), seeds))):
        raise ValueError('Full diagnostics require M/J checkpoints for each of two matched training seeds')
    protocol = dict(schema=DIAGNOSTIC_SCHEMA, phase='development', split='validation',
        policy_sha256=sha256(args.policy), dataset_manifest_sha256=manifest_hash,
        selection_sha256=sha256(args.selection), selection_schema=SELECTION_SCHEMA,
        states=args.states, models=provenances, particles=args.particles, repeats=args.repeats,
        selected_states=[selection_metadata(*selected) for selected in chosen],
        timing_state_indices=sorted(args.state_indices) if args.state_indices is not None else list(range(args.states)),
        timing_repeats=args.timing_repeats, beta=args.beta, betas=args.betas, depth=1,
        proposal_batches=args.proposal_batches, max_constraint_nodes=args.max_constraint_nodes,
        seed=args.seed, device_type=device.type, device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        torch_version=str(torch.__version__), numpy_version=str(np.__version__),
        script_sha256=sha256(__file__), replay_script_sha256=sha256(Path(__file__).with_name('benchmark_facts_belief.py')),
        inference_source_sha256=inference_source_sha256(), target_input=False, teacher_masks_used=False,
        inventory_backend=args.inventory_backend, joint_overlap_constraints=args.overlap_constraints,
        test_fold_used=False, smoke=args.smoke, runtime_parallel_workers=1,
        warmup_searches_per_process_per_model=1, rng_scheme='sha256(base,state,model,repeat,stream,particles); torch-generator')
    digest = bind_protocol(args.output, protocol)
    warmup_done = False
    for index, selected in enumerate(chosen):
        if index % args.workers != args.worker_index:
            continue
        if args.mode == 'timing' and index not in protocol['timing_state_indices']:
            continue
        item, original, shard, trace, row = selected
        metadata = selection_metadata(*selected)
        env, history = restore_public_position(trace, original, metadata['ply'])
        actual = blueprint.builder.build(env.state, None, env.state.turn).spatial
        if not np.allclose(actual, shard['spatial'][row], rtol=0, atol=1e-7):
            raise ValueError('Saved validation observation differs from replayed public root')
        if not warmup_done:
            config = SearchConfig(particles=max(args.particles), depth=1, beta=args.beta,
                                  proposal_batches=args.proposal_batches, max_constraint_nodes=args.max_constraint_nodes)
            for model_index, (model, path) in enumerate(models):
                search = FactsBeliefSearch(blueprint, model, config,
                    seed=seed_for(args.seed, index, model_index, 0, 'warmup'),
                    inventory_backend=args.inventory_backend,
                    overlap_constraints=args.overlap_constraints and model.belief_kind == 'joint')
                search.action(env, history)
            synchronize(device)
            warmup_done = True
        history_hash = hashlib.sha256(json.dumps(history, separators=(',', ':')).encode()).hexdigest()
        modes = ('shared', 'timing') if args.mode == 'all' else (args.mode,)
        for mode in modes:
            if mode == 'timing' and index not in protocol['timing_state_indices']:
                continue
            path = args.output / f'state-{index:05d}-{mode}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if saved.get('protocol_sha256') != digest or saved.get('state') != metadata or not saved.get('complete'):
                    raise ValueError('Existing state diagnostics belong to another/incomplete protocol')
                validate_state_record(saved, protocol, mode)
                continue
            record = dict(schema=DIAGNOSTIC_SCHEMA, protocol_sha256=digest, complete=True,
                          state_index=index, state=metadata, public_history_sha256=history_hash,
                          public_history_steps=len(history), measurement=mode)
            if mode == 'shared':
                record['models'] = [_run_shared(env, history, blueprint, model, model_index, index, args,
                                               provenances[model_index]) for model_index, (model, path) in enumerate(models)]
            else:
                record['searches'] = _run_timing(env, history, blueprint, models, index, args, provenances)
            write_json(path, record)
            print(json.dumps(dict(state_index=index, stage=metadata['stage'], measurement=mode,
                                  states_total=args.states, complete=True)), flush=True)
    # Multiworker aggregation is a separate command after all workers exit.
    if args.workers == 1:
        aggregate_output(args.output, protocol, args.mode)
    return dict(output=str(args.output), worker_index=args.worker_index, workers=args.workers)


if __name__ == '__main__':
    main()
