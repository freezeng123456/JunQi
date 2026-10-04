#!/usr/bin/env python3
"""Frozen validation action ordering against paired, independent pi0 continuations.

One common candidate set and an equal-artifact mixture of accepted neural worlds
are shared by the depth-one value and Monte Carlo estimators. Discovery chooses
one action after averaging worlds; the independent audit half scores that fixed
choice. These continuations are development diagnostics, never formal games.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from junqi_rl.analysis.facts_joint_search import (
    FactsBeliefSearch, SearchConfig, _finished_value, clone_world, step_world,
)
from junqi_rl.analysis.random_eval import WEIGHTED_SAMPLING_VERSION
from junqi_rl.env import unrotate_compact_action_id
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_core.rules import Seat

try:
    from scripts.benchmark_facts_belief import restore_public_position, synchronize
    from scripts.calibrate_facts_search_value import (
        discounted_terminal_return, pi0_continuations, root_team_leaf_value,
        value_training_target_contract,
    )
    from scripts.compare_facts_joint import sha256, write_json
    from scripts.diagnose_facts_belief_search import (
        DIAGNOSTIC_SCHEMA, prefix_estimates, rank_correlation, repeated_q_summary, rng_digest,
        validate_state_record as validate_saved_state,
    )
    from scripts.score_facts_belief_samples import diagnostic_setup, verify_frozen_diagnostic_artifacts
except ModuleNotFoundError as exc:
    # Direct execution from a scripts-only import path must not mask a missing
    # model/dependency import underneath one of these helpers.
    if exc.name != 'scripts' and not str(exc.name).startswith('scripts.'):
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_facts_belief import restore_public_position, synchronize
    from calibrate_facts_search_value import (
        discounted_terminal_return, pi0_continuations, root_team_leaf_value,
        value_training_target_contract,
    )
    from compare_facts_joint import sha256, write_json
    from diagnose_facts_belief_search import (
        DIAGNOSTIC_SCHEMA, prefix_estimates, rank_correlation, repeated_q_summary, rng_digest,
        validate_state_record as validate_saved_state,
    )
    from score_facts_belief_samples import diagnostic_setup, verify_frozen_diagnostic_artifacts

SCHEMA = 'junqi.facts-action-quality.v1'
VALIDATION_FLAGS = ('support_validated', 'public_replay_validated',
                    'root_observation_validated', 'root_legal_actions_validated')
FALSE_FLAGS = ('test_fold_used', 'weights_updated', 'outcome_used_for_selection',
               'targets_used_for_selection', 'truth_passed_to_inference', 'manual_probability_used',
               'strategy_fallback_used', 'oracle_per_world_action_selection')
RECENCY_CHANNELS = list(range(CHANNEL_LAYOUT['cm_recency'].start, CHANNEL_LAYOUT['cm_recency'].stop))
SAVED_OBSERVATION_VALIDATION = dict(continuous_feature='public cm_recency exp(-elapsed/tau)',
    continuous_channels=RECENCY_CHANNELS, atol=1e-7, rtol=0.,
    other_channels='exact array equality', sampled_world_root_observation='exact array equality')


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    json.dumps(value, allow_nan=False)  # Fail before replacing any durable output.
    write_json(Path(path), value)


def bind_action_protocol(output, protocol):
    """Serialize fresh-directory validation with the atomic protocol write."""
    output = Path(output)
    with (output / '.protocol.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = output / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError('Action quality output directory belongs to another protocol')
        else:
            if any(item.name != '.protocol.lock' for item in output.iterdir()):
                raise ValueError('New action quality output directory must be empty')
            atomic_json(path, protocol)
    return json_digest(protocol)


def stable_mode(actions, scores):
    ids, values = np.asarray(actions), np.asarray(scores, dtype=np.float64)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(set(ids.tolist())) != len(ids) or values.shape != ids.shape
            or not np.isfinite(values).all()):
        raise ValueError('Action modes require unique integer IDs and finite scores')
    return int(ids[values == values.max()].min())


def validate_saved_observation(actual, saved):
    actual, saved = np.asarray(actual), np.asarray(saved)
    if (actual.shape != (317, 17, 17) or saved.shape != actual.shape
            or actual.dtype != np.float32 or saved.dtype != np.float32
            or not np.isfinite(actual).all() or not np.isfinite(saved).all()):
        raise ValueError('Public root observation has invalid shape/dtype/values')
    difference = np.abs(actual.astype(np.float64) - saved.astype(np.float64))
    other = np.ones(317, dtype=bool)
    other[RECENCY_CHANNELS] = False
    if (np.any(difference[other] != 0)
            or np.any(difference[RECENCY_CHANNELS] > SAVED_OBSERVATION_VALIDATION['atol'])):
        raise ValueError('Public root observation differs beyond public recency floating precision')
    coordinates = np.argwhere(difference != 0)
    return dict(maximum_absolute_difference=float(difference.max()), different_entries=len(coordinates),
                different_channels=(np.unique(coordinates[:, 0]).tolist() if len(coordinates) else []),
                actual_dtype=str(actual.dtype), saved_dtype=str(saved.dtype),
                contract=SAVED_OBSERVATION_VALIDATION)


def select_common_actions(saved, *, max_actions=5):
    """pi0 mode plus each fixed artifact's mean depth-one Q mode, before MC."""
    if type(max_actions) is not int or not 2 <= max_actions <= 5 or len(saved['models']) != 4:
        raise ValueError('Four frozen models and a maximum of two to five actions required')
    first = saved['models'][0]['repetitions'][0]['decision']
    ids, logp = first['legal_compact_action_ids'], first['pi0_log_probabilities']
    prior = stable_mode(ids, logp)
    chosen, modes = {prior}, []
    for model in sorted(saved['models'], key=lambda row: row['model']['checkpoint_sha256']):
        scores = []
        for repetition in model['repetitions']:
            decision = repetition['decision']
            if (decision['legal_compact_action_ids'] != ids
                    or not np.allclose(decision['pi0_log_probabilities'], logp, rtol=0, atol=1e-6)):
                raise ValueError('Saved repetitions must share the root legal prior')
            scores.append(decision['q'])
        mean = np.asarray(scores, dtype=np.float64).mean(0)
        mode = stable_mode(ids, mean)
        chosen.add(mode)
        modes.append(dict(checkpoint_sha256=model['model']['checkpoint_sha256'],
                          q_mode=mode, mean_q_at_mode=float(mean[ids.index(mode)]),
                          repetitions=len(scores)))
    runner_up = None
    if len(chosen) == 1 and len(ids) > 1:
        ranked = sorted(zip(ids, logp), key=lambda row: (-row[1], row[0]))
        runner_up = int(ranked[1][0])
        chosen.add(runner_up)
    if len(chosen) > max_actions:
        raise ValueError('Action budget would omit a frozen model Q mode')
    return dict(actions=[prior] + sorted(chosen - {prior}), pi0_mode=prior,
                saved_candidate_log_pi0=[float(logp[ids.index(a)]) for a in [prior] + sorted(chosen - {prior})],
                model_q_modes=modes, runner_up_fill=runner_up,
                legal_actions=len(ids), pi0_mass_on_candidates=float(sum(
                    np.exp(logp[ids.index(a)]) for a in chosen)),
                rule='pi0 mode + four mean saved depth-one Q modes; stable smallest-ID ties; '
                     'pi0 runner-up when only one distinct mode',
                scope='candidate subset ordering; no all-legal MC optimum claim')


def sampling_seed(base, game_id, checkpoint):
    return int.from_bytes(hashlib.sha256(json.dumps(
        ['fresh-accepted-worlds', base, game_id, checkpoint]).encode()).digest()[:8], 'little') % (2**63 - 1)


def continuation_seeds(base, game_id, world_origins, rollouts):
    """One 32-bit counter seed per world/repeat; action is deliberately absent."""
    if type(rollouts) is not int or rollouts < 4 or rollouts % 2:
        raise ValueError('At least four even continuation replicates required')
    values = [[int.from_bytes(hashlib.sha256(json.dumps([
        'independent-pi0-Q', base, game_id, row['checkpoint_sha256'], row['particle'], repeat
    ]).encode()).digest()[:4], 'little') for repeat in range(rollouts)] for row in world_origins]
    answer = np.asarray(values, dtype=np.int64)
    if len(set(answer.ravel().tolist())) != answer.size:
        raise ValueError('Continuation counter-seed collision; change the declared base seed')
    return answer


def paired_difference(returns, left, right, columns, origin_indices):
    """Pair actions first; distinguish fixed-pool MC noise and world sampling."""
    difference = (returns[left] - returns[right])[:, columns]
    n_worlds, n_repeats = difference.shape
    mc_variance = float(difference.var(1, ddof=1).sum() / (n_worlds**2 * n_repeats))
    means = difference.mean(1)
    groups = [means[np.asarray(indices)] for indices in origin_indices]
    total_variance = (float(sum(group.var(ddof=1) / len(group) for group in groups) / len(groups)**2)
                      if all(len(group) > 1 for group in groups) else None)
    return dict(mean=float(means.mean()), conditional_pool_mc_standard_error=float(np.sqrt(mc_variance)),
                stratified_world_cluster_standard_error=(float(np.sqrt(total_variance)) if total_variance is not None else None),
                world_variance_mc_noise_corrected=(total_variance - mc_variance if total_variance is not None else None),
                corrected_variance_clipped=False,
                uncertainty_scope='paired action differences; fixed-world MC SE; within-artifact '
                                  'world-cluster SE includes finite MC noise; four fitted artifacts fixed')


def summarize_action_values(actions, leaf_values, returns, logp, origins, *, beta=.5):
    """All action decisions follow averaging the same worlds, never per-world max."""
    leaf, mc = np.asarray(leaf_values, dtype=np.float64), np.asarray(returns, dtype=np.float64)
    if (leaf.ndim != 2 or mc.ndim != 3 or leaf.shape != mc.shape[:2]
            or leaf.shape[0] != len(actions) or leaf.shape[1] != len(origins)
            or mc.shape[2] < 4 or mc.shape[2] % 2 or not np.isfinite(leaf).all()
            or not np.isfinite(mc).all() or len(logp) != len(actions)):
        raise ValueError('Complete finite action/world/replicate arrays required')
    half = mc.shape[2] // 2
    discovery, audit = slice(0, half), slice(half, None)
    q_leaf, q_discovery, q_audit = leaf.mean(1), mc[:, :, discovery].mean((1, 2)), mc[:, :, audit].mean((1, 2))
    index = {aid: i for i, aid in enumerate(actions)}
    prior, value_mode = stable_mode(actions, logp), stable_mode(actions, q_leaf)
    mc_mode, audit_mode = stable_mode(actions, q_discovery), stable_mode(actions, q_audit)
    leaf_regularized = stable_mode(actions, np.asarray(logp) + q_leaf / beta)
    mc_regularized = stable_mode(actions, np.asarray(logp) + q_discovery / beta)
    grouped = {}
    for i, origin in enumerate(origins):
        grouped.setdefault(origin['checkpoint_sha256'], []).append(i)
    groups = list(grouped.values())
    if len(groups) != 4 or len({len(group) for group in groups}) != 1 or not groups[0]:
        raise ValueError('Equal four-artifact mixture with nonempty world groups required')
    pairs = []
    for a, b in itertools.combinations(actions, 2):
        left, right = index[a], index[b]
        pairs.append(dict(left=a, right=b, leaf_delta=float(q_leaf[left] - q_leaf[right]),
                          discovery=paired_difference(mc, left, right, discovery, groups),
                          audit=paired_difference(mc, left, right, audit, groups)))
    audit_contrasts = {}
    for name, a, b in (
        ('discovery_choice_minus_leaf_choice', mc_mode, value_mode),
        ('discovery_choice_minus_pi0_mode', mc_mode, prior),
        ('leaf_choice_minus_pi0_mode', value_mode, prior),
        ('discovery_regularized_minus_leaf_regularized', mc_regularized, leaf_regularized),
    ):
        audit_contrasts[name] = dict(left=a, right=b, **paired_difference(mc, index[a], index[b], audit, groups))
    return dict(q_leaf=q_leaf.tolist(), q_mc_discovery=q_discovery.tolist(), q_mc_audit=q_audit.tolist(),
                pi0_mode=prior, leaf_mode=value_mode, discovery_mc_mode=mc_mode,
                audit_mc_mode_diagnostic_only=audit_mode, leaf_regularized_mode=leaf_regularized,
                discovery_mc_regularized_mode=mc_regularized, conditional_candidate_beta=beta,
                leaf_discovery_ranking_correlation=rank_correlation(q_leaf, q_discovery),
                leaf_audit_ranking_correlation=rank_correlation(q_leaf, q_audit),
                discovery_audit_ranking_correlation=rank_correlation(q_discovery, q_audit),
                discovery_audit_mode_agreement=mc_mode == audit_mode,
                leaf_discovery_mode_changed=value_mode != mc_mode,
                audit_contrasts=audit_contrasts, action_pairs=pairs,
                choice_scope='one common root action selected after equal-artifact/world averaging; '
                             'discovery choices fixed before independent audit half; candidate subset only')


def assert_flags(protocol):
    if (protocol.get('schema') != SCHEMA or protocol.get('phase') != 'development'
            or protocol.get('split') != 'validation' or any(protocol.get(key) is not False for key in FALSE_FLAGS)
            or protocol.get('new_training_games') != 0 or protocol.get('formal_match_games') != 0
            or protocol.get('identity_encoding') != 'known_only'
            or protocol.get('observation_channels') != 317 or protocol.get('observation_global_dims') != 0):
        raise ValueError('Action quality protocol violates the frozen information/game contract')


def function_ast_sha256(path, name):
    matches = [node for node in ast.parse(Path(path).read_text()).body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise ValueError('Frozen replay/validation function is missing or duplicated')
    return hashlib.sha256(ast.dump(matches[0], include_attributes=False).encode()).hexdigest()


def reference_source_provenance(saved, source_root=None):
    """Bind original source; allow consumer wiring only after function comparison."""
    current = Path(__file__).resolve().parents[1]
    reference = Path(source_root) if source_root is not None else current
    helpers = {'scripts/diagnose_facts_belief_search.py': saved['script_sha256'],
               'scripts/benchmark_facts_belief.py': saved['replay_script_sha256']}
    if any(not (reference / name).is_file() or sha256(reference / name) != digest
           for name, digest in helpers.items()):
        raise ValueError('Original diagnostic helper hashes require the frozen --reference-source-root')
    files = sorted(path for package in ('junqi_core', 'junqi_rl')
                   for path in (reference / package).rglob('*.py'))
    if not files or any(not (reference / package).is_dir() for package in ('junqi_core', 'junqi_rl')):
        raise ValueError('Frozen reference source requires both inference/rules packages')
    manifest = {str(path.relative_to(reference)): sha256(path) for path in files}
    inference = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if inference != saved['inference_source_sha256']:
        raise ValueError('Original diagnostic transitive inference source hash mismatch')
    functions = {'scripts/benchmark_facts_belief.py': 'restore_public_position',
                 'scripts/diagnose_facts_belief_search.py': 'validate_state_record'}
    ast_hashes = {}
    for filename, function in functions.items():
        before, after = function_ast_sha256(reference / filename, function), function_ast_sha256(current / filename, function)
        if before != after:
            raise ValueError('Current replay/validation function differs from the frozen reference source')
        ast_hashes[f'{filename}:{function}'] = before
    return dict(helper_files_sha256=helpers, inference_source_sha256=inference,
                current_reference_function_ast_sha256=ast_hashes,
                saved_derived_float_validation=dict(atol=1e-12, rtol=1e-10,
                    scope='portable derived prefix/repeated floats only; source SHAs, raw cells and all structure fixed'),
                compatibility='full original source hashes verified; current replay/validation function '
                              'ASTs identical; current transitive core separately bound to original hash')


def portable_derived_equal(left, right):
    """Only recomputed floating statistics allow platform roundoff."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(portable_derived_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(portable_derived_equal(a, b) for a, b in zip(left, right))
    if type(left) is bool or type(right) is bool or left is None or right is None:
        return left is right
    if type(left) is int and type(right) is int:
        return left == right
    if type(left) in (float, int) and type(right) in (float, int):
        return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-10, atol=1e-12))
    return type(left) is type(right) and left == right


def validate_portable_saved_state(record, saved):
    normalized = deepcopy(record)
    for old, model in zip(record['models'], normalized['models']):
        for before, repetition in zip(old['repetitions'], model['repetitions']):
            expected = prefix_estimates(repetition['decision'], saved['particles'], saved['betas'])
            if not portable_derived_equal(before['prefix_estimates'], expected):
                raise ValueError('Saved prefix derivations differ beyond floating roundoff')
            repetition['prefix_estimates'] = expected
        expected = repeated_q_summary(model['repetitions'], saved['particles'], saved['betas'])
        if not portable_derived_equal(old['repeated_summary'], expected):
            raise ValueError('Saved repeated derivations differ beyond floating roundoff')
        model['repeated_summary'] = expected
    validate_saved_state(normalized, saved, 'shared')


def load_saved_diagnostics(directory, protocol, selection_path, reference_source_root=None):
    directory = Path(directory)
    path = directory / 'protocol.json'
    saved = json.loads(path.read_text())
    selection = json.loads(Path(selection_path).read_text())
    if (saved.get('schema') != DIAGNOSTIC_SCHEMA or saved.get('phase') != 'development'
            or saved.get('split') != 'validation' or saved.get('depth') != 1
            or saved.get('target_input') is not False or saved.get('teacher_masks_used') is not False
            or saved.get('test_fold_used') is not False or saved.get('states') != selection['count']
            or saved.get('selected_states') != selection['states']
            or any(saved.get(key) != protocol[key] for key in (
                'policy_sha256', 'dataset_manifest_sha256', 'selection_sha256', 'inference_source_sha256'))
            or sorted(saved.get('models', []), key=lambda row: row['checkpoint_sha256']) != protocol['models']):
        raise ValueError('Saved diagnostics/model/data/source provenance mismatch')
    source_provenance = reference_source_provenance(saved, reference_source_root)
    digest = json_digest(saved)
    files, records = {'protocol.json': sha256(path)}, {}
    expected = {f'state-{i:05d}-shared.json' for i in range(saved['states'])}
    if {p.name for p in directory.glob('state-*-shared.json')} != expected:
        raise ValueError('Saved diagnostics require complete distinct state coverage')
    for i in range(saved['states']):
        path = directory / f'state-{i:05d}-shared.json'
        record = json.loads(path.read_text())
        if record.get('protocol_sha256') != digest:
            raise ValueError('Saved state belongs to another diagnostic protocol')
        validate_portable_saved_state(record, saved)
        files[path.name] = sha256(path)
        records[record['state']['game_id']] = record
    return saved, records, files, source_provenance


def validate_record(record, protocol, digest):
    assert_flags(protocol)
    i = record.get('state_index')
    if (record.get('schema') != SCHEMA or record.get('complete') is not True
            or record.get('protocol_sha256') != digest or type(i) is not int
            or not 0 <= i < protocol['states'] or record.get('state') != protocol['selected_rows'][i]
            or record.get('selection') != protocol['action_sets'][i]
            or record.get('public_history_steps') != record['state']['ply']
            or record.get('public_history_sha256') != protocol['public_history_sha256'][i]
            or record.get('new_training_games') != 0 or record.get('formal_match_games') != 0):
        raise ValueError('Action quality state provenance/coverage mismatch')
    observation = record.get('saved_observation_validation', {})
    if (observation.get('contract') != protocol.get('saved_observation_validation')
            or protocol.get('saved_observation_validation') != SAVED_OBSERVATION_VALIDATION
            or not set(observation.get('different_channels', [])) <= set(RECENCY_CHANNELS)
            or not 0 <= observation.get('maximum_absolute_difference', float('inf')) <= 1e-7):
        raise ValueError('Public observation precision validation provenance mismatch')
    world_count = len(protocol['models']) * protocol['worlds_per_model']
    origins = record['world_origins']
    expected = [(model['checkpoint_sha256'], p) for model in protocol['models']
                for p in range(protocol['worlds_per_model'])]
    if [(row['checkpoint_sha256'], row['particle']) for row in origins] != expected:
        raise ValueError('Equal-mixture world origin coverage mismatch')
    sampling = record['sampling']
    if len(sampling) != 4:
        raise ValueError('Sampling artifact coverage mismatch')
    for row, model in zip(sampling, protocol['models']):
        assignments = np.asarray(row.get('assignments'))
        if (row.get('model') != model or assignments.shape != (protocol['worlds_per_model'], 75)
                or assignments.dtype.kind not in 'iu' or (assignments < 0).any() or (assignments >= 12).any()
                or len(row.get('piece_ids', [])) != 75 or len(set(row['piece_ids'])) != 75
                or any(row.get(key) is not True for key in VALIDATION_FLAGS)
                or row.get('seed') != sampling_seed(protocol['seed'], record['state']['game_id'], model['checkpoint_sha256'])):
            raise ValueError('Sampled world validation/provenance mismatch')
        origin_rows = [origin for origin in origins if origin['checkpoint_sha256'] == model['checkpoint_sha256']]
        if any(origin['assignment_sha256'] != json_digest(row['assignments'][origin['particle']]) for origin in origin_rows):
            raise ValueError('World origin assignment digest mismatch')
    legal_ids = record.get('legal_compact_action_ids', [])
    if (len(legal_ids) != record['selection']['legal_actions'] or len(set(legal_ids)) != len(legal_ids)
            or not set(record['selection']['actions']) <= set(legal_ids)
            or not np.allclose(record['candidate_log_pi0'], record['selection']['saved_candidate_log_pi0'], rtol=0, atol=2e-5)):
        raise ValueError('Root candidate legal/prior support mismatch')
    seeds = continuation_seeds(protocol['seed'], record['state']['game_id'], origins, protocol['rollouts'])
    if record['continuation_seeds'] != seeds.tolist() or len(record['actions']) != len(record['selection']['actions']):
        raise ValueError('Continuation seed/action coverage mismatch')
    leaf, returns = [], []
    gamma = protocol['value_training_target_contract']['gamma']
    for action, aid in zip(record['actions'], record['selection']['actions']):
        if (action.get('canonical_action') != aid or len(action['worlds']) != world_count
                or action.get('world_action') != unrotate_compact_action_id(aid, Seat(record['state']['observer_seat']))):
            raise ValueError('Root action/world coverage mismatch')
        predictions, values = [], []
        for w, row in enumerate(action['worlds']):
            prediction = row['prediction']
            if row.get('world_index') != w or len(row['continuations']) != protocol['rollouts']:
                raise ValueError('Action continuation cell coverage mismatch')
            if (prediction.get('next_actor') not in (0, 1, 2, 3)
                    or prediction.get('root_team') != record['state']['observer_seat'] & 1):
                raise ValueError('Leaf actor/root-team provenance mismatch')
            if prediction['source'] == 'frozen_next_actor_value_head':
                sign = 1 if (prediction['next_actor'] & 1) == (record['state']['observer_seat'] & 1) else -1
                if prediction['sign'] != sign or prediction['value'] != sign * prediction['next_actor_value']:
                    raise ValueError('Leaf actor/team sign mismatch')
            elif prediction['source'] not in ('terminal_team_reward', 'declared_total_move_limit_draw'):
                raise ValueError('Unknown leaf value source')
            discount = gamma if prediction['source'] == 'frozen_next_actor_value_head' else 1.
            if (prediction.get('root_step_discount') != discount
                    or prediction.get('root_action_value') != discount * prediction['value']):
                raise ValueError('Root action value discount mismatch')
            lane_values = []
            for repeat, result in enumerate(row['continuations']):
                if (result.get('completed') is not True or result.get('seed') != int(seeds[w, repeat])
                        or type(result.get('moves')) is not int or result['moves'] < 0
                        or type(result.get('terminal')) is not bool or type(result.get('draw')) is not bool
                        or result.get('root_team_return') not in (-1., 0., 1.)
                        or result.get('root_team_discounted_return') != discounted_terminal_return(
                            result['root_team_return'], result['moves'], gamma)
                        or result.get('root_action_discounted_return') != float(
                            result['root_team_return'] * gamma ** result['moves'])
                        or (not protocol['acceptance_only'] and result.get('termination_reason') == 'continuation_move_cap_draw')):
                    raise ValueError('Completed continuation return/seed/protocol mismatch')
                reason = result.get('termination_reason')
                expected_reason = ('terminal_draw' if result['terminal'] and result['draw'] else
                                   'team_victory' if result['terminal'] else
                                   'original_total_move_limit_draw' if reason == 'original_total_move_limit_draw' else
                                   'continuation_move_cap_draw')
                if (reason != expected_reason or (not result['terminal'] and not result['draw'])
                        or (result['draw'] and result['root_team_return'] != 0)
                        or (not result['draw'] and result['root_team_return'] == 0)):
                    raise ValueError('Continuation termination classification mismatch')
                lane_values.append(result['root_action_discounted_return'])
            predictions.append(prediction['root_action_value'])
            values.append(lane_values)
        leaf.append(predictions)
        returns.append(values)
    if record['summary'] != summarize_action_values(record['selection']['actions'], leaf, returns,
                                                    record['candidate_log_pi0'], origins, beta=protocol['beta']):
        raise ValueError('Action summaries differ from paired raw-return recomputation')


def aggregate_output(output, *, require_complete=False):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    assert_flags(protocol)
    digest, records, seen = json_digest(protocol), [], set()
    for path in sorted(output.glob('state-*.json')):
        row = json.loads(path.read_text())
        validate_record(row, protocol, digest)
        i = row['state_index']
        if i in seen or path.name != f'state-{i:05d}.json':
            raise ValueError('Duplicate or renamed action quality state')
        seen.add(i)
        records.append(row)
    complete = seen == set(range(protocol['states']))
    if len({row['state']['game_id'] for row in records}) != len(records):
        raise ValueError('Action quality intervals require distinct source games')
    if require_complete and not complete:
        raise ValueError('Action quality frozen state budget is not completed')
    names = ('discovery_choice_minus_leaf_choice', 'discovery_choice_minus_pi0_mode',
             'leaf_choice_minus_pi0_mode', 'discovery_regularized_minus_leaf_regularized')
    aggregate = {}
    for name in names:
        values = np.asarray([row['summary']['audit_contrasts'][name]['mean'] for row in records])
        interval = None
        if len(values) > 1:
            rng = np.random.default_rng(protocol['seed'])
            choices = rng.integers(0, len(values), (2000, len(values)))
            interval = np.quantile(values[choices].mean(1), [.025, .975]).tolist()
        aggregate[name] = dict(mean=float(values.mean()) if len(values) else None,
                               paired_source_game_bootstrap_ci95=interval)
    continuations = [r for row in records for action in row['actions'] for world in action['worlds']
                     for r in world['continuations']]
    reasons = {}
    for row in continuations:
        reasons[row['termination_reason']] = reasons.get(row['termination_reason'], 0) + 1
    summary = dict(schema=SCHEMA, protocol_sha256=digest, complete=complete,
        states_completed=len(records), states_planned=protocol['states'], source_games=len(records),
        actual_root_actions=sum(len(row['actions']) for row in records),
        max_actions_per_state=protocol['max_actions'], worlds_per_model=protocol['worlds_per_model'],
        mixture_worlds=protocol['mixture_worlds'], continuations_per_action_world=protocol['rollouts'],
        maximum_planned_continuations=protocol['states'] * protocol['max_actions'] * protocol['mixture_worlds'] * protocol['rollouts'],
        completed_continuations=len(continuations),
        terminal_continuations=sum(row['terminal'] for row in continuations),
        cutoff_continuations=sum(not row['terminal'] for row in continuations),
        continuation_environment_plies=sum(row['moves'] for row in continuations),
        termination_reasons=reasons,
        source_game_paired_audit=aggregate,
        discovery_audit_mode_agreement_fraction=(float(np.mean([
            row['summary']['discovery_audit_mode_agreement'] for row in records])) if records else None),
        leaf_discovery_mode_change_fraction=(float(np.mean([
            row['summary']['leaf_discovery_mode_changed'] for row in records])) if records else None),
        uncertainty_scope='whole source games, with paired actions/worlds and all four fixed artifacts kept '
                          'together; discovery choices scored only by audit half; no training-seed population claim',
        target_scope=protocol['target_scope'], acceptance_only=protocol['acceptance_only'],
        new_training_games=0, formal_match_games=0)
    atomic_json(output / 'action-quality-summary.json', summary)
    return summary


def batched_leaf_predictions(blueprint, leaves, root_seat, chunk_size, *, gamma=1.):
    """Reuse exact production terminal/team handling with bounded value batches."""
    cache = {}
    active = [world for world in leaves if _finished_value(world, root_seat) is None]
    for begin in range(0, len(active), chunk_size):
        chunk = active[begin:begin + chunk_size]
        values = blueprint.values(chunk).detach().cpu().numpy()
        cache.update({id(world): float(value) for world, value in zip(chunk, values)})
    class CachedValues:
        def values(self, worlds):
            return torch.tensor([cache[id(world)] for world in worlds], dtype=torch.float64)
    predictions = [root_team_leaf_value(CachedValues(), world, root_seat) for world in leaves]
    for prediction in predictions:
        discount = gamma if prediction['source'] == 'frozen_next_actor_value_head' else 1.
        prediction.update(root_step_discount=discount, root_action_value=discount * prediction['value'])
    return predictions


def run_state(blueprint, models, selected, saved, action_set, protocol, digest, index, args):
    item, raw, shard, trace, row = selected
    # Offline replay reconstructs the observer fact ledger. No original hidden
    # setup or targets enter the sampler/value/continuation inputs below.
    env, history = restore_public_position(trace, raw, int(shard['ply'][row]))
    root_seat = env.state.turn
    logp, _, legal, obs = blueprint.distribution([env])
    logp, legal, obs = logp.clone(), legal.clone(), obs.clone()
    observation_validation = validate_saved_observation(obs[0].cpu().numpy(), shard['spatial'][row])
    decision = saved['models'][0]['repetitions'][0]['decision']
    ids = legal[0].nonzero().flatten().cpu().tolist()
    if (ids != decision['legal_compact_action_ids']
            or not np.allclose(logp[0, ids].cpu().numpy(), decision['pi0_log_probabilities'], rtol=0, atol=2e-5)):
        raise ValueError('Frozen saved root legal prior differs from current pi0 inference')
    history_hash = hashlib.sha256(json.dumps(history, separators=(',', ':')).encode()).hexdigest()
    if history_hash != saved['public_history_sha256']:
        raise ValueError('Frozen public history differs from saved diagnostics')
    worlds, origins, sampling = [], [], []
    config = SearchConfig(particles=args.worlds_per_model, depth=1, proposal_batches=args.proposal_batches,
                          max_constraint_nodes=args.max_constraint_nodes)
    for model, _, provenance in models:
        seed = sampling_seed(args.seed, item['game_id'], provenance['checkpoint_sha256'])
        search = FactsBeliefSearch(blueprint, model, config, seed=seed, capture_diagnostics=True,
                                  inventory_backend=args.inventory_backend, overlap_constraints=False)
        before = rng_digest(search.generator)
        accepted = search.particles(env, history, obs, legal)
        telemetry = search.last_particle_diagnostics
        if len(accepted) != args.worlds_per_model or any(telemetry.get(key) is not True for key in VALIDATION_FLAGS):
            raise ValueError('Fresh worlds must pass every exact observer/public validation')
        sampling.append(dict(model=provenance, seed=seed, rng_before_sha256=before,
                             rng_after_sha256=rng_digest(search.generator), report=search.report(), **telemetry))
        for particle, world in enumerate(accepted):
            origins.append(dict(checkpoint_sha256=provenance['checkpoint_sha256'], particle=particle,
                                assignment_sha256=json_digest(telemetry['assignments'][particle])))
            worlds.append(world)
    seeds = continuation_seeds(args.seed, item['game_id'], origins, args.rollouts)
    actions, leaf_values, returns = [], [], []
    for aid in action_set['actions']:
        leaves = [clone_world(world) for world in worlds]
        world_action = unrotate_compact_action_id(aid, root_seat)
        for leaf in leaves:
            step_world(leaf, world_action)
        gamma = protocol['value_training_target_contract']['gamma']
        predictions = batched_leaf_predictions(blueprint, leaves, root_seat, args.leaf_value_chunk_size, gamma=gamma)
        measured, action_returns = [], []
        for w, (leaf, prediction) in enumerate(zip(leaves, predictions)):
            continuation = pi0_continuations(blueprint, leaf, root_seat, seeds[w],
                max_continuation_moves=args.continuation_max_moves,
                gamma=gamma)
            for result in continuation:
                result['root_action_discounted_return'] = float(result['root_team_return'] * gamma ** result['moves'])
            measured.append(dict(world_index=w, prediction=prediction, continuations=continuation))
            action_returns.append([r['root_action_discounted_return'] for r in continuation])
        actions.append(dict(canonical_action=aid, world_action=world_action, worlds=measured))
        leaf_values.append([prediction['root_action_value'] for prediction in predictions])
        returns.append(action_returns)
    candidate_logp = [float(logp[0, aid].cpu()) for aid in action_set['actions']]
    record = dict(schema=SCHEMA, complete=True, protocol_sha256=digest, state_index=index,
        state=protocol['selected_rows'][index], selection=action_set, legal_compact_action_ids=ids,
        public_history_sha256=history_hash, public_history_steps=len(history), sampling=sampling,
        saved_observation_validation=observation_validation,
        world_origins=origins, continuation_seeds=seeds.tolist(), candidate_log_pi0=candidate_logp,
        actions=actions, summary=summarize_action_values(action_set['actions'], leaf_values, returns,
                                                        candidate_logp, origins, beta=args.beta),
        new_training_games=0, formal_match_games=0)
    validate_record(record, protocol, digest)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('run', 'aggregate'), default='run')
    parser.add_argument('--policy')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--belief', action='append')
    parser.add_argument('--diagnostics', '--diagnostics-dir', dest='diagnostics', type=Path)
    parser.add_argument('--reference-source-root', type=Path,
                        help='Original frozen diagnostic source tree for exact source/hash verification')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int)
    parser.add_argument('--state-indices', nargs='+', type=int)
    parser.add_argument('--worlds-per-model', type=int, default=4)
    parser.add_argument('--max-actions', type=int, default=5)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--continuation-max-moves', type=int, default=0)
    parser.add_argument('--leaf-value-chunk-size', type=int, default=16)
    parser.add_argument('--proposal-batches', type=int, default=8)
    parser.add_argument('--max-constraint-nodes', type=int, default=100_000)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=97100500)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--inventory-backend', choices=('scan', 'incremental'), default='scan')
    parser.add_argument('--acceptance-only', action='store_true',
                        help='Label reduced execution checks without quality inference')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'aggregate':
        result = aggregate_output(args.output, require_complete=args.require_complete)
        print(json.dumps(result), flush=True)
        return result
    if any(getattr(args, key) is None for key in ('policy', 'dataset', 'selection', 'belief', 'diagnostics')):
        raise ValueError('Run mode requires frozen policy/dataset/selection/four beliefs/saved diagnostics')
    if (not 1 <= args.worlds_per_model <= 4 or not 2 <= args.max_actions <= 5 or args.rollouts < 4 or args.rollouts % 2
            or args.continuation_max_moves < 0 or args.leaf_value_chunk_size < 1 or args.seed < 0
            or args.workers < 1 or not 0 <= args.worker_index < args.workers
            or not np.isfinite(args.beta) or args.beta <= 0):
        raise ValueError('Invalid bounded action/world/replicate/worker protocol')
    device, selected, blueprint, models, protocol = diagnostic_setup(args)
    models.sort(key=lambda row: row[2]['checkpoint_sha256'])
    protocol['models'] = [row[2] for row in models]
    kinds = [row[2]['belief_kind'] for row in models]
    trained_seeds = {row[2]['training_seed'] for row in models}
    if (len(models) != 4 or sorted(kinds) != ['joint', 'joint', 'marginal', 'marginal']
            or len(trained_seeds) != 2 or len({(row[2]['belief_kind'], row[2]['training_seed']) for row in models}) != 4
            or protocol['selection_states'] != 32):
        raise ValueError('Gate requires the frozen 32-game selection and matched four M/J artifacts')
    saved_protocol, saved, files, reference = load_saved_diagnostics(
        args.diagnostics, protocol, args.selection, args.reference_source_root)
    action_sets = [select_common_actions(saved[row[0]['game_id']], max_actions=args.max_actions) for row in selected]
    target = value_training_target_contract(args.policy)
    protocol.update(schema=SCHEMA, formal_match_games=0, worlds_per_model=args.worlds_per_model,
        mixture_worlds=4 * args.worlds_per_model,
        max_actions=args.max_actions, rollouts=args.rollouts, discovery_replicates=args.rollouts // 2,
        audit_replicates=args.rollouts // 2, continuation_max_moves=args.continuation_max_moves,
        leaf_value_chunk_size=args.leaf_value_chunk_size, continuation_batch_lanes=args.rollouts,
        continuation_leaf_chunk_size=1, beta=args.beta, workers=args.workers,
        proposal_batches=args.proposal_batches, max_constraint_nodes=args.max_constraint_nodes,
        action_sets=action_sets, public_history_sha256=[saved[row[0]['game_id']]['public_history_sha256'] for row in selected],
        saved_diagnostic_protocol_sha256=json_digest(saved_protocol), saved_diagnostic_files_sha256=files,
        original_diagnostic_source=reference,
        script_sha256=sha256(__file__), calibration_helper_sha256=sha256(Path(__file__).with_name('calibrate_facts_search_value.py')),
        saved_observation_validation=SAVED_OBSERVATION_VALIDATION,
        value_training_target_contract=target,
        acceptance_only=(args.acceptance_only or args.continuation_max_moves > 0
                         or args.worlds_per_model != 4 or args.rollouts < 8),
        inventory_backend=args.inventory_backend, overlap_constraints=False,
        smoke=len(selected) != 32, continuation_sampling=WEIGHTED_SAMPLING_VERSION, continuation_stream=47,
        sampling_rng='independent sha256(base,game,checkpoint,fresh-accepted-worlds) torch generator',
        continuation_rng='sha256(base,game,checkpoint,particle,repeat,independent-pi0-Q) low32; '
                         'same counter stream across root actions; separate discovery/audit seeds',
        mixture_scope='equal four-artifact mixture, declared accepted samples per artifact; duplicate '
                      'assignments retain multiplicity; not a normalized true posterior',
        information_scope='offline source replay reconstructs observer facts; all inference uses current '
                          'actor legitimate known_only 317/0 inputs; future own identities are sampled '
                          'private information; original hidden labels never enter inference or selection',
        target_scope='independent pi0 terminal-return Q on common accepted-world mixture and candidate '
                     'actions; not intrinsic information-set critic calibration or formal playing strength')
    protocol.update({key: False for key in FALSE_FLAGS})
    assert_flags(protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    digest = bind_action_protocol(args.output, protocol)
    for index, state in enumerate(selected):
        if index % args.workers != args.worker_index:
            continue
        path = args.output / f'state-{index:05d}.json'
        if path.exists():
            validate_record(json.loads(path.read_text()), protocol, digest)
            continue
        began = time.perf_counter()
        record = run_state(blueprint, models, state, saved[state[0]['game_id']], action_sets[index],
                           protocol, digest, index, args)
        synchronize(device)
        record['wall_seconds'] = time.perf_counter() - began
        atomic_json(path, record)
        print(json.dumps(dict(phase='action_quality', state_index=index, source_game=state[0]['game_id'],
                              actions=len(record['actions']), completed_continuations=len(record['actions']) * len(record['world_origins']) * args.rollouts,
                              wall_seconds=record['wall_seconds'])), flush=True)
    verify_frozen_diagnostic_artifacts(args, models, protocol)
    if reference_source_provenance(saved_protocol, args.reference_source_root) != reference:
        raise RuntimeError('Original diagnostic source changed during run')
    if (sha256(__file__) != protocol['script_sha256']
            or sha256(Path(__file__).with_name('calibrate_facts_search_value.py')) != protocol['calibration_helper_sha256']
            or any(sha256(args.diagnostics / name) != digest for name, digest in files.items())):
        raise RuntimeError('Frozen action-quality source or saved diagnostics changed during run')
    return aggregate_output(args.output, require_complete=True) if args.workers == 1 else dict(worker_complete=True)


if __name__ == '__main__':
    main()
