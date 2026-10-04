"""Q telemetry must preserve inference and separate shared-pool cost from timing."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch

from junqi_rl.analysis.facts_joint_search import FactsBlueprint, FactsBeliefSearch, SearchConfig, regularized_update
from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA
from scripts import diagnose_facts_belief_search as diagnostic
from scripts.compare_facts_joint import sha256, write_json
from tests.test_facts_dataset import cpu_trace
from tests.test_facts_joint_search import FixedLabels, UniformBlueprint, fixture, play_history


def test_cli_acceptance_gate_rejects_missing_state_files_but_allows_partial_inspection(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    write_json(output / 'protocol.json', dict(states=1, models=[], timing_state_indices=[0]))
    argv = ['--mode', 'aggregate', '--dataset', str(tmp_path), '--selection', str(tmp_path / 'selection.json'),
            '--output', str(output)]
    assert diagnostic.main(argv)['complete'] is False
    with pytest.raises(ValueError, match='not completed'):
        diagnostic.main([*argv, '--require-complete'])


def test_opt_in_telemetry_preserves_action_rng_and_root_and_covers_every_particle():
    env, truth = fixture()
    history = play_history(env, count=12)
    initial = env.state.to_dict()
    config = SearchConfig(particles=4, depth=1)
    ordinary = FactsBeliefSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels(truth), config, seed=714)
    instrumented = FactsBeliefSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels(truth), config,
                                    seed=714, capture_diagnostics=True)
    assert ordinary.action(env, history) == instrumented.action(env, history)
    assert torch.equal(ordinary.generator.get_state(), instrumented.generator.get_state())
    assert env.state.to_dict() == initial
    assert ordinary.last_decision is None
    record = instrumented.last_decision
    legal = record['legal_compact_action_ids']
    assert len(legal) == len(env.legal_action_ids())
    assert set(record['legal_world_action_ids']) == set(env.legal_action_ids())
    returns = np.asarray(record['particle_returns'])
    assert returns.shape == (len(legal), 4)
    np.testing.assert_allclose(returns.mean(1), record['q'], rtol=0, atol=1e-7)
    np.testing.assert_allclose(record['q'], ordinary.last_q[0, legal], rtol=0, atol=1e-7)
    assert sum(record['pi0_probabilities']) == pytest.approx(1, abs=1e-6)
    assert sum(record['search_probabilities']) == pytest.approx(1, abs=1e-6)
    assert len(record['particles']['piece_ids']) == 75
    assert np.asarray(record['particles']['assignments']).shape == (4, 75)
    assert all(record['particles'][key] for key in (
        'support_validated', 'public_replay_validated', 'root_observation_validated', 'root_legal_actions_validated'))
    # JSON contains no illegal-action NaNs or infinities.
    json.dumps(record, allow_nan=False)


def test_shared_prefix_estimators_use_coupled_returns_and_offline_beta_needs_no_model():
    decision = dict(particle_returns=[[1., 3., 5., 7.], [4., 2., 0., -2.]],
                    pi0_log_probabilities=np.log([.8, .2]).tolist(), pi0_probabilities=[.8, .2])
    estimates = diagnostic.prefix_estimates(decision, [1, 2, 4], [.25, .5, 1., 2.])
    assert [row['q'] for row in estimates] == [[1., 4.], [2., 3.], [4., 1.]]
    assert [row['particle_ids'] for row in estimates] == [[0], [0, 1], [0, 1, 2, 3]]
    for row in estimates:
        for beta in row['beta_sensitivity']:
            torch_result = regularized_update(torch.tensor([decision['pi0_log_probabilities']]),
                torch.tensor([row['q']]), torch.ones((1, 2), dtype=torch.bool), beta=beta['beta']).exp()[0]
            np.testing.assert_allclose(beta['probabilities'], torch_result, rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match='Prefix'):
        diagnostic.prefix_estimates(decision, [5], [.5])


def test_repeat_variance_rank_ties_and_policy_change_are_reported_separately():
    first = dict(particle_returns=[[0], [1]], pi0_log_probabilities=np.log([.8, .2]).tolist(),
                 pi0_probabilities=[.8, .2])
    second = {**first, 'particle_returns': [[1], [0]]}
    records = [dict(prefix_estimates=diagnostic.prefix_estimates(row, [1], [.5])) for row in (first, second)]
    summary = diagnostic.repeated_q_summary(records, [1], [.5])[0]
    assert summary['q_variance_per_legal_action'] == [.5, .5]
    assert summary['mean_pairwise_q_spearman'] == pytest.approx(-1)
    assert summary['beta_sensitivity'][0]['argmax_change_fraction'] == .5
    assert summary['beta_sensitivity'][0]['repeated_argmax_agreement'] == 0
    assert diagnostic.rank_correlation([1, 1], [1, 1]) is None
    np.testing.assert_array_equal(diagnostic.average_ranks([2, 1, 2, 3]), [1.5, 0, 1.5, 3])


def test_rng_streams_reproducible_and_separate_shared_runtime_and_particles():
    seed = diagnostic.seed_for(3, 4, 2, 1, 'shared-pool')
    assert seed == diagnostic.seed_for(3, 4, 2, 1, 'shared-pool')
    distinct = {seed, diagnostic.seed_for(3, 4, 2, 1, 'actual-runtime', 1),
                diagnostic.seed_for(3, 4, 2, 1, 'actual-runtime', 2)}
    assert len(distinct) == 3


def synthetic_dataset(tmp_path, games=3):
    manifest = dict(schema='junqi.facts-belief-dataset.v1', status='complete',
                    protocol=dict(constraint_schema=CONSTRAINT_SCHEMA), shards=[], raw_games=[])
    for game in range(games):
        trace, metadata = cpu_trace(seed=17 + game, game_id=game, max_moves=12)
        trace_path = tmp_path / f'raw-{game}.npz'
        np.savez(trace_path, **trace)
        raw = dict(metadata, split='validation', path=trace_path.name, sha256=sha256(trace_path))
        plies = np.asarray([0, 4, 8], dtype=np.int32)
        shard_path = tmp_path / f'shard-{game}.npz'
        np.savez(shard_path, ply=plies, observer_seats=trace['seats'][plies])
        shard = dict(game_id=game, split='validation', path=shard_path.name, sha256=sha256(shard_path),
                     initial_setup_sha256=metadata['initial_setup_sha256'])
        manifest['raw_games'].append(raw)
        manifest['shards'].append(shard)
    write_json(tmp_path / 'dataset-manifest.json', manifest)
    return manifest


def test_stage_selection_and_manifest_loading_use_distinct_games_and_hashes(tmp_path):
    manifest = synthetic_dataset(tmp_path)
    destination = tmp_path / 'selection.json'
    selected = diagnostic.create_selection(tmp_path, destination, count=3, seed=37)
    metadata = [diagnostic.selection_metadata(*row) for row in selected]
    assert len({row['game_id'] for row in metadata}) == 3
    assert [row['stage'] for row in metadata] == ['early', 'middle', 'late']
    loaded = diagnostic.load_selected_validation_states(tmp_path, destination)
    assert [diagnostic.selection_metadata(*row) for row in loaded] == metadata
    other = diagnostic.stratified_validation_states(tmp_path, manifest, 3, 37)
    assert [diagnostic.selection_metadata(*row) for row in other] == metadata
    with (tmp_path / metadata[0]['shard_path']).open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='hash changed'):
        diagnostic.load_selected_validation_states(tmp_path, destination)


def test_stage_quotas_never_duplicate_a_game_or_open_test_shards(tmp_path):
    manifest = synthetic_dataset(tmp_path, games=1)
    manifest['shards'].append(dict(game_id=999, split='test', path='must-not-open.npz'))
    with pytest.raises(ValueError, match='distinct'):
        diagnostic.stratified_validation_states(tmp_path, manifest, 3, 1)
    selected = diagnostic.stratified_validation_states(tmp_path, manifest, 1, 1)
    assert len(selected) == 1
    assert diagnostic.selection_metadata(*selected[0])['stage'] == 'early'


def test_selection_matching_keeps_games_available_for_scarce_stages(tmp_path, monkeypatch):
    manifest = synthetic_dataset(tmp_path)
    original = diagnostic._read_validation_game
    def narrow(dataset, item, raw):
        shard, trace = original(dataset, item, raw)
        # One all-stage game and two early-only games: count two needs one middle.
        if item['game_id'] != 0:
            shard = dict(ply=shard['ply'][:1], observer_seats=shard['observer_seats'][:1])
        return shard, trace
    monkeypatch.setattr(diagnostic, '_read_validation_game', narrow)
    for seed in range(8):
        selected = diagnostic.stratified_validation_states(tmp_path, manifest, 2, seed)
        metadata = [diagnostic.selection_metadata(*row) for row in selected]
        assert metadata[1]['game_id'] == 0
        assert metadata[1]['stage'] == 'middle'
        assert metadata[0]['game_id'] != 0


@pytest.mark.parametrize('ply,moves,stage', [(0, 12, 'early'), (3, 12, 'early'), (4, 12, 'middle'),
                                           (7, 12, 'middle'), (8, 12, 'late'), (11, 12, 'late')])
def test_stage_uses_full_game_length(ply, moves, stage):
    assert diagnostic.game_stage(ply, moves) == stage


def test_illegal_stage_and_overlap_contracts_fail_closed():
    with pytest.raises(ValueError, match='complete game'):
        diagnostic.game_stage(12, 12)
    env, truth = fixture()
    with pytest.raises(ValueError, match='joint sampler'):
        FactsBeliefSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels(truth), overlap_constraints=True)
    with pytest.raises(ValueError, match='backend'):
        FactsBeliefSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels(truth), inventory_backend='approximate')


def diagnostic_record_fixture():
    env, truth = fixture()
    search = FactsBeliefSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels(truth),
                              SearchConfig(particles=2), seed=3, capture_diagnostics=True)
    search.action(env, [])
    model = dict(checkpoint_sha256='a' * 64, belief_kind='joint', training_seed=7)
    state = dict(game_id=9, row=0, ply=0, stage='early')
    protocol = dict(states=1, selected_states=[state], models=[model], particles=[1, 2],
                    repeats=2, betas=[.5], beta=.5, timing_repeats=1, timing_state_indices=[0])
    repetitions = [dict(repeat=repeat, decision=search.last_decision,
        prefix_estimates=diagnostic.prefix_estimates(search.last_decision, [1, 2], [.5])) for repeat in range(2)]
    record = dict(schema=diagnostic.DIAGNOSTIC_SCHEMA, complete=True, measurement='shared',
        state_index=0, state=state, public_history_steps=0,
        models=[dict(model=model, repetitions=repetitions,
                     repeated_summary=diagnostic.repeated_q_summary(repetitions, [1, 2], [.5]))])
    return record, protocol


def test_aggregation_recomputes_q_and_rejects_duplicate_repeat_or_model_cells(tmp_path):
    record, protocol = diagnostic_record_fixture()
    diagnostic.validate_state_record(record, protocol, 'shared')
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    record['protocol_sha256'] = digest
    write_json(tmp_path / 'state-00000-shared.json', record)
    summary = diagnostic.aggregate_output(tmp_path, protocol, scope='shared')
    assert summary['complete'] is True
    assert summary['shared_q_estimates'] == 4
    assert summary['shared_sampler_searches'] == 2
    assert summary['true_runtime_searches'] == 0
    duplicate = deepcopy(record)
    duplicate['models'].append(duplicate['models'][0])
    with pytest.raises(ValueError, match='model coverage'):
        diagnostic.validate_state_record(duplicate, protocol, 'shared')
    duplicate = deepcopy(record)
    duplicate['models'][0]['repetitions'][1]['repeat'] = 0
    with pytest.raises(ValueError, match='repetition coverage'):
        diagnostic.validate_state_record(duplicate, protocol, 'shared')
    broken = deepcopy(record)
    broken['models'][0]['repetitions'][0]['prefix_estimates'][0]['q'][0] += 1
    with pytest.raises(ValueError, match='independent recomputation'):
        diagnostic.validate_state_record(broken, protocol, 'shared')


def test_timing_subset_does_not_redefine_full_selection_or_accept_missing_budget(tmp_path):
    shared, protocol = diagnostic_record_fixture()
    protocol['states'] = 3
    protocol['selected_states'] *= 3
    timing = dict(schema=diagnostic.DIAGNOSTIC_SCHEMA, complete=True, measurement='timing',
                  state_index=0, state=shared['state'], public_history_steps=0, searches=[])
    for particles in protocol['particles']:
        timing['searches'].append(dict(model=protocol['models'][0], particles=particles, repeat=0,
            measurement='independent-full-production-search', wall_seconds=1.,
            report=dict(config=dict(particles=particles, depth=1, beta=.5), successful_decisions=1,
                        candidate_actions=2, legal_actions=2)))
    timing['protocol_sha256'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    write_json(tmp_path / 'state-00000-timing.json', timing)
    summary = diagnostic.aggregate_output(tmp_path, protocol, scope='timing')
    assert summary['complete'] is True
    assert summary['protocol']['states'] == 3
    assert summary['protocol']['timing_state_indices'] == [0]
    assert summary['true_runtime_searches'] == 2
    broken = deepcopy(timing)
    broken['searches'].pop()
    with pytest.raises(ValueError, match='cell coverage'):
        diagnostic.validate_state_record(broken, protocol, 'timing')
