"""Action quality gates pair worlds/streams and keep discovery out of audit."""
from copy import deepcopy
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
import torch

from junqi_core.rules import Seat
from scripts import diagnose_facts_action_quality as quality


def origins():
    return [dict(checkpoint_sha256=letter * 64, particle=p) for letter in 'abcd' for p in range(4)]


def saved_modes(qs, logp=None):
    return dict(models=[dict(model={'checkpoint_sha256': letter * 64}, repetitions=[dict(decision=dict(
        legal_compact_action_ids=[9, 3, 7], pi0_log_probabilities=logp or [-.2, -2., -3.], q=q))])
                        for letter, q in zip('abcd', qs)])


def test_public_action_union_is_bounded_has_stable_ties_and_runner_up():
    record = saved_modes([[0, 1, 1]] * 4)
    selected = quality.select_common_actions(record)
    assert selected['actions'] == [9, 3]  # Equal Q chooses smallest compact ID.
    same = quality.select_common_actions(saved_modes([[1, 0, 0]] * 4))
    assert same['actions'] == [9, 3]
    assert same['runner_up_fill'] == 3
    record['models'][2]['repetitions'][0]['decision']['q'] = [0, 0, 1]
    assert quality.select_common_actions(record)['actions'] == [9, 3, 7]
    with pytest.raises(ValueError, match='omit'):
        quality.select_common_actions(record, max_actions=2)
    corrupt = deepcopy(record)
    corrupt['models'][0]['repetitions'][0]['decision']['legal_compact_action_ids'] = [9, 3, 6]
    with pytest.raises(ValueError, match='share'):
        quality.select_common_actions(corrupt)


def test_counter_stream_is_action_independent_world_order_independent_and_disjoint():
    source = origins()
    seeds = quality.continuation_seeds(72, 19, source, 8)
    np.testing.assert_array_equal(seeds, quality.continuation_seeds(72, 19, source, 8))
    np.testing.assert_array_equal(seeds[::-1], quality.continuation_seeds(72, 19, source[::-1], 8))
    assert len(set(seeds.ravel())) == 128
    assert set(seeds[:, :4].ravel()).isdisjoint(seeds[:, 4:].ravel())
    assert not np.array_equal(seeds, quality.continuation_seeds(73, 19, source, 8))
    assert quality.sampling_seed(72, 19, 'a' * 64) != int(seeds[0, 0])
    with pytest.raises(ValueError, match='even'):
        quality.continuation_seeds(72, 19, source, 3)


def test_saved_recency_precision_cannot_relax_fact_identity_planes():
    saved = np.zeros((317, 17, 17), dtype=np.float32)
    saved[253, 9, 15] = np.float32(.6257840394973755)
    actual = saved.copy()
    actual[253, 9, 15] = np.nextafter(saved[253, 9, 15], np.float32(0))
    report = quality.validate_saved_observation(actual, saved)
    assert report['different_entries'] == 1 and report['different_channels'] == [253]
    assert report['maximum_absolute_difference'] == 2**-24
    actual[0, 0, 0] = np.float32(1e-8)
    with pytest.raises(ValueError, match='precision'):
        quality.validate_saved_observation(actual, saved)
    actual = saved.copy()
    actual[253, 9, 15] += np.float32(1e-5)
    with pytest.raises(ValueError, match='precision'):
        quality.validate_saved_observation(actual, saved)


def test_root_averaging_precedes_choice_and_audit_cannot_choose_discovery_action():
    leaf = np.asarray([[1] + [-1] * 15, [0] * 16], dtype=float)
    mc = np.zeros((2, 16, 8))
    mc[0, :, :4] = 1  # Discovery chooses action 9.
    mc[0, :, 4:] = -1  # Audit must score that action despite preferring 3.
    result = quality.summarize_action_values([9, 3], leaf, mc, [-.1, -2.], origins())
    assert result['q_leaf'] == [-.875, 0]
    assert result['leaf_mode'] == 3
    assert result['discovery_mc_mode'] == 9
    assert result['audit_mc_mode_diagnostic_only'] == 3
    contrast = result['audit_contrasts']['discovery_choice_minus_leaf_choice']
    assert (contrast['left'], contrast['right'], contrast['mean']) == (9, 3, -1)
    assert result['discovery_audit_mode_agreement'] is False


def test_paired_mc_uncertainty_retains_covariance_and_world_clusters():
    values = np.tile([-1., 1., -1., 1., -1., 1., -1., 1.], (16, 1))
    returns = np.stack([values, values])
    groups = [list(range(i * 4, (i + 1) * 4)) for i in range(4)]
    identical = quality.paired_difference(returns, 0, 1, slice(0, 4), groups)
    assert identical['conditional_pool_mc_standard_error'] == 0
    assert identical['stratified_world_cluster_standard_error'] == 0
    returns[1] = -values
    opposite = quality.paired_difference(returns, 0, 1, slice(0, 4), groups)
    assert opposite['conditional_pool_mc_standard_error'] == pytest.approx(np.sqrt((16 * 16 / 3) / (256 * 4)))
    assert opposite['world_variance_mc_noise_corrected'] < 0
    assert opposite['corrected_variance_clipped'] is False


def test_batched_prediction_uses_actual_actor_sign_and_avoids_terminal_inference():
    def world(turn, terminal=False):
        return SimpleNamespace(max_num_moves=4000, state=SimpleNamespace(
            terminated=terminal, move_counter=4, turn=Seat(turn),
            team_rewards=lambda: (1, -1, 1, -1)))
    class Blueprint:
        batches = []
        def values(self, worlds):
            self.batches.append(len(worlds))
            return torch.tensor([.6] * len(worlds))
    blueprint = Blueprint()
    answer = quality.batched_leaf_predictions(blueprint, [world(1), world(2), world(1, True)], Seat(0), 1, gamma=.9)
    assert [row['value'] for row in answer] == pytest.approx([-.6, .6, 1])
    assert [row['root_action_value'] for row in answer] == pytest.approx([-.54, .54, 1])
    assert blueprint.batches == [1, 1]
    assert answer[2]['source'] == 'terminal_team_reward'


def protocol():
    return dict(schema=quality.SCHEMA, phase='development', split='validation', identity_encoding='known_only',
                observation_channels=317, observation_global_dims=0, new_training_games=0, formal_match_games=0,
                **{key: False for key in quality.FALSE_FLAGS})


def test_runtime_flags_reject_truth_fallback_and_false_completion(tmp_path):
    good = protocol()
    quality.assert_flags(good)
    for flag in quality.FALSE_FLAGS:
        bad = dict(good, **{flag: True})
        with pytest.raises(ValueError, match='contract'):
            quality.assert_flags(bad)
    good.update(states=32, seed=3, target_scope='fixed mixture', acceptance_only=False,
                max_actions=5, worlds_per_model=4, mixture_worlds=16, rollouts=8)
    quality.atomic_json(tmp_path / 'protocol.json', good)
    partial = quality.aggregate_output(tmp_path)
    assert partial['complete'] is False and partial['completed_continuations'] == 0
    with pytest.raises(ValueError, match='not completed'):
        quality.aggregate_output(tmp_path, require_complete=True)
    with pytest.raises(ValueError):
        quality.atomic_json(tmp_path / 'nonfinite.json', {'q': float('nan')})
    assert not (tmp_path / 'nonfinite.json').exists()


def test_eight_shared_writers_bind_fresh_protocol_under_atomic_write_lock(tmp_path, monkeypatch):
    original = quality.atomic_json
    writing, finish = threading.Event(), threading.Event()
    def held_write(path, value):
        temporary = tmp_path / '.protocol.json-own-atomic-write.tmp'
        temporary.write_text('writing')
        writing.set()
        assert finish.wait(5)
        temporary.unlink()
        original(path, value)
    monkeypatch.setattr(quality, 'atomic_json', held_write)
    p = protocol()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(quality.bind_action_protocol, tmp_path, p)]
        assert writing.wait(5)
        futures += [executor.submit(quality.bind_action_protocol, tmp_path, p) for _ in range(7)]
        finish.set()
        assert [future.result(timeout=5) for future in futures] == [quality.json_digest(p)] * 8
    with pytest.raises(ValueError, match='another protocol'):
        quality.bind_action_protocol(tmp_path, dict(p, seed=123))
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'state-00000.json').write_text('{}')
    with pytest.raises(ValueError, match='empty'):
        quality.bind_action_protocol(foreign, p)


def complete_record():
    p = protocol()
    models = [dict(checkpoint_sha256=letter * 64) for letter in 'abcd']
    selection = quality.select_common_actions(saved_modes([[0, 1, 0]] * 4))
    state = dict(game_id=19, ply=7, observer_seat=0)
    p.update(states=1, seed=72, selected_rows=[state], action_sets=[selection], models=models,
             public_history_sha256=['e' * 64], worlds_per_model=4, mixture_worlds=16, rollouts=8,
             beta=.5, max_actions=5, value_training_target_contract={'gamma': 1.},
             saved_observation_validation=quality.SAVED_OBSERVATION_VALIDATION,
             acceptance_only=False, target_scope='fixed mixture')
    world_origins = [dict(row, assignment_sha256=quality.json_digest([0] * 75)) for row in origins()]
    seeds = quality.continuation_seeds(p['seed'], state['game_id'], world_origins, 8)
    sampling = [dict(model=model, assignments=[[0] * 75] * 4, piece_ids=list(range(75)),
                     seed=quality.sampling_seed(p['seed'], state['game_id'], model['checkpoint_sha256']),
                     **{key: True for key in quality.VALIDATION_FLAGS}) for model in models]
    actions, leaf, returns = [], [], []
    for a, aid in enumerate(selection['actions']):
        measured = []
        for w in range(16):
            prediction = dict(source='frozen_next_actor_value_head', value=float(a), root_action_value=float(a),
                              root_step_discount=1., root_team=0, sign=-1, next_actor=1, next_actor_value=float(-a))
            continuation = [dict(completed=True, terminal=True, draw=not bool(a),
                                 seed=int(seeds[w, r]), moves=1, root_team_return=float(a),
                                 root_team_discounted_return=float(a), root_action_discounted_return=float(a),
                                 termination_reason='team_victory' if a else 'terminal_draw') for r in range(8)]
            measured.append(dict(world_index=w, prediction=prediction, continuations=continuation))
        actions.append(dict(canonical_action=aid,
                            world_action=quality.unrotate_compact_action_id(aid, Seat(state['observer_seat'])), worlds=measured))
        leaf.append([float(a)] * 16)
        returns.append([[float(a)] * 8] * 16)
    logp = selection['saved_candidate_log_pi0']
    row = dict(schema=quality.SCHEMA, complete=True, protocol_sha256=quality.json_digest(p), state_index=0,
               state=state, selection=selection, legal_compact_action_ids=[9, 3, 7],
               public_history_steps=7, public_history_sha256='e' * 64, world_origins=world_origins,
               saved_observation_validation=dict(contract=quality.SAVED_OBSERVATION_VALIDATION,
                   maximum_absolute_difference=0., different_channels=[], different_entries=0),
               sampling=sampling, continuation_seeds=seeds.tolist(), actions=actions, candidate_log_pi0=logp,
               summary=quality.summarize_action_values(selection['actions'], leaf, returns, logp, world_origins),
               new_training_games=0, formal_match_games=0)
    return p, row


def test_full_cell_coverage_return_recomputation_and_protocol_identity(tmp_path):
    p, row = complete_record()
    digest = quality.json_digest(p)
    quality.validate_record(row, p, digest)
    for mutate in (
        lambda r: r['actions'][0]['worlds'].pop(),
        lambda r: r['world_origins'].pop(),
        lambda r: r['world_origins'][0].update(assignment_sha256='f' * 64),
        lambda r: r['actions'][0].update(world_action=123),
        lambda r: r['actions'][0]['worlds'][0]['continuations'][0].update(seed=1),
        lambda r: r['summary']['audit_contrasts']['leaf_choice_minus_pi0_mode'].update(mean=99),
        lambda r: r.update(protocol_sha256='f' * 64),
    ):
        bad = deepcopy(row)
        mutate(bad)
        with pytest.raises(ValueError):
            quality.validate_record(bad, p, digest)
    quality.atomic_json(tmp_path / 'protocol.json', p)
    quality.atomic_json(tmp_path / 'state-00000.json', row)
    summary = quality.aggregate_output(tmp_path, require_complete=True)
    assert summary['complete'] is True
    assert summary['completed_continuations'] == 256
    assert summary['terminal_continuations'] == 256 and summary['cutoff_continuations'] == 0
    assert summary['termination_reasons'] == {'terminal_draw': 128, 'team_victory': 128}
    quality.atomic_json(tmp_path / 'state-00001.json', row)
    with pytest.raises(ValueError, match='Duplicate or renamed'):
        quality.aggregate_output(tmp_path, require_complete=True)
