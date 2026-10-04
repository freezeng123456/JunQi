"""Calibration uses the production team sign and independent complete continuations."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi_core.rules import Seat
from junqi_rl.analysis.facts_joint_search import FactsBlueprint
from scripts.calibrate_facts_search_value import (
    discounted_terminal_return, pi0_continuations, root_team_leaf_value, select_root_actions,
    summarize_calibration, value_training_target_contract,
)
from tests.test_facts_joint_search import UniformBlueprint, fixture


def test_root_team_sign_uses_actual_next_actor_and_terminal_reward():
    class Blueprint:
        calls = 0

        def values(self, worlds):
            self.calls += 1
            return torch.tensor([.6])

    blueprint = Blueprint()
    world = SimpleNamespace(max_num_moves=4000, state=SimpleNamespace(
        terminated=False, move_counter=8, turn=Seat(1)))
    opposite = root_team_leaf_value(blueprint, world, Seat(0))
    assert opposite['sign'] == -1
    assert opposite['value'] == pytest.approx(-.6)
    # Q12 can skip the nominal next opponent to a teammate.
    world.state.turn = Seat(2)
    same = root_team_leaf_value(blueprint, world, Seat(0))
    assert same['sign'] == 1
    assert same['value'] == pytest.approx(.6)
    world.state.terminated = True
    world.state.team_rewards = lambda: (1, -1, 1, -1)
    terminal = root_team_leaf_value(blueprint, world, Seat(1))
    assert terminal['value'] == -1
    assert terminal['source'] == 'terminal_team_reward'
    assert blueprint.calls == 2


def test_public_root_action_selection_is_frozen_and_has_no_illegal_alternative():
    logp = np.asarray([-2., -1., np.nan, -3.])
    legal = np.asarray([True, True, False, True])
    assert select_root_actions(logp, legal, 1, seed=3) == [1]
    selected = select_root_actions(logp, legal, 20, seed=3)
    assert selected[0] == 1
    assert set(selected) == {0, 1, 3}
    assert select_root_actions(logp, legal, 20, seed=3) == selected


def test_continuation_lane_rng_does_not_depend_on_lane_order_and_cap_is_draw():
    env, _ = fixture()
    blueprint = FactsBlueprint(UniformBlueprint(), 'cpu')
    initial = env.state.to_dict()
    seeds = np.asarray([131, 719], dtype=np.int64)
    first = pi0_continuations(blueprint, env, env.state.turn, seeds, max_continuation_moves=3)
    reversed_ = pi0_continuations(blueprint, env, env.state.turn, seeds[::-1].copy(), max_continuation_moves=3)
    assert first == list(reversed(reversed_))
    assert env.state.to_dict() == initial
    assert all(row['completed'] and not row['terminal'] and row['draw'] for row in first)
    assert all(row['moves'] == 3 and row['root_team_return'] == 0 for row in first)
    assert all(row['termination_reason'] == 'continuation_move_cap_draw' for row in first)
    assert first[0]['action_history_sha256'] != first[1]['action_history_sha256']


def test_terminal_continuations_are_not_cutoff_draws():
    env, _ = fixture()
    env.state.terminated = True
    env.state.draw = False
    env.state.winner_team = 0
    blueprint = FactsBlueprint(UniformBlueprint(), 'cpu')
    records = pi0_continuations(blueprint, env, Seat(1), np.asarray([100, 101]))
    assert all(row['terminal'] and not row['draw'] and row['moves'] == 0 for row in records)
    assert all(row['root_team_return'] == -1 for row in records)
    assert blueprint.query_counts['policy_calls'] == 0
    with pytest.raises(ValueError, match='distinct'):
        pi0_continuations(blueprint, env, Seat(0), np.asarray([100, 100]))


def test_calibration_errors_exclude_terminal_values_and_count_continuations():
    leaves = [dict(prediction={'source': 'frozen_next_actor_value_head', 'value': .5},
                   continuation_mean_return=0, continuation_discounted_mean_return=0,
                   continuation_discounted_sample_variance=.25,
                   continuations=[{'terminal': True}, {'terminal': False}]),
              dict(prediction={'source': 'terminal_team_reward', 'value': 1},
                   continuation_mean_return=1, continuation_discounted_mean_return=1,
                   continuation_discounted_sample_variance=0, continuations=[{'terminal': True}])]
    state = dict(game_id=0, models=[dict(checkpoint_sha256='a', belief_kind='marginal',
                 training_seed=1, leaves=leaves)])
    result = summarize_calibration([state])[0]
    assert result['neural_value_leaves'] == 1
    assert result['terminal_or_cutoff_leaves'] == 1
    assert result['errors']['rmse'] == .5
    assert result['errors']['mse_mc_noise_corrected'] == .125
    assert result['completed_continuations'] == 3
    assert result['terminal_continuations'] == 2
    assert result['cutoff_draw_continuations'] == 1
    assert result['errors']['game_bootstrap_ci95'] is None


def test_discount_is_per_environment_ply_with_immediate_reward_unshifted():
    assert discounted_terminal_return(1, 1, .9) == 1
    assert discounted_terminal_return(1, 3, .9) == pytest.approx(.81)
    assert discounted_terminal_return(-1, 3, .9) == pytest.approx(-.81)
    assert discounted_terminal_return(1, 0, .9) == 1
    assert discounted_terminal_return(0, 30, .9) == 0
    with pytest.raises(ValueError, match='gamma'):
        discounted_terminal_return(1, 1, True)


def test_target_contract_reads_parent_trace_instead_of_assuming_gamma(tmp_path):
    path = tmp_path / 'policy.pt'
    torch.save(dict(train_cfg=dict(ppo=dict(gamma=.9, gae_lambda=.5, td_lambda=.8),
                                   reward_shaping=False, random_opponent=False)), path)
    target = value_training_target_contract(path)
    assert target['gamma'] == .9
    assert target['td_lambda'] == .8
    assert target['discount_time_unit'] == 'one environment ply, including every acting seat'
    assert target['exact_finite_training_label'] is False
    torch.save({}, path)
    with pytest.raises(ValueError, match='explicit parent training'):
        value_training_target_contract(path)
