"""Facts search reconstructs complete worlds and evaluates every root action."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.knowledge import ObserverKnowledge
from junqi_core.rules import ALL_SEATS, PieceType, Seat, ShowMode
from junqi_core.state import GameState
from junqi_rl.analysis import facts_joint_search as search_module
from junqi_rl.analysis.facts_joint_search import (
    FactsBlueprint, FactsJointSearch, SamplingBudgetExceeded, SearchConfig,
    WorldReconstructionError, clone_world, materialize, public_result,
    regularized_update, sampled_setups, scalar_value, step_world,
)
from junqi_rl.belief.joint import build_joint_input
from junqi_rl.belief.constraints import ConstraintBudgetExceeded
from junqi_rl.env import JunqiEnv, rotate_to_compact_action_id, src_dst_to_action_id
from junqi_rl.networks.facts_joint_belief_net import FactsJointBeliefNet, FactsJointBeliefNetConfig
from tests.test_feature_information_boundaries import position


class UniformBlueprint(torch.nn.Module):
    identity_encoding = 'known_only'
    uses_global_summary = False

    def forward(self, obs, global_, legal, *, actions):
        assert global_.shape[-1] == 0
        logits = torch.zeros(legal.shape, device=obs.device).masked_fill(~legal, -torch.inf)
        return {'log_probs': logits.log_softmax(-1), 'value': self.forward_value(obs, global_)}

    def forward_value(self, obs, global_):
        return torch.tensor([.1, .2, .7], device=obs.device).log().expand(len(obs), -1)


class FixedLabels(torch.nn.Module):
    """Offline test oracle; production proposals always come from the network."""
    identity_encoding = 'known_only'
    uses_global_summary = False

    def __init__(self, truth):
        super().__init__()
        self.truth = truth
        self.calls = 0
        self.last_sample_metrics = {}

    def sample(self, obs, *, inputs, n_samples, **kwargs):
        self.calls += 1
        labels = [[self.truth[int(pid)] for pid in inputs.piece_ids[0]]]
        return torch.tensor(labels, device=obs.device)[:, None].expand(-1, n_samples, -1)


def fixture(seed=1703):
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                   identity_encoding='known_only', record_public_history=True)
    env.reset(seed)
    truth = {pid: TYPE_TO_INDEX[PieceType(int(kind))]
             for pid, kind in enumerate(env.state.piece_type_arr) if kind >= 2}
    return env, truth


def play_history(env, count=100):
    rng = np.random.default_rng(7)
    history = []
    for _ in range(count):
        if env.state.terminated:
            break
        action = int(rng.choice(env.legal_action_ids()))
        history.append(public_result(step_world(env, action)))
    return history


def q12_fixture(plies):
    """Frozen public trace; hidden labels are only offline simulator targets."""
    data = json.loads((Path(__file__).parent / 'fixtures' / 'facts_q12_public_trace.json').read_text())
    kinds = data['initial_piece_types']
    setups = [[PieceType(kind) if kind >= 2 else PieceType.NONE
               for kind in kinds[owner * 30:(owner + 1) * 30]] for owner in range(4)]
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                   identity_encoding='known_only', record_public_history=True)
    env._state = GameState.new_game(setups, show_mode=ShowMode.DARK)
    env._beliefs = {seat: None for seat in ALL_SEATS}
    env._knowledge = {seat: ObserverKnowledge(env.state, seat) for seat in ALL_SEATS}
    history = [public_result(step_world(env, action)) for action in data['world_actions'][:plies]]
    return env, history


def test_materialization_replays_dead_identities_and_every_public_field():
    env, truth = fixture()
    history = play_history(env)
    snapshot = env.knowledge_snapshot(env.state.turn)
    inputs = build_joint_input([snapshot])
    assert not inputs.alive.all()
    labels = [truth[int(pid)] for pid in inputs.piece_ids[0]]
    world = materialize(snapshot, inputs.piece_ids[0], labels, history)
    np.testing.assert_array_equal(world.state.cell_piece_id, env.state.cell_piece_id)
    np.testing.assert_array_equal(world.state.piece_type_arr, env.state.piece_type_arr)
    assert world._knowledge is None
    assert all(belief is None for belief in world.beliefs.values())
    blueprint = FactsBlueprint(UniformBlueprint(), 'cpu')
    real = blueprint.inputs([env])[0].clone()
    rebuilt = blueprint.inputs([world])[0].clone()
    torch.testing.assert_close(real, rebuilt)
    for field in (4, 5, 6, 7, 9):
        bad = list(history)
        last = list(bad[-1])
        last[field] = not last[field]
        bad[-1] = tuple(last)
        with pytest.raises(WorldReconstructionError, match='broadcast'):
            materialize(snapshot, inputs.piece_ids[0], labels, bad)
    for field, replacement in ((8, 1), (10, (0,))):
        bad = list(history)
        last = list(bad[-1])
        last[field] = replacement
        bad[-1] = tuple(last)
        with pytest.raises(WorldReconstructionError, match='broadcast'):
            materialize(snapshot, inputs.piece_ids[0], labels, bad)
    with pytest.raises(ValueError, match='75'):
        materialize(snapshot, inputs.piece_ids[0][:-1], labels[:-1], history)


def test_world_setup_has_no_access_to_real_hidden_types_or_private_memory():
    env, truth = fixture()
    snapshot = env.knowledge_snapshot(Seat.SOUTH)
    inputs = build_joint_input([snapshot])
    labels = [truth[int(pid)] for pid in inputs.piece_ids[0]]
    expected = sampled_setups(snapshot, inputs.piece_ids[0], labels)
    hidden = env.state.piece_seat_arr != Seat.SOUTH
    env.state.piece_type_arr[hidden] = -127
    for pos, ref in list(env.state.zero_board.items()):
        if ref.seat != Seat.SOUTH:
            del env.state.zero_board[pos]
    env.state.combat_memory.rank_floor[1:] = 127
    assert sampled_setups(snapshot, inputs.piece_ids[0], labels) == expected
    reconstructed = materialize(snapshot, inputs.piece_ids[0], labels, [])
    assert (reconstructed.state.combat_memory.rank_floor == 0).all()


def test_public_result_excludes_private_debug_identity_fields():
    env, _ = fixture()
    result = step_world(env, int(env.legal_action_ids()[0]))
    private = replace(result, src_type_revealed=PieceType.SILING, dst_type_revealed=PieceType.DILEI)
    assert public_result(private) == public_result(result)


def test_full_action_search_refreshes_samples_and_preserves_the_real_root():
    env, truth = fixture()
    joint = FixedLabels(truth)
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), joint,
                              SearchConfig(particles=2, depth=2), seed=41)
    initial = env.state.to_dict()
    n = len(env.legal_action_ids())
    assert search.action(env, []) in env.legal_action_ids()
    assert env.state.to_dict() == initial
    assert search.metrics['candidate_actions'] == n == search.metrics['legal_actions']
    assert search.metrics['rollout_world_steps'] == n * 2 * 2
    assert search.action(env, []) in env.legal_action_ids()
    assert joint.calls == search.metrics['joint_refreshes'] == 2
    assert search.report()['action_coverage'] == 1
    assert search.report()['acceptance_rate'] == 1


def test_leaf_value_uses_actual_next_team_after_player_elimination(monkeypatch):
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False, identity_encoding='known_only')
    env._state = position(PieceType.SHIZH, PieceType.JUNQI)
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels({}))
    monkeypatch.setattr(search, 'particles', lambda env, *args: [clone_world(env)])
    search.action(env, [])
    capture = src_dst_to_action_id((2, 7), (1, 7))
    capture_compact = rotate_to_compact_action_id(capture, Seat.SOUTH)
    assert search.last_q[0, capture_compact].item() == pytest.approx(.6)
    for aid in env.legal_action_ids():
        if int(aid) != capture:
            compact = rotate_to_compact_action_id(int(aid), Seat.SOUTH)
            assert search.last_q[0, compact].item() == pytest.approx(-.6)


def test_values_and_regularization_use_the_trained_prior_only():
    assert scalar_value(torch.tensor([[.1, .2, .7]]).log()).item() == pytest.approx(.6)
    assert scalar_value(torch.tensor([.2])).item() == pytest.approx(.2)
    legal = torch.tensor([[True, True, False]])
    prior = torch.tensor([[.2, .8, 0.0]]).log()
    q = torch.tensor([[0.0, 0.0, torch.nan]])
    torch.testing.assert_close(regularized_update(prior, q, legal, beta=.5), prior)
    q[0, 0] = .5
    result = regularized_update(prior, q, legal, beta=.5)
    assert result[0, 0] > prior[0, 0]
    assert result[0, 2].isneginf()
    with pytest.raises(ValueError, match='searched Q'):
        regularized_update(prior, torch.full_like(q, torch.nan), legal, beta=.5)


def test_history_rejection_exhausts_budget_without_selecting_baseline(monkeypatch):
    env, truth = fixture()
    joint = FixedLabels(truth)
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), joint,
                              SearchConfig(proposal_batches=2))
    def reject(*args, **kwargs):
        raise WorldReconstructionError('public_broadcast_mismatch', 0)
    monkeypatch.setattr(search_module, 'materialize', reject)
    with pytest.raises(SamplingBudgetExceeded, match='no action selected'):
        search.action(env, [])
    assert joint.calls == search.metrics['rejected_proposals'] == 2
    assert search.metrics['successful_decisions'] == 0
    assert search.metrics['sample_budget_failures'] == 1
    assert search.metrics['rollout_world_steps'] == 0


def test_invalid_sampler_output_is_an_error_not_a_rejected_or_fallback_sample():
    env, truth = fixture()
    bad_truth = dict.fromkeys(truth, 0)
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), FixedLabels(bad_truth))
    with pytest.raises(ValueError, match='domain|inventory'):
        search.action(env, [])
    assert search.metrics['successful_decisions'] == 0
    assert search.metrics['rejected_proposals'] == 0


def test_unknown_constraint_feasibility_aborts_search_without_fallback(monkeypatch):
    env, truth = fixture()
    joint = FixedLabels(truth)
    def exhausted(*args, **kwargs):
        raise ConstraintBudgetExceeded('bounded exact feasibility is unknown')
    monkeypatch.setattr(joint, 'sample', exhausted)
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), joint)
    with pytest.raises(ConstraintBudgetExceeded):
        search.action(env, [])
    assert search.metrics['constraint_budget_failures'] == 1
    assert search.metrics['successful_decisions'] == search.metrics['rollout_world_steps'] == 0


def test_real_facts_neural_sampler_and_all_actions_complete_an_opening_decision():
    env, _ = fixture()
    joint = FactsJointBeliefNet(FactsJointBeliefNetConfig(
        cnn_channels=8, cnn_layers=1, embed_dim=16, n_head=2,
        ff_factor=2, n_encoder_layer=1, n_decoder_layer=1)).eval()
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), joint)
    assert search.action(env, []) in env.legal_action_ids()
    assert search.metrics['accepted_proposals'] == 1
    assert search.metrics['candidate_actions'] == len(env.legal_action_ids())
    assert search.metrics['constraint_seconds'] > 0


@pytest.mark.parametrize('plies', [161, 200])
def test_late_q12_worlds_preserve_public_elimination_without_rejection(plies):
    env, history = q12_fixture(plies)
    # Step 161 eliminates SOUTH due to no legal move, with no flag capture.
    assert history[160][6] is False and history[160][10] == (0,)
    torch.manual_seed(731)
    joint = FactsJointBeliefNet(FactsJointBeliefNetConfig(
        cnn_channels=8, cnn_layers=1, embed_dim=16, n_head=2,
        ff_factor=2, n_encoder_layer=1, n_decoder_layer=1)).eval()
    search = FactsJointSearch(FactsBlueprint(UniformBlueprint(), 'cpu'), joint,
                              SearchConfig(particles=4, proposal_batches=1), seed=731)
    _, _, legal, obs = search.blueprint.distribution([env])
    worlds = search.particles(env, history, obs.clone(), legal.clone())
    assert len(worlds) == 4
    assert search.metrics['accepted_proposals'] == 4
    assert search.metrics['rejected_proposals'] == 0
    assert all(world.state.info[Seat.SOUTH].dead for world in worlds)


@pytest.mark.parametrize('kwargs', [{'particles': 0}, {'depth': 0}, {'proposal_batches': 0},
                                    {'max_constraint_nodes': 0}, {'beta': 0}, {'beta': float('nan')}])
def test_invalid_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)
