# ruff: noqa: E402
"""Real combat, information boundaries, and all policy entry points."""
from __future__ import annotations

import copy
import random

import pytest

torch = pytest.importorskip("torch")

from junqi_core.board import COMPACT_ACTION_DIM, FLAT_TO_COMPACT
from junqi_core.info_model import TRACKED_TYPES, BeliefTensor
from junqi_core.observation import build_observation
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import Event, PieceType, Seat, ShowMode, resolve_combat
from junqi_core.setup import generate_random_setup
from junqi_core.state import Action, GameState
from junqi_rl.networks.combat_features import CombatFeatureMetrics, CombatOutcomeHead
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from tests.test_feature_information_boundaries import position

EVENTS = (Event.EAT, Event.KILLED, Event.BOMB)


def inputs(state, belief, actions):
    observation = build_observation(state, belief, belief.observer).snapshot()
    ids = []
    for action in actions:
        sx, sy = world_to_canonical(*action.src, belief.observer)
        dx, dy = world_to_canonical(*action.dst, belief.observer)
        ids.append(int(FLAT_TO_COMPACT[sy*17+sx]) * 129 + int(FLAT_TO_COMPACT[dy*17+dx]))
    mask = torch.zeros(1, COMPACT_ACTION_DIM, dtype=torch.bool)
    mask[0, ids] = True
    return (torch.from_numpy(observation.spatial)[None],
            torch.from_numpy(observation.global_)[None], mask, torch.tensor(ids))


def all_features(head, data):
    obs, _, mask, actions = data
    return head.for_actions(obs.expand(len(actions), -1, -1, -1), actions,
                            mask.expand(len(actions), -1))


def test_all_120_mobile_combat_pairs_against_real_game_step():
    head = CombatOutcomeHead()
    count = 0
    for attacker in TRACKED_TYPES:
        if attacker.is_immobile:
            continue
        for defender in TRACKED_TYPES:
            state = position(attacker, defender)
            state.show_mode = ShowMode.BRIGHT
            action = Action(Seat.SOUTH, (2, 7), (1, 7))
            assert action in state.legal_actions()
            data = inputs(state, BeliefTensor.initial(state, Seat.SOUTH), [action])
            _, result = state.step(action)
            torch.testing.assert_close(all_features(head, data)[0],
                                       torch.eye(3)[EVENTS.index(result.event)], rtol=0, atol=0)
            count += 1
    assert count == 120


def test_legal_play_all_views_matches_observer_belief_oracle():
    head = CombatOutcomeHead()
    rng = random.Random(914100)
    state = GameState.new_game(generate_random_setup(rng), show_mode=ShowMode.DARK)
    beliefs = {s: BeliefTensor.initial(state, s) for s in Seat}
    seen = set()
    attacks = 0
    for _ in range(128):
        actions = state.legal_actions()
        if state.terminated or not actions:
            break
        belief = beliefs[state.turn]
        features = all_features(head, inputs(state, belief, actions))
        expected = torch.zeros(len(actions), 3)
        for row, action in enumerate(actions):
            if action.dst in state.pieces:
                attacks += 1
                own_type = state.pieces[action.src].piece_type
                for kind, probability in zip(TRACKED_TYPES, belief.get(action.dst), strict=True):
                    expected[row, EVENTS.index(resolve_combat(own_type, kind))] += float(probability)
        torch.testing.assert_close(features, expected, rtol=1e-6, atol=1e-7)
        seen.add(state.turn)
        after, result = state.step(rng.choice(actions))
        for b in beliefs.values():
            b.update(state, after, result)
        state = after
    assert seen == set(Seat)
    assert attacks > 100


def test_hidden_identity_control_and_snapshot_stability():
    head = CombatOutcomeHead()
    outputs = []
    for defender in (PieceType.DILEI, PieceType.LIANZH):
        state = position(PieceType.PAIZH, defender)
        action = Action(Seat.SOUTH, (2, 7), (1, 7))
        belief = BeliefTensor.initial(state, Seat.SOUTH)
        data = inputs(state, belief, [action])
        before = all_features(head, data).clone()
        after, result = state.step(action)
        belief.update(state, after, result)
        build_observation(after, belief, Seat.SOUTH)
        torch.testing.assert_close(all_features(head, data), before, rtol=0, atol=0)
        outputs.append((data[0], before))
    for a, b in zip(*outputs, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def tiny(enabled):
    return JunqiNet(JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1,
        embed_dim=16, n_head=2, ff_factor=2, action_key_dim=8,
        combat_outcome_features=enabled)).eval()


def tactical_data():
    state = position(PieceType.GONGB, PieceType.DILEI)
    return inputs(state, BeliefTensor.initial(state, Seat.SOUTH), state.legal_actions())


def test_disabled_compatibility_and_enabled_zero_initialization():
    torch.manual_seed(72)
    baseline = tiny(False)
    torch.manual_seed(72)
    candidate = tiny(True)
    assert candidate.num_parameters() - baseline.num_parameters() == 81
    assert not any('combat' in key for key in baseline.state_dict())
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
    sp, gl, mask, actions = tactical_data()
    for key, value in baseline(sp, gl, mask, actions=actions[:1]).items():
        torch.testing.assert_close(candidate(sp, gl, mask, actions=actions[:1])[key],
                                   value, rtol=0, atol=0)
    restored = tiny(False)
    restored.load_state_dict(baseline.state_dict(), strict=True)


def test_factorized_residual_matches_selected_features_and_masks_moves():
    candidate = tiny(True)
    with torch.no_grad():
        candidate.combat_head.residual[-1].weight.fill_(0.7)
        candidate.combat_head.residual[-1].bias.fill_(0.3)
    data = tactical_data()
    sp, _, mask, actions = data
    features = all_features(candidate.combat_head, data)
    expected = candidate.combat_head.residual(features).squeeze(-1) * (features.sum(-1) > 0)
    scores = candidate.combat_head(sp)[0, actions]
    torch.testing.assert_close(scores, expected)
    assert (features.sum(-1) == 0).any() and (features.sum(-1) > 0).any()
    assert (scores[features.sum(-1) == 0] == 0).all()
    torch.testing.assert_close(candidate.combat_head.for_actions(sp, actions[:1],
                              torch.zeros_like(mask)), torch.zeros(1, 3))


def test_collect_ppo_shared_ema_and_greedy_use_same_residual():
    model = tiny(True)
    with torch.no_grad():
        model.combat_head.residual[-1].weight.fill_(0.4)
    sp, gl, mask, actions = tactical_data()
    output = model(sp, gl, mask, actions=actions[:1])
    greedy = model.act_greedy(sp, gl, mask)
    torch.testing.assert_close(greedy, output['log_probs'].argmax(-1).int())
    chosen, lp, value = model.act(sp, gl, mask)
    reevaluated = model(sp, gl, mask, actions=chosen)
    torch.testing.assert_close(lp, reevaluated['action_log_prob'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(value, reevaluated['value'])
    shared = model.forward_policy_value_shared(sp.expand(2, -1, -1, -1),
        gl.expand(2, -1), torch.tensor([1]), mask, actions=actions[:1])
    torch.testing.assert_close(shared['log_probs'], output['log_probs'], atol=2e-6, rtol=1e-6)
    ema = copy.deepcopy(model)
    torch.testing.assert_close(ema(sp, gl, mask, actions=actions[:1])['log_probs'],
                               output['log_probs'], rtol=0, atol=0)
    illegal = ~mask
    assert torch.all(output['log_probs'].exp()[illegal] == 0)
    loss = -output['action_log_prob'].mean()
    loss.backward()
    grad = model.combat_head.residual[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_feature_metadata_and_rollout_metrics():
    from junqi_rl.checkpoint_compat import current_checkpoint_metadata, validate_policy_checkpoint
    model = tiny(True)
    assert current_checkpoint_metadata(model)['combat_feature_version'] == 1
    ckpt = {'policy': model.state_dict(), 'checkpoint_meta': current_checkpoint_metadata(model)}
    validate_policy_checkpoint(tiny(True), ckpt)
    with pytest.raises(ValueError, match='combat_feature_version'):
        validate_policy_checkpoint(tiny(False), ckpt)
    data = tactical_data()
    sp, _, legal, actions = data
    features = all_features(model.combat_head, data)
    attack = actions[(features.sum(-1) > 0).nonzero()[0, 0]].reshape(1)
    metrics = CombatFeatureMetrics(model)
    metrics.add(sp, attack, legal, torch.tensor([False]))
    metrics.add(sp, attack, legal)
    logged = metrics.finish()
    assert logged['combat/chosen_attacks'] == 1
    assert logged['combat/chosen_attack_fraction'] == 1
    assert abs(sum(logged[f'combat/chosen_{name}_mean'] for name in
                   ('eat_survive', 'own_dies', 'mutual_death')) - 1) < 1e-6


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_real_ppo_optimizer_updates_combat_residual(shared, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required for device optimizer validation')
    from junqi_rl.training.ppo import PPOConfig, PPOTrainer
    from junqi_rl.training.rollout import RolloutBatch
    model = tiny(True).to(device)
    trainer = PPOTrainer(model, PPOConfig(net=model.cfg, dtype='float32',
        num_epochs_per_rollout=1), device=device)
    sp, gl, mask, actions = (x.to(device) for x in tactical_data())
    features = all_features(model.combat_head, (sp, gl, mask, actions))
    attack = actions[(features.sum(-1) > 0).nonzero()[0, 0]].reshape(1)
    with torch.no_grad():
        old = trainer.policy(sp, gl, mask, actions=attack)['action_log_prob'].clone()
    batch = RolloutBatch(obs_spatial=sp, obs_global=gl, legal_mask=mask,
        actions=attack, old_log_probs=old, advantages=torch.tensor([0.25], device=device),
        returns=torch.tensor([0.5], device=device), values=torch.zeros(1, device=device),
        adv_mask=torch.ones(1, dtype=torch.bool, device=device), value_only_mask=torch.zeros(1, dtype=torch.bool, device=device))
    if shared:
        batch.value_obs_spatial = sp
        batch.value_obs_global = gl
        batch.value_returns = batch.returns
        batch.policy_value_indices = torch.arange(1, device=device)
    before = trainer.policy.combat_head.residual[-1].weight.detach().clone()
    trainer._update_step(batch)
    after = trainer.policy.combat_head.residual[-1].weight.detach()
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
