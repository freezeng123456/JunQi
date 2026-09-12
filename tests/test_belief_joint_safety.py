"""Bound confidence, preserve information boundaries, and count real learning."""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from junqi_core.batched_state import BatchedGameState
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import CHANNEL_LAYOUT, ObservationBuilder
from junqi_core.rules import Seat, ShowMode, PieceType
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from junqi_rl.belief.buffer import BeliefBuffer
from junqi_rl.belief.inference import constrain_neural_beliefs, rule_only_belief_observations
from junqi_rl.belief.sampling import MidgameBeliefSampler
from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig
from junqi_rl.training.belief_ppo import BeliefPPOConfig, BeliefPPOTrainer


class CPUWorld:
    num_envs = 1

    def __init__(self, setups):
        self.game = GameState.new_game(setups, show_mode=ShowMode.DARK)
        batch = BatchedGameState.from_game_states([self.game])
        self.state = SimpleNamespace(copy_to_host=lambda: {
            name: getattr(batch, name).copy() for name in
            ('piece_seat_arr', 'piece_type_arr', 'alive', 'pos_x', 'pos_y')})
        self.observations = []
        self.rules = torch.zeros(1, 4, 12, 289)
        for seat in Seat:
            belief = BeliefTensor.initial(self.game, seat)
            self.observations.append(ObservationBuilder().build(self.game, belief, seat).spatial)
            for (x, y), probs in belief.probs.items():
                self.rules[0, seat.value, :, y*17+x] = torch.from_numpy(probs)
        self.observations = torch.from_numpy(np.stack(self.observations)[None])

    def build_all_seat_observations_torch(self):
        return self.observations, None

    def rule_beliefs_torch(self):
        return self.rules


def test_bounded_mix_preserves_support_and_public_identity_under_extreme_predictions():
    torch.manual_seed(912)
    rules = torch.rand(2, 4, 12, 289)
    rules[:, :, 0] = 0
    rules /= rules.sum(dim=2, keepdim=True)
    rules[:, :, :, 0] = 0
    rules[:, :, 3, 0] = 1  # public identity
    rules[:, :, :, 1] = 0  # empty square
    probs = torch.zeros(2, 4, 289, 12)
    probs[..., 7] = 1
    live = rules.sum(2) > 0
    mixed = constrain_neural_beliefs(probs, rules, live, neural_weight=.25, max_kl=.05)
    assert not mixed[rules == 0].any()
    assert torch.all(mixed >= .75 * rules - 1e-7)
    torch.testing.assert_close(mixed[:, :, :, 0], rules[:, :, :, 0])
    torch.testing.assert_close(mixed.sum(2)[live], torch.ones_like(mixed.sum(2)[live]))
    kl = (mixed * (mixed.clamp_min(1e-30).log() - rules.clamp_min(1e-30).log())).sum(2)
    assert kl.max() <= .050001
    zero = constrain_neural_beliefs(probs, rules, live, neural_weight=0)
    torch.testing.assert_close(zero, rules)


def test_rule_input_discards_previous_neural_confidence_for_all_rotations():
    world = CPUWorld(generate_random_setup(random.Random(9)))
    expected = rule_only_belief_observations(world.observations, world.rules)
    altered = world.observations.clone()
    for name in ('belief_left_side', 'belief_right_side'):
        group = CHANNEL_LAYOUT[name]
        occupied = altered[:, :, group].sum(2, keepdim=True) > 0
        altered[:, :, group] = 0
        altered[:, :, group.start:group.start+1] = occupied
    actual = rule_only_belief_observations(altered, world.rules)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(expected, world.observations, atol=0, rtol=0)


def test_hidden_truth_changes_targets_without_changing_player_input():
    setups = generate_random_setup(random.Random(7))
    changed = [list(lineup) for lineup in setups]
    for owner in (Seat.WEST, Seat.EAST):
        indices = [i for i, typ in enumerate(changed[owner.value])
                   if PieceType.SILING <= typ <= PieceType.PAIZH]
        vals = [changed[owner.value][i] for i in indices]
        for i, typ in zip(indices, vals[1:] + vals[:1], strict=True):
            changed[owner.value][i] = typ
    buffers = []
    for source in (setups, changed):
        world = CPUWorld(source)
        buffer = BeliefBuffer(capacity=8, seed=1)
        sample = MidgameBeliefSampler(buffer, every_steps=1, envs_per_sample=1)
        original = world.observations.clone()
        sample(rollout_world=world)
        assert len(buffer) == 4
        assert set(buffer._seat[:4]) == {0, 1, 2, 3}
        assert (buffer._enemy[:4].sum(1) > 0).all()
        torch.testing.assert_close(world.observations, original, atol=0, rtol=0)
        for row in range(4):
            for cell in np.flatnonzero(buffer._enemy[row]):
                piece = world.game.pieces[(int(cell % 17), int(cell // 17))]
                assert piece.seat.team != Seat(row).team
                assert buffer._label[row, cell] == piece.piece_type.value - 2
        buffers.append(buffer)
    np.testing.assert_array_equal(buffers[0]._obs[0], buffers[1]._obs[0])
    assert np.any(buffers[0]._label[0] != buffers[1]._label[0])


def test_empty_supervision_skips_adam_momentum_and_ema():
    world = CPUWorld(generate_random_setup(random.Random(3)))
    buf = BeliefBuffer(capacity=8, seed=3)
    MidgameBeliefSampler(buf, every_steps=1)(rollout_world=world)
    net = BeliefNet(BeliefNetConfig(n_encoder_layer=1, n_head=2, embed_dim=16,
                                  cnn_channels=8, cnn_layers=1, ff_factor=2))
    trainer = BeliefPPOTrainer(net, BeliefPPOConfig(batch_size=2, epochs_per_rollout=1,
                                                  ema_decay=.5))
    assert trainer.train_epoch(buf)['belief_train/num_updates'] == 1
    before = copy.deepcopy(trainer.state_dict())
    buf._label[:len(buf)] = -1
    metrics = trainer.train_epoch(buf)
    assert metrics['belief_train/num_updates'] == 0
    assert metrics['belief_train/n_revealed'] == 0
    assert metrics['belief_train/empty_skip'] == 1
    assert trainer.num_train_step == 1
    after = trainer.state_dict()
    for group in ('net', 'ema'):
        for key in before[group]:
            torch.testing.assert_close(before[group][key], after[group][key], atol=0, rtol=0)
    for pid in before['optimizer']['state']:
        for key in before['optimizer']['state'][pid]:
            torch.testing.assert_close(before['optimizer']['state'][pid][key],
                                       after['optimizer']['state'][pid][key], atol=0, rtol=0)


@pytest.mark.parametrize('weight, cap', [(float('nan'), .1), (-.1, .1), (1.1, .1), (.2, 0)])
def test_invalid_coupling_settings_fail_closed(weight, cap):
    with pytest.raises(ValueError):
        constrain_neural_beliefs(torch.ones(1,4,289,12), torch.ones(1,4,12,289),
                                torch.ones(1,4,289,dtype=torch.bool),
                                neural_weight=weight, max_kl=cap)


@pytest.mark.parametrize('outcome', ['backoff', 'reject', 'exception'])
def test_publication_guard_restores_live_state_and_policy_mode(monkeypatch, outcome):
    from junqi_rl.belief import inference

    class PublicationWorld:
        def __init__(self):
            self.current = torch.tensor([.37, .63])
            self.weight = 0.0

        def current_beliefs_torch(self):
            return self.current

        def upload_beliefs(self, values):
            self.current = torch.as_tensor(values).clone()
            self.weight = 0.0

    world = PublicationWorld()
    original = world.current.clone()
    policy = torch.nn.Linear(1, 1)
    policy.train()
    weights = []

    def distribution(current_world, _policy):
        assert not _policy.training
        disruptive = current_world.weight > 0 and (
            outcome != 'backoff' or current_world.weight > .125)
        probabilities = torch.tensor([.999, .001] if disruptive else [.5, .5])
        return probabilities.log().reshape(1, 1, 2), torch.ones(1, 1, dtype=torch.bool)

    def refresh(current_world, _belief, *, neural_weight, **kwargs):
        weights.append(neural_weight)
        current_world.current = torch.tensor([neural_weight, 1-neural_weight])
        current_world.weight = neural_weight
        if outcome == 'exception':
            raise FloatingPointError('injected failure after upload')
        return {'belief_infer/neural_weight': neural_weight}

    monkeypatch.setattr(inference, '_publication_policy_distribution', distribution)
    monkeypatch.setattr(inference, 'refresh_beliefs_neural', refresh)
    rng_before = torch.random.get_rng_state().clone()
    if outcome == 'exception':
        with pytest.raises(FloatingPointError, match='injected failure'):
            inference.guarded_refresh_beliefs_neural(world, None, policy, neural_weight=.25)
    else:
        metrics = inference.guarded_refresh_beliefs_neural(world, None, policy, neural_weight=.25)
        if outcome == 'backoff':
            assert weights == [.25, .125]
            assert metrics['belief_guard/backoffs'] == 1
            assert metrics['belief_guard/rejected'] == 0
            torch.testing.assert_close(world.current, torch.tensor([.125, .875]))
        else:
            assert len(weights) == 9
            assert metrics['belief_guard/rejected'] == 1
    if outcome != 'backoff':
        torch.testing.assert_close(world.current, original, atol=0, rtol=0)
    assert policy.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, atol=0, rtol=0)
