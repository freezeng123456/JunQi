"""Facts-only input: no rule probabilities, no hidden truth, no silent migration."""
from dataclasses import replace
import numpy as np
import pytest
import torch

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import ObservationBuilder, CHANNEL_LAYOUT
from junqi_core.rules import PieceType, Seat, ShowMode
from junqi_core.state import Action
from junqi_rl.env import VectorJunqiEnv
from junqi_rl.checkpoint_compat import current_checkpoint_metadata, validate_policy_checkpoint
from junqi_rl.training.checkpoint import export_evaluation_policy, load_evaluation_policy
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from tests.test_feature_information_boundaries import position

def tiny():
    return JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1, embed_dim=16,
                         n_head=2, ff_factor=2, action_key_dim=8,
                         use_global_summary=False, identity_encoding='known_only')

def test_collection_never_creates_rules_and_unknowns_are_zero(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Rule probabilities must not be computed')
    monkeypatch.setattr(BeliefTensor, 'initial', forbidden)
    env = VectorJunqiEnv(2, show_mode=ShowMode.DARK, include_global=False,
                         identity_encoding='known_only')
    from junqi_rl.training.collector import collect_rollout
    from junqi_rl.training.rollout import RolloutBuffer
    from junqi_rl.training.ppo import PPOConfig, PPOTrainer
    policy = JunqiNet(tiny())
    initial = policy.cls_token.detach().clone()
    rollout = RolloutBuffer(num_envs=2, steps_per_env=4, obs_global_dims=0,
                            adv_filt_rate=1, adv_filt_thresh=0)
    collect_rollout(env, policy, rollout, device='cpu', seed_base=20261002)
    assert all(b is None for e in env._envs for b in e.beliefs.values())
    for name in ('prob_teammate', 'belief_left_side', 'belief_right_side'):
        x = env.obs_spatial[:, :, CHANNEL_LAYOUT[name]]
        assert np.all((x == 0) | (x == 1))
        assert np.all(x.sum(axis=2) <= 1)
    trainer = PPOTrainer(policy, PPOConfig(net=policy.cfg, dtype='float32',
        minibatch_size=4, num_epochs_per_rollout=1, adv_filt_rate=1, adv_filt_thresh=0), device='cpu')
    trainer.train_epoch(rollout, rng=np.random.default_rng(1))
    assert trainer.num_train_step > 0 and not torch.equal(initial, policy.cls_token)

def test_equal_public_history_does_not_disclose_unknown_rank():
    builder = ObservationBuilder(include_global=False, identity_encoding='known_only')
    views = []
    for kind in (PieceType.SHIZH, PieceType.JUNZH, PieceType.SILING):
        state = position(kind, PieceType.LVZH)
        after, _ = state.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
        obs = builder.build(after, None, Seat.WEST).snapshot()
        views.append(obs.spatial)
        floor = obs.spatial[CHANNEL_LAYOUT['cm_floor_ge']].reshape(9, -1).max(-1)
        np.testing.assert_array_equal(floor, [1, 1, 1, 1, 1, 1, 1, 0, 0])
        for name in ('belief_left_side', 'belief_right_side'):
            assert obs.spatial[CHANNEL_LAYOUT[name]].sum() == 0
    for view in views[1:]:
        np.testing.assert_array_equal(view, views[0])

def test_public_mine_reveal_and_engineer_reveal_survive():
    builder = ObservationBuilder(include_global=False, identity_encoding='known_only')
    state = position(PieceType.SILING, PieceType.DILEI)
    after, _ = state.step(Action(Seat.SOUTH, (2, 7), (1, 7)))
    obs = builder.build(after, None, Seat.SOUTH)
    from junqi_core.info_model import TRACKED_TYPES
    sl = CHANNEL_LAYOUT['belief_left_side']
    assert obs.spatial[sl.start + TRACKED_TYPES.index(PieceType.DILEI), 7, 1] == 1
    state = position(PieceType.GONGB, PieceType.LVZH)
    pid = state.pieces[(1, 7)].piece_id
    state.combat_memory.is_gongb[:, pid] = True
    obs = builder.build(state, None, Seat.SOUTH)
    assert obs.spatial[sl.start + TRACKED_TYPES.index(PieceType.GONGB), 7, 1] == 1

def test_checkpoint_semantics_and_forbidden_heads(tmp_path):
    model = JunqiNet(tiny())
    restored = load_evaluation_policy(export_evaluation_policy(model, tmp_path / 'facts.pt'))
    assert current_checkpoint_metadata(restored)['architecture_id'] == 'junqi.graph317.facts.v1'
    legacy = JunqiNet(replace(tiny(), identity_encoding='legacy_probabilities'))
    for meta in ({}, {'checkpoint_meta': current_checkpoint_metadata(legacy)}):
        with pytest.raises(ValueError, match='identity_encoding'):
            validate_policy_checkpoint(model, {'policy': legacy.state_dict(), **meta})
    for head in ('combat', 'tactical', 'relational'):
        with pytest.raises(ValueError, match='known_only'):
            JunqiNet(replace(tiny(), **{head + ('_outcome_features' if head == 'combat' else '_features'): True}))
    with pytest.raises(ValueError, match='include_global'):
        ObservationBuilder(identity_encoding='known_only')
