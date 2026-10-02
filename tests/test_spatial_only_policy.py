"""Spatial-only policy contracts, real CPU collection/update, and migration."""

from argparse import Namespace
from dataclasses import replace
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from junqi_core import observation as obs_module
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import CHANNEL_LAYOUT, OBS_CHANNELS, ObservationBuilder
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from junqi_rl.checkpoint_compat import current_checkpoint_metadata, validate_policy_checkpoint
from junqi_rl.env import VectorJunqiEnv
from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.checkpoint import (
    export_evaluation_policy, load_evaluation_policy,
    migrate_policy_without_global_summary, save_checkpoint,
)
from junqi_rl.training.collector import collect_rollout
from junqi_rl.training.config import TrainConfig, load_config, validate_config
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout import RolloutBuffer


def tiny(**kwargs):
    return JunqiNet(JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1,
        embed_dim=16, n_head=2, ff_factor=2, action_key_dim=8, **kwargs)).eval()


def inputs(batch=2):
    spatial = torch.rand(batch, OBS_CHANNELS, 17, 17)
    legal = torch.zeros(batch, 16641, dtype=torch.bool)
    legal[:, [5, 13, 25]] = True
    return spatial, legal, torch.full((batch,), 13, dtype=torch.long)


@pytest.mark.parametrize('features', [{}, {'combat': 1}, {'combat': 1, 'tactical': 2},
                                      {'combat': 1, 'tactical': 2, 'relational': 1}])
def test_spatial_only_forward_ignores_global_and_learns_cls(features):
    model = tiny(use_global_summary=False, policy_features=features)
    sp, legal, actions = inputs()
    out = model(sp, None, legal, actions=actions)
    assert model.global_proj is None and model.observation_global_dims == 0
    for ignored in (torch.empty(2, 0), torch.full((2, 28), float('nan'))):
        other = model(sp, ignored, legal, actions=actions)
        for key in out:
            torch.testing.assert_close(other[key], out[key], rtol=0, atol=0)
    value = model.forward_value(sp, None)
    torch.testing.assert_close(value, out['value'])
    shared = model.forward_policy_value_shared(sp, None, torch.tensor([1]), legal[1:], actions=actions[1:])
    torch.testing.assert_close(shared['log_probs'], out['log_probs'][1:], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(shared['value'], out['value'])
    loss = -out['action_log_prob'].mean() + out['value'].square().mean()
    loss.backward()
    assert torch.isfinite(model.cls_token.grad).all()
    assert torch.count_nonzero(model.cls_token.grad) > 0
    assert not any(key.startswith('global_proj.') for key in model.state_dict())
    assert legal.gather(1, model.act_greedy(sp, None, legal).long().unsqueeze(1)).all()


@pytest.mark.parametrize('mode', list(ShowMode))
def test_omitting_global_preserves_spatial_and_skips_aggregation(mode, monkeypatch):
    state = GameState.new_game(generate_random_setup(random.Random(317)), show_mode=mode)
    belief = BeliefTensor.initial(state, Seat.SOUTH)
    expected = ObservationBuilder().build(state, belief, Seat.SOUTH).snapshot()

    def forbidden(**kwargs):
        raise AssertionError('Unused global summaries must not be calculated')

    monkeypatch.setattr(obs_module, '_write_global_features', forbidden)
    builder = ObservationBuilder(include_global=False)
    actual = builder.build(state, belief, Seat.SOUTH)
    np.testing.assert_array_equal(actual.spatial, expected.spatial)
    assert actual.global_.shape == (0,)
    sp = np.empty((1, OBS_CHANNELS, 17, 17), dtype=np.float32)
    gl = np.empty((1, 0), dtype=np.float32)
    builder.build_observations_batch([state], [belief], [Seat.SOUTH], sp, gl)
    np.testing.assert_array_equal(sp[0], expected.spatial)


def test_migration_is_explicit_and_matches_zeroed_legacy_input(tmp_path):
    source = tiny(use_global_summary=True, policy_features={'combat': 1, 'tactical': 2})
    with torch.no_grad():
        source.global_proj.bias.uniform_(-0.2, 0.2)
    migrated = migrate_policy_without_global_summary(source)
    assert sum(p.numel() for p in source.parameters()) - sum(p.numel() for p in migrated.parameters()) == 28 * 16
    for key, tensor in source.state_dict().items():
        if not key.startswith('global_proj.'):
            torch.testing.assert_close(migrated.state_dict()[key], tensor, rtol=0, atol=0)
    sp, legal, actions = inputs()
    expected = source(sp, torch.zeros(2, 28), legal, actions=actions)
    actual = migrated(sp, None, legal, actions=actions)
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match='architecture_id'):
        validate_policy_checkpoint(migrated, {'policy': source.state_dict(),
            'checkpoint_meta': current_checkpoint_metadata(source)})
    path = export_evaluation_policy(migrated, tmp_path / 'spatial.pt')
    loaded = load_evaluation_policy(path)
    assert not loaded.uses_global_summary
    meta = current_checkpoint_metadata(loaded)
    assert meta['observation_global_dims'] == 0
    assert meta['architecture_id'] == 'junqi.graph317.cls.v1'
    for key, value in loaded(sp, None, legal, actions=actions).items():
        torch.testing.assert_close(value, actual[key], rtol=0, atol=0)


def test_old_checkpoint_missing_flag_keeps_28_dim_architecture(tmp_path):
    source = tiny()
    path = tmp_path / 'legacy.pt'
    export_evaluation_policy(source, path)
    state = torch.load(path, weights_only=True)
    del state['train_cfg']['net']['use_global_summary']
    del state['checkpoint_meta']['global_summary_enabled']
    torch.save(state, path)
    loaded = load_evaluation_policy(path)
    assert loaded.uses_global_summary and loaded.global_proj.in_features == 28
    sp, legal, _ = inputs()
    with pytest.raises(ValueError, match='requires obs_global'):
        loaded(sp, None, legal)


@pytest.mark.parametrize('categorical', [False, True])
def test_cpu_collection_update_save_and_resume_without_global(tmp_path, categorical):
    torch.manual_seed(317)
    policy = tiny(use_global_summary=False, use_cat_vf=categorical)
    env = VectorJunqiEnv(num_envs=2, max_num_moves=32, show_mode=ShowMode.DARK, include_global=False)
    rollout = RolloutBuffer(num_envs=2, steps_per_env=4, obs_global_dims=0,
                            adv_filt_rate=1, adv_filt_thresh=0)
    ppo = PPOConfig(net=policy.cfg, dtype='float32', minibatch_size=4,
                    num_epochs_per_rollout=1, lr_coef=1e-4, lr_floor=1e-4, lr_ceil=1e-4,
                    adv_filt_rate=1, adv_filt_thresh=0)
    trainer = PPOTrainer(policy, ppo, device='cpu')
    initial = policy.cls_token.detach().clone()
    collect_rollout(env, policy, rollout, device='cpu', seed_base=317)
    assert env.obs_global.shape == (2, 4, 0)
    assert rollout.obs_global.shape == (4, 2, 0) and rollout.obs_global.nbytes == 0
    metrics = trainer.train_epoch(rollout, rng=np.random.default_rng(317))
    assert trainer.num_train_step > 0 and trainer.num_rollout == 1
    assert torch.isfinite(policy.cls_token).all() and not torch.equal(policy.cls_token, initial)
    assert metrics['train/num_updates'] > 0
    path = save_checkpoint(trainer, TrainConfig(net=policy.cfg, ppo=ppo), 1, str(tmp_path))
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    restored = PPOTrainer(tiny(use_global_summary=False, use_cat_vf=categorical), ppo, device='cpu')
    restored.load_state_dict(checkpoint)
    assert restored.num_train_step == trainer.num_train_step
    assert restored.num_rollout == trainer.num_rollout
    torch.testing.assert_close(restored.policy.cls_token, policy.cls_token, rtol=0, atol=0)


def test_beliefnet_probabilities_feed_spatial_only_policy():
    from junqi_rl.belief.inference import (
        _live_enemy_world_mask, constrain_neural_beliefs, rule_only_belief_observations,
    )
    from junqi_rl.training.collector import _build_legal_mask
    env = VectorJunqiEnv(num_envs=1, show_mode=ShowMode.DARK, include_global=False)
    spatial, global_ = env.reset(seed_base=317)
    sp = torch.from_numpy(spatial.copy())
    rules = torch.zeros(1, 4, 12, 289)
    for observer, belief in env.envs[0]._beliefs.items():
        for (x, y), vector in belief.probs.items():
            rules[0, observer.value, :, y * 17 + x] = torch.from_numpy(vector)
    net = BeliefNet(BeliefNetConfig(cnn_channels=8, cnn_layers=1, embed_dim=16,
                                  n_encoder_layer=1, n_head=2, ff_factor=2)).eval()
    with torch.no_grad():
        logits = net(sp.flatten(0, 1), seat_idx=torch.arange(4))['logits']
        live = _live_enemy_world_mask(sp)
        upload = constrain_neural_beliefs(logits.softmax(-1).reshape(1, 4, 289, 12), rules, live)
        assert not torch.count_nonzero(upload[rules == 0])
        torch.testing.assert_close(upload.sum(2)[live], torch.ones_like(upload.sum(2)[live]))
        policy_sp = rule_only_belief_observations(sp, upload)
        legal = torch.from_numpy(_build_legal_mask(env, env.current_seats()))
        policy = tiny(use_global_summary=False, policy_features={'combat': 1, 'tactical': 2})
        out = policy(policy_sp[:, 0], None, legal)
        assert torch.isfinite(out['value']).all()
        assert legal.gather(1, out['action'].long().unsqueeze(1)).all()
    assert global_.shape == (1, 4, 0)


def test_new_v100_configs_disable_global_and_keep_matched_belief_control():
    root = Path(__file__).resolve().parents[1]
    configs = [load_config(Namespace(config=str(root / f'configs/v100_spatial_{arm}.yaml'), extra=[]))
               for arm in ('belief', 'rules')]
    assert configs[0].belief.enabled and not configs[1].belief.enabled
    for cfg in configs:
        assert not cfg.reward_shaping
        assert cfg.net.use_global_summary is False
        assert cfg.ppo.dtype == cfg.belief.ppo.autocast_dtype == 'float32'
        assert cfg.resume == cfg.init_policy == ''
        assert cfg.seed == 20261001
    invalid = replace(configs[0], resume='old.pt', init_policy='new.pt')
    with pytest.raises(ValueError, match='mutually exclusive'):
        validate_config(invalid)
