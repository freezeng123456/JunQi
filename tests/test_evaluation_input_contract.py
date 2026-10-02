"""Public evaluation and replay must use the loaded policy's input semantics."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from junqi_core.info_model import BeliefTensor
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_core.rules import ShowMode
from junqi_rl.analysis.evaluate import eval_head_to_head, eval_vs_random
from junqi_rl.analysis.record import record_game_with_policy
from junqi_rl.input_contract import policy_input_contract
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def tiny(**overrides):
    cfg = JunqiNetConfig(cnn_channels=8, cnn_layers=1, depth=1, embed_dim=16,
                         n_head=2, ff_factor=2, action_key_dim=8,
                         use_global_summary=False, identity_encoding='known_only')
    return JunqiNet(replace(cfg, **overrides)).eval()


@pytest.mark.parametrize('greedy', [False, True])
def test_public_facts_evaluators_and_replay_never_create_rules(monkeypatch, greedy):
    def forbidden(*args, **kwargs):
        raise AssertionError('Facts evaluation created a legacy belief')
    monkeypatch.setattr(BeliefTensor, 'initial', forbidden)
    first, second = tiny(), tiny()
    observed = []
    def inspect_input(model, args):
        spatial, global_ = args[:2]
        assert global_.shape[-1] == 0
        for group in ('prob_teammate', 'belief_left_side', 'belief_right_side'):
            planes = spatial[:, CHANNEL_LAYOUT[group]]
            assert torch.all((planes == 0) | (planes == 1))
        observed.append(True)
    # act_greedy encodes directly, so inspect the shared encoder entry point.
    for policy in (first, second):
        original = policy._encode
        def checked(spatial, global_, _original=original, _policy=policy):
            inspect_input(_policy, (spatial, global_))
            return _original(spatial, global_)
        monkeypatch.setattr(policy, '_encode', checked)
    metrics = eval_head_to_head(first, second, num_games=2, max_steps=4,
                               device='cpu', greedy=greedy, seed_base=7)
    assert metrics['league/num_games'] == 2
    stats = eval_vs_random(first, num_games=2, max_steps=4, device='cpu', greedy=greedy)
    assert stats['num_games'] == 2
    trajectory = record_game_with_policy(first, rng_seed=7, device='cpu', max_steps=4,
                                         show_mode=ShowMode.DARK, greedy=greedy)
    assert trajectory.meta['global_dims'] == 0 and trajectory.beliefs is None
    assert observed


def test_cross_schema_match_fails_before_environment_allocation(monkeypatch):
    from junqi_rl import env
    monkeypatch.setattr(env, 'VectorJunqiEnv', lambda **kw: pytest.fail('Environment allocated'))
    with pytest.raises(ValueError, match='Cross-schema'):
        eval_head_to_head(tiny(), tiny(identity_encoding='legacy_probabilities'), device='cpu')


def test_explicit_legacy_contract_can_share_optional_globals():
    without = tiny(identity_encoding='legacy_probabilities')
    with_globals = tiny(identity_encoding='legacy_probabilities', use_global_summary=True)
    assert policy_input_contract(without, with_globals).observation_kwargs() == {
        'identity_encoding': 'legacy_probabilities', 'include_global': True}
    result = eval_head_to_head(without, with_globals, device='cpu', num_games=1, max_steps=2)
    assert result['league/num_games'] == 1


def test_missing_or_invalid_contract_is_not_guessed():
    for policy in (SimpleNamespace(),
                   SimpleNamespace(identity_encoding='known_only', uses_global_summary=0),
                   SimpleNamespace(identity_encoding='known_only', uses_global_summary=True)):
        with pytest.raises(ValueError):
            policy_input_contract(policy)


@pytest.mark.parametrize('greedy', [False, True])
def test_public_head_to_head_rejects_nonfinite_policy(monkeypatch, greedy):
    policy = tiny()
    original = policy.forward
    def invalid(*args, **kwargs):
        output = original(*args, **kwargs)
        output['log_probs'].fill_(torch.nan)
        return output
    monkeypatch.setattr(policy, 'forward', invalid)
    with pytest.raises(ValueError, match='Invalid evaluation policy distribution'):
        eval_head_to_head(policy, policy, device='cpu', num_games=1,
                          max_steps=2, greedy=greedy)
