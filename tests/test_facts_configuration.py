"""Current CLI defaults are explicit; historical profiles retain their meaning."""
from pathlib import Path

from scripts.train import build_parser
from junqi_rl.training.config import load_config, load_yaml_config

ROOT = Path(__file__).resolve().parents[1]


def test_default_cli_uses_completed_game_facts_profile_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(build_parser().parse_args([]))
    assert cfg.net.identity_encoding == 'known_only'
    assert not cfg.net.use_global_summary and not cfg.net.policy_features
    assert not cfg.belief.enabled and not cfg.arr.enabled and not cfg.reward_shaping
    assert cfg.target_completed_games == 2000 and cfg.env.use_gpu_rollout


def test_explicit_historical_profile_and_cli_overrides_still_work():
    args = build_parser().parse_args([
        '--config', str(ROOT/'configs/default.yaml'), '--num_envs', '3'])
    cfg = load_config(args)
    assert cfg.net.identity_encoding == 'legacy_probabilities'
    assert cfg.net.use_global_summary and cfg.env.num_envs == 3


def test_facts_base_does_not_inherit_a_historical_experiment():
    import yaml
    base = yaml.safe_load((ROOT/'configs/facts_base.yaml').read_text())
    assert 'extends' not in base
    assert base['net']['identity_encoding'] == 'known_only'
    assert not base['belief']['enabled'] and not base['reward_shaping']
    cfg = load_yaml_config(ROOT/'configs/v100_facts.yaml')
    assert cfg['net'] == base['net'] and cfg['ppo'] == base['ppo']
    assert cfg['env']['num_envs'] == 32 and cfg['env']['steps_per_env'] == 128
    assert cfg['total_rollouts'] == 512  # Historical run, not 512 games.
