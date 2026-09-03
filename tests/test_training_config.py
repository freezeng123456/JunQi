from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from junqi_rl.training.config import (
    TrainConfig,
    _dataclass_to_dict,
    _dict_to_dataclass,
    load_config,
    parse_overrides,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _args(config: Path, *extra: str) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config),
        extra=list(extra),
        resume_cli="",
        validate_only=False,
    )


def test_iteration_baseline_is_sparse_and_valid() -> None:
    cfg = load_config(_args(ROOT / "configs" / "iteration_baseline.yaml"))

    assert cfg.env.use_gpu_rollout is True
    assert cfg.ppo.num_epochs_per_rollout == 1
    assert cfg.ppo.lr_schedule_unit == "rollout"
    assert cfg.arr.enabled is False
    assert cfg.belief.enabled is True
    assert cfg.mixed_setup is False
    assert cfg.fixed_setup_styles is None
    assert cfg.rollout.storage_mode == "compact_history"


def test_nested_cli_overrides_are_typed_yaml_values() -> None:
    parsed = parse_overrides(
        [
            "env__num_envs=16",
            "belief__enabled=true",
            "mixed_own_team_styles=[T, D]",
        ]
    )

    assert parsed == {
        "env": {"num_envs": 16},
        "belief": {"enabled": True},
        "mixed_own_team_styles": ["T", "D"],
    }


def test_load_config_applies_extra_after_yaml() -> None:
    cfg = load_config(
        _args(
            ROOT / "configs" / "iteration_baseline.yaml",
            "env__num_envs=8",
            "total_rollouts=2",
        )
    )

    assert cfg.env.num_envs == 8
    assert cfg.total_rollouts == 2


def test_current_h20_config_only_overrides_advantage_filter() -> None:
    base = load_config(_args(ROOT / "configs" / "ataraxos_selfplay.yaml"))
    current = load_config(_args(ROOT / "configs" / "h20_10m_current.yaml"))

    expected = _dataclass_to_dict(base)
    expected["ppo"]["adv_filt_rate"] = 1.0
    assert current.ppo.adv_filt_rate == 1.0
    assert _dataclass_to_dict(current) == expected


def test_ataraxos_config_uses_rollout_quantile_with_timestep_batches() -> None:
    cfg = load_config(_args(ROOT / "configs" / "ataraxos_selfplay.yaml"))

    assert cfg.ppo.num_epochs_per_rollout == 1
    assert cfg.ppo.minibatch_group == "timestep"
    assert cfg.ppo.adv_filter_scope == "rollout"
    assert cfg.ppo.adv_filt_rate == pytest.approx(0.25)
    assert cfg.ppo.adv_filt_thresh == pytest.approx(0.01)
    assert cfg.ppo.temperature_schedule_unit == "rollout"
    assert cfg.ppo.temperature_coef == pytest.approx(0.1)
    assert cfg.ppo.temperature_decay == pytest.approx(0.3)
    assert cfg.ppo.temperature_floor == pytest.approx(0.0)
    assert cfg.ppo.magnet_shape == "uniform_legal"


def test_value_all_config_only_decouples_value_sampling() -> None:
    base = load_config(_args(ROOT / "configs" / "ataraxos_selfplay.yaml"))
    value_all = load_config(
        _args(ROOT / "configs" / "ataraxos_selfplay_value_all.yaml")
    )

    expected = _dataclass_to_dict(base)
    expected["ppo"]["value_sample_scope"] = "all_valid"
    expected["ppo"]["value_minibatch_size"] = 2048
    assert _dataclass_to_dict(value_all) == expected


def test_ddp2_value_all_config_preserves_global_rollout_size() -> None:
    value_all = load_config(
        _args(ROOT / "configs" / "ataraxos_selfplay_value_all.yaml")
    )
    ddp2 = load_config(
        _args(ROOT / "configs" / "ataraxos_selfplay_value_all_ddp2.yaml")
    )

    expected = _dataclass_to_dict(value_all)
    expected["env"]["num_envs"] = value_all.env.num_envs // 2
    expected["save_dir"] = "exps/ataraxos_selfplay_value_all_ddp2"
    assert ddp2.rollout.storage_mode == "compact_history"
    assert ddp2.env.num_envs * 2 == value_all.env.num_envs
    assert _dataclass_to_dict(ddp2) == expected


def test_all_valid_value_scope_requires_timestep_minibatches() -> None:
    cfg = TrainConfig()
    cfg.ppo.value_sample_scope = "all_valid"
    cfg.ppo.minibatch_group = "global"

    with pytest.raises(ValueError, match="requires.*timestep"):
        validate_config(cfg)


def test_invalid_advantage_filter_scope_is_rejected() -> None:
    cfg = TrainConfig()
    cfg.ppo.adv_filter_scope = "rank"

    with pytest.raises(ValueError, match="adv_filter_scope"):
        validate_config(cfg)


def test_config_extends_cycle_fails_fast(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n", encoding="utf-8")
    second.write_text("extends: first.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="extends cycle"):
        load_config(_args(first))


def test_removed_sampled_proxy_config_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "sampled-proxy.yaml"
    config.write_text(
        "ppo:\n  kl_mode: sampled_proxy\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"ppo.*kl_mode"):
        load_config(_args(config))


def test_unknown_nested_key_fails_fast() -> None:
    with pytest.raises(ValueError, match=r"ppo.*clip_rnage"):
        _dict_to_dataclass(
            TrainConfig,
            {"ppo": {"clip_rnage": 0.2}},
        )


def test_incompatible_setup_modes_are_rejected() -> None:
    cfg = TrainConfig(
        mixed_setup=True,
        fixed_setup_styles=["T"],
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(cfg)


def test_compact_history_requires_gpu_rollout() -> None:
    cfg = TrainConfig()
    cfg.rollout.storage_mode = "compact_history"
    cfg.env.use_gpu_rollout = False

    with pytest.raises(ValueError, match="use_gpu_rollout"):
        validate_config(cfg)


def test_all_tracked_training_configs_remain_compatible() -> None:
    paths = sorted((ROOT / "configs").glob("*.yaml"))
    paths += sorted((ROOT / "exps").glob("**/cfg.yaml"))

    for path in paths:
        try:
            load_config(_args(path))
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{path.relative_to(ROOT)}: {exc}")
