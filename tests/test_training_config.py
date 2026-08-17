from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from junqi_rl.training.config import (
    TrainConfig,
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
    assert cfg.mixed_setup is True
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
        values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        try:
            cfg = _dict_to_dataclass(TrainConfig, values)
            cfg.ppo.net = cfg.net
            validate_config(cfg)
        except (TypeError, ValueError) as exc:
            pytest.fail(f"{path.relative_to(ROOT)}: {exc}")
