"""Typed configuration loading and validation for JunQi RL training."""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, get_args, get_type_hints

import torch
import yaml

from junqi_rl.networks.arrangement_net import ArrangementNetConfig
from junqi_rl.networks.belief_net import BeliefNetConfig
from junqi_rl.networks.junqi_net import JunqiNetConfig
from junqi_rl.training.arr_ppo import ArrangementPPOConfig
from junqi_rl.training.belief_ppo import BeliefPPOConfig
from junqi_rl.training.ppo import PPOConfig


@dataclass
class EnvConfig:
    """Environment and rollout dimensions."""

    num_envs: int = 32
    steps_per_env: int = 128
    max_num_moves: int = 4000
    seed: int = 42
    use_gpu_rollout: bool = False


@dataclass
class ArrangementTrainConfig:
    """Optional arrangement-policy training."""

    enabled: bool = False
    refresh_every: int = 1
    n_arr: int = 1024
    # Combined 4-seat boards uploaded to CUDA. 0 = zip n_arr//4 frozen
    # tuples (legacy). >0 independently re-pairs per-seat samples up to
    # this many rows so parallel games do not reuse a few hundred openings.
    pool_size: int = 10_000
    storage_duration: int = 4
    net: ArrangementNetConfig = field(default_factory=ArrangementNetConfig)
    ppo: ArrangementPPOConfig = field(default_factory=ArrangementPPOConfig)
    ema_decay: float = 0.999
    reg_temp_init: float = 0.1
    reg_temp_decay: float = 0.3
    reg_temp_floor: float = 0.0
    reg_norm: float = 10.0


@dataclass
class BeliefTrainConfig:
    """Belief-network training and inference (default on)."""

    enabled: bool = True
    refresh_every: int = 1
    buffer_capacity: int = 12_000
    warmup_rollouts: int = 20
    net: BeliefNetConfig | dict[str, Any] | None = None
    ppo: BeliefPPOConfig | dict[str, Any] | None = None
    ema_decay: float = 0.999
    infer_chunk_size: int = 128


@dataclass
class RolloutTrainConfig:
    """GPU rollout storage strategy."""

    storage_mode: str = "full_obs"
    csr_legal_mask: bool = True
    csr_k_max: int = 256


@dataclass
class TrainConfig:
    """Complete training configuration."""

    env: EnvConfig = field(default_factory=EnvConfig)
    net: JunqiNetConfig = field(default_factory=JunqiNetConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    arr: ArrangementTrainConfig = field(default_factory=ArrangementTrainConfig)
    belief: BeliefTrainConfig = field(default_factory=BeliefTrainConfig)
    rollout: RolloutTrainConfig = field(default_factory=RolloutTrainConfig)

    save_dir: str = "exps/default"
    total_rollouts: int = 10_000
    save_every: int = 100
    eval_every: int = 50
    eval_num_games: int = 128
    eval_record_games: int = 1
    eval_baseline_ckpt: str = ""
    eval_baseline_games: int = 0
    eval_record_beliefs: bool = False
    # GPU setup pools are process-global. Re-upload this fixed pool before
    # every primary evaluation so ArrangementNet refreshes cannot silently
    # change the test distribution from checkpoint to checkpoint.
    eval_fixed_setup_pool: bool = True
    eval_setup_seed: int = 20_260_817
    league_max_checkpoints: int = 12
    league_eval_games: int = 16

    early_stop_win_rate: float = 0.0
    early_stop_patience: int = 3
    early_stop_min_rollout: int = 100

    resume: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_wandb: bool = False
    wandb_project: str = "junqi-rl"
    wandb_run_name: str = ""
    log_every: int = 10
    seed: int = 42
    torch_deterministic: bool = False

    random_opponent: bool = True
    train_value_on_random_seats: bool = False
    reward_shaping: bool = True

    fixed_setup_styles: tuple[str, ...] | list[str] | None = None
    mixed_setup: bool = False
    mixed_own_team_styles: tuple[str, ...] | list[str] = ("T",)

    # Compatibility switches used by historical curriculum configs. New
    # experiments should disable the corresponding component instead.
    disable_arr_train: bool = False
    disable_belief_train: bool = False


def _nested_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` in place."""

    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _nested_update(base[key], value)
        else:
            base[key] = value
    return base


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert nested dataclasses to portable YAML values."""

    if dataclasses.is_dataclass(obj):
        return {
            item.name: _dataclass_to_dict(getattr(obj, item.name))
            for item in dataclasses.fields(obj)
        }
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(value) for value in obj]
    return obj


def _nested_dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    for candidate in get_args(annotation):
        if isinstance(candidate, type) and dataclasses.is_dataclass(candidate):
            return candidate
    return None


def _dict_to_dataclass(
    cls: type[Any],
    values: Any,
    *,
    path: str = "",
) -> Any:
    """Build a nested dataclass and reject misspelled configuration keys."""

    if not dataclasses.is_dataclass(cls) or not isinstance(values, dict):
        return values

    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        location = path or cls.__name__
        names = ", ".join(unknown)
        raise ValueError(f"unknown configuration key(s) at {location}: {names}")

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        annotation = hints.get(name, fields[name].type)
        nested_cls = _nested_dataclass_type(annotation)
        child_path = f"{path}.{name}" if path else name
        if nested_cls is not None and isinstance(value, dict):
            kwargs[name] = _dict_to_dataclass(
                nested_cls,
                value,
                path=child_path,
            )
        else:
            kwargs[name] = value
    return cls(**kwargs)


def parse_overrides(items: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Parse ``section__field=value`` command-line overrides."""

    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"invalid override {item!r}; expected section__field=value"
            )
        key, value_text = item.split("=", 1)
        parts = [part.strip() for part in key.split("__")]
        if not parts or any(not part for part in parts):
            raise ValueError(f"invalid override key in {item!r}")
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid YAML value in override {item!r}") from exc

        target = result
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"override path collides at {part!r}")
            target = existing
        target[parts[-1]] = value
    return result


_FLAT_TO_ENV_KEYS = {"num_envs", "steps_per_env", "use_gpu_rollout"}
_NON_CONFIG_ARGS = {"config", "extra", "resume_cli", "validate_only"}


def validate_config(cfg: TrainConfig) -> None:
    """Fail before allocating models when a configuration is inconsistent."""

    positive = {
        "env.num_envs": cfg.env.num_envs,
        "env.steps_per_env": cfg.env.steps_per_env,
        "env.max_num_moves": cfg.env.max_num_moves,
        "total_rollouts": cfg.total_rollouts,
        "save_every": cfg.save_every,
        "eval_every": cfg.eval_every,
        "eval_num_games": cfg.eval_num_games,
        "log_every": cfg.log_every,
        "early_stop_patience": cfg.early_stop_patience,
        "arr.refresh_every": cfg.arr.refresh_every,
        "belief.refresh_every": cfg.belief.refresh_every,
        "belief.buffer_capacity": cfg.belief.buffer_capacity,
        "belief.infer_chunk_size": cfg.belief.infer_chunk_size,
        "rollout.csr_k_max": cfg.rollout.csr_k_max,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(
            "configuration values must be positive: " + ", ".join(invalid)
        )
    if cfg.arr.n_arr <= 0 or cfg.arr.n_arr % 4:
        raise ValueError("arr.n_arr must be a positive multiple of 4")
    if cfg.arr.pool_size < 0:
        raise ValueError("arr.pool_size must be >= 0 (0 = zip-only pairing)")
    if cfg.ppo.lr_schedule_unit not in {"grad_step", "rollout"}:
        raise ValueError(
            "ppo.lr_schedule_unit must be 'grad_step' or 'rollout'"
        )
    if cfg.ppo.temperature_schedule_unit not in {"grad_step", "rollout"}:
        raise ValueError(
            "ppo.temperature_schedule_unit must be 'grad_step' or 'rollout'"
        )
    if cfg.ppo.kl_mode not in {"reverse_full", "sampled_proxy"}:
        raise ValueError(
            "ppo.kl_mode must be 'reverse_full' or 'sampled_proxy'"
        )
    if cfg.ppo.magnet_shape not in {"uniform_legal", "piece_then_dest"}:
        raise ValueError(
            "ppo.magnet_shape must be 'uniform_legal' or 'piece_then_dest'"
        )
    if cfg.ppo.minibatch_group not in {"global", "timestep"}:
        raise ValueError(
            "ppo.minibatch_group must be 'global' or 'timestep'"
        )
    cfg.ppo.get_dtype()
    if cfg.rollout.storage_mode not in {"full_obs", "compact_history"}:
        raise ValueError(
            "rollout.storage_mode must be 'full_obs' or 'compact_history'"
        )
    if cfg.rollout.storage_mode == "compact_history" and not cfg.env.use_gpu_rollout:
        raise ValueError(
            "compact_history requires env.use_gpu_rollout=true"
        )
    if cfg.mixed_setup and cfg.fixed_setup_styles:
        raise ValueError(
            "mixed_setup and fixed_setup_styles are mutually exclusive"
        )
    if cfg.mixed_setup and not cfg.mixed_own_team_styles:
        raise ValueError(
            "mixed_setup requires at least one mixed_own_team_styles entry"
        )


def load_config(args: argparse.Namespace) -> TrainConfig:
    """Resolve dataclass defaults, YAML, CLI flags, then ``--set`` values."""

    resolved = _dataclass_to_dict(TrainConfig())
    config_path = getattr(args, "config", "")
    if config_path:
        path = os.fspath(config_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"configuration file not found: {path}")
        with open(path, encoding="utf-8") as handle:
            yaml_values = yaml.safe_load(handle) or {}
        if not isinstance(yaml_values, dict):
            raise ValueError("top-level YAML configuration must be a mapping")
        _nested_update(resolved, yaml_values)
        print(f"[config] Loaded YAML: {path}")

    cli_values: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in _NON_CONFIG_ARGS or value is None:
            continue
        if "__" in key:
            target = cli_values
            parts = key.split("__")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        elif key in _FLAT_TO_ENV_KEYS:
            cli_values.setdefault("env", {})[key] = value
        else:
            cli_values[key] = value

    extra_values = parse_overrides(getattr(args, "extra", None) or [])
    _nested_update(cli_values, extra_values)
    _nested_update(resolved, cli_values)

    resume_cli = getattr(args, "resume_cli", "")
    if resume_cli:
        resolved["resume"] = resume_cli

    cfg = _dict_to_dataclass(TrainConfig, resolved)
    cfg.ppo.net = cfg.net
    validate_config(cfg)
    return cfg


__all__ = [
    "ArrangementTrainConfig",
    "BeliefTrainConfig",
    "EnvConfig",
    "RolloutTrainConfig",
    "TrainConfig",
    "_dataclass_to_dict",
    "_dict_to_dataclass",
    "_nested_update",
    "load_config",
    "parse_overrides",
    "validate_config",
]
