"""Validate all 8 H20 experiment configs: YAML loads, dataclass converts
cleanly, nets build on CPU, param counts are sane.

Run with:
    pytest tests/test_h20_configs.py -xvs
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import yaml


ROOT = pathlib.Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train.py"


def _load_train():
    spec = importlib.util.spec_from_file_location("train_mod", TRAIN)
    mod = importlib.util.module_from_spec(spec)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sys.modules["train_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_train()


H20_EXPS = [
    "h20_exp1_v32_bf16_base",
    "h20_exp2_belief_bigger",
    "h20_exp3_move_bigger",
    "h20_exp4_draw_penalty",
    "h20_exp5_self_play",
    "h20_exp6_arr_slow_refresh",
    "h20_exp7_longer_train",
    "h20_exp8_bigger_both",
]


@pytest.mark.parametrize("exp", H20_EXPS)
def test_config_loads(mod, exp):
    """Each H20 config must load + roundtrip through our dataclass."""
    cfg_path = ROOT / "exps" / exp / "cfg.yaml"
    assert cfg_path.is_file(), f"Missing config: {cfg_path}"
    with open(cfg_path) as f:
        d = yaml.safe_load(f)
    cfg = mod._dict_to_dataclass(mod.TrainConfig, d)

    # Core sanity checks shared across all 8.
    assert cfg.env.num_envs == 512, f"{exp}: num_envs should be 512 on H20"
    assert cfg.ppo.dtype == "bfloat16", f"{exp}: dtype should be bfloat16"
    assert cfg.env.seed in range(101, 109), f"{exp}: seed should be in 101-108 range"
    assert cfg.early_stop_win_rate > 0, f"{exp}: early_stop should be armed"


def test_seeds_are_unique(mod):
    """Each of the 8 experiments must use a distinct seed (101-108) so
    they can be compared against each other without seed confound."""
    seeds = []
    for exp in H20_EXPS:
        with open(ROOT / "exps" / exp / "cfg.yaml") as f:
            d = yaml.safe_load(f)
        cfg = mod._dict_to_dataclass(mod.TrainConfig, d)
        seeds.append(cfg.env.seed)
    assert len(set(seeds)) == 8, (
        f"Seeds must be unique across 8 experiments; got {sorted(seeds)}"
    )
    assert sorted(seeds) == list(range(101, 109))


@pytest.mark.parametrize("exp", H20_EXPS)
def test_junqi_net_builds(mod, exp):
    """JunqiNet has to actually build on the declared hyperparams."""
    import torch
    from junqi_rl.networks.junqi_net import JunqiNet
    from junqi_core.observation import OBS_CHANNELS

    with open(ROOT / "exps" / exp / "cfg.yaml") as f:
        d = yaml.safe_load(f)
    cfg = mod._dict_to_dataclass(mod.TrainConfig, d)
    net = JunqiNet(cfg.net)
    n_params = sum(p.numel() for p in net.parameters())
    # Print so we can eyeball the per-exp param counts
    print(f"  {exp}: JunqiNet = {n_params:,} params "
          f"(depth={cfg.net.depth}, embed={cfg.net.embed_dim})")
    # Min sanity bound: must be at least ~1M (baseline) and at most ~20M
    assert 900_000 < n_params < 20_000_000, (
        f"{exp}: JunqiNet param count {n_params:,} seems wrong"
    )

    # Quick forward-pass smoke test on CPU (small batch)
    B = 2
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17)
    obs_gl = torch.randn(B, 28)
    legal = torch.ones(B, 16641, dtype=torch.bool)
    legal[:, 0] = False
    net.eval()
    with torch.no_grad():
        out = net(obs_sp, obs_gl, legal)
    assert out["log_probs"].shape == (B, 16641)


def test_exp5_self_play_flag(mod):
    """Exp 5 must set random_opponent=False; all others must leave it True."""
    for exp in H20_EXPS:
        with open(ROOT / "exps" / exp / "cfg.yaml") as f:
            d = yaml.safe_load(f)
        cfg = mod._dict_to_dataclass(mod.TrainConfig, d)
        expected = (exp != "h20_exp5_self_play")
        assert cfg.random_opponent == expected, (
            f"{exp}: random_opponent={cfg.random_opponent}, expected {expected}"
        )


def test_exp2_and_exp8_have_belief_net_override(mod):
    """Exps 2 and 8 override belief.net; others use library defaults."""
    for exp in H20_EXPS:
        with open(ROOT / "exps" / exp / "cfg.yaml") as f:
            d = yaml.safe_load(f)
        cfg = mod._dict_to_dataclass(mod.TrainConfig, d)
        wants_override = exp in ("h20_exp2_belief_bigger", "h20_exp8_bigger_both")
        has_override = cfg.belief.net is not None
        assert has_override == wants_override, (
            f"{exp}: belief.net override presence {has_override}, expected {wants_override}"
        )


def test_all_have_bf16(mod):
    """On H20 we want bf16 everywhere (arr_train's autocast_dtype too)."""
    for exp in H20_EXPS:
        with open(ROOT / "exps" / exp / "cfg.yaml") as f:
            d = yaml.safe_load(f)
        cfg = mod._dict_to_dataclass(mod.TrainConfig, d)
        assert cfg.ppo.dtype == "bfloat16"
        if cfg.arr.enabled and cfg.arr.ppo is not None:
            assert cfg.arr.ppo.autocast_dtype == "bfloat16", (
                f"{exp}: arr.ppo.autocast_dtype should be bfloat16"
            )
