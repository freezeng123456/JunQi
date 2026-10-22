"""Regression gates for mobility-conditioned collapse detection and resume."""
from argparse import Namespace
from dataclasses import asdict
import json
import random

import numpy as np
import pytest
import torch

from junqi_rl.training.config import load_config
from junqi_rl.training.opening_monitor import (
    OpeningMonitor, OpeningMonitorConfig, preserve_probe_state, summarize_sources,
)


def cfg(**kw):
    return OpeningMonitorConfig(boards_per_bank=8, min_source_samples=4, **kw)


def test_mines_do_not_dilute_conditional_collapse():
    legal = np.ones((20, 5), dtype=bool)
    legal[::2, 2] = False
    p = np.zeros_like(legal, dtype=float)
    p[::2, 0] = 1
    p[1::2, 2] = 1
    result = summarize_sources(p, legal, cfg())
    assert result["collapsed"]
    assert result["max_conditional_source_mass"] == 1
    assert result["source_available_samples"][2] == 10
    assert result["source_probability_mean"][2] == .5


def test_uniform_sources_and_state_dependent_choices_do_not_collapse():
    legal = np.ones((20, 5), dtype=bool)
    uniform = summarize_sources(np.full((20, 5), .2), legal, cfg())
    assert not uniform["collapsed"]
    assert uniform["normalized_source_entropy"] == pytest.approx(1)
    p = np.eye(5)[np.arange(20) % 5]
    varied = summarize_sources(p, legal, cfg())
    assert varied["normalized_source_entropy"] == 0
    assert not varied["collapsed"]


def test_entropy_uses_the_same_mobility_condition_as_source_probability():
    legal = np.ones((20, 5), dtype=bool)
    legal[::2, 2] = False
    p = legal.astype(float) / legal.sum(1, keepdims=True)
    p[1::2] = [0, 0, 1, 0, 0]
    result = summarize_sources(p, legal, cfg())
    assert result["normalized_source_entropy"] == pytest.approx(.5)
    assert result["dominant_source_conditional_entropy"] == 0
    assert result["collapsed"]


def test_only_one_movable_piece_is_excluded():
    legal = np.ones((21, 5), dtype=bool)
    legal[-1] = [1, 0, 0, 0, 0]
    p = legal / legal.sum(1, keepdims=True)
    result = summarize_sources(p, legal, cfg())
    assert result["eligible_decisions"] == 20
    assert not result["collapsed"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1])
def test_invalid_probabilities_fail_closed(bad):
    p = np.full((20, 5), .2)
    p[0, 0] = bad
    with pytest.raises(ValueError):
        summarize_sources(p, np.ones_like(p, dtype=bool), cfg())


def test_probe_restores_all_rng_and_training_mode():
    model = torch.nn.Linear(2, 2).train()
    random.seed(11); np.random.seed(12); torch.manual_seed(13)
    py, npy, pt = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError), preserve_probe_state(model):
        assert not model.training
        random.random(); np.random.random(); torch.rand(3)
        raise RuntimeError("probe interrupted")
    assert model.training and random.getstate() == py
    assert np.random.get_state()[0] == npy[0]
    np.testing.assert_array_equal(np.random.get_state()[1], npy[1])
    assert torch.equal(torch.get_rng_state(), pt)


def test_persistent_collapse_streak_survives_resume(tmp_path):
    config = cfg(patience=3)
    model = torch.nn.Linear(2, 2)
    probe = OpeningMonitor(config, tmp_path)
    collapsed = dict(collapsed=True, dominant_source=2)
    probe._bank = lambda policy, seed: collapsed.copy()
    assert probe.check(model, completed_games=0, rollout=0)["status"] == "warning"
    assert not probe.due(999)
    assert probe.check(model, completed_games=1001, rollout=30)["status"] == "warning"
    saved = json.loads(json.dumps(probe.state_dict()))
    resumed = OpeningMonitor(config, tmp_path)
    resumed.load_state_dict(saved)
    resumed._bank = probe._bank
    assert resumed.check(model, completed_games=2000, rollout=60)["status"] == "collapse"
    assert len((tmp_path / "opening_progress.jsonl").read_text().splitlines()) == 3
    saved["config"]["source_mass_limit"] = .9
    with pytest.raises(ValueError, match="contract"):
        resumed.load_state_dict(saved)


def test_different_bank_dominant_sources_reset_streak(tmp_path):
    probe = OpeningMonitor(cfg(), tmp_path)
    probe.streak = 2
    rows = iter([dict(collapsed=True, dominant_source=2), dict(collapsed=True, dominant_source=3)])
    probe._bank = lambda policy, seed: next(rows)
    result = probe.check(torch.nn.Linear(2, 2), completed_games=2000, rollout=60)
    assert result["status"] == "healthy" and result["collapse_streak"] == 0


def test_fresh_config_resets_both_networks_and_exploration_covers_all_samples():
    config = load_config(Namespace(config="configs/v100_facts_scratch_opening.yaml", extra=[]))
    assert config.init_policy == config.resume == ""
    assert config.arr.enabled and not config.disable_arr_train
    assert config.ppo.adv_filt_rate == 1 and config.ppo.adv_filt_thresh == 0
    assert config.ppo.temperature_coef == config.ppo.temperature_floor == .1
    assert config.opening.enabled and config.target_completed_games == 10_000_000


def test_empty_or_unsupported_probe_cannot_report_healthy():
    with pytest.raises(ValueError, match="enough movable"):
        summarize_sources(np.eye(5), np.eye(5, dtype=bool), cfg())
    config = cfg()
    config.min_source_samples = 100
    with pytest.raises(ValueError):
        config.validate()


def test_short_diagnostic_counts_games_and_separates_distribution_from_matches():
    config = load_config(Namespace(config="configs/v100_opening_diagnostic.yaml", extra=[]))
    assert config.target_completed_games == 20000
    assert config.save_every_games == 5000 and config.opening.every_games == 1000
    assert not config.eval_enabled and config.opening.enabled
