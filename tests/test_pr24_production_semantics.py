from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from junqi_rl.analysis.protocol import (
    EvaluationCounts,
    completed_evaluation_score,
)
from junqi_rl.training.config import load_config
from junqi_rl.training.rollout_gpu import _rollout_advantage_keep_mask

ROOT = Path(__file__).resolve().parents[1]


def _args(config: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config),
        extra=[],
        resume_cli="",
        validate_only=False,
    )


def test_current_h20_profile_really_keeps_every_valid_transition() -> None:
    cfg = load_config(_args(ROOT / "configs" / "h20_10m_current.yaml"))

    assert cfg.ppo.adv_filt_rate == pytest.approx(1.0)
    assert cfg.ppo.adv_filt_thresh == pytest.approx(0.0)

    advantages = torch.tensor(
        [[0.0, 1.0e-6, -5.0e-3, 2.0e-2]],
        dtype=torch.float32,
    )
    valid = torch.ones_like(advantages, dtype=torch.bool)
    keep, threshold = _rollout_advantage_keep_mask(
        advantages,
        valid,
        keep_rate=cfg.ppo.adv_filt_rate,
        min_thresh=cfg.ppo.adv_filt_thresh,
    )

    assert threshold == pytest.approx(0.0)
    assert torch.equal(keep, valid)


def test_evaluation_game_seed_is_stable_config_state() -> None:
    cfg = load_config(_args(ROOT / "configs" / "h20_10m_current.yaml"))
    assert isinstance(cfg.eval_game_seed, int)
    assert cfg.eval_game_seed >= 0


def test_completed_h2h_score_counts_draws_as_half() -> None:
    metrics = EvaluationCounts(
        wins=2,
        losses=1,
        draws=1,
    ).as_metrics(prefix="h2h")

    assert completed_evaluation_score(metrics, prefix="h2h") == pytest.approx(0.625)


def test_incomplete_h2h_cannot_select_a_best_checkpoint() -> None:
    metrics = EvaluationCounts(
        wins=2,
        losses=1,
        draws=0,
        ongoing=1,
    ).as_metrics(prefix="h2h")

    assert completed_evaluation_score(metrics, prefix="h2h") is None
