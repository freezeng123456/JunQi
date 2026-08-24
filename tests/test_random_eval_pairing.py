from __future__ import annotations

import pytest

# This module exercises the RL evaluator, whose implementation imports Torch.
# The core test profile intentionally installs no RL extras, so skip during
# collection instead of failing before pytest can apply its test selection.
pytest.importorskip("torch")

from junqi_rl.analysis import random_eval
from junqi_rl.analysis.protocol import EvaluationCounts


def test_paired_evaluation_splits_odd_budget_and_reuses_seed(monkeypatch) -> None:
    calls: list[tuple[int, int, int]] = []

    def fake_evaluator(**kwargs):
        calls.append(
            (
                kwargs["trained_team"],
                kwargs["num_games"],
                kwargs["seed"],
            )
        )
        metrics = EvaluationCounts(
            wins=kwargs["num_games"],
            losses=0,
            draws=0,
        ).as_metrics()
        metrics["eval/avg_game_len"] = 10.0
        metrics["eval/avg_game_len_all"] = 10.0
        return metrics

    monkeypatch.setattr(
        random_eval,
        "evaluate_vs_random_gpu",
        fake_evaluator,
    )

    metrics = random_eval.evaluate_paired_vs_random(
        object(),
        num_games=5,
        num_envs=64,
        use_gpu=True,
        device="cuda",
        seed=123,
        max_moves=4000,
    )

    assert calls == [(0, 3, 123), (1, 2, 123)]
    assert metrics["eval/requested_games"] == 5.0
    assert metrics["eval/win_rate"] == 1.0
