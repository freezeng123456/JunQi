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

@pytest.mark.parametrize("head_to_head", [False, True])
@pytest.mark.parametrize("team", [0, 1])
def test_cpu_unique_game_seeds(monkeypatch, head_to_head, team):
    import torch
    from junqi_rl.analysis.evaluate import eval_head_to_head
    from junqi_rl.env import JunqiEnv

    started, finished = [], []
    original_reset = JunqiEnv.reset
    original_step = JunqiEnv._step_game_only

    def reset(self, *args, **kwargs):
        self._test_seed = kwargs.get("seed")
        started.append(self._test_seed)
        self.max_num_moves = 1 + self._test_seed % 3
        return original_reset(self, *args, **kwargs)

    def step(self, *args, **kwargs):
        was_done = self.state.terminated
        result = original_step(self, *args, **kwargs)
        if result[1] and not was_done:
            finished.append(self._test_seed)
        return result

    class Policy:
        def eval(self):
            return self

        def act_greedy(self, sp, gl, mask):
            return mask.long().argmax(-1)

    monkeypatch.setattr(JunqiEnv, "reset", reset)
    monkeypatch.setattr(JunqiEnv, "_step_game_only", step)
    torch.set_num_threads(1)
    if head_to_head:
        num_games = 19
        metrics = eval_head_to_head(
            Policy(),
            Policy(),
            num_games=num_games,
            max_steps=3,
            seed_base=100,
            device="cpu",
            first_team=team,
        )
        count = metrics["league/num_games"]
    else:
        num_games = 7
        metrics = random_eval.evaluate_vs_random_cpu(
            Policy(), num_envs=2, num_games=num_games, max_moves=3, seed=100, trained_team=team
        )
        count = metrics["eval/num_games"]
    assert sorted(started) == list(range(100, 100 + num_games))
    assert sorted(finished) == sorted(started)
    assert count == num_games
