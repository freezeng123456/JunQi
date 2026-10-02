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
        from junqi_core.rules import ShowMode
        assert self.show_mode is ShowMode.DARK
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
        identity_encoding = "legacy_probabilities"
        uses_global_summary = True

        def eval(self):
            return self

        def act_greedy(self, sp, gl, mask):
            return mask.long().argmax(-1)

        def __call__(self, sp, gl, mask, *, actions):
            return {"log_probs": mask.float().log().log_softmax(dim=-1)}

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


class DistributionPolicy:
    def __init__(self, log_probs):
        self.log_probs = log_probs

    def __call__(self, spatial, global_, legal, *, actions):
        # The evaluator must request the distribution without sampling.
        assert actions.shape == legal.shape[:1]
        return {"log_probs": self.log_probs}

    def act_greedy(self, *args):
        raise AssertionError("Unchecked greedy inference bypassed the distribution")


@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("bad_log_prob", [float("nan"), float("inf"), -float("inf"),
                                           1000.0, -1000.0])
def test_evaluation_rejects_invalid_legal_distribution(greedy, bad_log_prob):
    import torch

    policy = DistributionPolicy(torch.tensor([[-torch.inf, bad_log_prob, -torch.inf]]))
    legal = torch.tensor([[False, True, False]])
    with pytest.raises(ValueError, match="Invalid evaluation policy distribution"):
        random_eval._policy_actions(policy, None, None, legal, torch.tensor([9]),
                                    ply=0, greedy=greedy)


@pytest.mark.parametrize("greedy", [False, True])
def test_evaluation_rejects_empty_legal_row_in_batch(greedy):
    import torch

    policy = DistributionPolicy(torch.zeros(2, 3))
    legal = torch.tensor([[False, True, False], [False, False, False]])
    with pytest.raises(ValueError, match="Invalid evaluation policy distribution"):
        random_eval._policy_actions(policy, None, None, legal, torch.tensor([9, 10]),
                                    ply=0, greedy=greedy)


@pytest.mark.parametrize("greedy", [False, True])
def test_evaluation_accepts_illegal_negative_infinity(greedy):
    import torch

    legal = torch.tensor([[False, True, False], [False, False, True]])
    policy = DistributionPolicy(torch.where(legal, 0.0, -torch.inf))
    actions = random_eval._policy_actions(policy, None, None, legal,
                                         torch.tensor([9, 10]), ply=3, greedy=greedy)
    assert actions.tolist() == [1, 2]
    assert actions.dtype == torch.int32


def test_evaluation_greedy_and_seeded_sampling_use_checked_distribution():
    import torch

    legal = torch.tensor([[False, True, True]]).expand(128, -1)
    log_probs = torch.tensor([[0.0, 0.25, 0.75]]).log().expand(128, -1)
    policy = DistributionPolicy(log_probs)
    seeds = torch.arange(128)
    greedy = random_eval._policy_actions(policy, None, None, legal, seeds, 7, True)
    assert torch.equal(greedy, torch.full((128,), 2, dtype=torch.int32))
    first = random_eval._policy_actions(policy, None, None, legal, seeds, 7, False)
    torch.rand(16)
    second = random_eval._policy_actions(policy, None, None, legal, seeds, 7, False)
    assert torch.equal(first, second)
    assert set(first.tolist()) == {1, 2}


@pytest.mark.parametrize("greedy", [False, True])
def test_actual_policy_nonfinite_logits_abort_without_fallback(monkeypatch, greedy):
    import torch
    from junqi_core.board import COMPACT_ACTION_DIM
    from junqi_core.observation import OBS_CHANNELS
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    policy = JunqiNet(JunqiNetConfig(
        cnn_channels=8, cnn_layers=1, depth=1, embed_dim=16, n_head=2,
        ff_factor=2, action_key_dim=8, use_global_summary=False,
        identity_encoding="known_only")).eval()
    legal = torch.ones(1, COMPACT_ACTION_DIM, dtype=torch.bool)
    monkeypatch.setattr(policy, "_policy_logits",
                        lambda *args: torch.full(legal.shape, torch.nan))
    with pytest.raises(ValueError, match="Invalid evaluation policy distribution"):
        random_eval._policy_actions(policy, torch.zeros(1, OBS_CHANNELS, 17, 17),
                                    torch.zeros(1, 0), legal, torch.tensor([9]), 0, greedy)
    assert policy._nan_fwd_count == 0
