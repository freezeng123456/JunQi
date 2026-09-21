# ruff: noqa: E402
"""Run the production GPU evaluator with deterministic asynchronous games."""
from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from junqi_rl import gpu_rollout
from junqi_rl.analysis import random_eval


class Policy:
    def eval(self):
        return self

    def act_greedy(self, spatial, global_, legal):
        return legal.long().argmax(-1)


class FixedGames:
    started: ClassVar[list] = []
    finished: ClassVar[list] = []
    actions: ClassVar[dict] = defaultdict(list)
    step_calls = 0
    belief_updates = 0

    def __init__(self, num_envs, **kwargs):
        self.num_envs = num_envs
        self.state = self
        self.next_id = 0

    def reset(self, seed_base):
        self.ids = np.arange(seed_base, seed_base + self.num_envs)
        self.next_id = int(self.ids[-1]) + 1
        self.ages = np.zeros(self.num_envs, dtype=np.int64)
        self.done = np.zeros(self.num_envs, dtype=bool)
        type(self).started.extend(self.ids.tolist())

    def turn_torch(self):
        return torch.ones(self.num_envs, dtype=torch.int8)

    def build_acting_seat_observation_torch(self, acting):
        return torch.zeros(self.num_envs, 1, 1, 1), torch.zeros(self.num_envs, 1)

    def legal_mask_canonical_torch_device(self, acting):
        return torch.ones(self.num_envs, 17, dtype=torch.bool)

    def step_device_torch(self, actions, acting):
        type(self).step_calls += 1
        for i in range(self.num_envs):
            if not self.done[i]:
                self.ages[i] += 1
                type(self).actions[int(self.ids[i])].append(int(actions[i]))
                duration = 100 if self.ids[i] == 1 else 1
                if self.ages[i] >= duration:
                    self.done[i] = True
                    type(self).finished.append(int(self.ids[i]))
        return {
            "terminated": torch.from_numpy(self.done.copy()),
            "winner_team": torch.from_numpy((self.ids == 1).astype(np.int8)),
            "draw": torch.zeros(self.num_envs, dtype=torch.bool),
        }

    def update_beliefs_device(self, result, acting):
        type(self).belief_updates += 1

    def reset_terminated_device(self, seed):
        # The old evaluator uses this to keep restarting the short lane.
        for i in range(self.num_envs):
            if self.done[i]:
                self.ids[i] = self.next_id
                self.next_id += 1
                self.ages[i] = 0
                self.done[i] = False
                type(self).started.append(int(self.ids[i]))

    def copy_to_host(self):
        return {
            "piece_type_arr": np.repeat(self.ids[:, None], 120, axis=1).astype(np.int8),
            "move_counter": self.ages.copy(),
            "moves_since_last_combat": self.ages.copy(),
        }


@pytest.fixture(autouse=True)
def fake_gpu(monkeypatch):
    FixedGames.step_calls = FixedGames.belief_updates = 0
    FixedGames.started = []
    FixedGames.finished = []
    FixedGames.actions = defaultdict(list)
    random_eval._GPU_ROLLOUT_CACHE.clear()
    monkeypatch.setattr(gpu_rollout, "GpuRollout", FixedGames)
    yield
    random_eval._GPU_ROLLOUT_CACHE.clear()


@pytest.mark.parametrize("head_to_head", [False, True])
def test_long_game_is_counted_once_and_no_extra_games_start(head_to_head):
    fn = (random_eval.evaluate_head_to_head_gpu if head_to_head
          else random_eval.evaluate_vs_random_gpu)
    args = (Policy(), Policy()) if head_to_head else (Policy(),)
    metrics = fn(*args, num_envs=2, num_games=4, device="cpu", seed=0, max_moves=120)
    prefix = "h2h" if head_to_head else "eval"
    assert metrics[f"{prefix}/wins"] == 3
    assert metrics[f"{prefix}/losses"] == 1
    assert metrics[f"{prefix}/ongoing"] == 0
    assert metrics[f"{prefix}/avg_game_len"] == pytest.approx(103 / 4)
    assert sorted(FixedGames.started) == [0, 1, 2, 3]
    assert sorted(FixedGames.finished) == [0, 1, 2, 3]


def test_timeout_records_pending_game_instead_of_replacing_it():
    records = []
    metrics = random_eval.evaluate_vs_random_gpu(
        Policy(), num_envs=2, num_games=4, device="cpu", seed=0, max_moves=3,
        game_records=records,
    )
    assert metrics["eval/wins"] == 3
    assert metrics["eval/ongoing"] == 1
    assert metrics["eval/avg_game_len"] == 1
    assert metrics["eval/avg_game_len_all"] == 1.5
    pending = [r for r in records if r["outcome"] == "ongoing"]
    assert len(pending) == 1 and pending[0]["game_id"] == 1
    assert pending[0]["termination_reason"] == "evaluation_step_cap"
    assert len({r["setup_sha256"] for r in records}) == 4


def test_random_stream_for_each_game_does_not_depend_on_concurrency():
    trajectories = []
    for concurrency in (1, 2, 3):
        FixedGames.actions = defaultdict(list)
        random_eval.evaluate_vs_random_gpu(
            Policy(), num_envs=concurrency, num_games=4, device="cpu", seed=0, max_moves=120,
        )
        trajectories.append(dict(FixedGames.actions))
    assert trajectories[0] == trajectories[1] == trajectories[2]


def test_gpu_paired_wrapper_respects_requested_concurrency(monkeypatch):
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs["num_envs"])
        from junqi_rl.analysis.protocol import EvaluationCounts
        return EvaluationCounts(wins=kwargs["num_games"], losses=0, draws=0).as_metrics()
    monkeypatch.setattr(random_eval, "evaluate_vs_random_gpu", evaluate)
    random_eval.evaluate_paired_vs_random(
        Policy(), num_games=64, num_envs=2, use_gpu=True, device="cpu", seed=0, max_moves=120,
    )
    assert calls == [2, 2]


def test_evaluation_updates_rule_beliefs_after_every_device_step():
    random_eval.evaluate_vs_random_gpu(
        Policy(), num_envs=2, num_games=4, device='cpu', seed=0, max_moves=120)
    assert FixedGames.step_calls > 1
    assert FixedGames.belief_updates == FixedGames.step_calls
