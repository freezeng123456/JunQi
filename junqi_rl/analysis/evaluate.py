"""Policy evaluation entry points for JunQi RL."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from junqi_rl.input_contract import policy_input_contract
from junqi_rl.analysis.random_eval import (
    _policy_actions,
    evaluate_paired_vs_random,
    evaluate_vs_random_cpu,
    evaluate_vs_random_gpu,
)

if TYPE_CHECKING:
    from junqi_rl.networks.junqi_net import JunqiNet


@torch.no_grad()
def eval_vs_random(
    policy: JunqiNet,
    *,
    num_games: int = 128,
    trained_team: int = 0,           # 0 = RED (SOUTH+NORTH), 1 = BLUE (WEST+EAST)
    max_steps: int = 1000,
    device: str | torch.device = "cuda",
    seed_base: int = 0,
    greedy: bool = False,
) -> dict:
    """Compatibility wrapper around the shared bounded-batch evaluator."""

    evaluator = evaluate_vs_random_gpu if torch.device(device).type == "cuda" else evaluate_vs_random_cpu
    metrics = evaluator(
        policy,
        num_envs=min(64, num_games),
        num_games=num_games,
        trained_team=trained_team,
        max_moves=max_steps,
        device=device,
        seed=seed_base,
        greedy=greedy,
    )
    return {
        "num_games": int(metrics["eval/requested_games"]),
        "trained_team": int(trained_team),
        "trained_win_rate": metrics["eval/win_rate"],
        "trained_loss_rate": metrics["eval/loss_rate"],
        "draw_rate": metrics["eval/draw_rate"],
        "ongoing_rate": metrics["eval/ongoing_rate"],
        "mean_length": metrics["eval/avg_game_len"],
    }


@torch.no_grad()
def eval_head_to_head(
    first_policy: JunqiNet,
    second_policy: JunqiNet,
    *,
    num_games: int = 16,
    first_team: int = 0,
    max_steps: int = 4000,
    device: str | torch.device = "cuda",
    seed_base: int = 0,
    greedy: bool = True,
) -> dict[str, float]:
    """Evaluate two policies on the CPU rules engine with paired model inference.

    ``first_team`` controls which fixed team the first policy receives.  Call
    twice with the same seed and opposite teams, then merge the counts, to
    remove seat/setup bias.
    """

    from junqi_rl.analysis.protocol import EvaluationCounts
    from junqi_rl.env import VectorJunqiEnv, unrotate_compact_action_id
    from junqi_rl.training.collector import _build_legal_mask

    if first_team not in (0, 1):
        raise ValueError(f"first_team must be 0 or 1, got {first_team}")
    if num_games <= 0 or max_steps <= 0:
        raise ValueError("num_games and max_steps must be positive")

    inputs = policy_input_contract(first_policy, second_policy)
    dev = torch.device(device)
    first_policy.eval()
    second_policy.eval()
    n = min(16, num_games)
    from junqi_core.rules import ShowMode
    env = VectorJunqiEnv(num_envs=n, max_num_moves=max_steps, show_mode=ShowMode.DARK,
                        **inputs.observation_kwargs())
    obs_sp, obs_gl = env.reset(seed_base=seed_base)
    next_game_id = n
    active = np.ones(n, dtype=bool)
    game_moves = np.zeros(n, dtype=np.int32)
    game_seeds = np.arange(seed_base, seed_base + n, dtype=np.int64)
    wins = losses = draws = total_games = total_moves = 0

    while total_games < num_games:
        current_seats = env.current_seats()
        acting_idx = np.asarray([seat.value for seat in current_seats], dtype=np.int64)
        acting_sp = obs_sp[np.arange(n), acting_idx]
        acting_gl = obs_gl[np.arange(n), acting_idx]
        legal_mask = _build_legal_mask(env, current_seats)
        world_actions = np.zeros(n, dtype=np.int32)

        for i, seat in enumerate(current_seats):
            if env.done[i]:
                continue
            legal_world = env.envs[i].legal_action_ids(seat)
            if legal_world.size == 0:
                continue
            model = first_policy if (seat.value & 1) == first_team else second_policy
            sp_t = torch.from_numpy(acting_sp[i : i + 1]).to(dev)
            gl_t = torch.from_numpy(acting_gl[i : i + 1]).to(dev)
            mask_t = torch.from_numpy(legal_mask[i : i + 1]).to(dev)
            action = _policy_actions(
                model, sp_t, gl_t, mask_t,
                torch.as_tensor(game_seeds[i : i + 1], device=dev),
                int(game_moves[i]), greedy,
            )
            world_actions[i] = unrotate_compact_action_id(
                int(action[0].cpu()), seat
            )

        obs_sp, obs_gl, rewards, done, _infos = env.step(world_actions)
        game_moves += active
        for i, finished in enumerate(done):
            if not finished or not active[i]:
                continue
            team_zero_reward = float(rewards[i, 0])
            first_reward = (
                team_zero_reward if first_team == 0 else -team_zero_reward
            )
            if first_reward > 0:
                wins += 1
            elif first_reward < 0:
                losses += 1
            else:
                draws += 1
            total_games += 1
            total_moves += int(game_moves[i])
            game_moves[i] = 0
            if next_game_id >= num_games:
                active[i] = False
                continue
            env.envs[i].reset(seed=seed_base + next_game_id)
            game_seeds[i] = seed_base + next_game_id
            next_game_id += 1
            env._done[i] = False
            env._fill_all_obs()
            obs_sp = env.obs_spatial
            obs_gl = env.obs_global

    counts = EvaluationCounts(wins=wins, losses=losses, draws=draws)
    metrics = counts.as_metrics(prefix="league")
    denom = max(1, total_games)
    metrics["league/avg_game_len"] = total_moves / denom
    metrics["league/avg_game_len_all"] = total_moves / denom
    metrics["league/first_team"] = float(first_team)
    return metrics


__all__ = [
    "eval_head_to_head",
    "eval_vs_random",
    "evaluate_paired_vs_random",
    "evaluate_vs_random_cpu",
    "evaluate_vs_random_gpu",
]
