"""Shared CPU/GPU evaluation against a uniform-random opponent."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

from junqi_rl.analysis.protocol import EvaluationCounts, merge_evaluations

if TYPE_CHECKING:
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet


_GPU_ROLLOUT_CACHE: dict[tuple[int, int], GpuRollout] = {}


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype | None,
):
    if device.type != "cuda" or dtype is None or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


@torch.no_grad()
def evaluate_vs_random_gpu(
    policy: JunqiNet,
    *,
    num_envs: int = 64,
    num_games: int = 128,
    device: str | torch.device = "cuda",
    seed: int = 0,
    max_moves: int = 4000,
    autocast_dtype: torch.dtype | None = None,
    trained_team: int = 0,
    greedy: bool = True,
) -> dict[str, float]:
    """Evaluate one policy team with bounded, reusable GPU environments."""

    from junqi_rl.gpu_rollout import GpuRollout

    if trained_team not in (0, 1):
        raise ValueError(f"trained_team must be 0 or 1, got {trained_team}")
    if num_games <= 0 or num_envs <= 0:
        raise ValueError("num_games and num_envs must be positive")

    dev = torch.device(device)
    policy.eval()
    batch_size = min(num_envs, num_games)
    device_id = dev.index or 0
    cache_key = (batch_size, device_id)
    if cache_key not in _GPU_ROLLOUT_CACHE:
        _GPU_ROLLOUT_CACHE[cache_key] = GpuRollout(
            num_envs=batch_size,
            device_id=device_id,
        )
    rollout = _GPU_ROLLOUT_CACHE[cache_key]
    rollout.reset(seed_base=seed)

    wins = losses = draws = total_games = total_steps = 0
    generator = torch.Generator(device=dev)
    generator.manual_seed(seed)

    while total_games < num_games:
        acting = rollout.turn_torch().clone()
        obs_spatial, obs_global = rollout.build_acting_seat_observation_torch(
            acting
        )
        legal_mask = rollout.legal_mask_canonical_torch_device(acting)

        with _autocast_context(dev, autocast_dtype):
            if greedy:
                actions = policy.act_greedy(
                    obs_spatial,
                    obs_global,
                    legal_mask,
                )
            else:
                act = getattr(policy, "_orig_act", policy.act)
                actions, _, _ = act(obs_spatial, obs_global, legal_mask)
        actions = actions.to(torch.int32)

        is_enemy = (acting.to(torch.int64) & 1) != trained_team
        if is_enemy.any():
            uniform = torch.where(legal_mask, 0.0, float("-inf"))
            uniform_noise = torch.rand(
                uniform.shape,
                dtype=torch.float32,
                device=uniform.device,
                generator=generator,
            ).clamp_(1e-10, 1.0)
            gumbel = -torch.log(-torch.log(uniform_noise))
            random_actions = (uniform + gumbel).argmax(dim=-1).to(torch.int32)
            actions = torch.where(is_enemy, random_actions, actions)

        result = rollout.step_device_torch(actions, acting)
        terminated = result["terminated"]
        total_steps += batch_size

        if terminated.any():
            term_np = terminated.cpu().numpy()
            winner_np = result["winner_team"].cpu().numpy()
            draw_np = result["draw"].cpu().numpy()
            for env_idx in range(batch_size):
                if not term_np[env_idx] or total_games >= num_games:
                    continue
                total_games += 1
                if draw_np[env_idx]:
                    draws += 1
                elif winner_np[env_idx] == trained_team:
                    wins += 1
                else:
                    losses += 1
            rollout.reset_terminated_device(seed=seed + total_games)

        if total_steps > num_games * max_moves:
            break

    completed = min(total_games, num_games)
    requested = max(1, num_games)
    completed_denom = max(1, completed)
    metrics = EvaluationCounts(
        wins=wins,
        losses=losses,
        draws=draws,
        ongoing=max(0, num_games - completed),
    ).as_metrics()
    metrics.update(
        {
            "eval/avg_game_len": total_steps / completed_denom,
            "eval/avg_game_len_all": total_steps / requested,
            "eval/trained_team": float(trained_team),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_vs_random_cpu(
    policy: JunqiNet,
    *,
    num_envs: int = 16,
    num_games: int = 32,
    device: str | torch.device = "cpu",
    seed: int = 0,
    max_moves: int = 4000,
    trained_team: int = 0,
    greedy: bool = True,
) -> dict[str, float]:
    """Evaluate one policy team with the authoritative CPU environment."""

    from junqi_core.rules import Seat
    from junqi_rl.action_lut import build_legal_mask_batch
    from junqi_rl.env import VectorJunqiEnv, unrotate_compact_action_id

    if trained_team not in (0, 1):
        raise ValueError(f"trained_team must be 0 or 1, got {trained_team}")
    if num_games <= 0 or num_envs <= 0:
        raise ValueError("num_games and num_envs must be positive")

    dev = torch.device(device)
    policy.eval()
    batch_size = min(num_envs, num_games)
    env = VectorJunqiEnv(num_envs=batch_size, max_num_moves=max_moves)
    obs_spatial, obs_global = env.reset(seed_base=seed)
    rng = np.random.default_rng(seed)
    game_moves = np.zeros(batch_size, dtype=np.int32)
    wins = losses = draws = total_games = total_moves = 0
    env_indices = np.arange(batch_size)

    while total_games < num_games:
        current_seats = env.current_seats()
        acting_idx = np.asarray(
            [seat.value for seat in current_seats],
            dtype=np.int64,
        )
        acting_spatial = obs_spatial[env_indices, acting_idx]
        acting_global = obs_global[env_indices, acting_idx]
        legal_mask = build_legal_mask_batch(env, current_seats)
        world_actions = np.zeros(batch_size, dtype=np.int32)

        for env_idx, seat in enumerate(current_seats):
            if env.done[env_idx]:
                continue
            legal_world = env.envs[env_idx].legal_action_ids(seat)
            if len(legal_world) == 0:
                continue
            if (seat.value & 1) == trained_team:
                spatial_t = torch.from_numpy(
                    acting_spatial[env_idx : env_idx + 1]
                ).to(dev)
                global_t = torch.from_numpy(
                    acting_global[env_idx : env_idx + 1]
                ).to(dev)
                mask_t = torch.from_numpy(
                    legal_mask[env_idx : env_idx + 1]
                ).to(dev)
                if greedy:
                    action = policy.act_greedy(spatial_t, global_t, mask_t)
                else:
                    action, _, _ = policy.act(spatial_t, global_t, mask_t)
                world_actions[env_idx] = unrotate_compact_action_id(
                    int(action[0].cpu()),
                    seat,
                )
            else:
                world_actions[env_idx] = int(rng.choice(legal_world))

        obs_spatial, obs_global, rewards, done, _ = env.step(world_actions)
        game_moves += 1

        for env_idx, finished in enumerate(done):
            if not finished:
                continue
            team_zero_reward = float(rewards[env_idx, Seat.SOUTH.value])
            trained_reward = (
                team_zero_reward
                if trained_team == 0
                else -team_zero_reward
            )
            if trained_reward > 0:
                wins += 1
            elif trained_reward < 0:
                losses += 1
            else:
                draws += 1
            total_games += 1
            total_moves += int(game_moves[env_idx])
            game_moves[env_idx] = 0
            if total_games >= num_games:
                break
            env.envs[env_idx].reset(seed=seed + total_games)
            env._done[env_idx] = False
            env._fill_all_obs()
            obs_spatial = env.obs_spatial
            obs_global = env.obs_global

    completed = max(1, total_games)
    metrics = EvaluationCounts(
        wins=wins,
        losses=losses,
        draws=draws,
    ).as_metrics()
    metrics.update(
        {
            "eval/avg_game_len": total_moves / completed,
            "eval/avg_game_len_all": total_moves / completed,
            "eval/trained_team": float(trained_team),
        }
    )
    return metrics


def evaluate_paired_vs_random(
    policy: JunqiNet,
    *,
    num_games: int,
    num_envs: int,
    use_gpu: bool,
    device: str | torch.device,
    seed: int,
    max_moves: int,
    autocast_dtype: torch.dtype | None = None,
    greedy: bool = True,
) -> dict[str, float]:
    """Evaluate both team assignments on paired seeds and merge exact counts."""

    if num_games <= 0:
        raise ValueError("num_games must be positive")
    team_zero_games = (num_games + 1) // 2
    team_one_games = num_games - team_zero_games
    evaluator = evaluate_vs_random_gpu if use_gpu else evaluate_vs_random_cpu

    def evaluate_team(team: int, games: int) -> dict[str, float]:
        kwargs = {
            "policy": policy,
            "num_envs": min(64, games) if use_gpu else min(num_envs, 16, games),
            "num_games": games,
            "device": device,
            "seed": seed,
            "max_moves": max_moves,
            "trained_team": team,
            "greedy": greedy,
        }
        if use_gpu:
            kwargs["autocast_dtype"] = autocast_dtype
        return evaluator(**kwargs)

    shards = [evaluate_team(0, team_zero_games)]
    if team_one_games:
        shards.append(evaluate_team(1, team_one_games))
    return merge_evaluations(*shards)


__all__ = [
    "evaluate_paired_vs_random",
    "evaluate_vs_random_cpu",
    "evaluate_vs_random_gpu",
]
