"""Shared CPU/GPU evaluation against a uniform-random opponent."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

from junqi_core.rules import ShowMode
from junqi_rl.analysis.protocol import EvaluationCounts, merge_evaluations

if TYPE_CHECKING:
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet


_GPU_ROLLOUT_CACHE: dict[tuple[int, int, int], GpuRollout] = {}


def _autocast_context(
    device: torch.device,
    dtype: torch.dtype | None,
):
    if device.type != "cuda" or dtype is None or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def _game_uniform(seeds: torch.Tensor, ply: int, stream: int = 0) -> torch.Tensor:
    """Counter-based per-game random variates, independent of batch scheduling."""
    x = (seeds.to(torch.int64) + (ply + 1) * 0x9E3779B9 + stream * 0x85EBCA6B) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    x = ((x ^ (x >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return ((x.to(torch.float64) + 0.5) / 4294967296.0).float().clamp_max(1 - 2**-24)


def _sample_weights(weights, legal, seeds, ply, stream=0):
    weights = torch.where(legal, weights.float(), 0.0)
    cumulative = weights.cumsum(dim=-1)
    total = cumulative[:, -1:]
    threshold = _game_uniform(seeds, ply, stream).unsqueeze(-1) * total
    return (cumulative > threshold).to(torch.int32).argmax(dim=-1).to(torch.int32)


def _policy_actions(policy, spatial, global_, legal, seeds, ply, greedy, stream=0):
    if greedy:
        return policy.act_greedy(spatial, global_, legal).to(torch.int32)
    # Supplying actions prevents the forward path from sampling through the
    # process-global generator. Only its distribution is used here.
    placeholder = legal.long().argmax(dim=-1)
    output = policy(spatial, global_, legal, actions=placeholder)
    return _sample_weights(output["log_probs"].exp(), legal, seeds, ply, stream)


def _setup_hashes(rollout, batch_size):
    import hashlib
    host = rollout.state.copy_to_host()
    types = np.asarray(host["piece_type_arr"], dtype=np.int8).reshape(batch_size, 120)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in types]


@torch.no_grad()
def _evaluate_gpu_games(
    first_policy, second_policy, *, num_envs, num_games, device, seed,
    max_moves, autocast_dtype, first_team, greedy, prefix, game_records, setup_seed,
):
    """Run fixed seeded games in bounded waves; never replace an unfinished game.

    Every game uses the uniform CPU-seeded setup from reset(seed + game_id).
    The process-global training setup pool is not an evaluation input. Waiting
    for each wave trades some throughput for exact game budgets and pairing.
    """
    from junqi_rl.gpu_rollout import GpuRollout
    if first_team not in (0, 1):
        raise ValueError("team must be 0 or 1")
    if num_games <= 0 or num_envs <= 0 or max_moves <= 0:
        raise ValueError("num_games, num_envs and max_moves must be positive")
    dev = torch.device(device)
    first_policy.eval()
    if second_policy is not None:
        second_policy.eval()
    wins = losses = draws = completed_moves = total_work = 0
    for first_id in range(0, num_games, num_envs):
        batch_size = min(num_envs, num_games - first_id)
        device_id = dev.index or 0
        cache_key = (batch_size, device_id, max_moves)
        if cache_key not in _GPU_ROLLOUT_CACHE:
            _GPU_ROLLOUT_CACHE[cache_key] = GpuRollout(
                num_envs=batch_size, show_mode=ShowMode.DARK,
                device_id=device_id, max_num_moves=max_moves,
            )
        rollout = _GPU_ROLLOUT_CACHE[cache_key]
        setup_base = seed if setup_seed is None else setup_seed
        rollout.reset(seed_base=setup_base + first_id)
        hashes = _setup_hashes(rollout, batch_size)
        seeds = torch.arange(seed + first_id, seed + first_id + batch_size, device=dev)
        active = np.ones(batch_size, dtype=bool)
        game_moves = np.zeros(batch_size, dtype=np.int32)
        outcomes = ["ongoing"] * batch_size
        winner = np.full(batch_size, -1, dtype=np.int8)
        is_draw = np.zeros(batch_size, dtype=bool)
        for ply in range(max_moves):
            active_t = torch.as_tensor(active, device=dev)
            acting = torch.where(active_t, rollout.turn_torch(), 0).to(torch.int8)
            spatial, global_ = rollout.build_acting_seat_observation_torch(acting)
            legal = rollout.legal_mask_canonical_torch_device(acting).clone()
            # Finished lanes remain terminal until the whole wave is done.
            # Their dummy action is never executed, but makes policy input valid.
            legal[~active_t, 0] = True
            with _autocast_context(dev, autocast_dtype):
                first = _policy_actions(first_policy, spatial, global_, legal,
                                        seeds, ply, greedy, stream=1)
                if second_policy is None:
                    second = _sample_weights(legal.float(), legal, seeds, ply)
                else:
                    second = _policy_actions(second_policy, spatial, global_, legal,
                                             seeds, ply, greedy, stream=2)
            actions = torch.where((acting.long() & 1) == first_team, first, second)
            result = rollout.step_device_torch(actions, acting)
            rollout.update_beliefs_device(result, acting)
            game_moves += active
            total_work += int(active.sum())
            term = result["terminated"].cpu().numpy().astype(bool)
            newly_done = term & active
            if newly_done.any():
                winner = result["winner_team"].cpu().numpy()
                is_draw = result["draw"].cpu().numpy().astype(bool)
                for i in np.flatnonzero(newly_done):
                    if is_draw[i]:
                        draws += 1
                        outcomes[i] = "draw"
                    elif winner[i] == first_team:
                        wins += 1
                        outcomes[i] = "win"
                    else:
                        losses += 1
                        outcomes[i] = "loss"
                    completed_moves += int(game_moves[i])
                active[newly_done] = False
            if not active.any():
                break
        host = rollout.state.copy_to_host()
        since = np.asarray(host["moves_since_last_combat"]).reshape(batch_size)
        moves = np.asarray(host["move_counter"]).reshape(batch_size)
        if game_records is not None:
            for i in range(batch_size):
                reason = "team_victory"
                if active[i]:
                    reason = "evaluation_step_cap"
                elif outcomes[i] == "draw":
                    reason = ("total_move_limit" if moves[i] >= max_moves else
                              "no_combat_limit" if since[i] >= 200 else "draw")
                game_records.append({
                    "game_id": first_id + i, "setup_seed": setup_base + first_id + i,
                    "random_seed": seed + first_id + i, "setup_sha256": hashes[i],
                    "setup_source": "uniform_seeded_game", "first_team": first_team,
                    "opponent": "random" if second_policy is None else "policy",
                    "outcome": outcomes[i], "moves": int(game_moves[i]),
                    "termination_reason": reason, "backend": "gpu",
                })
    completed = wins + losses + draws
    metrics = EvaluationCounts(wins, losses, draws, num_games - completed).as_metrics(prefix=prefix)
    metrics.update({
        f"{prefix}/avg_game_len": completed_moves / max(1, completed),
        f"{prefix}/avg_game_len_all": total_work / num_games,
        f"{prefix}/environment_moves": float(total_work),
        f"{prefix}/num_envs": float(min(num_envs, num_games)),
        f"{prefix}/{'trained_team' if prefix == 'eval' else 'first_team'}": float(first_team),
    })
    return metrics


@torch.no_grad()
def evaluate_vs_random_gpu(
    policy: JunqiNet, *, num_envs: int = 64, num_games: int = 128,
    device: str | torch.device = "cuda", seed: int = 0, max_moves: int = 4000,
    autocast_dtype: torch.dtype | None = None, trained_team: int = 0,
    greedy: bool = True, game_records: list[dict] | None = None,
    setup_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate fixed game IDs; optional records include every requested game."""
    return _evaluate_gpu_games(
        policy, None, num_envs=num_envs, num_games=num_games, device=device,
        seed=seed, max_moves=max_moves, autocast_dtype=autocast_dtype,
        first_team=trained_team, greedy=greedy, prefix="eval", game_records=game_records, setup_seed=setup_seed,
    )


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
    game_records: list[dict] | None = None,
    setup_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate one policy team with the authoritative CPU environment."""

    from junqi_core.rules import Seat
    from junqi_rl.action_lut import build_legal_mask_batch
    from junqi_rl.env import VectorJunqiEnv, unrotate_compact_action_id

    if trained_team not in (0, 1):
        raise ValueError(f"trained_team must be 0 or 1, got {trained_team}")
    if num_games <= 0 or num_envs <= 0 or max_moves <= 0:
        raise ValueError("num_games, num_envs and max_moves must be positive")

    dev = torch.device(device)
    policy.eval()
    batch_size = min(num_envs, num_games)
    env = VectorJunqiEnv(num_envs=batch_size, max_num_moves=max_moves, show_mode=ShowMode.DARK)
    setup_base = seed if setup_seed is None else setup_seed
    obs_spatial, obs_global = env.reset(seed_base=setup_base)
    game_ids = np.arange(batch_size)
    import hashlib
    hashes = [hashlib.sha256(e.state.piece_type_arr.tobytes()).hexdigest() for e in env.envs]
    rngs = [np.random.default_rng(seed + i) for i in range(batch_size)]
    next_game_id = batch_size
    active = np.ones(batch_size, dtype=bool)
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
                action = _policy_actions(
                    policy, spatial_t, global_t, mask_t,
                    torch.tensor([seed + int(game_ids[env_idx])], device=dev),
                    int(game_moves[env_idx]), greedy, stream=1,
                )
                world_actions[env_idx] = unrotate_compact_action_id(
                    int(action[0].cpu()),
                    seat,
                )
            else:
                world_actions[env_idx] = int(rngs[env_idx].choice(legal_world))

        obs_spatial, obs_global, rewards, done, _ = env.step(world_actions)
        game_moves += active

        for env_idx, finished in enumerate(done):
            if not finished or not active[env_idx]:
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
            if game_records is not None:
                state = env.envs[env_idx].state
                reason = "team_victory"
                if state.draw:
                    reason = ("total_move_limit" if state.move_counter >= max_moves
                              else "no_combat_limit" if state.moves_since_last_combat >= 200
                              else "draw")
                gid = int(game_ids[env_idx])
                game_records.append({
                    "game_id": gid, "setup_seed": setup_base + gid,
                    "random_seed": seed + gid, "setup_sha256": hashes[env_idx],
                    "setup_source": "uniform_seeded_game", "first_team": trained_team,
                    "opponent": "random", "outcome": "win" if trained_reward > 0
                    else "loss" if trained_reward < 0 else "draw",
                    "moves": int(game_moves[env_idx]), "termination_reason": reason,
                    "backend": "cpu",
                })
            game_moves[env_idx] = 0
            if next_game_id >= num_games:
                active[env_idx] = False
                continue
            env.envs[env_idx].reset(seed=setup_base + next_game_id)
            game_ids[env_idx] = next_game_id
            hashes[env_idx] = hashlib.sha256(env.envs[env_idx].state.piece_type_arr.tobytes()).hexdigest()
            rngs[env_idx] = np.random.default_rng(seed + next_game_id)
            next_game_id += 1
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
    game_records: list[dict] | None = None,
    setup_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate both team assignments on paired seeds and merge exact counts."""

    if num_games <= 0 or num_envs <= 0 or max_moves <= 0:
        raise ValueError("num_games, num_envs and max_moves must be positive")
    team_zero_games = (num_games + 1) // 2
    team_one_games = num_games - team_zero_games
    evaluator = evaluate_vs_random_gpu if use_gpu else evaluate_vs_random_cpu

    def evaluate_team(team: int, games: int) -> dict[str, float]:
        kwargs = {
            "policy": policy,
            "num_envs": min(num_envs, games) if use_gpu else min(num_envs, 16, games),
            "num_games": games,
            "device": device,
            "seed": seed,
            "max_moves": max_moves,
            "trained_team": team,
            "greedy": greedy,
            "game_records": game_records,
            "setup_seed": setup_seed,
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

@torch.no_grad()
def evaluate_head_to_head_gpu(
    first_policy: JunqiNet, second_policy: JunqiNet, *,
    num_envs: int = 64, num_games: int = 128, device: str | torch.device = "cuda",
    seed: int = 0, max_moves: int = 4000, autocast_dtype: torch.dtype | None = None,
    first_team: int = 0, greedy: bool = True, game_records: list[dict] | None = None,
    setup_seed: int | None = None,
) -> dict[str, float]:
    """Evaluate fixed seeded games, from first_policy's point of view."""
    return _evaluate_gpu_games(
        first_policy, second_policy, num_envs=num_envs, num_games=num_games,
        device=device, seed=seed, max_moves=max_moves, autocast_dtype=autocast_dtype,
        first_team=first_team, greedy=greedy, prefix="h2h", game_records=game_records, setup_seed=setup_seed,
    )


def evaluate_paired_head_to_head(
    first_policy: "JunqiNet",
    second_policy: "JunqiNet",
    *,
    num_games: int,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    max_moves: int,
    autocast_dtype: torch.dtype | None = None,
    greedy: bool = True,
    game_records: list[dict] | None = None,
    setup_seed: int | None = None,
) -> dict[str, float]:
    """Both team assignments on paired seeds; metrics from first_policy."""

    if num_games <= 0 or num_envs <= 0 or max_moves <= 0:
        raise ValueError("num_games, num_envs and max_moves must be positive")
    team_zero_games = (num_games + 1) // 2
    team_one_games = num_games - team_zero_games
    shards = [
        evaluate_head_to_head_gpu(
            first_policy,
            second_policy,
            num_envs=num_envs,
            num_games=team_zero_games,
            device=device,
            seed=seed,
            max_moves=max_moves,
            autocast_dtype=autocast_dtype,
            first_team=0,
            greedy=greedy, game_records=game_records, setup_seed=setup_seed,
        )
    ]
    if team_one_games > 0:
        shards.append(
            evaluate_head_to_head_gpu(
                first_policy,
                second_policy,
                num_envs=num_envs,
                num_games=team_one_games,
                device=device,
                seed=seed,
                max_moves=max_moves,
                autocast_dtype=autocast_dtype,
                first_team=1,
                greedy=greedy, game_records=game_records, setup_seed=setup_seed,
            )
        )
    return merge_evaluations(*shards, prefix="h2h")
