"""Shared CPU/GPU evaluation against a uniform-random opponent."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import numpy as np
import torch

from junqi_core.rules import ShowMode
from junqi_rl.analysis.protocol import EvaluationCounts, merge_evaluations
from junqi_rl.input_contract import policy_input_contract

if TYPE_CHECKING:
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.networks.junqi_net import JunqiNet


_GPU_ROLLOUT_CACHE: dict[tuple[int, int, int, bool, str], GpuRollout] = {}


def _refresh_evaluation_belief(rollout, policy, belief_net, *, team, options):
    """Publish learned beliefs only for the evaluated policy's own observers."""
    from junqi_rl.belief.inference import refresh_beliefs_neural, guarded_refresh_beliefs_neural
    kwargs = dict(options or {})
    max_policy_kl = kwargs.pop('max_policy_kl', 0.0)
    min_entropy_ratio = kwargs.pop('min_entropy_ratio', .95)
    kwargs['observer_seats'] = tuple(seat for seat in range(4) if seat % 2 == team)
    if max_policy_kl:
        return guarded_refresh_beliefs_neural(rollout, belief_net, policy,
            max_policy_kl=max_policy_kl, min_entropy_ratio=min_entropy_ratio, **kwargs)
    return refresh_beliefs_neural(rollout, belief_net, **kwargs)


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


def _sample_uniform(legal, seeds, ply):
    # Counts are exact integers. CUDA floating cumsum is not supported by
    # torch's strict deterministic mode; no floating reduction is needed here.
    cumulative = legal.cumsum(dim=-1, dtype=torch.int32)
    threshold = _game_uniform(seeds, ply).unsqueeze(-1) * cumulative[:, -1:]
    return (cumulative > threshold).to(torch.int32).argmax(dim=-1).to(torch.int32)


def _policy_actions(policy, spatial, global_, legal, seeds, ply, greedy, stream=0):
    # Supplying actions prevents the forward path from sampling through the
    # process-global generator or replacing nonfinite logits with a fallback.
    # Read the distribution in greedy mode too: act_greedy returns an argmax
    # without exposing whether its logits were finite.
    placeholder = legal.long().argmax(dim=-1)
    output = policy(spatial, global_, legal, actions=placeholder)
    log_probs = output["log_probs"].float()
    if log_probs.shape != legal.shape:
        raise ValueError("Evaluation policy distribution must match the legal mask")
    weights = torch.where(legal, log_probs.exp(), 0.0)
    mass = weights.sum(dim=-1)
    valid = (legal.any(dim=-1)
             & (torch.isfinite(log_probs) | ~legal).all(dim=-1)
             & torch.isfinite(mass) & (mass > 0))
    # One scalar check per batch, including empty masks and zero/overflowed
    # mass. Illegal actions may normally carry -inf log-probabilities.
    if not bool(valid.all()):
        raise ValueError("Invalid evaluation policy distribution: every row requires "
                         "legal actions, finite legal log-probabilities, and positive "
                         "finite probability mass")
    if greedy:
        return log_probs.masked_fill(~legal, -torch.inf).argmax(dim=-1).to(torch.int32)
    return _sample_weights(weights, legal, seeds, ply, stream)


def _setup_hashes(rollout, batch_size):
    import hashlib
    host = rollout.state.copy_to_host()
    types = np.asarray(host["piece_type_arr"], dtype=np.int8).reshape(batch_size, 120)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in types]


@torch.no_grad()
def _evaluate_gpu_games(
    first_policy, second_policy, *, num_envs, num_games, device, seed,
    max_moves, autocast_dtype, first_team, greedy, prefix, game_records, setup_seed,
    belief_net=None, belief_refresh_steps=128, belief_options=None,
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
    if belief_refresh_steps <= 0:
        raise ValueError("belief_refresh_steps must be positive")
    dev = torch.device(device)
    first_policy.eval()
    if second_policy is not None:
        second_policy.eval()
    inputs = policy_input_contract(first_policy, *(() if second_policy is None else (second_policy,)))
    identity_encoding = inputs.identity_encoding
    if identity_encoding == "known_only" and belief_net is not None:
        raise ValueError("known_only does not publish legacy belief probabilities")
    include_global = inputs.include_global
    wins = losses = draws = completed_moves = total_work = 0
    for first_id in range(0, num_games, num_envs):
        batch_size = min(num_envs, num_games - first_id)
        device_id = dev.index or 0
        cache_key = (batch_size, device_id, max_moves, include_global, identity_encoding)
        if cache_key not in _GPU_ROLLOUT_CACHE:
            _GPU_ROLLOUT_CACHE[cache_key] = GpuRollout(
                num_envs=batch_size, show_mode=ShowMode.DARK,
                device_id=device_id, max_num_moves=max_moves,
                include_global=include_global, identity_encoding=identity_encoding,
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
            if belief_net is not None and ply % belief_refresh_steps == 0:
                _refresh_evaluation_belief(rollout, first_policy, belief_net,
                    team=first_team, options=belief_options)
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
                    second = _sample_uniform(legal, seeds, ply)
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
                    "first_policy_neural_belief": belief_net is not None,
                    "belief_refresh_steps": belief_refresh_steps if belief_net is not None else None,
                })
    completed = wins + losses + draws
    metrics = EvaluationCounts(wins, losses, draws, num_games - completed).as_metrics(prefix=prefix)
    metrics.update({
        f"{prefix}/neural_belief_enabled": float(belief_net is not None),
        f"{prefix}/belief_refresh_steps": float(belief_refresh_steps if belief_net is not None else 0),
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
    belief_net=None, belief_refresh_steps: int = 128, belief_options: dict | None = None,
) -> dict[str, float]:
    """Evaluate fixed game IDs; optional records include every requested game."""
    return _evaluate_gpu_games(
        policy, None, num_envs=num_envs, num_games=num_games, device=device,
        seed=seed, max_moves=max_moves, autocast_dtype=autocast_dtype,
        first_team=trained_team, greedy=greedy, prefix="eval", game_records=game_records, setup_seed=setup_seed,
        belief_net=belief_net, belief_refresh_steps=belief_refresh_steps, belief_options=belief_options,
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
    inputs = policy_input_contract(policy)
    env = VectorJunqiEnv(num_envs=batch_size, max_num_moves=max_moves, show_mode=ShowMode.DARK,
                        **inputs.observation_kwargs())
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
    belief_net=None, belief_refresh_steps: int = 128, belief_options: dict | None = None,
) -> dict[str, float]:
    """Evaluate both team assignments on paired seeds and merge exact counts."""

    if num_games <= 0 or num_envs <= 0 or max_moves <= 0:
        raise ValueError("num_games, num_envs and max_moves must be positive")
    if belief_net is not None and not use_gpu:
        raise ValueError("Learned-belief evaluation requires the GPU publication backend")
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
            if belief_net is not None:
                kwargs.update(belief_net=belief_net, belief_refresh_steps=belief_refresh_steps,
                              belief_options=belief_options)
        return evaluator(**kwargs)

    shards = [evaluate_team(0, team_zero_games)]
    if team_one_games:
        shards.append(evaluate_team(1, team_one_games))
    metrics = merge_evaluations(*shards)
    metrics['eval/neural_belief_enabled'] = float(belief_net is not None)
    metrics['eval/belief_refresh_steps'] = float(belief_refresh_steps if belief_net is not None else 0)
    return metrics


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
    belief_net=None, belief_refresh_steps: int = 128, belief_options: dict | None = None,
) -> dict[str, float]:
    """Evaluate fixed seeded games, from first_policy's point of view."""
    return _evaluate_gpu_games(
        first_policy, second_policy, num_envs=num_envs, num_games=num_games,
        device=device, seed=seed, max_moves=max_moves, autocast_dtype=autocast_dtype,
        first_team=first_team, greedy=greedy, prefix="h2h", game_records=game_records, setup_seed=setup_seed,
        belief_net=belief_net, belief_refresh_steps=belief_refresh_steps, belief_options=belief_options,
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
    belief_net=None, belief_refresh_steps: int = 128, belief_options: dict | None = None,
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
            belief_net=belief_net, belief_refresh_steps=belief_refresh_steps, belief_options=belief_options,
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
                belief_net=belief_net, belief_refresh_steps=belief_refresh_steps, belief_options=belief_options,
            )
        )
    metrics = merge_evaluations(*shards, prefix="h2h")
    metrics['h2h/neural_belief_enabled'] = float(belief_net is not None)
    metrics['h2h/belief_refresh_steps'] = float(belief_refresh_steps if belief_net is not None else 0)
    return metrics
