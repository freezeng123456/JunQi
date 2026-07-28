"""junqi_rl.analysis.evaluate — Match trained policy vs a random opponent.

Self-play in 四国军棋 is per-seat.  Teams:
  * RED  (team 0) = SOUTH + NORTH
  * BLUE (team 1) = WEST  + EAST

``eval_vs_random`` plays N parallel games where one team uses the trained
policy and the other team plays uniformly-random legal actions.  Returns
per-team win/loss/draw stats.

This is the T-06 acceptance gate: **trained team beats random >= 55%**.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.training.gpu_collector import (
    _UNROTATE_LUT_STACK,
    build_legal_mask_batch_gpu_torch,
)

if TYPE_CHECKING:
    from junqi_rl.networks.junqi_net import JunqiNet


# Seat ↔ team (SOUTH=0, WEST=1, NORTH=2, EAST=3)
_SEAT_TEAM = np.array([0, 1, 0, 1], dtype=np.int8)


def _sample_random_from_mask(
    mask_t: torch.Tensor,  # (N, FLAT) bool on CUDA
) -> torch.Tensor:
    """Uniformly sample one True index per row.  Rows with no True return 0."""
    # Convert to float, sample via Categorical; rows with all-zero get a
    # uniform fallback that we'll mask out at the action-apply stage.
    probs = mask_t.to(torch.float32)
    any_legal = probs.any(dim=-1)
    # Avoid 0/0 — give dead rows a uniform distribution (their actions are
    # filtered anyway by done_flags in the caller).
    probs = torch.where(any_legal[:, None], probs, torch.ones_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True)
    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample()  # (N,) int64


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
    """Play ``num_games`` parallel games; return per-team win/loss/draw rates.

    Both the trained policy and the random opponent run on GPU.  The
    games advance in lock-step: each env has exactly one acting seat
    per step (from the engine's turn field), and we dispatch to either
    the trained policy or random sampling based on that seat's team.

    Returns
    -------
    dict with keys::
        trained_win_rate, trained_loss_rate, draw_rate, ongoing_rate,
        mean_length, num_games

    Note that ``trained_win_rate + trained_loss_rate + draw_rate +
    ongoing_rate == 1.0``.
    """
    if trained_team not in (0, 1):
        raise ValueError(f"trained_team must be 0 or 1, got {trained_team}")

    dev = torch.device(device)
    policy.eval()

    world = GpuRollout(num_envs=num_games)
    world.reset(seed_base=seed_base)

    # Which seats belong to each side
    trained_seats_mask = (trained_team == _SEAT_TEAM)   # (4,) bool

    # Per-env termination + steps-played tracker (host numpy for simple state)
    done_flags = world.read_termination()["terminated"].copy()
    result_winner = np.full(num_games, -1, dtype=np.int8)
    result_draw   = np.zeros(num_games, dtype=bool)
    steps_played  = np.zeros(num_games, dtype=np.int32)

    for _step in range(max_steps):
        if done_flags.all():
            break
        alive_mask = ~done_flags

        # ---- Who's acting per env? ----
        turns = world.state.copy_turn_to_host()
        turns = np.asarray(turns, dtype=np.int8).reshape(num_games)
        acting_seats = np.where(alive_mask, turns, np.int8(0)).astype(np.int8)

        trained_env = alive_mask & trained_seats_mask[acting_seats]
        # ---- Build legal mask on device (both sides need it) ----
        lm_t = build_legal_mask_batch_gpu_torch(
            world, acting_seats, done_flags, dev,
        )

        # ---- Sample actions from both sides, compose per-env ----
        # Trained side
        sp_full, gl_full = world.build_all_seat_observations_torch()
        env_idx = torch.arange(num_games, device=dev)
        acting_t = torch.from_numpy(acting_seats).to(dev, dtype=torch.long)
        sp_t = sp_full[env_idx, acting_t].contiguous()
        gl_t = gl_full[env_idx, acting_t].contiguous()

        if greedy:
            from junqi_rl.networks.junqi_net import JunqiNet  # noqa: F401
            actions_trained = policy.act_greedy(sp_t, gl_t, lm_t).to(torch.int64)
        else:
            actions_trained, _, _ = policy.act(sp_t, gl_t, lm_t)
            actions_trained = actions_trained.to(torch.int64)

        # Random side — sample uniformly from legal mask
        actions_random = _sample_random_from_mask(lm_t)

        # Compose: trained_env rows take actions_trained, others take actions_random
        trained_env_t = torch.from_numpy(trained_env).to(dev)
        actions_can = torch.where(trained_env_t, actions_trained, actions_random)

        # ---- Canonical compact → world-full (numpy) ----
        # Since commit 1826873 ROTATE_LUT / UNROTATE_LUT live in the compact
        # 129×129 frame.  The ``world.step`` API still expects world-full ids
        # (src*289 + dst), so we:
        #   1. Unrotate compact canonical → compact world (via UNROTATE_LUT).
        #   2. Decompose into (src_compact, dst_compact).
        #   3. Map each through COMPACT_TO_FLAT → (src_full, dst_full).
        #   4. Recompose: world_full = src_full * 289 + dst_full.
        from junqi_core.board import COMPACT_TO_FLAT, NUM_ON_BOARD_CELLS
        actions_can_np = actions_can.detach().cpu().numpy().astype(np.int64)
        compact_world = _UNROTATE_LUT_STACK[
            acting_seats.astype(np.int64), actions_can_np
        ].astype(np.int64)
        src_c = compact_world // NUM_ON_BOARD_CELLS
        dst_c = compact_world %  NUM_ON_BOARD_CELLS
        c2f = np.asarray(COMPACT_TO_FLAT, dtype=np.int64)
        actions_world = (c2f[src_c] * 289 + c2f[dst_c]).astype(np.int32)
        # Terminated envs: zero action
        actions_world = np.where(alive_mask, actions_world, np.int32(0))

        # ---- Step ----
        result = world.step(actions_world)
        new_term = result["terminated"].astype(bool, copy=False)
        new_win = result["winner_team"].astype(np.int8, copy=False)
        new_draw = result["draw"].astype(bool, copy=False)

        fired = new_term & ~done_flags
        for i in np.where(fired)[0]:
            result_winner[i] = new_win[i]
            result_draw[i]   = new_draw[i]
        steps_played[alive_mask] += 1
        done_flags = new_term.copy()

    # Tally
    finished = done_flags
    ongoing = (~finished).sum()
    wins = ((result_winner == trained_team) & finished & ~result_draw).sum()
    draws = (result_draw & finished).sum()
    losses = (finished & ~result_draw & (result_winner != trained_team) &
              (result_winner >= 0)).sum()
    mean_len = float(steps_played[finished].mean()) if finished.any() else float("nan")

    return {
        "num_games":         int(num_games),
        "trained_team":      int(trained_team),
        "trained_win_rate":  float(wins)   / num_games,
        "trained_loss_rate": float(losses) / num_games,
        "draw_rate":         float(draws)  / num_games,
        "ongoing_rate":      float(ongoing)/ num_games,
        "mean_length":       mean_len,
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
    if num_games <= 0:
        raise ValueError("num_games must be positive")

    dev = torch.device(device)
    first_policy.eval()
    second_policy.eval()
    n = min(16, num_games)
    env = VectorJunqiEnv(num_envs=n, max_num_moves=max_steps)
    obs_sp, obs_gl = env.reset(seed_base=seed_base)
    game_moves = np.zeros(n, dtype=np.int32)
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
            if greedy:
                action = model.act_greedy(sp_t, gl_t, mask_t)
            else:
                action, _, _ = model.act(sp_t, gl_t, mask_t)
            world_actions[i] = unrotate_compact_action_id(
                int(action[0].cpu()), seat
            )

        obs_sp, obs_gl, rewards, done, _infos = env.step(world_actions)
        game_moves += 1
        for i, finished in enumerate(done):
            if not finished:
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
            if total_games >= num_games:
                break
            env.envs[i].reset(seed=seed_base + total_games)
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


__all__ = ["eval_head_to_head", "eval_vs_random"]
