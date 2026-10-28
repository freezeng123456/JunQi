"""Returns along each observer's own decision sequence.

Only a subsequent decision by that same seat, or the final team result,
closes an action. Rollout boundaries and individual elimination do not.
"""
from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def same_seat_returns(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    seats: Tensor,
    valid: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    td_lambda: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute GAE and TD(lambda) without changing the observer.

    Inputs have shape ``(time, environments)``. Rewards are terminal-only,
    expressed from the terminating actor's team's perspective. ``dones``
    means the entire game ended, never merely that one seat was eliminated.
    Invalid rows permit a four-row prefix containing pending actions from
    the previous rollout, one row per seat. Such actions must retain their
    original observations, values and behavior probabilities in the caller.

    Returns ``(advantages, returns, ready, pending_indices)``. Pending indices
    have shape ``(environments, 4)`` and equal -1 if that seat has no pending
    action in the last unfinished game. An unresolved row has advantage zero,
    return equal to its stored value, and ready=False; its stored value still
    bootstraps an earlier action by the same seat. Train only ready rows and
    carry only pending rows, so every action is learned from at most once.

    Gamma and both lambdas count *own decisions*, not intervening opponent or
    teammate actions. A terminal result closes the final action with that
    result directly. The current experiment uses gamma=1. The recurrence is
    vectorized over environments and seats; only time is scanned in Python.
    """
    if values.ndim != 2 or not values.is_floating_point():
        raise ValueError("values must be a floating point (time, environments) tensor")
    for name, tensor in (("rewards", rewards), ("dones", dones),
                         ("seats", seats), ("valid", valid)):
        if tensor.shape != values.shape or tensor.device != values.device:
            raise ValueError(f"{name} must match values shape and device")
    if dones.dtype != torch.bool or valid.dtype != torch.bool:
        raise ValueError("dones and valid must be boolean tensors")
    if seats.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("seats must contain integer seat indices")
    if not all(0.0 <= x <= 1.0 for x in (gamma, gae_lambda, td_lambda)):
        raise ValueError("gamma and lambdas must lie in [0, 1]")
    # Validate once per rollout, never once per environment or scan step.
    if bool((valid & ((seats < 0) | (seats > 3))).any()):
        raise ValueError("valid seat indices must lie in [0, 3]")
    if bool((dones & ~valid).any()):
        raise ValueError("a terminal event must belong to a valid action")
    if bool((valid & ~dones & (rewards != 0)).any()):
        raise ValueError("same_seat_returns supports terminal-only rewards")

    time, num_envs = values.shape
    dev = values.device
    value = values.float()
    reward = rewards.float()
    advantages = torch.zeros_like(value)
    returns = value.clone()
    ready = torch.zeros_like(valid)
    pending = torch.full((num_envs, 4), -1, dtype=torch.int64, device=dev)
    next_value = torch.zeros((num_envs, 4), dtype=torch.float32, device=dev)
    next_gae = torch.zeros_like(next_value)
    next_td = torch.zeros_like(next_value)
    next_terminal = torch.zeros((num_envs, 4), dtype=torch.bool, device=dev)
    has_next = torch.zeros_like(next_terminal)
    # A terminal encountered while scanning backwards separates earlier
    # games from the final unfinished game whose tails may be carried.
    final_game = torch.ones(num_envs, dtype=torch.bool, device=dev)
    env_ids = torch.arange(num_envs, device=dev)
    team_ids = torch.arange(4, device=dev) & 1

    for t in range(time - 1, -1, -1):
        active = valid[t]
        seat = seats[t].long().clamp(0, 3)
        terminal = dones[t]
        team_sign = torch.where(
            team_ids.unsqueeze(0) == (seat & 1).unsqueeze(1), 1.0, -1.0,
        )
        outcome = reward[t].unsqueeze(1) * team_sign
        terminal_all = terminal.unsqueeze(1)
        # A terminal token closes every observer's last action, including
        # a player eliminated many turns earlier who never acts again.
        next_value = torch.where(terminal_all, outcome, next_value)
        next_gae = torch.where(terminal_all, 0.0, next_gae)
        next_td = torch.where(terminal_all, 0.0, next_td)
        next_terminal |= terminal_all
        has_next |= terminal_all
        final_game &= ~terminal

        closed = has_next[env_ids, seat]
        terminal_successor = next_terminal[env_ids, seat]
        discount = torch.where(terminal_successor, 1.0, gamma)
        delta = discount * next_value[env_ids, seat] - value[t]
        continuation = (~terminal_successor).float() * gamma
        gae = delta + continuation * gae_lambda * next_gae[env_ids, seat]
        td = delta + continuation * td_lambda * next_td[env_ids, seat]
        ready_t = active & closed
        ready[t] = ready_t
        advantages[t] = torch.where(ready_t, gae, 0.0)
        returns[t] = torch.where(ready_t, value[t] + td, value[t])

        unresolved = active & ~closed & final_game
        pending[env_ids, seat] = torch.where(
            unresolved, t, pending[env_ids, seat],
        )
        next_value[env_ids, seat] = torch.where(active, value[t], next_value[env_ids, seat])
        next_gae[env_ids, seat] = torch.where(active, advantages[t], next_gae[env_ids, seat])
        next_td[env_ids, seat] = torch.where(active, returns[t] - value[t], next_td[env_ids, seat])
        next_terminal[env_ids, seat] &= ~active
        has_next[env_ids, seat] |= active

    return advantages, returns, ready, pending
