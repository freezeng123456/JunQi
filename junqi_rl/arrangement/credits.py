"""Batch terminal layout rewards using the reset pool recorded at game start."""
from __future__ import annotations

import numpy as np
import torch

from junqi_rl.networks.arrangement_net import PIECE_TYPE_VALUE_TO_VOCAB_IDX


def credit_completed_games(buffer, setups, records, *, random_opponent=False):
    """One CPU batch per rollout; no per-move engine readbacks or callbacks."""
    if not records:
        return 0
    setups = np.asarray(setups).reshape(len(records), 4, 30)
    lut = np.full(14, -1, dtype=np.int64)
    for value, vocab in PIECE_TYPE_VALUE_TO_VOCAB_IDX.items():
        lut[value] = vocab
    lineups = lut[setups]
    if (lineups < 0).any():
        raise ValueError("completed game contains an unknown setup piece type")
    seats = np.array((0, 2) if random_opponent else (0, 1, 2, 3))
    winners = np.array([row["winner_team"] for row in records])[:, None]
    rewards = np.where(winners < 0, 0, np.where(winners == seats[None, :] % 2, 1, -1))
    return buffer.add_rewards(
        torch.from_numpy(lineups[:, seats].reshape(-1, 30)),
        torch.from_numpy(np.broadcast_to(seats, rewards.shape).copy().reshape(-1)),
        torch.ones(rewards.size, dtype=torch.bool),
        torch.from_numpy(rewards.astype(np.float32).reshape(-1)),
    )
