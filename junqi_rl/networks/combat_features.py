"""Observation-only attack outcome probabilities and a small policy residual.

The outcome order is attacker survives, attacker alone dies, mutual death.
The table is shared across all source cells with the same known piece type:
12*129 triples replace 129*129 triples. No rollout history fields are added.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from junqi_core.board import COMPACT_TO_FLAT, NUM_ON_BOARD_CELLS
from junqi_core.info_model import TRACKED_TYPES
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_core.rules import Event, resolve_combat

COMBAT_FEATURE_VERSION = 1
OUTCOME_NAMES = ("eat_survive", "own_dies", "mutual_death")


class CombatOutcomeHead(nn.Module):
    """3 -> 16 -> 1 residual, initialized to preserve the baseline policy."""

    def __init__(self) -> None:
        super().__init__()
        table = torch.zeros(12, 12, 3)
        events = (Event.EAT, Event.KILLED, Event.BOMB)
        for a, attacker in enumerate(TRACKED_TYPES):
            if attacker.is_immobile:
                continue
            for d, defender in enumerate(TRACKED_TYPES):
                table[a, d, events.index(resolve_combat(attacker, defender))] = 1
        self.register_buffer("outcome_table", table, persistent=False)
        self.register_buffer("on_board", torch.tensor(COMPACT_TO_FLAT, dtype=torch.long),
                             persistent=False)
        self.residual = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def _pieces(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        # All inputs come from the actor's canonical observation; no game
        # state, piece IDs, or enemy legal-action queries enter this path.
        flat = obs.flatten(-2)
        own = flat[:, CHANNEL_LAYOUT["piece_own"]].index_select(-1, self.on_board).float()
        enemy = (flat[:, CHANNEL_LAYOUT["belief_left_side"]]
                 + flat[:, CHANNEL_LAYOUT["belief_right_side"]]).index_select(-1, self.on_board).float()
        occupancy = (flat[:, CHANNEL_LAYOUT["piece_left_side_enemy"]]
                     + flat[:, CHANNEL_LAYOUT["piece_right_side_enemy"]]).index_select(-1, self.on_board) > 0
        enemy = torch.where(occupancy, enemy, 0.0)
        return own, enemy

    def forward(self, obs: Tensor) -> Tensor:
        own, enemy = self._pieces(obs)  # B,12,129
        # B, attacker type, destination, outcome; fixed shapes compile cleanly.
        features = torch.einsum("bdj,ado->bajo", enemy, self.outcome_table)
        scores = self.residual(features).squeeze(-1).float()  # B,12,129
        # Zero moves even after MLP biases change; immobile types stay zero.
        scores = scores * (features.sum(-1) > 0)
        return torch.bmm(own.transpose(1, 2), scores).flatten(1)

    def for_actions(self, obs: Tensor, actions: Tensor, legal_mask: Tensor) -> Tensor:
        """Return (B,3) for selected legal attacks, zeros for moves/dummy rows."""
        own, enemy = self._pieces(obs)
        source = actions.long() // NUM_ON_BOARD_CELLS
        target = actions.long() % NUM_ON_BOARD_CELLS
        a = own.gather(2, source[:, None, None].expand(-1, 12, 1)).squeeze(-1)
        d = enemy.gather(2, target[:, None, None].expand(-1, 12, 1)).squeeze(-1)
        features = torch.einsum("ba,bd,ado->bo", a, d, self.outcome_table)
        legal = legal_mask.gather(1, actions.long()[:, None])
        return features * legal


class CombatFeatureMetrics:
    """Accumulate chosen learner attacks on device; synchronize once per rollout."""

    def __init__(self, policy: nn.Module) -> None:
        self.head = getattr(policy, "combat_head", None)
        self.totals: Tensor | None = None

    @torch.no_grad()
    def add(self, obs: Tensor, actions: Tensor, legal: Tensor, rows: Tensor | None = None) -> None:
        if self.head is None:
            return
        features = self.head.for_actions(obs, actions, legal)
        valid = legal.gather(1, actions.long()[:, None]).squeeze(1)
        if rows is not None:
            valid = valid & rows
        features = features * valid[:, None]
        attack = features.sum(-1) > 0
        total = torch.cat((features.sum(0), attack.sum().reshape(1), valid.sum().reshape(1)))
        self.totals = total if self.totals is None else self.totals + total

    def finish(self) -> dict[str, float]:
        if self.totals is None:
            return {}
        eat, die, mutual, attacks, actions = self.totals.cpu().tolist()
        return {
            "combat/chosen_attack_fraction": attacks / max(actions, 1),
            "combat/chosen_attacks": attacks,
            **{f"combat/chosen_{name}_mean": value / max(attacks, 1)
               for name, value in zip(OUTCOME_NAMES, (eat, die, mutual), strict=True)},
        }
