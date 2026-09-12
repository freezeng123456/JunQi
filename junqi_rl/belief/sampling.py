"""Sample current player information with separate simulator-only labels.

The model input is snapshotted before any labels are read. Supervision uses
the simulator's hidden types at the same positions, solely as loss targets.
This is centralized training; neither policy inference nor BeliefNet forward
receives the hidden types. Sampling all four observers avoids a cadence that
accidentally trains only one seat. Empty and publicly determined targets are
excluded instead of presenting zero loss as successful learning.
"""
from __future__ import annotations

import numpy as np
import torch

from junqi_core.info_model import TRACKED_TYPES
from junqi_rl.belief.buffer import BeliefBuffer
from junqi_rl.belief.inference import rule_only_belief_observations


def current_hidden_labels(soa, rules: np.ndarray, env_ids: np.ndarray):
    """World labels/masks for selected envs × four observers, in that order."""
    n = rules.shape[0]
    arrays = {key: np.asarray(soa[key]).reshape(n, 120) for key in
              ("piece_seat_arr", "piece_type_arr", "alive", "pos_x", "pos_y")}
    labels = np.full((len(env_ids), 4, 289), -1, dtype=np.int64)
    mask = np.zeros(labels.shape, dtype=bool)
    type_to_idx = {int(typ.value): idx for idx, typ in enumerate(TRACKED_TYPES)}
    for row, env in enumerate(env_ids):
        for pid in np.flatnonzero(arrays["alive"][env]):
            owner = int(arrays["piece_seat_arr"][env, pid])
            x, y = int(arrays["pos_x"][env, pid]), int(arrays["pos_y"][env, pid])
            if owner not in range(4) or x not in range(17) or y not in range(17):
                raise ValueError("Invalid live piece in belief-label source")
            label = type_to_idx[int(arrays["piece_type_arr"][env, pid])]
            cell = y * 17 + x
            for observer in range(4):
                if owner % 2 == observer % 2:
                    continue
                if np.count_nonzero(rules[env, observer, :, cell]) <= 1:
                    continue
                labels[row, observer, cell] = label
                mask[row, observer, cell] = True
    return labels.reshape(-1, 289), mask.reshape(-1, 289)


class MidgameBeliefSampler:
    def __init__(self, buffer: BeliefBuffer, *, every_steps: int = 32,
                 envs_per_sample: int = 4, seed: int = 0):
        if every_steps <= 0 or envs_per_sample <= 0:
            raise ValueError("Belief sampling cadence and env count must be positive")
        self.buffer = buffer
        self.every_steps = every_steps
        self.envs_per_sample = envs_per_sample
        self.rng = np.random.default_rng(seed)
        self.steps = self.inserted = self.valid_labels = self.empty_rows = 0

    def __call__(self, *, rollout_world, **_):
        sample = self.steps % self.every_steps == 0
        self.steps += 1
        if not sample:
            return
        n = rollout_world.num_envs
        ids = np.sort(self.rng.choice(n, min(n, self.envs_per_sample), replace=False))
        observations, _ = rollout_world.build_all_seat_observations_torch()
        rules = rollout_world.rule_beliefs_torch()
        # Snapshot player inputs first. Reading labels below cannot mutate it.
        clean = rule_only_belief_observations(observations, rules)
        selected = torch.as_tensor(ids, device=clean.device, dtype=torch.long)
        obs = clean.index_select(0, selected).detach().cpu().numpy().copy()
        rule_np = rules.detach().cpu().numpy().copy()
        soa = rollout_world.state.copy_to_host()
        labels, mask = current_hidden_labels(soa, rule_np, ids)
        keep = mask.any(axis=1)
        seats = np.tile(np.arange(4, dtype=np.int64), len(ids))
        self.empty_rows += int((~keep).sum())
        if not keep.any():
            return
        self.buffer.add(obs.reshape(-1, *obs.shape[2:])[keep], seats[keep],
                        labels[keep], mask[keep])
        self.inserted += int(keep.sum())
        self.valid_labels += int(mask[keep].sum())

    def stats(self):
        return {
            "belief_sample/rows": float(self.inserted),
            "belief_sample/valid_labels": float(self.valid_labels),
            "belief_sample/empty_rows_skipped": float(self.empty_rows),
        }
