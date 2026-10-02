"""Legacy v1 joint inputs; consumes old rule-probability support.

All three other armies are hidden in DARK, including the teammate. Generate
all 25 identities per army, including dead pieces, using known initial counts.
Never read hidden surviving counts or use hidden types to build support.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from junqi_core.board import index_to_pos
from junqi_core.identity import ADMISSIBLE_BY_INDEX, TRACKED_TYPES
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import CAMP_INDICES, PIECE_COUNTS, Seat


@dataclass
class JointBeliefInput:
    cells: Tensor
    owners: Tensor
    public_features: Tensor
    support: Tensor
    inventory: Tensor
    piece_ids: Tensor

    def to(self, device) -> 'JointBeliefInput':
        return JointBeliefInput(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def kwargs(self):
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != 'piece_ids'}


_SLOTS = [i for i in range(30) if i not in CAMP_INDICES]
_COUNTS = np.array([PIECE_COUNTS[t] for t in TRACKED_TYPES], dtype=np.int64)
_INITIAL_SUPPORT = np.array([[t in ADMISSIBLE_BY_INDEX[i] for t in TRACKED_TYPES]
                             for i in _SLOTS], dtype=bool)


def build_joint_input(soa: dict, rule_probabilities: np.ndarray,
                      env_ids: np.ndarray, observers: np.ndarray) -> JointBeliefInput:
    """Build CPU metadata from public locations/counters and observer beliefs.

    ``piece_type_arr`` is deliberately neither requested nor accessed here.
    World rules have shape (N,4,12,289). Sequence order is relative owner
    1,2,3, then initial slot; current positions are canonical encoder indices.
    """
    n = rule_probabilities.shape[0]
    arrays = {name: np.asarray(soa[name]).reshape(n, 120) for name in
              ('alive', 'pos_x', 'pos_y', 'move_count_arr', 'active_eat_arr', 'passive_surv_arr')}
    rows = len(env_ids)
    cells = np.empty((rows, 75), dtype=np.int64)
    owners = np.tile(np.repeat(np.arange(1, 4), 25), (rows, 1)).astype(np.int64)
    features = np.empty((rows, 75, 4), dtype=np.float32)
    support = np.tile(_INITIAL_SUPPORT, (rows, 3, 1))
    pids = np.empty((rows, 75), dtype=np.int64)
    inventory = np.zeros((rows, 4, 12), dtype=np.int64)
    inventory[:, 1:] = _COUNTS
    for row, (env, observer) in enumerate(zip(env_ids, observers)):
        observer = Seat(int(observer))
        for q in range(3):
            owner = Seat((int(observer)+q+1) % 4)
            for j, slot in enumerate(_SLOTS):
                i, pid = q*25+j, int(owner)*30+slot
                pids[row, i] = pid
                live = bool(arrays['alive'][env, pid])
                moved = bool(arrays['move_count_arr'][env, pid]
                             or arrays['active_eat_arr'][env, pid])
                features[row, i] = (live, moved, arrays['active_eat_arr'][env, pid] > 0,
                                    arrays['passive_surv_arr'][env, pid] > 0)
                initial = index_to_pos(owner, slot)
                if live:
                    x, y = int(arrays['pos_x'][env, pid]), int(arrays['pos_y'][env, pid])
                    public = rule_probabilities[env, int(observer), :, y*17+x]
                    if public.sum() > 0:
                        support[row, i] &= public > 0
                else:
                    x, y = initial
                if moved:
                    support[row, i, :2] = False  # flag and mine cannot have moved
                cx, cy = world_to_canonical(x, y, observer)
                cells[row, i] = cy*17+cx
    if not support.any(-1).all():
        raise ValueError('Public joint-belief constraints have empty support')
    return JointBeliefInput(*(torch.from_numpy(x) for x in
                             (cells, owners, features, support, inventory, pids)))


def joint_training_targets(soa: dict, env_ids: np.ndarray, inputs: JointBeliefInput) -> Tensor:
    """Read truth solely after the public input has been frozen."""
    types = np.asarray(soa['piece_type_arr']).reshape(-1, 120)
    labels = types[np.asarray(env_ids)[:, None], inputs.piece_ids.numpy()] - 2
    if not ((labels >= 0) & (labels < 12)).all():
        raise ValueError('Invalid joint training labels')
    return torch.from_numpy(labels.astype(np.int64))
