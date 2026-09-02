"""Static memory accounting for full-observation and compact rollout modes."""

from __future__ import annotations

from dataclasses import dataclass

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.training.rollout import BOARD_SIZE, FLAT_ACTION_DIM


@dataclass(frozen=True)
class RolloutStorageEstimate:
    bytes_per_transition: int
    total_bytes: int
    observation_bytes: int
    legal_bytes: int
    scalar_bytes: int

    @property
    def total_gib(self) -> float:
        return self.total_bytes / float(1024**3)


# action, old_log_prob, value, reward, done, seat, return, advantage
_SCALAR_BYTES = 4 + 4 + 4 + 4 + 1 + 1 + 4 + 4

# Exact fields allocated by DeviceRolloutHistory per (step, env).
_COMPACT_CORE_BYTES = (
    7 * 120  # piece_seat/type, alive, pos_x/y, zero_x/y
    + 3 * 120 * 2  # move/eat/survive counters
    + 120  # death reason
    + 120 * 2  # death location
    + 289 * 2  # cell_piece_id
    + 2 * 4  # seat dead / flag revealed
    + 1  # turn
    + 2 * 4  # move counters
    + 32 * 2 * 2  # move-history ring
    + 2 * 4  # history write index / count
)
_OBSERVER_BELIEF_BYTES = 12 * 289 * 4
_ALL_OBSERVERS_COMBAT_MEMORY_BYTES = 4 * (
    6 * 120 * 8  # six uint64 pid bitmaps
    + 2 * 120 * 2  # direct/chain type uint16
    + 4 * 120 * 2  # four int16 counters/steps
    + 120  # rank floor int8
    + 3 * 120  # three bool flags
)
COMPACT_HISTORY_BYTES_PER_TRANSITION = (
    _COMPACT_CORE_BYTES
    + _OBSERVER_BELIEF_BYTES
    + _ALL_OBSERVERS_COMBAT_MEMORY_BYTES
)


def estimate_rollout_storage(
    *,
    num_envs: int,
    steps_per_env: int,
    storage_mode: str,
    obs_dtype_bytes: int = 2,
    csr_legal_mask: bool = True,
    csr_k_max: int = 256,
) -> RolloutStorageEstimate:
    """Estimate persistent storage allocated per completed rollout."""

    if num_envs <= 0 or steps_per_env <= 0:
        raise ValueError("num_envs and steps_per_env must be positive")
    transitions = num_envs * steps_per_env

    if storage_mode == "compact_history":
        observation_bytes = COMPACT_HISTORY_BYTES_PER_TRANSITION
        legal_bytes = 0
    elif storage_mode == "full_obs":
        observation_bytes = (
            OBS_CHANNELS * BOARD_SIZE * BOARD_SIZE + OBS_GLOBAL_DIMS
        ) * obs_dtype_bytes
        legal_bytes = (
            csr_k_max * 4 + 4
            if csr_legal_mask
            else FLAT_ACTION_DIM
        )
    else:
        raise ValueError(
            "storage_mode must be 'full_obs' or 'compact_history'"
        )

    per_transition = observation_bytes + legal_bytes + _SCALAR_BYTES
    return RolloutStorageEstimate(
        bytes_per_transition=per_transition,
        total_bytes=per_transition * transitions,
        observation_bytes=observation_bytes,
        legal_bytes=legal_bytes,
        scalar_bytes=_SCALAR_BYTES,
    )


__all__ = [
    "COMPACT_HISTORY_BYTES_PER_TRANSITION",
    "RolloutStorageEstimate",
    "estimate_rollout_storage",
]
