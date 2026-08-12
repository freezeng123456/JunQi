"""junqi_rl.training — PPO training pipeline for 四国军棋.

Exports
-------
PPOConfig
    Training hyper-parameters.
PPOTrainer
    PPO algorithm implementation with EMA.
RolloutBuffer
    Fixed-size trajectory store with GAE.
RolloutBatch
    Minibatch container for gradient updates.
collect_rollout
    Rollout collection from VectorJunqiEnv.
EMAPolicy
    Exponential moving average of model weights.
power_schedule
    Learning rate / temperature annealing schedule.
"""

from .collector import collect_rollout
from .config import (
    ArrangementTrainConfig,
    BeliefTrainConfig,
    EnvConfig,
    RolloutTrainConfig,
    TrainConfig,
    load_config,
    validate_config,
)
from .gpu_collector import (
    build_legal_mask_batch_gpu,
    collect_rollout_gpu,
    collect_rollout_gpu_v2,
)
from .ppo import EMAPolicy, PPOConfig, PPOTrainer, power_schedule
from .rollout import RolloutBatch, RolloutBuffer
from .rollout_gpu import RolloutBufferGPU
from .rollout_storage import RolloutStorageEstimate, estimate_rollout_storage

__all__ = [
    "ArrangementTrainConfig",
    "BeliefTrainConfig",
    "EMAPolicy",
    "EnvConfig",
    "PPOConfig",
    "PPOTrainer",
    "RolloutBatch",
    "RolloutBuffer",
    "RolloutBufferGPU",
    "RolloutStorageEstimate",
    "RolloutTrainConfig",
    "TrainConfig",
    "build_legal_mask_batch_gpu",
    "collect_rollout",
    "collect_rollout_gpu",
    "collect_rollout_gpu_v2",
    "estimate_rollout_storage",
    "load_config",
    "power_schedule",
    "validate_config",
]
