"""junqi_rl — Reinforcement-learning adapters for the junqi core engine.

This package provides stable RL entry points built on top of
``junqi_core``.  Phase 0.4 ships :class:`JunqiEnv` (single-env),
:class:`VectorJunqiEnv` (N parallel envs sharing one obs slab), and
:class:`ExperienceBuffer` (fixed-capacity ring buffer for rollouts).

Phase 0.5 adds:

* ``junqi_rl.networks`` — :class:`JunqiNet` Transformer policy + value network.
* ``junqi_rl.training`` — PPO trainer, rollout buffer, rollout collector, and
  logging utilities.
* ``junqi_rl.action_lut`` — Pre-computed action-id rotation LUTs for fast
  world ↔ canonical frame conversion (vectorised NumPy, no Python loop).

Phase GPU adds:

* :class:`VectorJunqiEnvGPU` — GPU-accelerated drop-in replacement for
  :class:`VectorJunqiEnv` that offloads observation building to the CUDA
  backend (requires the ``junqi_cuda`` extension to be compiled).

See `docs/DECISIONS.md` ADR-122.
"""

from .action_lut import (
    FLAT_ACTION_DIM,
    ROTATE_LUT,
    UNROTATE_LUT,
    batch_rotate_action_ids,
    batch_unrotate_action_ids,
    build_legal_mask_batch,
)
from .buffer import ExperienceBuffer, LegalMaskMode
from .env import (
    JunqiEnv,
    JunqiStepInfo,
    VectorJunqiEnv,
    action_id_to_src_dst,
    rotate_action_id,
    src_dst_to_action_id,
    unrotate_action_id,
)

# CPU facts and metadata consumers do not need to load a CUDA backend.
_GPU_EXPORTS = {
    "VectorJunqiEnvGPU": "env_gpu",
    "GpuRollout": "gpu_rollout",
    "GpuRolloutHistory": "gpu_rollout",
    "GpuWorld": "gpu_world",
}


def __getattr__(name):
    if name not in _GPU_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{_GPU_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    # action LUT (fast batch rotation)
    "FLAT_ACTION_DIM",
    "ROTATE_LUT",
    "UNROTATE_LUT",
    "batch_rotate_action_ids",
    "batch_unrotate_action_ids",
    "build_legal_mask_batch",
    # env
    "ExperienceBuffer",
    "GpuRollout",
    "GpuRolloutHistory",
    "GpuWorld",
    "JunqiEnv",
    "JunqiStepInfo",
    "LegalMaskMode",
    "VectorJunqiEnv",
    "VectorJunqiEnvGPU",
    "action_id_to_src_dst",
    "rotate_action_id",
    "src_dst_to_action_id",
    "unrotate_action_id",
    # subpackages (imported on demand to avoid hard torch dependency at import time)
    # "networks",
    # "training",
]
