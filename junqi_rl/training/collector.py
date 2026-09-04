"""junqi_rl.training.collector — Rollout collection from VectorJunqiEnv.

The :func:`collect_rollout` function drives ``num_envs`` parallel
:class:`~junqi_rl.VectorJunqiEnv` instances for ``steps_per_env`` steps
each, storing experience into a :class:`~junqi_rl.training.rollout.RolloutBuffer`.

4-Player Considerations
-----------------------
四国军棋 is a turn-based game: exactly one seat acts per environment step.
Each step produces exactly ONE training tuple (for the acting seat).  The
reward is the acting seat's team reward (+1/−1/0) and is non-zero only at
terminal steps.

Action frame convention
~~~~~~~~~~~~~~~~~~~~~~~
* The policy network operates in **canonical** frame (observer = acting seat
  at SOUTH).
* :func:`collect_rollout` converts observations to canonical frame (already
  handled by :meth:`VectorJunqiEnv._fill_all_obs`) and passes them to the
  policy.
* The resulting canonical-frame action is **un-rotated** to world frame
  before being passed to :meth:`VectorJunqiEnv.step`.
* **Both** canonical and world-frame action ids are stored; the rollout
  buffer stores canonical ids (for policy gradient) while the env receives
  world ids (for game logic).

Legal mask construction
~~~~~~~~~~~~~~~~~~~~~~~
The environment exposes ``legal_action_ids`` per seat (world-frame).  The
collector rotates them to canonical frame and builds a dense bool mask over
the 16,641-dim compact action space.  Rotation is performed using a pre-computed
LUT (see :mod:`junqi_rl.action_lut`) — one NumPy fancy-index per env,
replacing the old per-action Python loop.

Episode resets
~~~~~~~~~~~~~~
When an environment episode ends (``done=True``), the env is reset
automatically at the start of the next step.  The final value bootstrap
(used in ``compute_returns``) is taken as 0 for terminal states and from
the policy's value head for non-terminal states.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from junqi_core.rules import ALL_SEATS, Seat
from junqi_rl.action_lut import ROTATE_LUT, UNROTATE_LUT, build_legal_mask_batch
from junqi_rl.env import VectorJunqiEnv, rotate_action_id, unrotate_action_id
from junqi_rl.networks.junqi_net import JunqiNet
from junqi_rl.training.rollout import FLAT_ACTION_DIM, RolloutBuffer


# ---------------------------------------------------------------------------
# Legacy helper — kept for backward compatibility (used in train.py eval)
# ---------------------------------------------------------------------------

def _build_legal_mask(
    env: VectorJunqiEnv,
    current_seats: list[Seat],
) -> np.ndarray:
    """Build dense legal-action mask in canonical frame.

    .. deprecated::
        Use :func:`~junqi_rl.action_lut.build_legal_mask_batch` which avoids
        the Python loop over individual legal action ids.
    """
    return build_legal_mask_batch(env, current_seats)


def _get_actor_rewards(
    rewards: np.ndarray,       # (N, 4)  per-seat rewards
    seats: list[Seat],         # acting seat per env
    done: np.ndarray,          # (N,)  bool
) -> np.ndarray:
    """Extract the acting-seat reward for each environment.

    Returns float32 array of shape (N,).  Uses vectorised NumPy indexing
    instead of a Python loop.
    """
    seat_indices = np.array([s.value for s in seats], dtype=np.intp)
    env_indices = np.arange(len(seats), dtype=np.intp)
    # Gather per-acting-seat reward; zero out non-terminal envs
    actor_rewards = rewards[env_indices, seat_indices].astype(np.float32)
    actor_rewards[~done] = 0.0
    return actor_rewards


@torch.no_grad()
def collect_rollout(
    env: VectorJunqiEnv,
    policy: JunqiNet,
    rollout: RolloutBuffer,
    *,
    device: str | torch.device = "cpu",
    seed_base: int | None = None,
    auto_reset_done: bool = True,
) -> None:
    """Collect one rollout epoch from ``env`` using ``policy``.

    Fills ``rollout`` with ``steps_per_env`` transitions per environment.
    After filling, calls ``rollout.compute_returns(last_values)`` internally
    using zero-bootstrap for terminal envs.

    Performance notes
    -----------------
    * Legal masks are built via :func:`~junqi_rl.action_lut.build_legal_mask_batch`
      (LUT-based NumPy fancy-index, O(K) per env, no Python loop over actions).
    * Acting-seat observation slicing uses NumPy advanced indexing over the
      pre-allocated ``(N, 4, C, H, W)`` slab.
    * World ↔ canonical action conversion uses
      :data:`~junqi_rl.action_lut.ROTATE_LUT` /
      :data:`~junqi_rl.action_lut.UNROTATE_LUT` vectorised over the full N-dim.

    Parameters
    ----------
    env
        :class:`VectorJunqiEnv` (already reset or will be reset here).
    policy
        :class:`JunqiNet` used for action selection and value estimation.
        Should be the **EMA** model for stable behaviour.
    rollout
        :class:`RolloutBuffer` to populate (will call ``rollout.reset()``).
    device
        Device for policy inference tensors.
    seed_base
        Optional base seed for environment resets.
    auto_reset_done
        If True, done environments are immediately reset so collection
        continues uninterrupted.
    """
    _device = torch.device(device)
    policy.eval()
    rollout.reset()

    N = env.num_envs
    T = rollout.steps_per_env

    # Ensure all envs are live at the start
    obs_sp, obs_gl = env.reset(seed_base=seed_base)

    # Pre-allocate per-step index vector (reused each step)
    _env_range = np.arange(N, dtype=np.intp)

    step = 0

    while step < T:
        # Current seats for each env
        current_seats = env.current_seats()

        # ---- Build legal action mask (canonical frame, LUT-based) ----
        legal_mask_np = build_legal_mask_batch(env, current_seats)

        # ---- Gather observations for the acting seat per env ----
        # obs_sp shape: (N, 4, OBS_CHANNELS, 17, 17)
        acting_idx = np.array([s.value for s in current_seats], dtype=np.intp)
        act_obs_sp = obs_sp[_env_range, acting_idx]   # (N, C, 17, 17)
        act_obs_gl = obs_gl[_env_range, acting_idx]   # (N, G)

        # ---- Policy inference ----
        sp_t = torch.from_numpy(act_obs_sp).to(_device)
        gl_t = torch.from_numpy(act_obs_gl).to(_device)
        lm_t = torch.from_numpy(legal_mask_np).to(_device)

        actions_can, log_probs, values = policy.act(sp_t, gl_t, lm_t)

        # ---- Convert values to numpy ----
        values_np = values.cpu().numpy()
        if values_np.ndim > 1:
            values_np = values_np[:, 0]
        values_np = values_np.astype(np.float32)

        # ---- Vectorised canonical → world action conversion ----
        actions_can_np = actions_can.cpu().numpy().astype(np.int32)
        actions_world = np.empty(N, dtype=np.int32)
        for i in range(N):
            if env.done[i]:
                actions_world[i] = 0
            else:
                # LUT unrotate: single array index (O(1) per env)
                actions_world[i] = UNROTATE_LUT[current_seats[i].value][actions_can_np[i]]

        # ---- Store experience BEFORE stepping ----
        rollout.add(
            obs_spatial=act_obs_sp,
            obs_global=act_obs_gl,
            legal_mask=legal_mask_np,
            actions=actions_can_np,
            log_probs=log_probs.cpu().numpy().astype(np.float32),
            values=values_np,
            rewards=np.zeros(N, dtype=np.float32),  # filled after step
            dones=env.done.copy(),
            seats=acting_idx.astype(np.int8),
        )

        # ---- Environment step ----
        obs_sp, obs_gl, rewards_np, done_np, infos = env.step(
            actions_world
        )

        # ---- Patch rewards into the just-written rollout slot ----
        actor_rewards = _get_actor_rewards(rewards_np.astype(np.float32), current_seats, done_np)
        rollout.rewards[step] = actor_rewards
        rollout.dones[step] = done_np.copy()

        # ---- Auto-reset done envs ----
        if auto_reset_done and done_np.any():
            for i in range(N):
                if done_np[i]:
                    sd = (seed_base + i) if seed_base is not None else None
                    env._envs[i].reset(seed=sd)
                    env._done[i] = False
            env._fill_all_obs()
            obs_sp = env.obs_spatial
            obs_gl = env.obs_global

        step += 1

    # ---- Compute bootstrap values for the last step ----
    current_seats = env.current_seats()
    acting_idx = np.array([s.value for s in current_seats], dtype=np.intp)
    act_obs_sp = obs_sp[_env_range, acting_idx]
    act_obs_gl = obs_gl[_env_range, acting_idx]
    legal_mask_np = build_legal_mask_batch(env, current_seats)

    sp_t = torch.from_numpy(act_obs_sp).to(_device)
    gl_t = torch.from_numpy(act_obs_gl).to(_device)
    lm_t = torch.from_numpy(legal_mask_np).to(_device)

    with torch.no_grad():
        _, _, last_values = policy.act(sp_t, gl_t, lm_t)

    if last_values.dim() == 1:
        last_val_np = last_values.cpu().numpy().astype(np.float32)
    else:
        last_val_np = last_values.cpu().numpy()[:, 0].astype(np.float32)

    # Zero bootstrap for terminal envs
    last_val_np = last_val_np * (~env.done).astype(np.float32)

    rollout.compute_returns(last_val_np)
