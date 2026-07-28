"""junqi_rl.training.rollout — Rollout buffer with GAE advantage estimation.

This module provides :class:`RolloutBuffer`, a fixed-size trajectory store
that mirrors Ataraxos' ``CircularBuffer`` but is adapted for the 4-player,
turn-based structure of 四国军棋.

Key design decisions
--------------------
* **Per-actor view**: each slot in the buffer corresponds to ONE actor's
  transition at ONE step — i.e., ``(obs, action, reward, done, value, logp)``.
  Because 四国军棋 is strictly sequential (one player moves per step), every
  environment step produces exactly one training tuple for the acting seat.

* **GAE over episode**: after collecting a full rollout, ``compute_returns``
  walks backwards through each environment's trajectory and computes TD(λ)
  generalised advantage estimates.  Trajectories span episode boundaries via
  a terminal-mask bootstrap (value = 0 at done).

* **Action mask storage**: legal-action masks are stored sparsely as int32
  id lists to save memory (≈100 B/step instead of ≈84 KB/step in dense mode).

* **Canonical frame**: observations and action ids are stored in canonical
  (observer-seat) frame, matching what the policy network expects.  World-frame
  conversion is the caller's responsibility (``unrotate_action_id``).

References
----------
Ataraxos ``pyengine/core/buffer.py`` — original Stratego rollout buffer.
ADR-122 — JunqiEnv / VectorJunqiEnv design contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLAT_ACTION_DIM: int = 129 * 129   # 16,641 (compact on-board action space)
BOARD_SIZE: int = 17


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    """A minibatch sampled from :class:`RolloutBuffer`.

    All tensors share the first dimension ``B`` (batch size).

    Attributes
    ----------
    obs_spatial     : (B, OBS_CHANNELS, 17, 17)  float32
    obs_global      : (B, OBS_GLOBAL_DIMS)        float32
    legal_mask      : (B, FLAT_ACTION_DIM)        bool
    actions         : (B,)                         int32
    old_log_probs   : (B,)                         float32
    advantages      : (B,)                         float32   (normalised)
    returns         : (B,)                         float32
    values          : (B,)                         float32   (old value estimates)
    adv_mask        : (B,)                         bool      high-|adv| filter
    value_only_mask : (B,)                         bool      True ⇒ this transition
                                                              should contribute to
                                                              ``value_loss`` only,
                                                              not ``policy_loss`` /
                                                              ``entropy_loss`` /
                                                              ``kl_loss``.
                                                              Used by F-4 to keep
                                                              the value head trained
                                                              on enemy-seat (random
                                                              policy) transitions
                                                              even when policy
                                                              gradient is gated to
                                                              policy-controlled
                                                              transitions.
    """

    obs_spatial: Tensor
    obs_global: Tensor
    legal_mask: Tensor
    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    values: Tensor
    adv_mask: Tensor
    # F-4: optional, defaults to "everything is policy-active" (i.e. all-False).
    # Older minibatches() implementations that don't yet emit this field will
    # leave it unset; callers MUST tolerate ``None`` and treat it as all-False.
    value_only_mask: Tensor | None = None


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Fixed-size buffer for one PPO rollout epoch.

    Collects transitions from ``num_envs`` parallel environments over
    ``steps_per_env`` steps each, then computes GAE advantages in-place.

    Parameters
    ----------
    num_envs
        Number of parallel :class:`~junqi_rl.VectorJunqiEnv` environments.
    steps_per_env
        Number of steps to collect from each environment before training.
    gamma
        Discount factor for returns.
    gae_lambda
        λ parameter for Generalised Advantage Estimation.
    td_lambda
        λ parameter for TD-return mixture (0 = one-step TD, 1 = Monte-Carlo).
        This follows Ataraxos' convention of having two separate λ values.
    adv_filt_thresh
        Minimum |advantage| to include a transition in training.  Transitions
        with |A| < threshold are masked out of the minibatch by ``adv_mask``.
    adv_filt_rate
        Fraction of transitions allowed to be filtered.  Prevents the filter
        from removing everything when advantages are near-zero early training.
    device
        Torch device for returned tensors.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        steps_per_env: int,
        gamma: float = 1.0,
        gae_lambda: float = 0.5,
        td_lambda: float = 0.8,
        adv_filt_thresh: float = 0.01,
        adv_filt_rate: float = 0.75,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.td_lambda = td_lambda
        self.adv_filt_thresh = adv_filt_thresh
        self.adv_filt_rate = adv_filt_rate
        self.device = torch.device(device)

        N = num_envs
        T = steps_per_env

        # Observations (canonical frame)
        self.obs_spatial = np.zeros(
            (T, N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self.obs_global = np.zeros((T, N, OBS_GLOBAL_DIMS), dtype=np.float32)

        # Actions (canonical frame)
        self.actions = np.zeros((T, N), dtype=np.int32)

        # Legal mask (bool, dense — for mask-based loss)
        self.legal_mask = np.zeros((T, N, FLAT_ACTION_DIM), dtype=bool)

        # Policy log-probs at collection time
        self.log_probs = np.zeros((T, N), dtype=np.float32)

        # Value estimates at collection time (acting seat's perspective)
        self.values = np.zeros((T, N), dtype=np.float32)

        # Rewards and terminal flags
        self.rewards = np.zeros((T, N), dtype=np.float32)
        self.dones = np.zeros((T, N), dtype=bool)

        # Seat of acting player (int, 0-3) — stored for debugging
        self.seats = np.zeros((T, N), dtype=np.int8)

        # Computed by compute_returns()
        self.returns_ = np.zeros((T, N), dtype=np.float32)
        self.advantages_ = np.zeros((T, N), dtype=np.float32)

        self._ptr: int = 0
        self._full: bool = False

    # -------------------------------------------------------------------------
    # Data insertion
    # -------------------------------------------------------------------------

    def add(
        self,
        *,
        obs_spatial: np.ndarray,   # (N, OBS_CHANNELS, 17, 17)
        obs_global: np.ndarray,    # (N, OBS_GLOBAL_DIMS)
        legal_mask: np.ndarray,    # (N, FLAT_ACTION_DIM)
        actions: np.ndarray,       # (N,)  canonical-frame flat ids
        log_probs: np.ndarray,     # (N,)
        values: np.ndarray,        # (N,)
        rewards: np.ndarray,       # (N,)  acting-seat reward for this step
        dones: np.ndarray,         # (N,)  bool
        seats: np.ndarray,         # (N,)  acting seat Seat.value
    ) -> None:
        """Insert one step of experience from all N environments."""
        t = self._ptr
        self.obs_spatial[t] = obs_spatial
        self.obs_global[t] = obs_global
        self.legal_mask[t] = legal_mask
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.seats[t] = seats
        self._ptr += 1
        if self._ptr == self.steps_per_env:
            self._ptr = 0
            self._full = True

    def reset(self) -> None:
        """Clear the buffer (does NOT zero memory)."""
        self._ptr = 0
        self._full = False

    @property
    def is_ready(self) -> bool:
        """True when a full rollout has been collected."""
        return self._full

    # -------------------------------------------------------------------------
    # GAE computation
    # -------------------------------------------------------------------------

    def compute_returns(
        self, last_values: np.ndarray  # (N,) bootstrap value at end of rollout
    ) -> None:
        """Compute GAE(λ) advantages and TD(λ) returns in-place.

        Implements Ataraxos-style two-λ advantage estimation:

            δ_t    = r_t + γ · V(s_{t+1}) − V(s_t)
            A_t    = Σ_{l≥0} (γ · gae_λ)^l · δ_{t+l}   [GAE]
            G_t    = A_t + V(s_t)                         [TD(λ) return]

        Terminal steps have their bootstrap zeroed (``done=True → V_next=0``).

        Parameters
        ----------
        last_values
            Value estimate for the state immediately after the last collected
            step.  Used as the bootstrap value for the final TD residual.
        """
        T = self.steps_per_env
        N = self.num_envs
        gae = np.zeros(N, dtype=np.float32)
        next_val = last_values.astype(np.float32)  # (N,)

        for t in reversed(range(T)):
            mask = (~self.dones[t]).astype(np.float32)   # 0 at terminal
            delta = self.rewards[t] + self.gamma * next_val * mask - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            self.advantages_[t] = gae
            self.returns_[t] = gae + self.values[t]
            next_val = self.values[t]

    # -------------------------------------------------------------------------
    # Minibatch iteration
    # -------------------------------------------------------------------------

    def minibatches(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Iterator[RolloutBatch]:
        """Iterate over minibatches of the full rollout.

        Applies the advantage magnitude filter (``adv_filt_thresh``), then
        yields :class:`RolloutBatch` objects ready for PPO loss computation.

        Parameters
        ----------
        batch_size
            Number of transitions per minibatch.
        shuffle
            Whether to shuffle transitions before batching.
        rng
            Optional NumPy random generator for reproducibility.
        """
        T = self.steps_per_env
        N = self.num_envs

        # Flatten (T, N) → (T*N,)
        obs_sp = self.obs_spatial.reshape(T * N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE)
        obs_gl = self.obs_global.reshape(T * N, OBS_GLOBAL_DIMS)
        lm = self.legal_mask.reshape(T * N, FLAT_ACTION_DIM)
        act = self.actions.reshape(T * N)
        lp = self.log_probs.reshape(T * N)
        adv = self.advantages_.reshape(T * N)
        ret = self.returns_.reshape(T * N)
        val = self.values.reshape(T * N)

        # Normalise advantages (zero-mean, unit-variance)
        adv_std = adv.std() + 1e-8
        adv_norm = (adv - adv.mean()) / adv_std

        # Advantage magnitude filter — keep large-|A| transitions only
        abs_adv = np.abs(adv_norm)
        thresh = self.adv_filt_thresh
        adv_mask = abs_adv >= thresh
        # Safety: don't filter more than adv_filt_rate fraction
        n_total = T * N
        n_kept = adv_mask.sum()
        min_kept = int(n_total * (1.0 - self.adv_filt_rate))
        if n_kept < min_kept:
            # Lower threshold until we keep enough
            sorted_abs = np.sort(abs_adv)[::-1]
            if min_kept < n_total:
                thresh = float(sorted_abs[min_kept - 1])
            adv_mask = abs_adv >= thresh

        # Build index array
        indices = np.where(adv_mask)[0]
        if shuffle:
            g = rng if rng is not None else np.random.default_rng()
            g.shuffle(indices)

        dev = self.device

        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            if len(idx) == 0:
                continue

            batch = RolloutBatch(
                obs_spatial=torch.from_numpy(obs_sp[idx]).to(dev),
                obs_global=torch.from_numpy(obs_gl[idx]).to(dev),
                legal_mask=torch.from_numpy(lm[idx]).to(dev),
                actions=torch.from_numpy(act[idx].astype(np.int64)).to(dev),
                old_log_probs=torch.from_numpy(lp[idx]).to(dev),
                advantages=torch.from_numpy(adv_norm[idx]).to(dev),
                returns=torch.from_numpy(ret[idx]).to(dev),
                values=torch.from_numpy(val[idx]).to(dev),
                adv_mask=torch.ones(len(idx), dtype=torch.bool, device=dev),
            )
            yield batch

    def num_valid_transitions(self) -> int:
        """Count transitions passing the advantage filter (for logging)."""
        adv = self.advantages_.reshape(-1)
        adv_std = adv.std() + 1e-8
        adv_norm = (adv - adv.mean()) / adv_std
        return int((np.abs(adv_norm) >= self.adv_filt_thresh).sum())

    def stats(self) -> dict[str, float]:
        """Summary statistics for logging."""
        adv = self.advantages_.reshape(-1)
        ret = self.returns_.reshape(-1)
        rew = self.rewards.reshape(-1)
        return {
            "rollout/mean_reward": float(rew.mean()),
            "rollout/mean_return": float(ret.mean()),
            "rollout/mean_advantage": float(adv.mean()),
            "rollout/std_advantage": float(adv.std() + 1e-8),
            "rollout/num_valid": float(self.num_valid_transitions()),
        }
