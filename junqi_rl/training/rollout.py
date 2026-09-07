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
    value_obs_spatial : (Bv, OBS_CHANNELS, 17, 17) float32, optional
    value_obs_global  : (Bv, OBS_GLOBAL_DIMS)      float32, optional
    value_returns     : (Bv,)                      float32, optional
    policy_value_indices : (Bp,)                   int64, optional
        A separate all-valid value batch. When present, the main fields hold
        only advantage-filtered policy samples and PPO evaluates the value
        objective on all samples. ``policy_value_indices`` maps the filtered
        policy rows into this batch, allowing the learner encoder to run once
        while the 16,641-action head still sees only policy samples.
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
    value_obs_spatial: Tensor | None = None
    value_obs_global: Tensor | None = None
    value_returns: Tensor | None = None
    policy_value_indices: Tensor | None = None



def timestep_keep_env_indices(
    abs_adv: np.ndarray,
    *,
    rate: float,
    thresh: float,
    own_mask: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Per-collect-row env indices to keep for PPO.

    ``abs_adv`` is ``|A_norm|`` with shape ``(T, N)``.  Each row is the
    same simulator step across parallel envs.  Within a row, keep the
    top ``round(N * rate)`` entries that also satisfy ``|A| >= thresh``
    and ``own_mask``.  Empty rows yield an empty index array so the
    caller can skip that Adam step.

    This is the legacy JunQi row-local quantile behaviour. Ataraxos also
    trains one batch per simulator row, but computes a single advantage
    threshold over the complete rollout before applying it to each row.
    """
    if abs_adv.ndim != 2:
        raise ValueError(f"abs_adv must be (T, N), got {abs_adv.shape}")
    t_steps, n_envs = abs_adv.shape
    if own_mask is None:
        own = np.ones((t_steps, n_envs), dtype=bool)
    else:
        own = np.asarray(own_mask, dtype=bool)
        if own.shape != abs_adv.shape:
            raise ValueError(
                f"own_mask shape {own.shape} != abs_adv {abs_adv.shape}"
            )
    if rate >= 1.0:
        k = n_envs
    elif rate <= 0.0:
        k = 1
    else:
        k = max(1, int(round(n_envs * float(rate))))
    out: list[np.ndarray] = []
    for t in range(t_steps):
        cand = own[t] & (abs_adv[t] >= float(thresh))
        idx = np.flatnonzero(cand)
        if idx.size == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        if idx.size > k:
            order = np.argsort(-abs_adv[t, idx], kind="stable")[:k]
            idx = idx[order]
        out.append(idx.astype(np.int64, copy=False))
    return out


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
        adv_filter_scope: str = "timestep",
        value_sample_scope: str = "policy",
        device: str | torch.device = "cpu",
        minibatch_group: str = "global",
    ) -> None:
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.td_lambda = td_lambda
        self.adv_filt_thresh = adv_filt_thresh
        self.adv_filt_rate = adv_filt_rate
        if adv_filter_scope not in {"timestep", "rollout"}:
            raise ValueError(
                "adv_filter_scope must be 'timestep' or 'rollout'; "
                f"got {adv_filter_scope!r}"
            )
        self.adv_filter_scope = adv_filter_scope
        if value_sample_scope not in {"policy", "all_valid"}:
            raise ValueError(
                "value_sample_scope must be 'policy' or 'all_valid'; "
                f"got {value_sample_scope!r}"
            )
        if value_sample_scope == "all_valid" and minibatch_group != "timestep":
            raise ValueError(
                "value_sample_scope='all_valid' requires "
                "minibatch_group='timestep'"
            )
        self.value_sample_scope = value_sample_scope
        self.device = torch.device(device)
        if minibatch_group not in {"global", "timestep"}:
            raise ValueError(
                "minibatch_group must be 'global' or 'timestep'; "
                f"got {minibatch_group!r}"
            )
        self.minibatch_group = minibatch_group

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
        self,
        last_values: np.ndarray,
        *,
        last_seats: np.ndarray | None = None,
    ) -> None:
        """Compute acting-team GAE and independent TD(lambda) value targets.

        Values, rewards and advantages are in the acting seat's perspective.
        At an enemy-team boundary BOTH the value bootstrap and the recursive
        trace change sign. Seat IDs, rather than a hard-coded alternating
        sign, also handle an eliminated seat being skipped.

        ``gae_lambda`` controls advantages; ``td_lambda`` independently
        controls the trace used in value targets. Setting them equal restores
        ``returns = advantages + values`` in self-play.

        Pass the actual post-rollout ``last_seats`` for exact bootstrapping.
        Omission preserves the GPU API's opposite-team fallback, which is
        not exact when the next surviving actor is a teammate.
        """
        T = self.steps_per_env
        N = self.num_envs
        next_val = np.asarray(last_values, dtype=np.float32)
        if next_val.shape != (N,):
            raise ValueError(f"last_values must have shape ({N},)")
        if last_seats is None:
            next_team = (self.seats[T - 1].astype(np.int64) & 1) ^ 1
        else:
            last_seats = np.asarray(last_seats)
            if last_seats.shape != (N,):
                raise ValueError(f"last_seats must have shape ({N},)")
            next_team = last_seats.astype(np.int64) & 1
        gae = np.zeros(N, dtype=np.float32)
        td_trace = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            mask = (~self.dones[t]).astype(np.float32)
            cur_team = self.seats[t].astype(np.int64) & 1
            flip = (cur_team == next_team).astype(np.float32) * 2.0 - 1.0
            delta = (
                self.rewards[t]
                + self.gamma * flip * next_val * mask
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * mask * flip * gae
            td_trace = delta + self.gamma * self.td_lambda * mask * flip * td_trace
            self.advantages_[t] = gae
            self.returns_[t] = td_trace + self.values[t]
            next_val = self.values[t]
            next_team = cur_team

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

        # Advantage magnitude filter — train on the largest-|A|
        # transitions. ``adv_filt_rate`` is the fraction kept, matching
        # RolloutBufferGPU; this buffer used to read it as a floor on how
        # many survive ``adv_filt_thresh``, i.e. the opposite knob.
        abs_adv = np.abs(adv_norm)
        thresh = self.adv_filt_thresh
        dev = self.device

        def _emit(idx: np.ndarray, value_only_mask: np.ndarray | None = None):
            if idx.size == 0:
                return
            if value_only_mask is None:
                value_only_mask = np.zeros(idx.size, dtype=bool)
            yield RolloutBatch(
                obs_spatial=torch.from_numpy(obs_sp[idx]).to(dev),
                obs_global=torch.from_numpy(obs_gl[idx]).to(dev),
                legal_mask=torch.from_numpy(lm[idx]).to(dev),
                actions=torch.from_numpy(act[idx].astype(np.int64)).to(dev),
                old_log_probs=torch.from_numpy(lp[idx]).to(dev),
                advantages=torch.from_numpy(adv_norm[idx]).to(dev),
                returns=torch.from_numpy(ret[idx]).to(dev),
                values=torch.from_numpy(val[idx]).to(dev),
                adv_mask=torch.ones(len(idx), dtype=torch.bool, device=dev),
                value_only_mask=torch.from_numpy(value_only_mask).to(dev),
            )

        def _emit_all_valid_row(t: int, policy_env_idx: np.ndarray):
            value_idx = t * N + np.arange(N, dtype=np.int64)
            policy_idx = (t * N + policy_env_idx).astype(np.int64, copy=False)
            yield RolloutBatch(
                obs_spatial=torch.from_numpy(obs_sp[policy_idx]).to(dev),
                obs_global=torch.from_numpy(obs_gl[policy_idx]).to(dev),
                legal_mask=torch.from_numpy(lm[policy_idx]).to(dev),
                actions=torch.from_numpy(
                    act[policy_idx].astype(np.int64, copy=False)
                ).to(dev),
                old_log_probs=torch.from_numpy(lp[policy_idx]).to(dev),
                advantages=torch.from_numpy(adv_norm[policy_idx]).to(dev),
                returns=torch.from_numpy(ret[policy_idx]).to(dev),
                values=torch.from_numpy(val[policy_idx]).to(dev),
                adv_mask=torch.ones(
                    policy_idx.size, dtype=torch.bool, device=dev,
                ),
                value_only_mask=torch.zeros(
                    policy_idx.size, dtype=torch.bool, device=dev,
                ),
                value_obs_spatial=torch.from_numpy(obs_sp[value_idx]).to(dev),
                value_obs_global=torch.from_numpy(obs_gl[value_idx]).to(dev),
                value_returns=torch.from_numpy(ret[value_idx]).to(dev),
                policy_value_indices=torch.from_numpy(
                    policy_env_idx.astype(np.int64, copy=False)
                ).to(dev),
            )

        if self.minibatch_group == "timestep":
            if self.adv_filter_scope == "rollout":
                # Ataraxos computes its single rollout-level threshold from
                # raw |A|. Advantages are still normalised for the PPO loss;
                # only the membership decision uses unnormalised values.
                filter_abs = np.abs(adv)
                if self.adv_filt_rate < 1.0:
                    q_thresh = float(np.quantile(
                        filter_abs, 1.0 - self.adv_filt_rate,
                    ))
                    thresh = max(thresh, q_thresh)
                rows = [
                    np.flatnonzero(row >= thresh).astype(np.int64, copy=False)
                    for row in filter_abs.reshape(T, N)
                ]
            else:
                rows = timestep_keep_env_indices(
                    abs_adv.reshape(T, N),
                    rate=self.adv_filt_rate,
                    thresh=self.adv_filt_thresh,
                )
            sizes = [int(r.size) for r in rows]
            self._last_n_total = int(T * N)
            self._last_n_own = int(T * N)
            self._last_n_policy = int(sum(sizes))
            self._last_n_value = (
                int(T * N)
                if self.value_sample_scope == "all_valid"
                else self._last_n_policy
            )
            self._last_thresh_used = float(thresh)
            nonempty = [s for s in sizes if s > 0]
            self._last_kept_mean = float(np.mean(sizes)) if sizes else 0.0
            self._last_kept_min = float(min(nonempty) if nonempty else 0)
            self._last_kept_max = float(max(sizes) if sizes else 0)
            self._last_n_empty_steps = float(sum(1 for s in sizes if s == 0))
            for t, env_idx in enumerate(rows):
                if self.value_sample_scope == "all_valid":
                    yield from _emit_all_valid_row(t, env_idx)
                elif env_idx.size:
                    yield from _emit((t * N + env_idx).astype(np.int64))
            return

        if self.adv_filt_rate < 1.0:
            q_thresh = float(np.quantile(abs_adv, 1.0 - self.adv_filt_rate))
            thresh = max(thresh, q_thresh)
        adv_mask = abs_adv >= thresh

        indices = np.where(adv_mask)[0]
        if shuffle:
            g = rng if rng is not None else np.random.default_rng()
            g.shuffle(indices)

        self._last_n_total = int(T * N)
        self._last_n_own = int(T * N)
        self._last_n_policy = int(len(indices))
        self._last_n_value = self._last_n_policy
        self._last_thresh_used = float(thresh)
        self._last_kept_mean = float("nan")
        self._last_kept_min = float("nan")
        self._last_kept_max = float("nan")
        self._last_n_empty_steps = 0.0

        for start in range(0, len(indices), batch_size):
            idx = indices[start : start + batch_size]
            yield from _emit(idx)

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
        out = {
            "rollout/mean_reward": float(rew.mean()),
            "rollout/mean_return": float(ret.mean()),
            "rollout/mean_advantage": float(adv.mean()),
            "rollout/std_advantage": float(adv.std() + 1e-8),
            "rollout/num_valid": float(self.num_valid_transitions()),
        }
        if hasattr(self, "_last_n_total"):
            out["rollout/n_total"] = float(self._last_n_total)
            out["rollout/n_own_seat"] = float(self._last_n_own)
            out["rollout/n_policy_kept"] = float(self._last_n_policy)
            out["rollout/n_value_samples"] = float(self._last_n_value)
            out["rollout/keep_frac_own"] = (
                float(self._last_n_policy) / max(1.0, float(self._last_n_own))
            )
            out["rollout/adv_thresh_used"] = float(self._last_thresh_used)
            out["rollout/kept_mean"] = float(self._last_kept_mean)
            out["rollout/kept_min"] = float(self._last_kept_min)
            out["rollout/kept_max"] = float(self._last_kept_max)
            out["rollout/n_empty_steps"] = float(self._last_n_empty_steps)
        return out

__all__ = ["RolloutBuffer", "RolloutBatch", "timestep_keep_env_indices", "FLAT_ACTION_DIM"]
