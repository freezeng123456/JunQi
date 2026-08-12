"""junqi_rl.training.ppo — PPO trainer for 四国军棋.

Implements Proximal Policy Optimisation (PPO) with:
  - Clipped policy gradient (PPO-Clip)
  - Value function loss (MSE or categorical cross-entropy)
  - Entropy bonus (magnet loss in Ataraxos terminology)
  - KL divergence penalty (to stabilise training)
  - Generalised Advantage Estimation (GAE)
  - EMA (Exponential Moving Average) policy for evaluation
  - Mixed-precision training (bfloat16 / float32 autocast)
  - Gradient clipping
  - Power-schedule learning rate and temperature annealing

The overall structure mirrors Ataraxos ``pyengine/core/rl.py``.

Usage
-----
See ``scripts/train.py`` for the full training entry point.

    cfg = PPOConfig()
    policy = JunqiNet(cfg.net)
    ppo    = PPOTrainer(policy, cfg, device="cuda:0")
    ...
    for epoch in range(total_epochs):
        rollout.reset()
        collect_rollout(env, policy_ema, rollout)
        rollout.compute_returns(last_values)
        metrics = ppo.train_epoch(rollout)
        policy_ema.update()
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel

# GradScaler moved between modules across torch versions:
#   * torch < 2.3: only ``torch.cuda.amp.GradScaler`` exists; constructor
#                  takes no positional device argument (``GradScaler()``).
#   * torch >= 2.3: ``torch.amp.GradScaler("cuda")`` is the canonical API,
#                  with ``torch.cuda.amp.GradScaler`` kept as an alias.
# We keep both code paths working so a single source tree runs on the
# H20 boxes (this repo's primary target, torch 2.1) and on newer machines.
try:
    from torch.amp import GradScaler as _TorchAmpGradScaler  # torch >= 2.3
    def _make_grad_scaler() -> "_TorchAmpGradScaler":
        return _TorchAmpGradScaler("cuda")
except ImportError:
    from torch.cuda.amp import GradScaler as _TorchAmpGradScaler  # torch < 2.3
    def _make_grad_scaler() -> "_TorchAmpGradScaler":
        return _TorchAmpGradScaler()

from junqi_rl.networks.junqi_net import N_VF_CAT, JunqiNet, JunqiNetConfig
from junqi_rl.training.rollout import RolloutBatch, RolloutBuffer


def _is_distributed() -> bool:
    """Return True iff a NCCL/gloo process group is initialised."""
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


# ---------------------------------------------------------------------------
# Power schedule (mirrors Ataraxos)
# ---------------------------------------------------------------------------


def power_schedule(
    coef: float,
    step: int,
    decay: float,
    ceil: float,
    floor: float,
) -> float:
    """Smooth annealing schedule — matches Ataraxos exactly.

    ``value(t) = clamp(coef / (1 + t)^decay, floor, ceil)``

    With ``decay > 1`` the schedule decays quickly; ``decay < 1`` decays
    slowly.  Ataraxos default is ``decay=1.1`` (moderate decay).

    .. warning:: v12 and earlier used ``(1+t)^(1/decay)`` which inverted
       the decay semantics.  Old configs (e.g. ``lr_decay=10.0``) will
       decay **catastrophically fast** under the new formula.  v13+
       configs use Ataraxos-aligned values like ``lr_decay=1.1``.
    """
    raw = coef / (1.0 + step) ** decay
    return float(min(max(raw, floor), ceil))


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) policy
# ---------------------------------------------------------------------------


class EMAPolicy:
    """Maintains an exponential moving average of model weights.

    After each gradient update call :meth:`update`.  Use :attr:`model` for
    evaluation / checkpointing (rather than the learner model).

    Parameters
    ----------
    model
        The learner (training) model whose weights are tracked.
    decay
        EMA decay coefficient (typically 0.999).
    """

    def __init__(self, model: JunqiNet, decay: float = 0.999) -> None:
        self.decay = decay
        # Shadow copy — kept on same device as model
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: JunqiNet) -> None:
        """Update shadow weights: ``θ_ema ← decay·θ_ema + (1−decay)·θ``."""
        d = self.decay
        for s_param, m_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            s_param.data.mul_(d).add_(m_param.data, alpha=1.0 - d)

    @property
    def model(self) -> JunqiNet:
        return self.shadow

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "shadow": self.shadow.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.decay = sd["decay"]
        self.shadow.load_state_dict(sd["shadow"])


# ---------------------------------------------------------------------------
# PPO configuration
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """All hyper-parameters for the PPO trainer.

    Defaults follow Ataraxos ``RLConfig`` with minor adjustments for
    the 4-player 四国军棋 game.
    """

    # --- Network ---
    net: JunqiNetConfig = field(default_factory=JunqiNetConfig)

    # --- PPO core ---
    clip_range: float = 0.2
    """PPO clipping epsilon (ε in clip(ratio, 1−ε, 1+ε))."""

    vf_coef: float = 1.0
    """Weight for value loss in total loss."""

    policy_coef: float = 1.0
    """Weight for policy gradient loss."""

    # --- Entropy / temperature ---
    temperature_coef: float = 0.05
    """Initial temperature (entropy bonus weight)."""

    temperature_ceil: float = 0.1
    """Maximum temperature."""

    temperature_floor: float = 0.001
    """Minimum temperature."""

    temperature_decay: float = 0.3
    """Power-schedule decay for temperature."""

    uniform_magnet: bool = True
    """If True, entropy bonus encourages uniform distribution.
    If False, it simply maximises entropy."""

    # --- KL divergence ---
    kl_coef: float = 0.1
    """Weight for KL(π_new ‖ π_old) regularisation term."""

    # --- GAE ---
    gamma: float = 1.0
    """Discount factor."""

    gae_lambda: float = 0.5
    """λ for GAE advantage estimation."""

    td_lambda: float = 0.8
    """λ for TD-return mixture (0 = TD, 1 = MC)."""

    adv_filt_thresh: float = 0.01
    """Minimum normalised |advantage| to train on."""

    adv_filt_rate: float = 0.75
    """Maximum fraction of transitions to filter out."""

    # --- Optimiser ---
    lr_coef: float = 0.5
    """Initial learning rate coefficient (power schedule)."""

    lr_decay: float = 1.1
    """Learning rate decay exponent."""

    lr_ceil: float = 1e-4
    """Maximum learning rate."""

    lr_floor: float = 5e-6
    """Minimum learning rate."""

    lr_schedule_unit: str = "grad_step"
    """Time unit driving the lr power schedule.

    * ``"grad_step"`` (legacy default, used by all v17–v35 runs): step =
      ``num_train_step``, i.e. one count per PPO minibatch update. This
      makes the actual *rollout-pace* of decay sensitive to
      ``num_envs * steps_per_env``, ``adv_filt_rate``, ``minibatch_size``
      and DDP world_size, because all of them change how many minibatches
      land per rollout. On H20 v35 (num_envs=384/rank, world_size=2)
      ``lr`` hit ``lr_floor`` at rollout 124 even though training was
      planned for 1500 rollouts — i.e. ~92% of the planned schedule
      ran with a frozen learning rate. See
      ``exps/v35_ddp_combat_memory/launch.log.v35_buggy``.

    * ``"rollout"`` (recommended new default for H20 / DDP runs): step =
      ``num_rollout``. The schedule's pace is independent of batch size,
      DDP world_size and adv-filter rate — ``lr_decay=1.1`` then means
      "halve every ~600 rollouts" regardless of cluster shape.

    Migration: keep your existing ``lr_decay`` value but switch
    ``lr_schedule_unit: rollout`` for any DDP / H20 run; if you intentionally
    want the per-grad-step pace (e.g. for bit-reproducing v32 on T4) set
    ``"grad_step"`` and keep the old ``lr_decay``.
    """

    weight_decay: float = 0.0
    """AdamW weight decay."""

    max_grad_norm: float = 0.5
    """Gradient clipping norm."""

    # --- EMA ---
    ema_decay: float = 0.999
    """EMA decay for evaluation/checkpoint policy."""

    # --- Training schedule ---
    num_epochs_per_rollout: int = 4
    """Number of optimisation epochs per collected rollout."""

    minibatch_size: int = 512
    """Minibatch size for PPO gradient updates."""

    # --- Mixed precision ---
    dtype: str = "bfloat16"
    """Training dtype: 'bfloat16' or 'float32'."""

    # --- Compilation ---
    torch_compile: bool = False
    """Whether to apply torch.compile() to the network."""

    def get_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"unknown dtype {self.dtype!r}")


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """PPO training algorithm for 四国军棋.

    Parameters
    ----------
    policy
        The :class:`~junqi_rl.networks.JunqiNet` to train.
    cfg
        :class:`PPOConfig` hyper-parameters.
    device
        Torch device for training (e.g. ``"cuda:0"``).

    Attributes
    ----------
    ema
        EMA shadow model.  Use ``trainer.ema.model`` for evaluation.
    num_train_step
        Total number of gradient update steps taken.
    """

    def __init__(
        self,
        policy: JunqiNet,
        cfg: PPOConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_train_step: int = 0
        self.num_rollout: int = 0
        self._nan_skip_count: int = 0
        """Cumulative count of minibatches skipped due to NaN/Inf in forward pass.
        Incremented inside ``_update_step`` when fp16 autocast overflows."""
        self._grad_nan_skip_count: int = 0
        """Cumulative count of minibatches whose ``optimizer.step()`` was skipped
        because the gradient norm was non-finite (NaN or Inf). Without this guard
        the next ``optimizer.step()`` would write NaN into the parameters and
        permanently poison the EMA shadow — exactly what crashed v33a at R107
        (`exps/_crash_diagnosis_v33a/train.log`). Under DDP we ``all_reduce(MAX)``
        the per-rank "is bad" flag so every rank takes the same skip decision,
        otherwise some ranks would step while others wouldn't and the parameter
        replicas would diverge."""

        # Move policy to device
        policy = policy.to(self.device)

        # Save the **unwrapped** JunqiNet reference. External callers
        # (``trainer.policy.state_dict()``, ``trainer.policy.load_state_dict``,
        # ``EMAPolicy.update``) operate on this unwrapped module so that
        # checkpoints stay compatible across single-process / DDP / different
        # world sizes.
        self._policy_unwrapped: JunqiNet = policy
        self.policy: JunqiNet = policy

        # Build the module actually used for forward+backward inside
        # ``_update_step``. Wrapping order is:
        #   raw → torch.compile → DDP
        # This matches Ataraxos (compile first, then DDP wraps the
        # OptimizedModule). DDP is only added when a process group is
        # initialised; in single-process mode this is a no-op.
        train_module: torch.nn.Module = policy
        if cfg.torch_compile:
            train_module = torch.compile(train_module)  # type: ignore[assignment]
        if _is_distributed():
            ddp_kwargs: dict[str, Any] = {
                "find_unused_parameters": False,
                # Gradient bucketing in MB; default 25 MB is fine for ~1-15 M
                # params we run.
            }
            if self.device.type == "cuda" and self.device.index is not None:
                ddp_kwargs["device_ids"] = [self.device.index]
                ddp_kwargs["output_device"] = self.device.index
            train_module = DistributedDataParallel(train_module, **ddp_kwargs)
        self._policy_for_train: torch.nn.Module = train_module

        # EMA tracks the **unwrapped** module; update() zips
        # ``shadow.parameters()`` with ``model.parameters()`` and DDP /
        # torch.compile both keep the same parameter list as the underlying
        # module, so this is safe.
        self.ema = EMAPolicy(self._policy_unwrapped, cfg.ema_decay)

        # Optimiser.  ``foreach=True`` (default on CUDA in recent torch) uses
        # fused multi-tensor ops — about 20-30% faster than the per-param
        # fallback for ~1M-param models.
        lr0 = power_schedule(
            cfg.lr_coef, 0, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor
        )
        self.optimizer = torch.optim.AdamW(
            self._policy_unwrapped.parameters(),
            lr=lr0,
            weight_decay=cfg.weight_decay,
            foreach=True,
        )

        # Mixed-precision scaler (only for float16; bfloat16 doesn't need it).
        # GradScaler adds ~1-2 ms/step of python-level bookkeeping on T4.
        # bfloat16 has equivalent accuracy and skips the scaler entirely, so
        # it's generally the faster choice unless the model genuinely needs
        # fp16's extra precision.
        self._use_amp = cfg.get_dtype() != torch.float32
        self._amp_dtype = cfg.get_dtype()
        self._scaler = (
            _make_grad_scaler()
            if (self._use_amp and self._amp_dtype == torch.float16)
            else None
        )

    # -------------------------------------------------------------------------
    # Learning-rate update
    # -------------------------------------------------------------------------

    def _update_lr(self) -> float:
        cfg = self.cfg
        # Pick the schedule's time unit. ``grad_step`` is the legacy v17–v35
        # behaviour. ``rollout`` removes the pace-vs-batch-size coupling that
        # caused v35 to hit lr_floor at R=124 of 1500. See cfg field doc.
        unit = getattr(cfg, "lr_schedule_unit", "grad_step")
        if unit == "rollout":
            step = self.num_rollout
        elif unit == "grad_step":
            step = self.num_train_step
        else:
            raise ValueError(
                f"lr_schedule_unit must be 'rollout' or 'grad_step'; got {unit!r}"
            )
        lr = power_schedule(
            cfg.lr_coef, step, cfg.lr_decay, cfg.lr_ceil, cfg.lr_floor
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def _get_temperature(self) -> float:
        cfg = self.cfg
        return power_schedule(
            cfg.temperature_coef,
            self.num_train_step,
            cfg.temperature_decay,
            cfg.temperature_ceil,
            cfg.temperature_floor,
        )

    # -------------------------------------------------------------------------
    # Loss computation
    # -------------------------------------------------------------------------

    def _policy_loss(
        self,
        new_log_probs: Tensor,   # (B,)  log π_new(a|s)
        old_log_probs: Tensor,   # (B,)  log π_old(a|s)
        advantages: Tensor,      # (B,)
        *,
        weight_per: Tensor | None = None,
    ) -> Tensor:
        """PPO-Clip policy gradient loss.

        ``weight_per`` (optional, shape (B,) float32 ∈ {0, 1}): per-sample
        weight. Samples with weight=0 are excluded from the gradient. Used
        by F-4 to silence the policy gradient on value-only samples
        (enemy-seat transitions in vs-random mode).
        """
        ratio = torch.exp(new_log_probs - old_log_probs)
        eps = self.cfg.clip_range
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantages
        per_sample = -torch.min(surr1, surr2)  # (B,)
        if weight_per is None:
            return per_sample.mean()
        # Weighted mean over policy-active samples; clamp denominator to 1
        # so an all-value-only minibatch returns 0 instead of NaN.
        w_sum = weight_per.sum().clamp_min(1.0)
        return (per_sample * weight_per).sum() / w_sum

    def _value_loss(
        self,
        values: Tensor,     # (B,) or (B, N_VF_CAT) — log_softmax output for cat-vf
        returns: Tensor,    # (B,) scalar return ∈ [-1, 1]
    ) -> Tensor:
        """Value function loss.

        For ``use_cat_vf=True`` we use Ataraxos's **soft** cross-entropy
        formulation (paper Appendix D.4 / `pyengine/core/rl.py` line 558-561):
        the scalar return is encoded as a soft 3-bin distribution by linearly
        interpolating between the two adjacent bin centres, then we minimise
        the cross-entropy with the predicted log-probs.

        Why soft, not hard?
        -------------------
        The previous implementation used ``F.nll_loss`` against a hard
        bucketised target. When the policy was confidently wrong (e.g. 0.005
        prob on the true bin) the per-sample CE was -log(0.005) ≈ 5.3 and
        the gradient through that single sample was ~1/p ≈ 200. In bf16
        under DDP self-play with high-variance returns this produced the
        loss_v=0.218 spike at v33a R69 and ultimately the R107 NaN crash
        (see ``exps/_crash_diagnosis_v33a/train.log``).

        Soft CE bounds the per-sample contribution, smoothes the gradient
        landscape, and matches Ataraxos exactly. Combined with the gradient-
        NaN guard above, this fixes both the immediate symptom and the
        underlying numerical instability.
        """
        if self.cfg.net.use_cat_vf:
            # ``values`` is already log_softmax(B, N_VF_CAT) from JunqiNet._value.
            # Build the soft target: linear interpolation between adjacent bins.
            #
            # Bins are at positions {-1, 0, +1} (the win/draw/loss centres).
            # For r ∈ [-1, 0]: target = [(-r), (1+r), 0]  (loss & draw mix)
            # For r ∈ [0, +1]: target = [0, (1-r), r]     (draw & win mix)
            n_bins = N_VF_CAT
            r = returns.clamp(-1.0, 1.0)
            # Map [-1, +1] → [0, n_bins-1] continuous position.
            pos = (r + 1.0) * 0.5 * (n_bins - 1)
            lower = pos.floor().clamp(max=n_bins - 2).long()    # (B,) bin index of left neighbour
            upper = lower + 1
            upper_w = pos - lower.to(pos.dtype)                  # (B,) ∈ [0, 1]
            lower_w = 1.0 - upper_w
            # Scatter weights into a one-hot-like soft distribution.
            target = torch.zeros(
                (returns.shape[0], n_bins),
                device=returns.device, dtype=values.dtype,
            )
            target.scatter_(1, lower.unsqueeze(1), lower_w.unsqueeze(1).to(values.dtype))
            target.scatter_(1, upper.unsqueeze(1), upper_w.unsqueeze(1).to(values.dtype))
            # Soft cross-entropy. ``values`` is already log-prob.
            return -(target * values).sum(dim=-1).mean()
        else:
            return F.mse_loss(values, returns)

    def _entropy_loss(
        self,
        log_probs: Tensor,   # (B, FLAT_ACTION_DIM)
        legal_mask: Tensor,  # (B, FLAT_ACTION_DIM) bool
        *,
        weight_per: Tensor | None = None,
    ) -> Tensor:
        """Magnet / entropy loss: encourages uniform distribution over legal moves.

        Ataraxos calls this the "magnet" loss.  When ``uniform_magnet=True``
        the loss pushes the policy toward a *uniform* distribution over legal
        actions (maximise entropy relative to uniform), rather than just
        maximising entropy.

        ``weight_per`` (optional, shape (B,) float32 ∈ {0, 1}): per-sample
        weight. Used by F-4 to silence the entropy gradient on value-only
        samples in vs-random mode.
        """
        # Compute entropy: H = -Σ p · log p  (only over legal actions).
        # NOTE: log_probs has -inf entries on illegal actions (legal_mask=False).
        # Naive `probs * log_probs` would give 0 * (-inf) = NaN for those. Use
        # `where` to zero out the contribution of illegal actions explicitly.
        probs = log_probs.exp()
        plogp = torch.where(
            legal_mask,
            probs * log_probs,
            torch.zeros_like(log_probs),
        )
        entropy = -plogp.sum(dim=-1)  # (B,)

        if self.cfg.uniform_magnet:
            # Relative entropy: H(π) − H(uniform) = log|A| − H(π)
            n_legal = legal_mask.float().sum(dim=-1).clamp(min=1.0)  # (B,)
            log_n = n_legal.log()
            # Magnet loss: encourage π → uniform ↔ minimise -H(π) + const
            per_sample = -(entropy - log_n)  # (B,)
        else:
            per_sample = -entropy

        if weight_per is None:
            return per_sample.mean()
        w_sum = weight_per.sum().clamp_min(1.0)
        return (per_sample * weight_per).sum() / w_sum

    def _kl_loss(
        self,
        new_log_probs: Tensor,   # (B, FLAT_ACTION_DIM)
        old_log_probs: Tensor,   # (B, FLAT_ACTION_DIM)
    ) -> Tensor:
        """Forward KL: KL(π_old ‖ π_new) = Σ π_old · (log π_old − log π_new)."""
        old_probs = old_log_probs.exp()
        return (old_probs * (old_log_probs - new_log_probs)).sum(dim=-1).mean()

    # -------------------------------------------------------------------------
    # One gradient update step
    # -------------------------------------------------------------------------

    def _update_step(self, batch: RolloutBatch) -> dict[str, float]:
        """Apply one PPO gradient update step.

        Returns a dict of scalar losses for logging.
        """
        cfg = self.cfg
        temp = self._get_temperature()

        self._policy_for_train.train()

        ctx_device = self.device.type if hasattr(self.device, "type") else str(self.device).split(":")[0]

        # RolloutBufferGPU stores obs_* as fp16 to save memory. When the
        # trainer runs in fp32 (or bf16 on devices without native bf16
        # storage in the buffer), the fp16 input mismatches the weight
        # dtype and conv2d/linear raise. Up-cast to the trainer's compute
        # dtype here so the forward path is dtype-consistent regardless
        # of buffer storage.
        compute_dtype = self._amp_dtype if self._use_amp else torch.float32
        if batch.obs_spatial.dtype != compute_dtype:
            obs_sp_in = batch.obs_spatial.to(compute_dtype)
            obs_gl_in = batch.obs_global.to(compute_dtype)
        else:
            obs_sp_in = batch.obs_spatial
            obs_gl_in = batch.obs_global

        with autocast(device_type=ctx_device, dtype=self._amp_dtype, enabled=self._use_amp):
            out = self._policy_for_train(
                obs_sp_in,
                obs_gl_in,
                batch.legal_mask,
            )
            new_log_probs_all = out["log_probs"]       # (B, FLAT_ACTION_DIM)
            value = out["value"]                        # (B,) or (B, N_VF_CAT)

            # --- NaN/Inf guard (fp16 numeric failure mode) ----------------
            # Under autocast_dtype=float16 a single outlier batch can produce
            # Inf/NaN in log_probs or value, which then poisons the rest of
            # training (the Categorical sampler later crashes with
            # ``probability tensor contains either `inf`, `nan` or element < 0``).
            # We detect it here, skip this minibatch, and increment a counter
            # so train_epoch can surface it. Zero-loss short-circuit keeps
            # the graph / EMA update consistent without stepping the
            # optimiser on poisoned gradients.
            # NaN check: -inf entries in log_probs are LEGAL (they encode the
            # legal_mask: illegal actions get logit=-inf so softmax gives 0
            # probability). Only NaN is a real numeric failure. Using
            # `isfinite` (which rejects both NaN and -inf) was a project-level
            # bug that caused 100% of fp16 minibatches to be short-circuited
            # in v17-v32 (silent ceiling = no actual training happened).
            if (torch.isnan(new_log_probs_all).any()
                    or torch.isnan(value).any()
                    or torch.isinf(value).any()):
                self._nan_skip_count += 1
                # Return a zero-gradient "no-op" result: zero losses, no
                # backward/step. The caller's aggregator averages these as
                # zero entries, which is fine because the frequency is
                # logged separately.
                zero = torch.zeros((), device=self.device)
                self.num_train_step += 1
                lr = self._update_lr()
                return {
                    "train/policy_loss": zero.detach(),
                    "train/value_loss": zero.detach(),
                    "train/entropy_loss": zero.detach(),
                    "train/kl_loss": zero.detach(),
                    "train/total_loss": zero.detach(),
                    "train/temperature": temp,
                    "train/lr": lr,
                    "train/batch_size": int(batch.actions.shape[0]),
                }

            # Log-prob of the chosen action
            act_idx = batch.actions.unsqueeze(1)       # (B, 1)
            new_log_prob = new_log_probs_all.gather(1, act_idx).squeeze(1)  # (B,)

            # Old log-probs (scalar per action, stored in buffer)
            old_log_prob = batch.old_log_probs.to(self.device)

            # Old full distribution for KL (re-derive as detached baseline)
            # We use the stored log_probs as the "old" distribution approximation
            # by treating old_log_prob as the anchor; for KL we approximate
            # using the action taken (importance-weighted).
            # Full old distribution is not stored to save memory; use action KL.
            adv = batch.advantages.to(self.device)
            ret = batch.returns.to(self.device)

            # F-4: ``value_only_mask[i]=True`` means the i-th sample comes from
            # an enemy seat (random policy in vs-random mode). It contributes
            # only to ``value_loss``; its policy / entropy / kl gradient is
            # silenced via a per-sample weight. This keeps V(s) trained on
            # ALL states (so GAE bootstrap on seat-0/2 transitions reads a
            # meaningful V(s_{t+1})) while still gating the policy gradient
            # to policy-controlled samples.
            value_only_mask = getattr(batch, "value_only_mask", None)
            if value_only_mask is None:
                policy_weight_per = None  # legacy path: no masking
            else:
                value_only_mask = value_only_mask.to(self.device)
                # 1 = include in policy/entropy/kl loss; 0 = skip.
                policy_weight_per = (~value_only_mask).to(torch.float32)

            # --- Losses ---
            policy_loss = self._policy_loss(
                new_log_prob, old_log_prob, adv,
                weight_per=policy_weight_per,
            )
            value_loss = self._value_loss(value, ret)
            entropy_loss = self._entropy_loss(
                new_log_probs_all, batch.legal_mask,
                weight_per=policy_weight_per,
            )

            # Approximate KL against old policy.
            # Full-distribution KL requires storing the old log_probs
            # distribution (16641 dims) which is prohibitive on T4.
            # Use π_new-weighted point estimate: Σ π_new · (log π_new − log π_old_action)
            # This is a cheap proxy that penalises large deviations.
            kl_per = (new_log_prob - old_log_prob.detach())
            if policy_weight_per is not None:
                # Mean only over policy-active samples; if all samples are
                # value-only (unlikely), fall back to plain mean (= 0 dist).
                w_sum = policy_weight_per.sum().clamp_min(1.0)
                kl_loss = (kl_per * policy_weight_per).sum() / w_sum
            else:
                kl_loss = kl_per.mean()  # ≈ E[log(π_new/π_old)] at sampled actions

            total_loss = (
                cfg.policy_coef * policy_loss
                + cfg.vf_coef * value_loss
                + temp * entropy_loss
                + cfg.kl_coef * kl_loss
            )

        # Avoid clearing every gradient buffer with a kernel before backward;
        # autograd will allocate/write gradients for all used parameters.
        self.optimizer.zero_grad(set_to_none=True)
        if self._scaler is not None:
            self._scaler.scale(total_loss).backward()
            self._scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._policy_unwrapped.parameters(), cfg.max_grad_norm
            )
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._policy_unwrapped.parameters(), cfg.max_grad_norm
            )

        # ---- Gradient NaN/Inf guard (DDP-safe) ----
        # Without this check, a single bad backward (e.g. value-loss spike on
        # an outlier batch) writes NaN into the params, after which every
        # forward returns NaN, the EMA shadow goes NaN, and eval permanently
        # collapses to 0.5 — exactly the v33a R107 failure mode. Symptoms
        # match: see ``exps/_crash_diagnosis_v33a/train.log`` lines 59-90.
        #
        # DDP correctness: every rank MUST take the same skip-or-step decision.
        # We compute a 0/1 flag locally (1 = bad grad on this rank), then
        # all_reduce(MAX) so any rank's NaN propagates to "all skip".
        bad_grad_local = torch.tensor(
            0 if torch.isfinite(grad_norm) else 1,
            device=self.device, dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad_grad_local, op=dist.ReduceOp.MAX)
        if int(bad_grad_local.item()) != 0:
            # Some rank produced NaN/Inf gradients. Drop the gradients on
            # every rank and DO NOT step the optimiser. Params remain at
            # their pre-backward values, so the next forward will succeed
            # and EMA stays clean.
            self.optimizer.zero_grad(set_to_none=True)
            # Critical: even on skip the GradScaler needs an `update()`
            # call to maintain its scale-factor state machine. Without
            # this the next iteration's `scaler.unscale_(optimizer)`
            # raises "unscale_() has already been called on this
            # optimizer since the last update()". GradScaler.update()
            # internally records that no step occurred and decays the
            # scale (this is the documented "skip" path).
            if self._scaler is not None:
                self._scaler.update()
            self._grad_nan_skip_count += 1
            self.num_train_step += 1
            lr = self._update_lr()
            zero = torch.zeros((), device=self.device)
            return {
                "train/policy_loss": zero.detach(),
                "train/value_loss": zero.detach(),
                "train/entropy_loss": zero.detach(),
                "train/kl_loss": zero.detach(),
                "train/total_loss": zero.detach(),
                "train/temperature": temp,
                "train/lr": lr,
                "train/batch_size": int(batch.actions.shape[0]),
                "train/grad_skip": torch.ones((), device=self.device),
            }

        # Gradients are finite on every rank — safe to step.
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()

        self.num_train_step += 1
        lr = self._update_lr()

        # NOTE: Returning raw GPU tensors (detached) avoids forcing a D2H sync
        # per minibatch.  With 248 minibatches/iteration and 5 scalars each,
        # the old `float(x)` path incurred ~1240 device syncs per iteration,
        # each stalling the CUDA stream for 1-2 ms on T4 (≈2 s lost per iter).
        # The caller (``train_epoch``) aggregates these into python floats
        # once at the end of the epoch.
        return {
            "train/policy_loss": policy_loss.detach(),
            "train/value_loss": value_loss.detach(),
            "train/entropy_loss": entropy_loss.detach(),
            "train/kl_loss": kl_loss.detach(),
            "train/total_loss": total_loss.detach(),
            "train/temperature": temp,
            "train/lr": lr,
            "train/batch_size": int(batch.actions.shape[0]),
        }

    # -------------------------------------------------------------------------
    # Full epoch
    # -------------------------------------------------------------------------

    def train_epoch(
        self,
        rollout: RolloutBuffer,
        *,
        rng: "np.random.Generator | None" = None,  # type: ignore[name-defined]  # noqa: F821
    ) -> dict[str, float]:
        """Run ``num_epochs_per_rollout`` passes over the rollout buffer.

        Parameters
        ----------
        rollout
            A completed :class:`RolloutBuffer` with computed returns.
        rng
            Optional NumPy RNG for reproducible minibatch shuffling.

        Returns
        -------
        Aggregated metrics dict for logging (means over all gradient steps).
        """
        cfg = self.cfg
        all_metrics: list[dict] = []
        if _is_distributed() and getattr(
            rollout,
            "uses_compact_history",
            False,
        ):
            raise RuntimeError(
                "compact_history currently supports single-GPU PPO only"
            )

        for _ in range(cfg.num_epochs_per_rollout):
            batches = rollout.minibatches(
                cfg.minibatch_size,
                shuffle=True,
                rng=rng,
            )
            if _is_distributed():
                # DDP ranks must execute the same number of backward passes.
                # Materialise only in distributed mode so we can all-reduce
                # the local batch counts and truncate to the minimum.
                #
                # In the common single-GPU path, keeping this as a generator
                # is important: RolloutBatch fields are index_select copies,
                # not views. Eagerly retaining 30-60 observation batches can
                # consume many GiB with the 412-channel observation schema.
                materialized = list(batches)
                n_local = torch.tensor(
                    len(materialized),
                    device=self.device,
                    dtype=torch.long,
                )
                dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
                n_to_use = int(n_local.item())
                batches = iter(materialized[:n_to_use])

            for batch in batches:
                metrics = self._update_step(batch)
                all_metrics.append(metrics)
                # Update EMA after each gradient step. Use the unwrapped
                # policy reference; DDP / torch.compile share parameter
                # storage so this picks up the just-stepped values.
                self.ema.update(self._policy_unwrapped)

        self.num_rollout += 1

        # Aggregate — single D2H sync at epoch end.
        # Each `metrics` dict holds GPU tensors (from _update_step) plus
        # python scalars (lr, temperature, batch_size).  We stack the GPU
        # tensors, mean on GPU, then transfer one small scalar each.
        if not all_metrics:
            return {}
        agg: dict[str, float] = {}
        m0 = all_metrics[0]
        for k, v0 in m0.items():
            if isinstance(v0, torch.Tensor):
                # Stack once, .mean() on GPU, single D2H.
                stacked = torch.stack([m[k] for m in all_metrics])
                agg[k] = float(stacked.float().mean().item())
            else:
                # Python scalar path (unchanged): lr, temperature, batch_size.
                agg[k] = float(sum(m[k] for m in all_metrics) / len(all_metrics))
        agg["train/num_updates"] = float(len(all_metrics))
        # Surface cumulative NaN-skip count (from _nan_skip_count counter).
        # Individual per-step 'train/nan_skip' entries live in all_metrics only
        # when that minibatch skipped; aggregating on the cumulative counter
        # is more useful because it persists across epochs.
        agg["train/nan_skip_total"] = float(self._nan_skip_count)
        # Same idea for the gradient-NaN guard (added after the v33a R107
        # crash). When this counter increments faster than 0, recent training
        # has been hitting bad gradients — useful early-warning signal that
        # something is wrong well before win_rate decays.
        agg["train/grad_skip_total"] = float(self._grad_nan_skip_count)
        agg.update(rollout.stats())
        return agg

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            # NOTE: Always saves the **unwrapped** JunqiNet state. DDP and
            # torch.compile both share parameter storage with this module;
            # cross-world-size resume works because the saved keys never
            # include the ``module.`` / ``_orig_mod.`` prefixes.
            "policy": self._policy_unwrapped.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_train_step": self.num_train_step,
            "num_rollout": self.num_rollout,
            "cfg": self.cfg,
            "world_size_at_save": _world_size(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self._policy_unwrapped.load_state_dict(sd["policy"])
        self.ema.load_state_dict(sd["ema"])
        self.optimizer.load_state_dict(sd["optimizer"])
        self.num_train_step = sd["num_train_step"]
        self.num_rollout = sd["num_rollout"]

    def save(self, path: str) -> None:
        """Save checkpoint to ``path``."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        """Load checkpoint from ``path``."""
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(sd)
