"""Supervised BeliefNet trainer using current parameters for inference.

Cross-entropy is computed only where enemy labels are supplied. Labels may
come from public reveal tracking or simulator truth used solely as a training
target; hidden truth is never added to the observation. The diagnostic baseline
is uniform over all 12 types, not a remaining-inventory prior.

The optimizer defaults are local engineering choices. This is supervised
learning on a replay buffer, with no policy-gradient loss or parameter EMA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from junqi_rl.belief.buffer import BeliefBuffer, BeliefSample
from junqi_rl.networks.belief_net import BeliefNet, N_BELIEF_TYPES


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


__all__ = [
    "BeliefPPOConfig",
    "BeliefPPOTrainer",
    "compute_belief_loss",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BeliefPPOConfig:
    """Local hyper-parameters for supervised belief learning."""

    lr: float = 5e-5
    """Adam learning rate."""

    weight_decay: float = 0.0
    """Adam weight decay (0 = pure Adam)."""

    max_grad_norm: float = 0.5
    """Gradient clipping L2 norm."""

    batch_size: int = 64
    """Minibatch size during belief training."""

    epochs_per_rollout: int = 2
    """Number of minibatches drawn per ``train_epoch`` invocation. With
    batch_size=64 and epochs_per_rollout=2, each main-PPO rollout
    contributes 128 samples worth of belief gradient steps."""

    autocast_dtype: str = "float32"
    """Autocast precision for the forward pass. Use "float16" on T4 for
    throughput, "bfloat16" on A100/H100. Set to "float32" to disable
    autocast entirely (safest default for supervised learning)."""


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_belief_loss(
    logits: Tensor,               # (B, 289, 12)
    true_type_idx: Tensor,        # (B, 289) int64; -1 for unknown
    enemy_mask: Tensor,           # (B, 289) bool
    *,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Per-cell CE + uniform-baseline diagnostic.

    Only cells with ``(true_type_idx >= 0) & enemy_mask`` contribute to
    the gradient.

    Parameters
    ----------
    logits
        ``(B, 289, 12)`` raw logits from :class:`BeliefNet.forward`.
    true_type_idx
        ``(B, 289)`` int64 where each entry is either ``-1`` (unknown,
        sentinel) or a valid vocab index in ``[0, N_BELIEF_TYPES)``.
    enemy_mask
        ``(B, 289)`` bool indicating which cells hold enemy pieces.

    Returns
    -------
    dict with differentiable ``ce_loss`` plus diagnostic scalars:
    * ``ce_loss``       — main training signal (per-cell CE, masked mean).
    * ``uniform_ce``    — CE of the uniform-over-12 baseline (constant
                           ≈ log(12) ≈ 2.48).
    * ``uniform_kl``    — CE_predicted − CE_uniform (should go NEGATIVE
                           once the net beats uniform).
    * ``n_revealed``    — number of cells that contributed to the loss.
    * ``accuracy``      — top-1 classification accuracy over revealed cells.
    """
    if logits.dim() != 3 or logits.shape[-1] != N_BELIEF_TYPES:
        raise ValueError(
            f"logits must be (B, 289, {N_BELIEF_TYPES}); got {tuple(logits.shape)}"
        )
    if true_type_idx.shape != logits.shape[:-1]:
        raise ValueError(
            f"true_type_idx must be {logits.shape[:-1]}; got {tuple(true_type_idx.shape)}"
        )
    if enemy_mask.shape != logits.shape[:-1]:
        raise ValueError(
            f"enemy_mask must be {logits.shape[:-1]}; got {tuple(enemy_mask.shape)}"
        )

    # log_softmax over type axis.
    log_prob = F.log_softmax(logits, dim=-1)           # (B, 289, 12)

    # Per-cell CE: gather the log-prob of the true type. For unknown cells
    # we clamp to 0 so gather doesn't choke; the mask below zeroes them.
    safe_label = true_type_idx.clamp(min=0).unsqueeze(-1)      # (B, 289, 1)
    ce_per_cell = -log_prob.gather(-1, safe_label).squeeze(-1)  # (B, 289)

    revealed = (true_type_idx >= 0) & enemy_mask                # (B, 289) bool
    n_revealed = revealed.sum().float()

    ce_loss = (ce_per_cell * revealed.float()).sum() / n_revealed.clamp(min=1.0)

    # --- Diagnostics ---
    # Uniform baseline: CE under p = 1/12 for each of 12 types.
    uniform_ce = torch.full_like(ce_loss, math.log(N_BELIEF_TYPES))

    # uniform_kl > 0 means the net is WORSE than uniform; < 0 means BETTER.
    uniform_kl = torch.where(n_revealed > 0, ce_loss - uniform_ce, 0.0)

    # Accuracy: argmax of logits vs true type, masked.
    with torch.no_grad():
        pred = logits.argmax(dim=-1)                       # (B, 289)
        correct = ((pred == true_type_idx) & revealed).float().sum()
        accuracy = correct / n_revealed.clamp(min=1.0)

    return {
        "ce_loss": ce_loss,
        "uniform_ce": uniform_ce.detach(),
        "uniform_kl": uniform_kl.detach(),
        "n_revealed": n_revealed.detach(),
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class BeliefPPOTrainer:
    """Supervised-only trainer for BeliefNet.

    Not actually PPO — the name mirrors :class:`ArrangementPPOTrainer` /
    :class:`PPOTrainer` for codebase consistency. The actual loss is
    pure cross-entropy.

    Lifecycle
    ~~~~~~~~~

    .. code-block:: python

        net = BeliefNet(BeliefNetConfig())
        trainer = BeliefPPOTrainer(net, BeliefPPOConfig())
        # During rollout, fill the buffer via reveal_tracker callbacks.
        # After each rollout:
        metrics = trainer.train_epoch(belief_buffer)
        # belief_buffer is the BeliefBuffer from junqi_rl.belief.buffer.
    """

    def __init__(
        self,
        net: BeliefNet,
        cfg: BeliefPPOConfig | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cfg = cfg or BeliefPPOConfig()
        self.device = torch.device(device)

        net = net.to(self.device)
        # ``self.net`` is the unwrapped BeliefNet (used by save/load,
        # and inference paths like ``refresh_beliefs_neural``). DDP wraps a
        # parallel copy whose only purpose is the supervised CE forward/
        # backward inside ``_update_step``.
        self.net = net
        if _is_distributed():
            ddp_kwargs: dict[str, Any] = {"find_unused_parameters": False}
            if self.device.type == "cuda" and self.device.index is not None:
                ddp_kwargs["device_ids"] = [self.device.index]
                ddp_kwargs["output_device"] = self.device.index
            self._net_for_train: torch.nn.Module = DistributedDataParallel(
                net, **ddp_kwargs,
            )
        else:
            self._net_for_train = net

        self.optimizer = torch.optim.Adam(
            net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.num_train_step: int = 0
        self.num_rollout: int = 0
        self._nan_skip_count: int = 0
        self._empty_skip_count: int = 0
        self._grad_skip_count: int = 0

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.cfg.autocast_dtype not in dtype_map:
            raise ValueError(
                f"autocast_dtype must be one of {list(dtype_map)}; "
                f"got {self.cfg.autocast_dtype!r}"
            )
        self._amp_dtype = dtype_map[self.cfg.autocast_dtype]
        self._use_amp = (
            self.cfg.autocast_dtype in ("float16", "bfloat16")
            and self.device.type == "cuda"
        )

    # ------------------------------------------------------------------
    # One gradient update step
    # ------------------------------------------------------------------

    def _update_step(self, batch: BeliefSample) -> dict[str, Tensor]:
        cfg = self.cfg
        # A zero-gradient Adam step can still move parameters through stored
        # momentum. Empty supervision must skip the forward pass and optimizer.
        # Decide before DDP forward, on every rank, to keep collectives aligned.
        n_labels = ((batch.true_type_idx >= 0) & batch.enemy_mask).sum().to(self.device)
        has_labels = (n_labels > 0).to(torch.long)
        if _is_distributed():
            dist.all_reduce(has_labels, op=dist.ReduceOp.MIN)
        if not bool(has_labels.item()):
            self._empty_skip_count += 1
            self.optimizer.zero_grad(set_to_none=True)
            zero = torch.zeros((), device=self.device)
            return {
                "belief_train/ce_loss": zero,
                "belief_train/uniform_kl": zero,
                "belief_train/accuracy": zero,
                "belief_train/n_revealed": n_labels.float(),
                "belief_train/empty_skip": zero + 1,
                "belief_train/updated": zero,
            }
        self._net_for_train.train()

        ctx_device = self.device.type if hasattr(self.device, "type") else str(self.device).split(":")[0]

        # Tensors are materialised on self.device already by BeliefBuffer.sample.
        with torch.amp.autocast(
            device_type=ctx_device, dtype=self._amp_dtype, enabled=self._use_amp,
        ):
            out = self._net_for_train(
                batch.obs_spatial.to(self.device),
                seat_idx=batch.seat_idx.to(self.device),
            )
            logits = out["logits"]
            losses = compute_belief_loss(
                logits,
                batch.true_type_idx.to(self.device),
                batch.enemy_mask.to(self.device),
            )
            ce_loss = losses["ce_loss"]

        # NaN-guard (DDP-safe).  One rank hitting non-finite CE while others
        # proceed would desync all_reduce in DDP — mirror ppo.py's grad guard:
        # every rank must take the same skip-or-step decision.
        finite_local = torch.tensor(
            1 if torch.isfinite(ce_loss) else 0,
            device=self.device,
            dtype=torch.long,
        )
        if _is_distributed():
            dist.all_reduce(finite_local, op=dist.ReduceOp.MIN)
        all_finite = int(finite_local.item()) != 0

        if not all_finite:
            self._nan_skip_count += 1
            # Zero-loss backward so every rank still enters the same collective.
            self.optimizer.zero_grad()
            zero_loss = sum(
                (p.sum() * 0.0) for p in self._net_for_train.parameters()
            )
            zero_loss.backward()
            zero = torch.zeros((), device=self.device)
            return {
                "belief_train/ce_loss": zero.detach(),
                "belief_train/uniform_kl": zero.detach(),
                "belief_train/accuracy": zero.detach(),
                "belief_train/n_revealed": losses["n_revealed"].detach(),
                "belief_train/nan_skip": torch.ones((), device=self.device),
                "belief_train/updated": zero,
            }

        self.optimizer.zero_grad()
        ce_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
        grad_finite = torch.isfinite(grad_norm).to(device=self.device, dtype=torch.long)
        if _is_distributed():
            dist.all_reduce(grad_finite, op=dist.ReduceOp.MIN)
        if not bool(grad_finite.item()):
            self._grad_skip_count += 1
            self.optimizer.zero_grad(set_to_none=True)
            zero = torch.zeros((), device=self.device)
            return {
                "belief_train/ce_loss": ce_loss.detach(),
                "belief_train/uniform_kl": losses["uniform_kl"].detach(),
                "belief_train/accuracy": losses["accuracy"].detach(),
                "belief_train/n_revealed": losses["n_revealed"].detach(),
                "belief_train/grad_skip": zero + 1,
                "belief_train/updated": zero,
            }
        self.optimizer.step()

        self.num_train_step += 1

        return {
            "belief_train/ce_loss": ce_loss.detach(),
            "belief_train/uniform_kl": losses["uniform_kl"].detach(),
            "belief_train/accuracy": losses["accuracy"].detach(),
            "belief_train/n_revealed": losses["n_revealed"].detach(),
            "belief_train/grad_norm": grad_norm.detach(),
            "belief_train/updated": torch.ones((), device=self.device),
        }

    # ------------------------------------------------------------------
    # Full epoch
    # ------------------------------------------------------------------

    def train_epoch(self, buffer: BeliefBuffer) -> dict[str, float]:
        """Run ``epochs_per_rollout`` minibatch updates from ``buffer``.

        Returns an aggregated-mean dict ready for logging. Keys are
        namespaced under ``belief_train/``.
        """
        cfg = self.cfg
        # DDP correctness: all ranks must execute the same number of
        # forward+backward passes. We poll min-buffer-size across ranks; if
        # any rank's buffer is empty we no-op on every rank this round.
        local_size = len(buffer)
        if _is_distributed():
            n_local = torch.tensor(
                local_size, device=self.device, dtype=torch.long,
            )
            dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
            min_size = int(n_local.item())
        else:
            min_size = local_size
        if min_size == 0:
            return {
                "belief_train/ce_loss": float("nan"),
                "belief_train/uniform_kl": float("nan"),
                "belief_train/accuracy": float("nan"),
                "belief_train/n_revealed": 0.0,
                "belief_train/num_updates": 0.0,
                "belief_train/buffer_size": float(local_size),
            }

        all_metrics: list[dict[str, Tensor]] = []
        for batch in buffer.sample(
            batch_size=cfg.batch_size,
            device=self.device,
            n_batches=cfg.epochs_per_rollout,
        ):
            metrics = self._update_step(batch)
            for key in ("empty_skip", "nan_skip", "grad_skip"):
                metrics.setdefault("belief_train/" + key, torch.zeros((), device=self.device))
            all_metrics.append(metrics)

        self.num_rollout += 1

        # Aggregate: mean over updates, single D2H sync at end.
        agg: dict[str, float] = {}
        if all_metrics:
            keys = set().union(*(m.keys() for m in all_metrics))
            for k in keys:
                v0 = next(m[k] for m in all_metrics if k in m)
                if isinstance(v0, torch.Tensor):
                    stacked = torch.stack([m[k] for m in all_metrics if k in m])
                    agg[k] = float(stacked.float().mean().item())
                else:
                    agg[k] = float(v0)
        agg["belief_train/num_updates"] = sum(float(m["belief_train/updated"].item()) for m in all_metrics)
        agg["belief_train/num_attempts"] = float(len(all_metrics))
        agg["belief_train/buffer_size"] = float(len(buffer))
        return agg

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "num_train_step": self.num_train_step,
            "num_rollout": self.num_rollout,
            "empty_skip_count": self._empty_skip_count,
            "nan_skip_count": self._nan_skip_count,
            "grad_skip_count": self._grad_skip_count,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.net.load_state_dict(sd["net"])
        # Legacy EMA entries are ignored; only the trained net is restored.
        self.optimizer.load_state_dict(sd["optimizer"])
        self.num_train_step = int(sd["num_train_step"])
        self.num_rollout = int(sd["num_rollout"])
        self._empty_skip_count = int(sd.get("empty_skip_count", 0))
        self._nan_skip_count = int(sd.get("nan_skip_count", 0))
        self._grad_skip_count = int(sd.get("grad_skip_count", 0))
