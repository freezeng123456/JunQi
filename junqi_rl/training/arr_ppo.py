"""junqi_rl.training.arr_ppo — PPO trainer for the arrangement network.

Mirrors Ataraxos ``pyengine/core/rl.py::arr_train`` (lines 616-700) with
JunQi adaptations:

* DDP wraps the train-time network and matches minibatch counts across ranks.
* Forward pass takes ``(seq, seat_idx)`` — the shared net conditions on
  per-sample seat via embedding.
* No ``force_handedness`` / flip — JunQi has no left-right symmetry.

4-term loss
-----------
    loss = arr_policy_coef * policy_loss    (PPO clip surrogate)
         + arr_vf_coef     * value_loss     (MSE vs val_est, or cat-CE)
         + arr_ent_pred_coef * entropy_loss (MSE of ent_pred vs reg_val_est)
         + arr_kl_coef     * kl_loss        (KL(new || old_sampled))

All four per-epoch means + clip fraction + grad norm + LR are returned
as stats.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW

from junqi_rl.arrangement.buffer import ArrangementBuffer, Batch
from junqi_rl.networks.arrangement_net import ArrangementNet


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass
class ArrangementPPOConfig:
    """Hyper-parameters for :class:`ArrangementPPOTrainer`."""

    lr: float = 3e-4
    """AdamW learning rate for the arrangement net."""

    weight_decay: float = 0.0
    """AdamW weight decay."""

    clip_range: float = 0.2
    """PPO policy clip range."""

    policy_coef: float = 1.0
    """Coefficient on policy (surrogate) loss."""

    vf_coef: float = 0.5
    """Coefficient on value loss."""

    ent_pred_coef: float = 0.5
    """Coefficient on ent_pred MSE loss."""

    kl_coef: float = 0.01
    """Coefficient on KL(new || old_sampled) regulariser."""

    max_grad_norm: float = 1.0
    """Gradient clipping threshold."""

    batch_size: int = 256
    """Number of arrangements per minibatch."""

    num_epoch_per_train: int = 4
    """Number of passes over the ready-rows of the buffer per train_epoch()."""

    autocast_dtype: str = "float32"
    """Use fp16/bf16 via torch.amp.autocast inside the forward pass.
    Accepted values: 'float32', 'float16', 'bfloat16'. Default fp32 for
    stability; the collector is the latency-critical path, not the
    arrangement trainer (buffer sizes ~1k rows per training phase)."""


class ArrangementPPOTrainer:
    """PPO trainer for :class:`ArrangementNet`.

    Lifecycle
    ---------
    ::

        trainer = ArrangementPPOTrainer(net, cfg)
        # After the buffer has ready rows AND process_data() has filled val_est/adv_est:
        stats = trainer.train_epoch(buffer)
        # stats: {'arr_train/policy_loss': float, ..., 'arr_train/g_norm': float}

    The trainer owns an AdamW optimizer over the net's parameters. It does
    NOT own an EMA — that's expected to be an external object that tracks
    the net's parameters and is updated by the caller after each training
    step (to match Ataraxos's ``self.arr_ema_policy.update()`` pattern).
    """

    def __init__(
        self,
        net: ArrangementNet,
        cfg: ArrangementPPOConfig | None = None,
    ) -> None:
        # ``self.net`` always holds the **unwrapped** ArrangementNet so that
        # external callers (EMAPolicy, save/load, generate_arrangements which
        # calls model.forward directly) don't need to know whether DDP is on.
        self.net = net
        self.cfg = cfg or ArrangementPPOConfig()
        self.optim = AdamW(
            self.net.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            foreach=True,
        )
        # Cumulative count of arr-PPO minibatches whose optimiser.step() was
        # skipped because gradient norm was non-finite. Same rationale as
        # ``PPOTrainer._grad_nan_skip_count`` — added after v33a R107 crash.
        self._grad_nan_skip_count: int = 0
        # DDP wrapper only used inside _step's forward+backward so the
        # gradient hooks fire exactly where they should.
        if _is_distributed():
            param_device = next(self.net.parameters()).device
            ddp_kwargs: dict[str, Any] = {"find_unused_parameters": False}
            if param_device.type == "cuda" and param_device.index is not None:
                ddp_kwargs["device_ids"] = [param_device.index]
                ddp_kwargs["output_device"] = param_device.index
            self._net_for_train: torch.nn.Module = DistributedDataParallel(
                self.net, **ddp_kwargs,
            )
        else:
            self._net_for_train = self.net
        self._total_batches: int = 0

    def train_epoch(
        self,
        buffer: ArrangementBuffer,
        *,
        on_optimizer_step: Callable[[ArrangementNet], None] | None = None,
    ) -> dict[str, float]:
        """Run ``cfg.num_epoch_per_train`` passes over the buffer's ready rows.

        Returns a dict of averaged per-stat values over ALL minibatches in
        this call. Empty dict if the buffer had zero ready rows.

        Under DDP, the "is the buffer ready" decision is made COLLECTIVELY:
        if any rank has zero ready rows we no-op on every rank. This avoids
        the deadlock where rank A enters the gradient backward+all_reduce
        loop while rank B short-circuits on an empty buffer (different
        terminations across rank-local envs is normal at rollout 0 — games
        last 200+ moves and not every rank's envs finish their first game
        within steps_per_env).
        """
        if buffer.need_arrangements:
            raise RuntimeError("Buffer has no arrangements; call add_arrangements first")
        any_local = bool(buffer.ready_flags.any())
        if _is_distributed():
            ref_param = next(self.net.parameters())
            local_ok = torch.tensor(
                1 if any_local else 0,
                device=ref_param.device, dtype=torch.long,
            )
            # MIN: any rank with 0 ready rows pulls the whole cluster to 0.
            dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
            if int(local_ok.item()) == 0:
                return {}
        elif not any_local:
            return {}

        # Ensure net is in training mode (touches both wrapped and unwrapped
        # since they share parameters/buffers).
        self._net_for_train.train()

        stats: dict[str, list[float]] = {
            "arr_train/policy_loss": [],
            "arr_train/value_loss": [],
            "arr_train/entropy_loss": [],
            "arr_train/kl_loss": [],
            "arr_train/total_loss": [],
            "arr_train/entropy": [],
            "arr_train/clip_fraction": [],
            "arr_train/g_norm": [],
            "arr_train/lr": [],
            "arr_train/n_batches": [],
            "arr_train/grad_skip": [],  # 1.0 per minibatch where grad was NaN
        }

        # DDP requires the same number of forward+backward passes on every
        # rank or all-reduce deadlocks. We materialise this epoch's
        # minibatches into a list, all_reduce(MIN) the per-rank counts, and
        # truncate. Buffer.sample() returns Batch objects on-device that hold
        # index_select views — so the materialised list is light (~kB each).
        for _epoch in range(self.cfg.num_epoch_per_train):
            batches = list(buffer.sample(self.cfg.batch_size))
            if _is_distributed():
                ref_param = next(self.net.parameters())
                n_local = torch.tensor(
                    len(batches), device=ref_param.device, dtype=torch.long,
                )
                dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
                batches = batches[: int(n_local.item())]
            for batch in batches:
                stepped = self._step(batch, stats)
                if stepped and on_optimizer_step is not None:
                    on_optimizer_step(self.net)

        # Aggregate.
        out = {k: (sum(v) / len(v)) for k, v in stats.items() if len(v) > 0 and k != "arr_train/n_batches"}
        out["arr_train/n_batches"] = float(sum(stats["arr_train/n_batches"]))
        # Cumulative grad-NaN-skip count, persists across rollouts so a slow
        # leak (e.g. 1 skip every 10 rollouts) is visible.
        out["arr_train/grad_skip_total"] = float(self._grad_nan_skip_count)
        return out

    # ----------------------------------------------------------- single batch

    def _step(self, batch: Batch, stats: dict[str, list[float]]) -> bool:
        """One PPO update; return True iff ``optimizer.step()`` ran."""
        cfg = self.cfg
        net = self._net_for_train  # DDP-wrapped (or plain) module for fwd+bwd
        # Use the unwrapped reference for non-tensor metadata access.
        # ``DistributedDataParallel`` only forwards tensor / parameter
        # attributes; ``net.cfg`` (a dataclass) is not visible through it
        # unless we go via ``net.module.cfg``. Reading ``self.net.cfg``
        # gives the same answer with no DDP-vs-single-process branching.
        unwrapped = self.net

        # --- Forward ---
        dtype_map = {"float32": torch.float32,
                     "float16": torch.float16,
                     "bfloat16": torch.bfloat16}
        amp_dtype = dtype_map.get(cfg.autocast_dtype, torch.float32)
        use_amp = amp_dtype != torch.float32 and batch.arrangements.is_cuda
        with autocast(
            device_type="cuda" if batch.arrangements.is_cuda else "cpu",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            # Net forward: samples (B, 30, 13) one-hot, seats (B,) int64.
            out = net(batch.arrangements, batch.seat_idx)
            logits = out["logits"]         # (B, T', 13)
            values_pred = out["value"]     # (B, T', C|1)
            regs_pred = out["ent_pred"]    # (B, T', 1)

            # --- Policy (PPO clip) ---
            # We slice the first T=30 rows (the ones aligned with the buffer).
            T = batch.arrangements.size(1)
            log_probs = F.log_softmax(logits[:, :T], dim=-1)   # (B, 30, 13)
            chosen = batch.arrangements.argmax(dim=-1, keepdim=True)  # (B, 30, 1)
            log_prob_new = log_probs.gather(-1, chosen).squeeze(-1)    # (B, 30)
            log_prob_old = batch.log_probs.gather(-1, chosen).squeeze(-1)
            ratio = torch.exp(log_prob_new - log_prob_old)             # (B, 30)

            advantages = batch.advantages                              # (B, 30)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(
                ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
            )
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # --- KL (new || old) across the full vocab ---
            # KL = Σ p_new * (log p_new - log p_old).
            # Evaluate in probability space to match Ataraxos (eq. batch.log_probs
            # is already log-softmax, so log_probs.exp() = p_new).
            kl_loss = (
                log_probs.exp() * (log_probs - batch.log_probs)
            ).sum(dim=-1).mean()

            # --- Value loss ---
            if unwrapped.cfg.use_cat_vf:
                # batch.returns has shape (B, 30, C) — a soft target (val_est of
                # the categorical distribution). Categorical cross-entropy loss.
                value_pred_slice = values_pred[:, :T]
                value_loss = (
                    -(batch.returns * F.log_softmax(value_pred_slice, dim=-1))
                    .sum(-1).mean()
                )
            else:
                value_pred_slice = values_pred[:, :T].squeeze(-1)
                value_loss = F.mse_loss(value_pred_slice, batch.returns)

            # --- ent_pred (regularised advantage target) ---
            reg_pred_slice = regs_pred[:, :T].squeeze(-1)               # (B, 30)
            entropy_loss = F.mse_loss(reg_pred_slice, batch.reg_returns)

            # --- Total ---
            total_loss = (
                cfg.policy_coef * policy_loss
                + cfg.vf_coef * value_loss
                + cfg.ent_pred_coef * entropy_loss
                + cfg.kl_coef * kl_loss
            )

        # --- Backprop ---
        self.optim.zero_grad(set_to_none=True)
        total_loss.backward()
        # Grad clip on the unwrapped params (they share storage with DDP's,
        # so either works, but unwrapped is the safe canonical choice).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), cfg.max_grad_norm,
        )

        # ---- Gradient NaN/Inf guard (DDP-safe) ----
        # See PPOTrainer._update_step for the full rationale. Briefly: a
        # single bad backward writes NaN into the params, after which every
        # subsequent forward returns NaN, the EMA shadow goes NaN, and the
        # arrangement pool quality decays to random. This guard short-
        # circuits the optimiser.step() so params stay clean.
        bad_grad_local = torch.tensor(
            0 if torch.isfinite(grad_norm) else 1,
            device=grad_norm.device, dtype=torch.long,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad_grad_local, op=dist.ReduceOp.MAX)
        if int(bad_grad_local.item()) != 0:
            self.optim.zero_grad(set_to_none=True)
            self._grad_nan_skip_count += 1
            stats["arr_train/grad_skip"].append(1.0)
            stats["arr_train/n_batches"].append(1.0)
            self._total_batches += 1
            return False

        self.optim.step()

        # --- Record stats ---
        stats["arr_train/policy_loss"].append(float(policy_loss.detach()))
        stats["arr_train/value_loss"].append(float(value_loss.detach()))
        stats["arr_train/entropy_loss"].append(float(entropy_loss.detach()))
        stats["arr_train/kl_loss"].append(float(kl_loss.detach()))
        stats["arr_train/total_loss"].append(float(total_loss.detach()))
        stats["arr_train/entropy"].append(float(-log_prob_new.sum(dim=-1).mean().detach()))
        stats["arr_train/clip_fraction"].append(
            float(((ratio - 1.0).abs() > cfg.clip_range).float().mean().detach())
        )
        stats["arr_train/g_norm"].append(float(grad_norm))
        stats["arr_train/lr"].append(float(self.optim.param_groups[0]["lr"]))
        stats["arr_train/n_batches"].append(1.0)
        self._total_batches += 1
        return True

    # ----------------------------------------------------------- checkpointing

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state for the arrangement trainer."""
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optim.state_dict(),
            "grad_nan_skip_count": self._grad_nan_skip_count,
            "total_batches": self._total_batches,
            "cfg": self.cfg,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore checkpoint state for the arrangement trainer."""
        self.net.load_state_dict(sd["net"])
        self.optim.load_state_dict(sd["optimizer"])
        self._grad_nan_skip_count = int(sd.get("grad_nan_skip_count", 0))
        self._total_batches = int(sd.get("total_batches", 0))


__all__ = [
    "ArrangementPPOConfig",
    "ArrangementPPOTrainer",
]
