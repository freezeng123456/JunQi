"""junqi_rl.arrangement.buffer — Ataraxos-faithful ArrangementBuffer for JunQi.

Stores generated arrangements + creation-time (log_probs, value, ent_pred)
plus terminal rewards from any games that used those arrangements. Provides
TD(λ)/GAE(λ) + regularised-advantage processing identical to Ataraxos
(``pyengine/arrangement/buffer.py``).

Adaptations for JunQi
---------------------
* **No pystratego** — we hash vocab-idx lineups with ``blake2b`` (digest 8
  bytes → int64) for dedup. Collisions within ``storage_duration`` are
  astronomically unlikely.
* **No flip / handedness** — JunQi 4-seat layout has no left-right symmetry,
  so ``needs_flip`` is dropped entirely.
* **Per-seat conditioning** — we additionally store ``seat_idx`` so the
  training forward pass can re-use the right seat embedding.
* **Categorical aggregation** — ``[-1, 0, 1]`` (lose/draw/win) as default,
  matching the scalar reward domain.

The lifecycle mirrors Ataraxos:

    buffer = ArrangementBuffer(storage_duration=..., ...)
    # ---- outer loop ----
    buffer.add_arrangements(samples, values, ents, log_probs, seat_idx, step=step)
    # ---- inner rollout loop: after each env.step ----
    buffer.add_rewards(env_arrangements, env_seats, is_terminal, rewards)
    # ---- training ----
    stats = buffer.process_data(td_lambda, gae_lambda, reg_temp, reg_norm)
    for batch in buffer.sample(batch_size):
        ... PPO update ...
    # ---- maintenance ----
    buffer.filter(current_step)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Generator

import torch
import torch.nn.functional as F
from torch import Tensor

from junqi_rl.networks.arrangement_net import (
    ARRANGEMENT_SIZE,
    N_PIECE_TYPE_WITH_NONE,
    N_VF_CAT_DEFAULT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default categorical-VF aggregation: maps (lose, draw, win) probs to a scalar
# reward expectation in [-1, 1]. Used when ``use_cat_vf=True`` to compute
# scalar advantages from categorical value distributions.
DEFAULT_CATEGORICAL_AGGREGATION: Tensor = torch.tensor(
    [-1.0, 0.0, 1.0], dtype=torch.float32
)


def _arrangement_ids(arr_idx: Tensor) -> list[int]:
    """Hash each lineup to a stable 64-bit int via blake2b.

    ``arr_idx`` is ``(N, ARRANGEMENT_SIZE)`` uint8 (vocab indices, 0..12).
    Identical lineups get identical IDs regardless of the seat.

    Using blake2b rather than Python's ``hash(tuple(...))`` keeps IDs stable
    across Python invocations (``hash()`` is randomised) — useful for
    offline debugging & for the storage-duration dedup across refreshes.
    """
    idx_np = arr_idx.cpu().to(torch.uint8).numpy()
    out: list[int] = []
    for row in idx_np:
        digest = hashlib.blake2b(row.tobytes(), digest_size=8).digest()
        out.append(int.from_bytes(digest, "little", signed=False))
    return out


def _mark_most_recent_appearance(values: list[int], timestamps: Tensor) -> Tensor:
    """Return a (N,) bool mask selecting exactly one row per unique value.

    The selected row is always the one with the largest timestamp; ties are
    broken by natural iteration order (the first one seen). Matches
    Ataraxos ``mark_most_recent_appearance`` semantics.
    """
    if timestamps.ndim != 1 or timestamps.size(0) != len(values):
        raise ValueError(
            f"timestamps must be 1-D with length {len(values)}, "
            f"got shape {tuple(timestamps.shape)}"
        )

    max_ts: dict[int, int] = {}
    ts_list = timestamps.tolist()
    for v, t in zip(values, ts_list):
        if v not in max_ts or t > max_ts[v]:
            max_ts[v] = int(t)

    seen: set[int] = set()
    mask = torch.zeros(len(values), dtype=torch.bool)
    for i, (v, t) in enumerate(zip(values, ts_list)):
        if int(t) == max_ts[v] and v not in seen:
            seen.add(v)
            mask[i] = True
    return mask


# ---------------------------------------------------------------------------
# Batch dataclass
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    """One minibatch yielded by :meth:`ArrangementBuffer.sample`."""

    arrangements: Tensor   # (B, 30, 13) one-hot float
    seat_idx: Tensor       # (B,) int64
    log_probs: Tensor      # (B, 30, 13) float — log-softmax at generation time
    returns: Tensor        # (B, *value_shape) float — val_est (training target for VF head)
    reg_returns: Tensor    # (B, 30) float — reg_val_est (training target for ent_pred)
    advantages: Tensor     # (B, 30) float — scalar advantage per slot (for policy loss)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------


class ArrangementBuffer:
    """Fixed-duration replay of generated arrangements + their game outcomes.

    See module docstring for the intended lifecycle.
    """

    def __init__(
        self,
        *,
        storage_duration: int,
        device: torch.device | str = "cpu",
        use_cat_vf: bool = True,
        n_vf_cat: int = N_VF_CAT_DEFAULT,
        categorical_aggregation: Tensor | None = None,
    ) -> None:
        if not isinstance(storage_duration, int):
            raise TypeError(f"storage_duration must be int, got {type(storage_duration)}")
        if storage_duration <= 0:
            raise ValueError(f"storage_duration must be positive, got {storage_duration}")

        device = torch.device(device) if not isinstance(device, torch.device) else device

        self.storage_duration = storage_duration
        self.device = device
        self.use_cat_vf = use_cat_vf
        self.n_vf_cat = n_vf_cat

        if use_cat_vf:
            if categorical_aggregation is None:
                categorical_aggregation = DEFAULT_CATEGORICAL_AGGREGATION
            if categorical_aggregation.shape != (n_vf_cat,):
                raise ValueError(
                    f"categorical_aggregation must have shape ({n_vf_cat},), "
                    f"got {categorical_aggregation.shape}"
                )
            self.categorical_aggregation = categorical_aggregation.to(device)
            self.value_shape: tuple[int, ...] = (ARRANGEMENT_SIZE, n_vf_cat)
            self.reward_shape: tuple[int, ...] = (n_vf_cat,)
            self.count_shape: tuple[int, ...] = (1,)
        else:
            self.categorical_aggregation = None
            self.value_shape = (ARRANGEMENT_SIZE,)
            self.reward_shape = ()
            self.count_shape = ()

        self.need_arrangements: bool = True

        # Tensors updated via add_arrangements()
        self.arrangements = torch.zeros(
            (0, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE), device=device,
        )
        self.seat_idx = torch.zeros(0, device=device, dtype=torch.long)
        self.values = torch.zeros((0, *self.value_shape), device=device, dtype=torch.float32)
        self.ents = torch.zeros((0, ARRANGEMENT_SIZE), device=device, dtype=torch.float32)
        self.log_probs = torch.zeros(
            (0, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE), device=device, dtype=torch.float32,
        )
        self.step_added = torch.zeros(0, device=device, dtype=torch.long)
        # Hash of each row → int64 (used for dedup and env→buffer lookup)
        self._row_ids: list[int] = []

        # Tensors sized against the buffer current-size, re-allocated inside
        # add_arrangements(). Kept as attrs so callers can introspect.
        self.counts: Tensor
        self.rewards: Tensor
        self.ready_flags: Tensor
        self.adv_est: Tensor
        self.val_est: Tensor
        self.reg_val_est: Tensor

        self._lookup_index: dict[int, int] = {}  # row_id → row in buffer

    # --------------------------------------------------------------- properties

    def __len__(self) -> int:
        return int(self.arrangements.size(0))

    @property
    def n_ready(self) -> int:
        return int(self.ready_flags.sum().item()) if len(self) else 0

    # --------------------------------------------------------------- add_arrangements

    @torch.no_grad()
    def add_arrangements(
        self,
        arrangements: Tensor,    # (B, 30, 13) one-hot
        values: Tensor,          # (B, *value_shape) — network prediction
        ents: Tensor,            # (B, 30)
        log_probs: Tensor,       # (B, 30, 13)
        seat_idx: Tensor,        # (B,) int64
        step: int,
    ) -> None:
        """Append a fresh batch of arrangements to the buffer.

        Memory-conscious implementation (since long runs with mis-sized
        ``storage_duration`` can accumulate 100k+ rows, the naive
        ``cat-then-mask`` pattern would peak at ~3× buffer footprint and
        OOM on T4):

        1. Hash new rows and dedup within the batch (CPU-only work).
        2. Drop *old* rows whose id collides with any kept new id. This
           mask-gather runs on the existing buffer only (not on buffer+new),
           so the peak is 1× buffer size, not 2×.
        3. ``cat`` the deduped new rows onto the filtered old buffer
           (Ataraxos convention: newest at index 0).

        Net peak memory during add: ``max(old_buffer, new_rows)`` rather
        than ``2 × old_buffer + new_rows``.

        Dedup rule: if two rows share the same blake2b(arrangement)-id,
        keep only the one with the largest ``step_added``. Ties within the
        same ``step`` keep the earliest-seen index.
        """
        self._check_onehot(arrangements)
        N = arrangements.shape[0]
        self._check_shape(values, (N, *self.value_shape), "values", torch.float32)
        self._check_shape(ents, (N, ARRANGEMENT_SIZE), "ents", torch.float32)
        self._check_shape(log_probs, (N, ARRANGEMENT_SIZE, N_PIECE_TYPE_WITH_NONE),
                          "log_probs", torch.float32)
        if seat_idx.shape != (N,):
            raise ValueError(f"seat_idx must have shape ({N},), got {seat_idx.shape}")
        if seat_idx.dtype != torch.long:
            raise ValueError(f"seat_idx must be int64, got {seat_idx.dtype}")
        if not isinstance(step, int):
            raise TypeError(f"step must be int, got {type(step)}")

        # When use_cat_vf, the network emits UN-softmaxed logits in `values`.
        # We store the softmaxed distribution so downstream TD-λ math is over
        # probability simplices (matches Ataraxos). When NOT cat_vf, we store
        # the scalar value as-is.
        if self.use_cat_vf:
            values = torch.softmax(values, dim=-1)

        # ---- 1. Hash new rows and dedup within the new batch ------------
        new_arr_idx = arrangements.argmax(dim=-1).to(torch.uint8)
        new_ids_all = _arrangement_ids(new_arr_idx)

        # Dedup within the new batch: when the same hash appears twice,
        # keep the *last* occurrence (matches paper's "most recent" rule
        # when ties are resolved within a single ``step``; within one batch
        # the later-emitted sample represents a later Monte-Carlo draw).
        last_seen: dict[int, int] = {}
        for i, nid in enumerate(new_ids_all):
            last_seen[nid] = i
        new_keep_idx_list = sorted(last_seen.values())
        new_ids_kept = [new_ids_all[i] for i in new_keep_idx_list]
        new_id_set = set(new_ids_kept)

        # Gather deduped new rows on device (small: at most N=1024).
        if len(new_keep_idx_list) < N:
            idx_keep = torch.tensor(
                new_keep_idx_list, device=self.device, dtype=torch.long,
            )
            new_arrangements = arrangements.to(self.device).index_select(0, idx_keep)
            new_seat_idx = seat_idx.to(self.device).index_select(0, idx_keep)
            new_values = values.to(self.device).index_select(0, idx_keep)
            new_ents = ents.to(self.device).index_select(0, idx_keep)
            new_log_probs = log_probs.to(self.device).index_select(0, idx_keep)
        else:
            new_arrangements = arrangements.to(self.device)
            new_seat_idx = seat_idx.to(self.device)
            new_values = values.to(self.device)
            new_ents = ents.to(self.device)
            new_log_probs = log_probs.to(self.device)

        N_kept = len(new_keep_idx_list)
        new_step = torch.full((N_kept,), step, device=self.device, dtype=torch.long)

        # ---- 2. Drop old rows superseded by new ids (in-place on old) ----
        # This mask-gather runs on the OLD buffer only. Peak memory:
        # 1× old_buffer, not 2× as with cat-then-mask.
        if len(self._row_ids) > 0 and new_id_set:
            old_keep_cpu = [rid not in new_id_set for rid in self._row_ids]
            if not all(old_keep_cpu):
                keep = torch.tensor(old_keep_cpu, dtype=torch.bool, device=self.device)
                self.arrangements = self.arrangements[keep]
                self.seat_idx = self.seat_idx[keep]
                self.values = self.values[keep]
                self.ents = self.ents[keep]
                self.log_probs = self.log_probs[keep]
                self.step_added = self.step_added[keep]
                self._row_ids = [rid for rid, k in zip(self._row_ids, old_keep_cpu) if k]

        # ---- 3. Prepend new rows (Ataraxos convention: newest at idx 0) --
        self.arrangements = torch.cat([new_arrangements, self.arrangements], dim=0)
        self.seat_idx = torch.cat([new_seat_idx, self.seat_idx], dim=0)
        self.values = torch.cat([new_values, self.values], dim=0)
        self.ents = torch.cat([new_ents, self.ents], dim=0)
        self.log_probs = torch.cat([new_log_probs, self.log_probs], dim=0)
        self.step_added = torch.cat([new_step, self.step_added], dim=0)
        self._row_ids = new_ids_kept + self._row_ids

        # --- Re-allocate per-row state for reward collection / post-processing.
        cur_N = self.arrangements.size(0)
        self.counts = torch.zeros((cur_N, *self.count_shape), device=self.device)
        self.rewards = torch.zeros(cur_N, *self.reward_shape,
                                   device=self.device, dtype=torch.float32)
        self.ready_flags = torch.zeros(cur_N, dtype=torch.bool, device=self.device)
        self.adv_est = torch.zeros((cur_N, ARRANGEMENT_SIZE),
                                   device=self.device, dtype=torch.float32)
        self.val_est = torch.zeros((cur_N, *self.value_shape),
                                   device=self.device, dtype=torch.float32)
        self.reg_val_est = torch.zeros((cur_N, ARRANGEMENT_SIZE),
                                       device=self.device, dtype=torch.float32)

        # Update the id→row lookup (used by add_rewards).
        self._lookup_index = {rid: i for i, rid in enumerate(self._row_ids)}

        self.need_arrangements = False

    # --------------------------------------------------------------- add_rewards

    @torch.no_grad()
    def add_rewards(
        self,
        env_arrangements: Tensor,     # (E, 30) int64 vocab indices of each env's arrangement
        is_newly_terminal: Tensor,    # (E,) bool
        rewards: Tensor,              # (E,) float in {-1, 0, 1}
    ) -> None:
        """Record the terminal reward for each env that just finished.

        ``env_arrangements[e]`` is the (integer-index) lineup that env ``e``
        is currently playing. Only rows with ``is_newly_terminal[e]==True``
        contribute — their row in the buffer gets a running-mean update of
        the reward.

        Rows whose id doesn't appear in the buffer (e.g. stale env that
        hasn't yet been refreshed) are silently skipped — they'll never be
        `ready`, so training ignores them.
        """
        if self.need_arrangements:
            raise RuntimeError("Buffer needs new arrangements — call add_arrangements first")

        E = env_arrangements.size(0)
        if env_arrangements.shape != (E, ARRANGEMENT_SIZE):
            raise ValueError(
                f"env_arrangements must be ({E}, {ARRANGEMENT_SIZE}), "
                f"got {env_arrangements.shape}"
            )
        if is_newly_terminal.shape != (E,) or is_newly_terminal.dtype != torch.bool:
            raise ValueError(
                f"is_newly_terminal must be ({E},) bool, "
                f"got {tuple(is_newly_terminal.shape)}/{is_newly_terminal.dtype}"
            )
        if rewards.shape != (E,):
            raise ValueError(f"rewards must be ({E},), got {rewards.shape}")

        if not bool(is_newly_terminal.any()):
            return

        term_mask = is_newly_terminal
        arr_term = env_arrangements[term_mask].to(torch.uint8)
        rewards_term = rewards[term_mask].to(self.device).float()

        if self.use_cat_vf:
            # Map scalar reward in {-1, 0, 1} to categorical one-hot in {0, 1, 2}.
            rewards_term = F.one_hot(
                (rewards_term + 1).long(), num_classes=self.n_vf_cat,
            ).to(torch.float32)

        # Map each terminal arrangement to its buffer row via hash.
        ids = _arrangement_ids(arr_term)
        rows: list[int] = []
        kept: list[int] = []
        for i, rid in enumerate(ids):
            row = self._lookup_index.get(rid)
            if row is not None:
                rows.append(row)
                kept.append(i)
        if not rows:
            return

        idx = torch.tensor(rows, device=self.device, dtype=torch.long)
        kept_idx = torch.tensor(kept, device=self.device, dtype=torch.long)

        # Running-mean reward update (matches Ataraxos).
        r_in = rewards_term[kept_idx]                                  # (K, *reward_shape)
        c = self.counts[idx]                                           # (K, *count_shape)
        self.rewards[idx] = (c * self.rewards[idx] + r_in) / (c + 1)
        self.counts[idx] = c + 1
        self.ready_flags[idx] = True

    # --------------------------------------------------------------- process_data

    @torch.no_grad()
    def process_data(
        self,
        *,
        td_lambda: float = 1.0,
        gae_lambda: float = 1.0,
        reg_temp: float = 0.02,
        reg_norm: float = 10.0,
    ) -> dict[str, float]:
        """Compute val_est / reg_val_est / adv_est for every ready row.

        Defaults (td_lambda=1.0, gae_lambda=1.0) match Ataraxos arrangement
        training — i.e. pure Monte-Carlo backup. The per-slot "rollout"
        within an arrangement has zero per-step reward until the last slot,
        so MC = GAE(1) = TD(1).

        Matches the math in Ataraxos ``buffer.py::process_data``.

        Returns
        -------
        dict of logging scalars:
            "arr_buf/init_abs_adv_q90", "arr_buf/init_abs_val_q90",
            "arr_buf/init_nll_q90".
        """
        if self.need_arrangements:
            raise RuntimeError("Buffer needs new arrangements")
        if not bool(self.ready_flags.any()):
            return {}

        ready = self.ready_flags
        rewards = self.rewards[ready]                             # (N, *reward_shape)
        values = self.values[ready]                               # (N, *value_shape)
        ents = reg_norm * self.ents[ready]                        # (N, 30)  — undo net's 1/reg_norm scale

        N = int(ready.sum().item())
        adv_est = torch.zeros(N, *self.value_shape, device=self.device)
        val_est = torch.zeros(N, *self.value_shape, device=self.device)
        reg_val_est = torch.zeros(N, ARRANGEMENT_SIZE, device=self.device)

        # --- Grounded value/advantage (backward recursion in t) ---
        td_trace: Tensor
        gae_trace: Tensor
        for step in range(ARRANGEMENT_SIZE - 1, -1, -1):
            if step == ARRANGEMENT_SIZE - 1:
                delta = rewards - values[:, step]                 # (N, *reward_shape)
                td_trace = delta
                gae_trace = delta
            else:
                delta = values[:, step + 1] - values[:, step]
                td_trace = delta + td_lambda * td_trace
                gae_trace = delta + gae_lambda * gae_trace
            val_est[:, step] = td_trace + values[:, step]
            adv_est[:, step] = gae_trace

        # Categorical VF → scalar advantage via expected-value aggregation.
        if self.use_cat_vf:
            adv_est = adv_est @ self.categorical_aggregation       # (N, 30)

        # --- NLL of the actually-chosen pieces per slot ---
        nll_all = -self.log_probs[ready]                           # (N, 30, 13)
        arr = self.arrangements[ready].argmax(dim=-1).long()       # (N, 30)
        nll = nll_all.gather(-1, arr.unsqueeze(-1)).squeeze(-1)    # (N, 30)

        # --- Regularised value/advantage (entropy-seeking term) ---
        reg_td_trace = torch.zeros(1, device=self.device)
        reg_gae_trace = torch.zeros(1, device=self.device)
        for step in range(ARRANGEMENT_SIZE - 1, -1, -1):
            if step == ARRANGEMENT_SIZE - 1:
                delta = nll[:, step] - ents[:, step]
                reg_td_trace = delta
                reg_gae_trace = delta
            else:
                delta = nll[:, step] + ents[:, step + 1] - ents[:, step]
                reg_td_trace = delta + td_lambda * reg_td_trace
                reg_gae_trace = delta + gae_lambda * reg_gae_trace
            reg_val_est[:, step] = reg_td_trace + ents[:, step]
            adv_est[:, step] += reg_temp * reg_gae_trace

        # Re-scale reg_val_est to the network's prediction scale (1/reg_norm).
        reg_val_est = reg_val_est / reg_norm

        # Write back.
        self.adv_est[ready] = adv_est
        self.val_est[ready] = val_est
        self.reg_val_est[ready] = reg_val_est

        # --- Logging (q90 quantiles of adv, val, NLL sums) ---
        abs_adv = adv_est.abs()
        scalar_values = (values @ self.categorical_aggregation) if self.use_cat_vf else values
        abs_val = scalar_values.abs()
        nll_sum = nll.sum(dim=-1)
        q = torch.tensor([0.1, 0.5, 0.9], device=self.device)

        def _qstats(name: str, x: Tensor) -> dict[str, float]:
            if x.numel() == 0:
                return {}
            qv = x.float().flatten().quantile(q)
            return {
                f"arr_buf/{name}_q10": float(qv[0].item()),
                f"arr_buf/{name}_q50": float(qv[1].item()),
                f"arr_buf/{name}_q90": float(qv[2].item()),
            }

        stats: dict[str, float] = {}
        stats.update(_qstats("abs_adv", abs_adv))
        stats.update(_qstats("abs_val", abs_val))
        stats.update(_qstats("nll_sum", nll_sum))
        stats["arr_buf/n_ready"] = float(N)
        stats["arr_buf/n_rows"] = float(len(self))
        return stats

    # --------------------------------------------------------------- sample

    def sample(self, batch_size: int) -> Generator[Batch, None, None]:
        """Yield shuffled minibatches of ready rows."""
        if self.need_arrangements:
            raise RuntimeError("Buffer is not ready to sample")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be positive int, got {batch_size}")
        if not bool(self.ready_flags.any()):
            return

        ready = self.ready_flags
        arr = self.arrangements[ready]
        seat = self.seat_idx[ready]
        logp = self.log_probs[ready]
        val_est = self.val_est[ready]
        reg_val_est = self.reg_val_est[ready]
        adv_est = self.adv_est[ready]

        N = arr.size(0)
        perm = torch.randperm(N, device=self.device)
        for i in range(0, N, batch_size):
            bi = perm[i:i + batch_size]
            yield Batch(
                arrangements=arr[bi],
                seat_idx=seat[bi],
                log_probs=logp[bi],
                returns=val_est[bi],
                reg_returns=reg_val_est[bi],
                advantages=adv_est[bi],
            )

    # --------------------------------------------------------------- filter

    def filter(self, current_step: int) -> None:
        """Drop rows older than ``storage_duration`` steps. Forces reset.

        ``current_step`` and ``storage_duration`` must share units. In the
        default ``scripts/train.py`` wiring both are *rollouts* (``step_added
        = rollout_idx`` and ``filter(current_step=rollout_idx)``), so a
        ``storage_duration`` of e.g. 4 means "keep the last 4 rollouts
        worth of arrangements".

        Memory note: after a large prune (>=50% dropped) we also call
        ``torch.cuda.empty_cache()`` so the released segments become
        available to the rest of the training step. Without this the CUDA
        caching allocator may hold 2-3 GB reserved-but-unallocated while
        the move-net PPO forward is trying to grab a fresh large buffer.
        """
        if not isinstance(current_step, int):
            raise TypeError(f"current_step must be int, got {type(current_step)}")
        if len(self) == 0:
            self.need_arrangements = True
            return

        expiration_step = self.step_added + self.storage_duration
        keep = expiration_step >= current_step
        n_before = int(self.arrangements.size(0))
        self.arrangements = self.arrangements[keep]
        self.seat_idx = self.seat_idx[keep]
        self.values = self.values[keep]
        self.ents = self.ents[keep]
        self.log_probs = self.log_probs[keep]
        self.step_added = self.step_added[keep]

        # Rebuild row_ids and lookup with the kept rows.
        keep_cpu = keep.cpu().tolist()
        self._row_ids = [rid for rid, k in zip(self._row_ids, keep_cpu) if k]
        self._lookup_index = {rid: i for i, rid in enumerate(self._row_ids)}

        # Release fragments if we dropped a substantial fraction. Guards
        # against the long-run fragmentation OOM that bit v17 at rollout
        # 374 — the naive cat-then-mask peaked at 3× buffer and the
        # allocator never gave back the freed segments.
        n_after = int(self.arrangements.size(0))
        if (
            n_before > 0
            and n_after < n_before // 2
            and self.device.type == "cuda"
        ):
            torch.cuda.empty_cache()

        self.need_arrangements = True

    # --------------------------------------------------------------- internals

    @staticmethod
    def _check_onehot(arrangements: Tensor) -> None:
        if arrangements.ndim != 3 or arrangements.size(1) != ARRANGEMENT_SIZE or \
                arrangements.size(2) != N_PIECE_TYPE_WITH_NONE:
            raise ValueError(
                f"arrangements must be (B, {ARRANGEMENT_SIZE}, {N_PIECE_TYPE_WITH_NONE}), "
                f"got {tuple(arrangements.shape)}"
            )
        if not torch.allclose(arrangements.sum(dim=-1),
                              torch.ones_like(arrangements.sum(dim=-1))):
            raise ValueError("arrangements must be one-hot along the last dim")

    @staticmethod
    def _check_shape(x: Tensor, shape: tuple[int, ...], name: str,
                     dtype: torch.dtype | None = None) -> None:
        if tuple(x.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(x.shape)}")
        if dtype is not None and x.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {x.dtype}")


__all__ = [
    "ArrangementBuffer",
    "Batch",
    "DEFAULT_CATEGORICAL_AGGREGATION",
]
