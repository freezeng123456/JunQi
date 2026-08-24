"""junqi_rl.training.rollout_gpu — Fully GPU-resident rollout buffer.

Mirrors :class:`~junqi_rl.training.rollout.RolloutBuffer` field-for-field
but stores every tensor on a CUDA device.  Eliminates:

  * 11 GB of host RAM at (T=128, N=1024)
  * Every-step ``.cpu()`` copy of the legal mask
  * The ``torch.from_numpy(...).to(device)`` roundtrip in
    :meth:`RolloutBuffer.minibatches`

Use this in place of :class:`RolloutBuffer` when the collector already
produces device tensors (i.e. when running :func:`collect_rollout_gpu`
with ``device="cuda"``).

The API is identical except ``add(...)`` accepts torch tensors in
addition to numpy arrays (numpy is auto-uploaded for parity), and
``obs_storage_dtype`` can match the learner's float16/bfloat16 AMP dtype.

Legal mask storage
------------------
When ``csr_legal_mask=True`` (default), the per-step legal mask is stored
as a sparse (indices, counts) pair instead of a dense
``(N, FLAT_ACTION_DIM) bool`` tensor.  At FLAT_ACTION_DIM=16641 and an
average of ~27 legal actions per env, this is a ~40x memory reduction
for the mask.  The dense mask is reconstructed lazily in
:meth:`minibatches` only for the selected transitions, so the gradient
path sees the same tensor shape as before.

Compact history
---------------
With ``storage_mode="compact_history"``, observations and legal masks are not
stored per transition.  A :class:`~junqi_rl.gpu_rollout.GpuRolloutHistory`
keeps compact simulator state on-device and reconstructs only selected PPO
minibatches.  ``full_obs`` remains the compatibility default.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor

from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
from junqi_rl.training.rollout import (
    BOARD_SIZE,
    FLAT_ACTION_DIM,
    RolloutBatch,
    timestep_keep_env_indices,
)

# Upper bound on per-env legal action count.  Empirically the kernel
# emits < 200 actions; the compact flat action space is 16641 so 256 is
# a conservative safety margin.  A single fallback path handles overflow.
CSR_K_MAX = 256
_OBS_STORAGE_DTYPES = frozenset((torch.float16, torch.bfloat16))


def observation_storage_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    """Choose compact observation storage without per-minibatch AMP casts.

    H20/Ampere-class training uses bfloat16 compute, so storing observations
    in bfloat16 avoids converting every selected minibatch from float16.
    Float32 training keeps the historical float16 storage to avoid doubling
    the already-dominant rollout-buffer memory.
    """

    if compute_dtype in _OBS_STORAGE_DTYPES:
        return compute_dtype
    if compute_dtype == torch.float32:
        return torch.float16
    raise ValueError(f"unsupported observation compute dtype: {compute_dtype}")


def _to_tensor(x, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Accept either a numpy array or a torch tensor and return a tensor on
    ``device`` with ``dtype``.  Cheap fast-path when ``x`` is already a
    matching CUDA tensor."""
    if isinstance(x, Tensor):
        if x.device == device and x.dtype == dtype:
            return x
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _dense_mask_to_csr(
    mask: Tensor,         # (N, FLAT) bool
    out_ids: Tensor,      # (N, K_MAX) int32, pre-allocated
    out_counts: Tensor,   # (N,) int32, pre-allocated
) -> None:
    """Write the True positions of each row of ``mask`` into ``out_ids``
    and the per-row count into ``out_counts``.  The row order is preserved.

    Unused slots in ``out_ids`` are filled with 0 (benign — they are
    ignored via ``out_counts``).  Rows with more than ``K_MAX`` True
    entries are truncated (the first K_MAX positions are kept).
    """
    N, K_MAX = out_ids.shape
    # Per-row TRUE counts (unclamped).  Used below to compute per-row
    # offsets into the flat ``nonzero()`` list.  We clamp separately when
    # writing ``out_counts`` so the reconstruction path only reads the
    # first K_MAX slots.
    counts_full = mask.sum(dim=1, dtype=torch.int32)           # (N,) int32

    # Clamped counts — what callers read.
    out_counts.copy_(torch.clamp(counts_full, max=K_MAX))

    # Offsets for flat scatter.  indices = nonzero(mask).
    nz = mask.nonzero(as_tuple=False)                          # (total_nz, 2) int64
    out_ids.zero_()
    if nz.numel() == 0:
        return
    rows = nz[:, 0]                                            # (total_nz,)
    cols = nz[:, 1].to(torch.int32)                            # (total_nz,) int32

    # In-row position for each nonzero.
    row_starts = torch.zeros(N + 1, dtype=torch.int64, device=mask.device)
    row_starts[1:] = counts_full.to(torch.int64).cumsum(0)
    global_idx = torch.arange(nz.shape[0], dtype=torch.int64,
                              device=mask.device)
    local_idx = global_idx - row_starts[rows]                  # 0..counts_full[row]-1

    # Truncate rows whose local_idx exceeds K_MAX. Always apply the mask:
    # branching on ``keep.all()`` synchronises the CUDA stream every env step,
    # while these index tensors contain only the sparse legal entries.
    keep = local_idx < K_MAX
    rows = rows[keep]
    cols = cols[keep]
    local_idx = local_idx[keep]

    out_ids[rows, local_idx] = cols


def _csr_to_dense_selected(
    legal_ids: Tensor,       # (T*N, K_MAX) int32
    legal_counts: Tensor,    # (T*N,) int32
    idx: Tensor,             # (B,) int64 — selected flat indices
    flat_action_dim: int,
) -> Tensor:
    """Reconstruct a dense ``(B, FLAT_ACTION_DIM) bool`` mask for the
    transitions selected by ``idx`` — **only** those transitions, not the
    full buffer.  At B=512 this costs ~42 MB vs 1 GB for the full buffer.
    """
    B = idx.numel()
    dev = legal_ids.device
    sel_ids = legal_ids.index_select(0, idx)      # (B, K_MAX)
    sel_cnt = legal_counts.index_select(0, idx)   # (B,)
    mask = torch.zeros((B, flat_action_dim), dtype=torch.bool, device=dev)

    # Build per-slot validity: col_idx < counts
    K_MAX = sel_ids.shape[1]
    col_idx = torch.arange(K_MAX, device=dev, dtype=torch.int32)
    valid = col_idx[None, :] < sel_cnt[:, None]            # (B, K_MAX)

    # Flatten and scatter via a 1-D index.
    row_offsets = torch.arange(B, device=dev, dtype=torch.int64) * flat_action_dim
    flat_idx = (row_offsets[:, None] + sel_ids.to(torch.int64)).reshape(-1)
    flat_valid = valid.reshape(-1)
    # Empty advanced-index assignments are safe; avoid a device synchronisation
    # from ``if flat_valid.any()`` on every PPO minibatch.
    mask.view(-1)[flat_idx[flat_valid]] = True
    return mask


class RolloutBufferGPU:
    """Fully GPU-resident drop-in replacement for :class:`RolloutBuffer`.

    All per-slot storage lives on ``device`` as torch tensors.  GAE
    computation runs on-device too (no host-side Python loop).

    Parameters mirror :class:`RolloutBuffer`.  The only addition is that
    ``device`` must be a CUDA device.
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
        minibatch_group: str = "global",
        device: str | torch.device = "cuda",
        csr_legal_mask: bool = True,
        csr_k_max: int = CSR_K_MAX,
        random_opponent: bool = True,
        train_value_on_random_seats: bool = False,
        obs_storage_dtype: torch.dtype = torch.float16,
        storage_mode: str = "full_obs",
        history=None,
    ) -> None:
        self.num_envs = num_envs
        self.steps_per_env = steps_per_env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.td_lambda = td_lambda
        self.adv_filt_thresh = adv_filt_thresh
        self.adv_filt_rate = adv_filt_rate
        if minibatch_group not in {"global", "timestep"}:
            raise ValueError(
                "minibatch_group must be 'global' or 'timestep'; "
                f"got {minibatch_group!r}"
            )
        self.minibatch_group = minibatch_group
        self.device = torch.device(device)
        self.csr_legal_mask = bool(csr_legal_mask)
        self.csr_k_max = int(csr_k_max)
        if storage_mode not in {"full_obs", "compact_history"}:
            raise ValueError(
                "storage_mode must be 'full_obs' or 'compact_history'; "
                f"got {storage_mode!r}"
            )
        self.storage_mode = storage_mode
        self.uses_compact_history = storage_mode == "compact_history"
        if self.uses_compact_history and history is None:
            raise ValueError("compact_history storage requires a history object")
        self.history = history
        if obs_storage_dtype not in _OBS_STORAGE_DTYPES:
            raise ValueError(
                "obs_storage_dtype must be torch.float16 or torch.bfloat16; "
                f"got {obs_storage_dtype}"
            )
        self.obs_storage_dtype = obs_storage_dtype
        # ``random_opponent=True`` (vs-random training) causes ``compute_returns``
        # to zero out advantages for seats 1, 3 (WEST, EAST = team 1) since
        # their actions came from a uniform-random policy and contain no
        # learnable signal. ``random_opponent=False`` (self-play) keeps every
        # seat's advantages — every transition was generated by the trainable
        # policy. See ``compute_returns`` for the cross-team perspective flip
        # also gated on this flag.
        self.random_opponent = bool(random_opponent)
        # F-4: when ``random_opponent=True``, the legacy code path also nuked
        # the *returns* of seat 1/3 transitions (set them to ``self.values`` so
        # value_loss became 0). That left the value head trained ONLY on seat
        # 0/2 states, while GAE bootstrap kept reading V(s_{t+1}) at
        # seat 1/3 states — so the bootstrap term was ~ noise, advantages
        # collapsed to a 1-step TD signal, and ``ret`` in the v35 logs sat
        # at ±0.003 while win_rate stayed flat. See bug-report F-4.
        # Setting this flag True keeps the *returns* alive for seat 1/3
        # transitions so value_loss can train V(s) on those states; the
        # PPO trainer gates its policy/entropy/kl losses to policy-controlled
        # transitions via ``RolloutBatch.value_only_mask``.
        self.train_value_on_random_seats = bool(train_value_on_random_seats)
        if self.device.type != "cuda":
            raise ValueError(
                f"RolloutBufferGPU requires a CUDA device; got {self.device}"
            )

        N = num_envs
        T = steps_per_env
        dev = self.device

        if self.uses_compact_history:
            # Observation/legal tensors are rebuilt from device state history
            # only for the selected PPO minibatch.
            self.obs_spatial = None
            self.obs_global = None
        else:
            # Observations (canonical frame) use the configured 16-bit AMP
            # dtype. Matching H20's bfloat16 compute avoids a full observation
            # conversion for every PPO minibatch.
            self.obs_spatial = torch.zeros(
                (T, N, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE),
                dtype=self.obs_storage_dtype,
                device=dev,
            )
            self.obs_global = torch.zeros(
                (T, N, OBS_GLOBAL_DIMS),
                dtype=self.obs_storage_dtype,
                device=dev,
            )

        # Actions (canonical frame)
        self.actions = torch.zeros((T, N), dtype=torch.int32, device=dev)

        # Compact history rebuilds legal masks from restored state.
        if self.uses_compact_history:
            self.legal_mask = None
            self.legal_ids = None
            self.legal_counts = None
        elif self.csr_legal_mask:
            # Full-observation CSR mode: (T,N,K_MAX) ids + (T,N) counts.
            self.legal_ids = torch.zeros(
                (T, N, self.csr_k_max), dtype=torch.int32, device=dev,
            )
            self.legal_counts = torch.zeros(
                (T, N), dtype=torch.int32, device=dev,
            )
            # ``legal_mask`` kept as None; legacy callers reconstruct on demand.
            self.legal_mask = None
        else:
            self.legal_mask = torch.zeros(
                (T, N, FLAT_ACTION_DIM), dtype=torch.bool, device=dev,
            )
            self.legal_ids = None
            self.legal_counts = None

        # Policy log-probs at collection time
        self.log_probs = torch.zeros((T, N), dtype=torch.float32, device=dev)

        # Value estimates at collection time
        self.values = torch.zeros((T, N), dtype=torch.float32, device=dev)

        # Rewards and terminal flags
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=dev)
        self.dones = torch.zeros((T, N), dtype=torch.bool, device=dev)

        # Seat of acting player
        self.seats = torch.zeros((T, N), dtype=torch.int8, device=dev)

        # Computed by compute_returns()
        self.returns_ = torch.zeros((T, N), dtype=torch.float32, device=dev)
        self.advantages_ = torch.zeros((T, N), dtype=torch.float32, device=dev)

        self._ptr: int = 0
        self._full: bool = False

    def snapshot_history(
        self,
        rollout_world,
        acting_seats: Tensor,
        step: int,
    ) -> None:
        """Save current pre-action state when compact history is enabled."""

        if self.uses_compact_history:
            self.history.snapshot(
                rollout_world.state,
                acting_seats,
                step,
            )

    def storage_bytes(self) -> int:
        """Return persistent rollout/history tensor bytes (excluding scratch)."""

        tensor_names = (
            "obs_spatial",
            "obs_global",
            "actions",
            "legal_mask",
            "legal_ids",
            "legal_counts",
            "log_probs",
            "values",
            "rewards",
            "dones",
            "seats",
            "returns_",
            "advantages_",
        )
        total = 0
        for name in tensor_names:
            value = getattr(self, name, None)
            if isinstance(value, Tensor):
                total += value.numel() * value.element_size()
        if self.uses_compact_history:
            total += int(self.history.history_bytes)
        return total

    # -------------------------------------------------------------------------
    # Data insertion
    # -------------------------------------------------------------------------

    def add(
        self,
        *,
        obs_spatial,     # (N, OBS_CHANNELS, 17, 17)
        obs_global,      # (N, OBS_GLOBAL_DIMS)
        legal_mask,      # (N, FLAT_ACTION_DIM) dense bool
        actions,         # (N,)
        log_probs,       # (N,)
        values,          # (N,)
        rewards,         # (N,) or None when caller patches after env.step()
        dones,           # (N,) or None when caller patches after env.step()
        seats,           # (N,)
    ) -> None:
        t = self._ptr
        dev = self.device
        if not self.uses_compact_history:
            self.obs_spatial[t] = _to_tensor(
                obs_spatial,
                device=dev,
                dtype=self.obs_storage_dtype,
            )
            self.obs_global[t] = _to_tensor(
                obs_global,
                device=dev,
                dtype=self.obs_storage_dtype,
            )

            # Dense input mask is retained as CSR or dense in full-obs mode.
            lm = _to_tensor(legal_mask, device=dev, dtype=torch.bool)
            if self.csr_legal_mask:
                _dense_mask_to_csr(
                    lm,
                    self.legal_ids[t],
                    self.legal_counts[t],
                )
            else:
                self.legal_mask[t] = lm

        self.actions[t]     = _to_tensor(actions,     device=dev, dtype=torch.int32)
        self.log_probs[t]   = _to_tensor(log_probs,   device=dev, dtype=torch.float32)
        self.values[t]      = _to_tensor(values,      device=dev, dtype=torch.float32)
        if rewards is not None:
            self.rewards[t] = _to_tensor(
                rewards,
                device=dev,
                dtype=torch.float32,
            )
        if dones is not None:
            self.dones[t] = _to_tensor(
                dones,
                device=dev,
                dtype=torch.bool,
            )
        self.seats[t]       = _to_tensor(seats,       device=dev, dtype=torch.int8)
        self._ptr += 1
        if self._ptr == self.steps_per_env:
            self._ptr = 0
            self._full = True

    def reset(self) -> None:
        self._ptr = 0
        self._full = False

    @property
    def is_ready(self) -> bool:
        return self._full

    # -------------------------------------------------------------------------
    # GAE
    # -------------------------------------------------------------------------

    def compute_returns(self, last_values, *, last_seats=None) -> None:
        """GAE(λ) and TD(λ) — all on-device in a Python loop over T.

        Multi-seat perspective handling
        -------------------------------
        Stored ``self.values[t]`` is the value head's prediction in the
        *acting seat's canonical frame* — i.e. ``V(s_t)`` is the expected
        return **from S_t's viewpoint** (positive ⇒ "I expect to win").
        Stored ``self.rewards[t]`` likewise is from S_t's viewpoint
        (computed by ``compute_rewards_kernel`` as +1 if acting team
        wins / -1 if loses).

        The GAE delta couples ``V(s_t)`` and ``V(s_{t+1})``, but
        ``V(s_{t+1})`` is from a DIFFERENT seat's frame. In JunQi the
        4-seat rotation S→W→N→E means consecutive actors usually belong
        to opposite teams — so the bootstrapped ``V_{t+1}`` is the
        opponent's expected return, which from S_t's frame must be
        **negated** before it can be added to ``r_t``.

        Concretely::

            same_team(t, t+1) = (S_t & 1) == (S_{t+1} & 1)
            flip = +1 if same_team else -1
            δ_t = r_t + γ · flip · V_{t+1} · (1-done_t) - V_t

        Q12 (a seat dies and gets skipped) can break the regular S→W→N→E
        rotation, so we don't hard-code ``flip = -1`` even though
        consecutive seats DO usually differ. We compute it dynamically
        from the buffered seat tags.

        ``last_seats`` is the seat that would act AFTER the last stored
        step (i.e. for bootstrapping at t=T-1). When omitted we fall back
        to assuming "different team" (``flip=-1``); callers that have
        easy access to ``rollout_world.turn_torch()`` should pass it for
        correctness.

        Random-opponent gating
        ----------------------
        When ``self.random_opponent=True`` (vs-random training), seats
        1 and 3 (= team 1 = WEST/EAST) acted via uniform-random sampling
        and their advantages are not learnable. We zero them after GAE
        so the advantage filter naturally skips those transitions. In
        self-play (``random_opponent=False``) the policy controlled all
        seats, so every transition is kept.

        Each iteration is a handful of (N,) tensor ops so loop-overhead
        per step is ~10 μs; at T ≤ 1024 total cost is negligible.
        """
        T = self.steps_per_env
        last_val_t = _to_tensor(last_values, device=self.device,
                                dtype=torch.float32)
        # If caller didn't supply ``last_seats`` (the seat that would act
        # at t=T), assume opposite-team for the bootstrap. In a regular
        # 4-seat cycle this is correct; in the rare Q12 case where one
        # team has been wiped before T-1, this introduces a small bias on
        # the very last transition — tolerable.
        if last_seats is None:
            # Best-effort: derive from S_{T-1} ^ 1 (XOR with 1 flips team
            # parity). Stored seat[T-1] is int8.
            last_seat_team = (self.seats[T - 1].to(torch.int64) & 1) ^ 1
        else:
            last_seats_t = _to_tensor(last_seats, device=self.device,
                                       dtype=torch.int64)
            last_seat_team = last_seats_t & 1

        gae = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        # next_val and next_team correspond to step t+1 as we iterate
        # backwards. Initialised to the bootstrap state (after T-1).
        next_val = last_val_t.clone()
        next_team = last_seat_team.clone()  # int64 (N,)

        for t in reversed(range(T)):
            mask = (~self.dones[t]).to(torch.float32)
            cur_team = (self.seats[t].to(torch.int64) & 1)
            # +1 when t and t+1 are same team, -1 when different. Cast to
            # float32 to match the value tensor's dtype.
            flip = (cur_team == next_team).to(torch.float32) * 2.0 - 1.0  # (N,)

            delta = (
                self.rewards[t]
                + self.gamma * flip * next_val * mask
                - self.values[t]
            )
            # The recurrence advantage_{t} = δ_t + γλ·(1-done_t)·advantage_{t+1}
            # propagates ``advantage_{t+1}`` back through one step. That
            # advantage was computed in S_{t+1}'s frame, so we similarly
            # flip it when teams differ at the t/t+1 boundary.
            gae = delta + self.gamma * self.gae_lambda * mask * flip * gae
            self.advantages_[t] = gae
            self.returns_[t] = gae + self.values[t]

            next_val = self.values[t]
            next_team = cur_team

        # Random-opponent: zero out the seats whose actions were uniform-random.
        # In self-play we keep every seat's advantages (all transitions are
        # policy-controlled and informative).
        #
        # F-4 (2026-05-10): we KEEP the *returns* alive for seat 1/3 even in
        # random_opponent mode so that ``value_loss`` can still update V(s) on
        # those states — bootstrap of seat-0/2 advantages will then read a
        # meaningful V(s_{t+1}) instead of a near-zero untrained head. We only
        # zero advantages, which gates the policy / entropy / kl gradient.
        # Legacy (v17–v35) behaviour is restored by setting
        # ``train_value_on_random_seats=False``.
        if self.random_opponent:
            enemy_mask = (self.seats == 1) | (self.seats == 3)
            if enemy_mask.any():
                self.advantages_[enemy_mask] = 0.0
                if not self.train_value_on_random_seats:
                    # Legacy behaviour: also kill the value-loss signal on
                    # enemy seats by setting returns := values (loss = 0).
                    self.returns_[enemy_mask] = self.values[enemy_mask]

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
        """Yield minibatches as :class:`RolloutBatch` with **device tensors**.

        All heavy tensors are device-side views (no H2D here).  Index arrays
        are built on-device using ``torch.randperm``.
        """
        T = self.steps_per_env
        N = self.num_envs
        total = T * N
        dev = self.device

        # Flatten persistent scalar data. Full-observation mode also exposes
        # zero-copy observation views; compact mode reconstructs them below.
        if self.uses_compact_history:
            obs_sp = None
            obs_gl = None
        else:
            obs_sp = self.obs_spatial.view(
                total,
                OBS_CHANNELS,
                BOARD_SIZE,
                BOARD_SIZE,
            )
            obs_gl = self.obs_global.view(total, OBS_GLOBAL_DIMS)
        act    = self.actions.view(total)
        lp     = self.log_probs.view(total)
        adv    = self.advantages_.view(total)
        ret    = self.returns_.view(total)
        val    = self.values.view(total)
        # Legal mask: compact mode rebuilds from state; full mode uses stored
        # CSR/dense data.
        if self.uses_compact_history:
            flat_ids = None
            flat_cnt = None
            lm = None
        elif self.csr_legal_mask:
            flat_ids = self.legal_ids.view(total, self.csr_k_max)
            flat_cnt = self.legal_counts.view(total)
            lm = None  # reconstructed per-minibatch below
        else:
            lm = self.legal_mask.view(total, FLAT_ACTION_DIM)

        # "Own-seat" mask: transitions whose advantage carries learnable
        # signal. In vs-random training (random_opponent=True), seats 1/3
        # are random-opponent transitions whose policy gradient is not
        # learnable (the action came from a uniform sampler, not the
        # policy). In self-play (random_opponent=False) every transition
        # is policy-controlled.
        #
        # Historical note (2026-05-11): until today this was computed as
        # ``adv != 0.0`` ("cheap proxy"). That proxy also caught
        # legitimate own-seat transitions whose normalised advantage
        # happened to round to 0 (rare but possible when V is well-fit
        # and reward propagated cleanly). We now use the exact seat-id
        # check, which is also cheaper (no comparison against 0 on a
        # large flat tensor).
        seats_flat = self.seats.view(total)
        if self.random_opponent:
            own_mask = (seats_flat == 0) | (seats_flat == 2)
        else:
            own_mask = torch.ones_like(adv, dtype=torch.bool)
        if own_mask.any():
            own_adv  = adv[own_mask]
            adv_mean = own_adv.mean()
            adv_std  = own_adv.std() + 1e-8
        else:
            adv_mean = adv.mean()
            adv_std  = adv.std() + 1e-8
        adv_norm = (adv - adv_mean) / adv_std

        # Magnitude filter — keep policy-controlled transitions with
        # |A_norm| >= thresh. In vs-random mode this still excludes the
        # zeroed enemy transitions; in self-play it trims the bottom
        # (1 - adv_filt_rate) of |advantage| as the paper does.
        abs_adv = adv_norm.abs()
        thresh = self.adv_filt_thresh
        keep = own_mask & (abs_adv >= thresh)

        def _emit(idx: Tensor, vo: Tensor):
            if idx.numel() == 0:
                return
            seats_batch = seats_flat.index_select(0, idx)
            if self.uses_compact_history:
                obs_sp_batch, obs_gl_batch, lm_batch = self.history.reconstruct(
                    idx,
                    seats_batch,
                    dtype=self.obs_storage_dtype,
                )
            elif self.csr_legal_mask:
                lm_batch = _csr_to_dense_selected(
                    flat_ids, flat_cnt, idx, FLAT_ACTION_DIM,
                )
                obs_sp_batch = obs_sp.index_select(0, idx)
                obs_gl_batch = obs_gl.index_select(0, idx)
            else:
                lm_batch = lm.index_select(0, idx)
                obs_sp_batch = obs_sp.index_select(0, idx)
                obs_gl_batch = obs_gl.index_select(0, idx)
            yield RolloutBatch(
                obs_spatial=obs_sp_batch,
                obs_global=obs_gl_batch,
                legal_mask=lm_batch,
                actions=act.index_select(0, idx).to(torch.int64),
                old_log_probs=lp.index_select(0, idx),
                advantages=adv_norm.index_select(0, idx),
                returns=ret.index_select(0, idx),
                values=val.index_select(0, idx),
                adv_mask=torch.ones(idx.numel(), dtype=torch.bool, device=dev),
                value_only_mask=vo,
            )

        if self.minibatch_group == "timestep":
            # One Adam step per collect row.  Filter only shrinks the row.
            # shuffle is ignored so t=0..T-1 stay in order.
            abs_np = abs_adv.detach().view(T, N).cpu().numpy()
            own_np = own_mask.detach().view(T, N).cpu().numpy()
            rows = timestep_keep_env_indices(
                abs_np,
                rate=self.adv_filt_rate,
                thresh=self.adv_filt_thresh,
                own_mask=own_np,
            )
            sizes = [int(r.size) for r in rows]
            nonempty = [s for s in sizes if s > 0]
            n_policy = int(sum(sizes))
            self._last_n_total = int(total)
            self._last_n_own = int(own_mask.sum().item())
            self._last_n_policy = n_policy
            self._last_thresh_used = float(self.adv_filt_thresh)
            self._last_kept_mean = float(np.mean(sizes)) if sizes else 0.0
            self._last_kept_min = float(min(nonempty) if nonempty else 0)
            self._last_kept_max = float(max(sizes) if sizes else 0)
            self._last_n_empty_steps = float(sum(1 for s in sizes if s == 0))
            for t, env_idx in enumerate(rows):
                if env_idx.size == 0:
                    continue
                flat = torch.from_numpy(
                    (t * N + env_idx).astype(np.int64)
                ).to(dev)
                vo = torch.zeros(flat.numel(), dtype=torch.bool, device=dev)
                yield from _emit(flat, vo)
            return

        # --- Avoid per-call .item() syncs ---
        # The old path called .item() on n_total and n_kept per minibatches()
        # call, forcing 2 device syncs.  With 4 epochs × 248 minibatches these
        # added up.  Since the rate-based fallback is identical behaviour for
        # any fixed adv_filt_rate < 1.0, we compute the quantile-based
        # threshold unconditionally, on GPU, and only sync once if we actually
        # need to lower the threshold.
        if self.adv_filt_rate < 1.0:
            # own_abs: (M,) where M = own_mask.sum()
            own_abs = abs_adv[own_mask]
            if own_abs.numel() > 0:
                # Pick the (1-adv_filt_rate) quantile — gives us min_kept guarantee.
                # torch.quantile for cheap; topk is O(n log k) which is fine here.
                q = 1.0 - self.adv_filt_rate
                # Use a single GPU quantile op (no python loop, no sync).
                q_thresh = torch.quantile(own_abs, q).item()
                thresh = max(self.adv_filt_thresh, q_thresh)
                keep = own_mask & (abs_adv >= thresh)

        indices = keep.nonzero(as_tuple=False).squeeze(-1)  # (K,)
        n_policy = int(indices.numel())

        # F-5: stash sample-count diagnostics so PPOTrainer / scripts can
        # surface them in per-rollout logs. Without this, "loss_p=0.0 because
        # adv_filt killed every sample" looks identical to "loss_p=0.0 because
        # the policy already converged" in the logs.
        self._last_n_total      = int(total)
        self._last_n_own        = int(own_mask.sum().item())
        self._last_n_policy     = n_policy
        self._last_thresh_used  = float(thresh)
        self._last_kept_mean = float("nan")
        self._last_kept_min = float("nan")
        self._last_kept_max = float("nan")
        self._last_n_empty_steps = 0.0

        # F-4: optionally mix in seat 1/3 transitions as VALUE-ONLY samples.
        # They contribute only to value_loss (no policy / kl / entropy gradient)
        # so the V-head learns V(s) on those states. The PPO trainer reads
        # ``value_only_mask`` from RolloutBatch to gate the policy loss.
        # Only relevant in random_opponent mode; in self-play every transition
        # is already policy-active.
        #
        # DDP note: each rank shuffles independently, so a given minibatch index
        # may be value-heavy on one rank and policy-heavy on another. DDP
        # all-reduces the gradient sum, so policy-head gradients land halfway
        # between "averaged over rank-B's policy samples" and zero (rank-A's
        # contribution). The direction stays correct; effective policy batch
        # size on that step is approximately halved. This is acceptable and
        # in practice averages out across minibatches; a stricter guarantee
        # would require sorting indices to keep value-only samples in the
        # tail of the epoch (a future optimisation, not a correctness fix).
        emit_value_only = (
            self.random_opponent
            and getattr(self, "train_value_on_random_seats", False)
        )
        if emit_value_only:
            # Collect indices of seat 1/3 transitions whose adv is exactly zero
            # (i.e. the ones compute_returns just zeroed). We cap their count
            # at n_policy to avoid drowning the policy gradient in value-only
            # backward passes (typical mix is then ~50/50, matching the
            # collect distribution).
            enemy_idx = (~own_mask).nonzero(as_tuple=False).squeeze(-1)
            if enemy_idx.numel() > 0 and n_policy > 0:
                if enemy_idx.numel() > n_policy:
                    perm_e = torch.randperm(enemy_idx.numel(), device=dev)[:n_policy]
                    enemy_idx = enemy_idx[perm_e]
                # value_only flag for each combined sample.
                vo_flags = torch.cat([
                    torch.zeros(indices.numel(), dtype=torch.bool, device=dev),
                    torch.ones(enemy_idx.numel(),  dtype=torch.bool, device=dev),
                ])
                indices_full = torch.cat([indices, enemy_idx])
            else:
                vo_flags     = torch.zeros(indices.numel(), dtype=torch.bool, device=dev)
                indices_full = indices
        else:
            vo_flags     = torch.zeros(indices.numel(), dtype=torch.bool, device=dev)
            indices_full = indices

        if shuffle:
            perm = torch.randperm(indices_full.numel(), device=dev)
            indices_full = indices_full[perm]
            vo_flags     = vo_flags[perm]

        for start in range(0, indices_full.numel(), batch_size):
            idx = indices_full[start : start + batch_size]
            vo  = vo_flags[start : start + batch_size]
            if idx.numel() == 0:
                continue
            seats_batch = seats_flat.index_select(0, idx)
            if self.uses_compact_history:
                obs_sp_batch, obs_gl_batch, lm_batch = self.history.reconstruct(
                    idx,
                    seats_batch,
                    dtype=self.obs_storage_dtype,
                )
            elif self.csr_legal_mask:
                lm_batch = _csr_to_dense_selected(
                    flat_ids, flat_cnt, idx, FLAT_ACTION_DIM,
                )
                obs_sp_batch = obs_sp.index_select(0, idx)
                obs_gl_batch = obs_gl.index_select(0, idx)
            else:
                lm_batch = lm.index_select(0, idx)
                obs_sp_batch = obs_sp.index_select(0, idx)
                obs_gl_batch = obs_gl.index_select(0, idx)
            # advantages: value-only samples carry adv=0 (already zeroed by
            # compute_returns) which is also harmless after normalisation —
            # the PPO trainer additionally masks them out.
            yield RolloutBatch(
                obs_spatial=obs_sp_batch,
                obs_global=obs_gl_batch,
                legal_mask=lm_batch,
                actions=act.index_select(0, idx).to(torch.int64),
                old_log_probs=lp.index_select(0, idx),
                advantages=adv_norm.index_select(0, idx),
                returns=ret.index_select(0, idx),
                values=val.index_select(0, idx),
                adv_mask=torch.ones(idx.numel(), dtype=torch.bool, device=dev),
                value_only_mask=vo,
            )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def num_valid_transitions(self) -> int:
        adv = self.advantages_.view(-1)
        adv_std = adv.std() + 1e-8
        adv_norm = (adv - adv.mean()) / adv_std
        return int((adv_norm.abs() >= self.adv_filt_thresh).sum().item())

    def stats(self) -> dict[str, float]:
        adv = self.advantages_.view(-1)
        ret = self.returns_.view(-1)
        rew = self.rewards.view(-1)
        # Single synchronisation — compute all scalars on-device, stack,
        # then D2H once.  Avoids 5 separate .item() syncs.
        adv_std = (adv.std() + 1e-8)
        adv_mean = adv.mean()
        adv_norm_abs = ((adv - adv_mean) / adv_std).abs()
        n_valid = (adv_norm_abs >= self.adv_filt_thresh).sum().to(torch.float32)
        stacked = torch.stack([
            rew.mean(), ret.mean(), adv_mean, adv_std, n_valid,
        ]).cpu().tolist()
        out = {
            "rollout/mean_reward":    float(stacked[0]),
            "rollout/mean_return":    float(stacked[1]),
            "rollout/mean_advantage": float(stacked[2]),
            "rollout/std_advantage":  float(stacked[3]),
            "rollout/num_valid":      float(stacked[4]),
            "rollout/storage_gib": (
                float(self.storage_bytes()) / float(1024**3)
            ),
        }
        # F-5: filter pipeline diagnostics. Surfaced in per-rollout logs so
        # "loss_p=0.0" is debuggable: was it the adv filter killing every
        # sample, or did the policy actually converge?
        if hasattr(self, "_last_n_total"):
            out["rollout/n_total"]        = float(self._last_n_total)
            out["rollout/n_own_seat"]     = float(self._last_n_own)
            out["rollout/n_policy_kept"]  = float(self._last_n_policy)
            out["rollout/keep_frac_own"]  = (
                float(self._last_n_policy) / max(1.0, float(self._last_n_own))
            )
            out["rollout/adv_thresh_used"] = float(self._last_thresh_used)
        if hasattr(self, "_collect_entropy"):
            out["collect/entropy"] = float(self._collect_entropy)
            if hasattr(self, "_last_kept_mean"):
                out["rollout/kept_mean"] = float(self._last_kept_mean)
                out["rollout/kept_min"] = float(self._last_kept_min)
                out["rollout/kept_max"] = float(self._last_kept_max)
                out["rollout/n_empty_steps"] = float(self._last_n_empty_steps)
        return out


__all__ = ["RolloutBufferGPU", "observation_storage_dtype"]
