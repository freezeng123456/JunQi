"""Original behavior transitions awaiting the same seat's next decision.

The first dimension is always seat, so a rollout can prepend a snapshot as
four rows. Slots survive player elimination and rollout boundaries; only a
whole-game terminal clears an environment. This cache never recomputes a
behavior probability from a newer policy.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from junqi_core.board import BOARD_SIZE, COMPACT_ACTION_DIM
from junqi_core.observation import OBS_CHANNELS


class PendingSeatTransitions:
    """Fixed ``[4, num_envs, ...]`` storage for legal self-play decisions.

    The collector owns legality and seat-range checks on the hot path.
    ``record`` uses vector indexing and performs no tensor-to-host reads.
    Checkpoint restore validates the complete storage contract before it
    changes any slot. V100 callers use float16 observation storage.
    """

    SCHEMA = "junqi.pending_seat_transitions.v1"
    TENSOR_FIELDS = (
        "obs_spatial", "obs_global", "legal_mask", "actions", "log_probs",
        "values", "seats", "behavior_log_probs", "valid",
    )

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        obs_dtype: torch.dtype = torch.float16,
        obs_global_dims: int = 0,
    ) -> None:
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if type(obs_global_dims) is not int or obs_global_dims < 0:
            raise ValueError("obs_global_dims must be a nonnegative integer")
        if obs_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("obs_dtype must be float16, bfloat16, or float32")
        self.num_envs = num_envs
        self.obs_dtype = obs_dtype
        self.obs_global_dims = obs_global_dims
        prefix = (4, num_envs)
        self.valid = torch.zeros(prefix, dtype=torch.bool, device=device)
        self.device = self.valid.device
        self.obs_spatial = torch.zeros(
            (*prefix, OBS_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=obs_dtype, device=self.device,
        )
        self.obs_global = torch.zeros(
            (*prefix, obs_global_dims), dtype=obs_dtype, device=self.device,
        )
        self.legal_mask = torch.zeros(
            (*prefix, COMPACT_ACTION_DIM), dtype=torch.bool, device=self.device,
        )
        self.actions = torch.zeros(prefix, dtype=torch.int32, device=self.device)
        self.log_probs = torch.zeros(prefix, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(prefix, dtype=torch.float32, device=self.device)
        self.seats = torch.arange(4, dtype=torch.int8, device=self.device)[:, None].expand(prefix).clone()
        self.behavior_log_probs = torch.zeros(
            (*prefix, COMPACT_ACTION_DIM), dtype=torch.float32, device=self.device,
        )
        self._env_indices = torch.arange(num_envs, device=self.device)

    @torch.no_grad()
    def begin_rollout(self) -> dict[str, Tensor]:
        """Freeze all four prefix rows independently of future collection."""
        return {name: getattr(self, name).clone() for name in self.TENSOR_FIELDS}

    def _check_input(self, name: str, value: Tensor, dtypes: tuple[torch.dtype, ...]) -> None:
        expected = getattr(self, name)
        if not isinstance(value, Tensor) or value.shape != expected.shape[1:]:
            raise ValueError(f"{name} must have shape {tuple(expected.shape[1:])}")
        if value.dtype not in dtypes:
            raise ValueError(f"{name} has unsupported dtype {value.dtype}")
        if value.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")

    @torch.no_grad()
    def record(
        self,
        obs_spatial: Tensor,
        obs_global: Tensor,
        legal_mask: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        seats: Tensor,
        behavior_log_probs: Tensor,
    ) -> None:
        """Copy one already-validated legal action per environment.

        Inputs may be views of simulator memory: assignments copy immediately.
        Full log probabilities must already be float32, avoiding an unnoticed
        half-precision round trip in the old-policy KL reference.
        """
        data = dict(obs_spatial=obs_spatial, obs_global=obs_global,
                    legal_mask=legal_mask, actions=actions, log_probs=log_probs,
                    values=values, seats=seats, behavior_log_probs=behavior_log_probs)
        floating = (torch.float16, torch.bfloat16, torch.float32)
        integer = (torch.int8, torch.int16, torch.int32, torch.int64)
        for name, value in data.items():
            allowed = (floating if name in ("obs_spatial", "obs_global") else
                       integer if name in ("actions", "seats") else
                       (torch.bool,) if name == "legal_mask" else (torch.float32,))
            self._check_input(name, value, allowed)
        rows = seats.to(dtype=torch.long)
        for name, value in data.items():
            storage = getattr(self, name)
            storage[rows, self._env_indices] = value.to(dtype=storage.dtype)
        self.valid[rows, self._env_indices] = True

    @torch.no_grad()
    def terminal(self, dones: Tensor) -> None:
        """Invalidate all seats only where the complete game has ended."""
        if not isinstance(dones, Tensor) or dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},)")
        if dones.dtype != torch.bool or dones.device != self.device:
            raise ValueError("dones must be boolean and share the cache device")
        self.valid.logical_and_(~dones[None, :])

    @torch.no_grad()
    def clear(self) -> None:
        """Invalidate every slot; stale payloads remain unreachable."""
        self.valid.zero_()

    def state_dict(self) -> dict[str, object]:
        """Return independent CPU storage, with no derived indices or RNG."""
        state: dict[str, object] = dict(
            schema=self.SCHEMA, num_envs=self.num_envs,
            obs_dtype=str(self.obs_dtype), obs_global_dims=self.obs_global_dims,
        )
        state.update({name: getattr(self, name).detach().cpu().clone()
                      for name in self.TENSOR_FIELDS})
        return state

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore exactly, rejecting incompatible or malformed pending data."""
        if not isinstance(state, Mapping):
            raise ValueError("pending state must be a mapping")
        metadata = dict(schema=self.SCHEMA, num_envs=self.num_envs,
                        obs_dtype=str(self.obs_dtype), obs_global_dims=self.obs_global_dims)
        if set(state) != set(metadata) | set(self.TENSOR_FIELDS):
            raise ValueError("pending state fields do not match the schema")
        for name, expected in metadata.items():
            if type(state[name]) is not type(expected) or state[name] != expected:
                raise ValueError(f"pending state {name} mismatch")
        tensors: dict[str, Tensor] = {}
        for name in self.TENSOR_FIELDS:
            value = state[name]
            expected = getattr(self, name)
            if not isinstance(value, Tensor) or value.shape != expected.shape:
                raise ValueError(f"pending state {name} shape mismatch")
            if value.dtype != expected.dtype or value.layout != torch.strided:
                raise ValueError(f"pending state {name} dtype/layout mismatch")
            tensors[name] = value.detach().cpu()

        # Loading is infrequent: check semantics here instead of synchronizing
        # the GPU after every collected action. Invalid slots are unreachable.
        valid = tensors["valid"]
        expected_seats = torch.arange(4, dtype=torch.int8)[:, None].expand_as(valid)
        if not torch.equal(tensors["seats"][valid], expected_seats[valid]):
            raise ValueError("pending state seats do not match their slots")
        actions = tensors["actions"][valid].long()
        if bool(((actions < 0) | (actions >= COMPACT_ACTION_DIM)).any()):
            raise ValueError("pending state action out of range")
        legal = tensors["legal_mask"][valid]
        if not bool(legal.gather(1, actions[:, None]).all()):
            raise ValueError("pending state chosen action is not legal")
        for name in ("obs_spatial", "obs_global", "log_probs", "values"):
            if not bool(torch.isfinite(tensors[name][valid]).all()):
                raise ValueError(f"pending state {name} contains nonfinite values")
        behavior = tensors["behavior_log_probs"][valid]
        if bool((torch.isnan(behavior) | torch.isposinf(behavior)).any()):
            raise ValueError("pending state behavior_log_probs contains invalid values")
        if not torch.allclose(behavior.logsumexp(-1), torch.zeros(actions.shape), atol=5e-5, rtol=0):
            raise ValueError("pending state behavior_log_probs is not normalized")
        if bool((behavior.exp().masked_fill(legal, 0).sum(-1) > 1e-7).any()):
            raise ValueError("pending state behavior_log_probs gives mass to illegal actions")
        chosen = behavior.gather(1, actions[:, None]).squeeze(1)
        if not torch.equal(chosen, tensors["log_probs"][valid]):
            raise ValueError("pending state chosen log_probs differ from behavior_log_probs")

        # Stage every device copy before changing the live cache, and never
        # retain aliases into the caller's checkpoint tensors.
        restored = {name: value.to(device=self.device).clone() for name, value in tensors.items()}
        for name, value in restored.items():
            getattr(self, name).copy_(value)
