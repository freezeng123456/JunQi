"""Lazy channel gathering, shared only within one policy-head invocation."""

from __future__ import annotations

import torch
from torch import Tensor

from junqi_core.observation import CHANNEL_LAYOUT


class PolicyFeatureContext:
    """Gather only requested channel groups; never retain another actor's data.

    Keeps tensors on their original device and preserves autograd. The context
    is created only when a policy has enabled heads, and is discarded after
    those heads finish. No CPU transfers or cross-forward cache invalidation.
    """

    def __init__(
        self, spatial: Tensor, on_board: Tensor, *,
        channel_slices: dict[str, slice] | None = None,
        gather_indices: Tensor | None = None,
    ) -> None:
        self.spatial = spatial
        self.on_board = on_board
        self._channel_slices = channel_slices if channel_slices is not None else {}
        self._gather_indices = gather_indices
        self._gathered_raw: Tensor | None = None
        self._gathered_float: Tensor | None = None
        self._channels: dict[str, Tensor] = {}
        self._pieces: tuple[Tensor, Tensor, Tensor] | None = None

    def _gather(self) -> None:
        if self._gathered_raw is None:
            cells = self.on_board.numel()
            width = self._gather_indices.numel() // cells
            self._gathered_raw = self.spatial.flatten(1).index_select(
                1, self._gather_indices,
            ).reshape(self.spatial.shape[0], width, cells)
            self._gathered_float = self._gathered_raw.float()

    def raw_channels(self, name: str) -> Tensor:
        if name in self._channel_slices:
            self._gather()
            return self._gathered_raw[:, self._channel_slices[name]]
        # Standalone heads and undeclared extension channels retain a safe
        # lazy fallback; declaring dependencies enables the fused model path.
        return self.spatial.flatten(-2)[:, CHANNEL_LAYOUT[name]].index_select(-1, self.on_board)

    def channels(self, name: str) -> Tensor:
        if name not in self._channels:
            if name in self._channel_slices:
                self._gather()
                self._channels[name] = self._gathered_float[:, self._channel_slices[name]]
            else:
                self._channels[name] = self.raw_channels(name).float()
        return self._channels[name]

    def pieces(self) -> tuple[Tensor, Tensor, Tensor]:
        """Own types, enemy belief and occupancy with tactical-v2 arithmetic."""
        if self._pieces is None:
            own = self.channels("piece_own")
            occupied = (
                self.channels("piece_left_side_enemy")
                + self.channels("piece_right_side_enemy")
            ) > 0
            enemy = (
                self.channels("belief_left_side") + self.channels("belief_right_side")
            ) * occupied
            self._pieces = own, enemy, occupied
        return self._pieces

    def combat_pieces(self) -> tuple[Tensor, Tensor]:
        """Preserve combat-v1's addition before casting for reduced-dtype obs."""
        own = self.channels("piece_own")
        enemy = (
            self.raw_channels("belief_left_side") + self.raw_channels("belief_right_side")
        ).float()
        occupied = (
            self.raw_channels("piece_left_side_enemy") + self.raw_channels("piece_right_side_enemy")
        ) > 0
        return own, torch.where(occupied, enemy, 0.0)
