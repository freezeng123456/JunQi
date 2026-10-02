"""Autoregressive hidden-army distributions with cached batched generation.

The board encoder is shared with the marginal model. Teacher forcing uses a
shifted causal prefix. Generation encodes once, caches cross-attention K/V
once, and appends self-attention K/V one token at a time. No policy rollout
calls this module unless joint inference/search is explicitly enabled.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from junqi_core.board import COMPACT_TO_FLAT, FLAT_TO_COMPACT

from .belief_net import BeliefNet, BeliefNetConfig, N_BELIEF_TYPES
from .causal_decoder import _DecoderBlock


@dataclass
class JointBeliefNetConfig(BeliefNetConfig):
    n_decoder_layer: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.n_decoder_layer<1:
            raise ValueError('Joint decoder must contain at least one layer')


class JointBeliefNet(BeliefNet):
    architecture_id = 'junqi.belief317.autoregressive.v1'

    def __init__(self, cfg: JointBeliefNetConfig | None = None):
        cfg = cfg or JointBeliefNetConfig()
        super().__init__(cfg)
        self.type_embedding = nn.Embedding(N_BELIEF_TYPES+1, cfg.embed_dim)
        self.owner_embedding = nn.Embedding(4, cfg.embed_dim)
        self.public_projection = nn.Linear(4, cfg.embed_dim)
        self.decoder = nn.ModuleList(_DecoderBlock(cfg) for _ in range(cfg.n_decoder_layer))
        self.decoder_norm = nn.LayerNorm(cfg.embed_dim)
        subsets = torch.arange(1 << N_BELIEF_TYPES)
        members = (subsets[:, None] & (1 << torch.arange(N_BELIEF_TYPES))) != 0
        self.register_buffer('subset_members', members.float(), persistent=False)
        self.register_buffer('on_board_cells', torch.tensor(COMPACT_TO_FLAT.copy(),dtype=torch.long),persistent=False)
        self.register_buffer('flat_to_compact', torch.tensor(FLAT_TO_COMPACT.copy(),dtype=torch.long),persistent=False)

    def encode(self, obs: Tensor, seats: Tensor) -> Tensor:
        return super().encode(obs, seats, cell_indices=self.on_board_cells)

    def future_demand(self, support: Tensor, owners: Tensor) -> Tensor:
        """Exact capacitated matching feasibility over a 12-type vocabulary.

        Hall's condition: every type subset must have enough inventory for
        all remaining pieces restricted to that subset. This static demand
        is computed once per input batch, outside the sampling loop.
        """
        bits = (support.long() * (1 << torch.arange(12, device=support.device))).sum(-1)
        subsets = torch.arange(1 << 12, device=support.device)
        contained = ((bits[..., None] & subsets) == bits[..., None]).to(torch.int16)
        demand = torch.zeros_like(contained)
        for owner in range(4):
            selected = contained * (owners == owner)[..., None]
            suffix = selected.flip(1).cumsum(1, dtype=torch.int16).flip(1) - selected
            demand = torch.where((owners == owner)[..., None], suffix, demand)
        return demand

    def feasible_types(self, remaining: Tensor, demand: Tensor) -> Tensor:
        capacity = remaining.float() @ self.subset_members.T
        tight = (capacity <= demand).float()
        forbidden = (tight @ self.subset_members) > 0
        return (remaining > 0) & ~forbidden

    def token_context(self, memory: Tensor, cells: Tensor, owners: Tensor,
                      public_features: Tensor) -> Tensor:
        compact = self.flat_to_compact[cells]
        if (compact < 0).any():
            raise ValueError('Joint piece positions must be on-board cells')
        return (memory.gather(1, compact[..., None].expand(-1, -1, memory.shape[-1]))
                + self.owner_embedding(owners) + self.public_projection(public_features))

    def decode(self, memory: Tensor, context: Tensor, targets: Tensor) -> Tensor:
        prefix = F.pad(targets[:, :-1], (1, 0), value=N_BELIEF_TYPES)
        x = context + self.type_embedding(prefix)
        for block in self.decoder:
            x = block(x, memory)
        return self.head(self.decoder_norm(x))

    def teacher_logits(self, obs: Tensor, seats: Tensor, cells: Tensor, owners: Tensor,
                       public_features: Tensor, targets: Tensor, support: Tensor,
                       inventory: Tensor) -> Tensor:
        memory = self.encode(obs, seats)
        context = self.token_context(memory, cells, owners, public_features)
        logits = self.decode(memory, context, targets)
        # Only preceding target identities participate in inventory masks.
        used = (F.one_hot(targets, N_BELIEF_TYPES)[:, :, None, :]
                * F.one_hot(owners, 4)[:, :, :, None]).cumsum(1)
        used = used - (F.one_hot(targets, N_BELIEF_TYPES)[:, :, None, :]
                       * F.one_hot(owners, 4)[:, :, :, None])
        remaining = inventory[:, None] - used
        remaining = remaining.gather(2, owners[:, :, None, None].expand(-1, -1, 1, 12)).squeeze(2)
        allowed = support & self.feasible_types(remaining, self.future_demand(support, owners))
        if not allowed.gather(-1, targets[..., None]).all():
            raise ValueError('Targets violate public support or initial inventory')
        return logits.masked_fill(~allowed, -torch.inf)

    def forward(self, obs: Tensor, seats: Tensor, *, inputs, targets: Tensor):
        """Joint training contract deliberately requires a causal target prefix."""
        logits = self.teacher_logits(obs, seats, targets=targets, **inputs.kwargs())
        return {'logits': logits, 'log_probs': logits.log_softmax(-1)}

    @torch.no_grad()
    def sample(self, obs: Tensor, seats: Tensor, cells: Tensor, owners: Tensor,
               public_features: Tensor, support: Tensor, inventory: Tensor, *,
               n_samples: int, generator: torch.Generator | None = None) -> Tensor:
        """Return (B, S, K) valid assignments, including unknown dead pieces.

        Exact lookahead rejects identity choices with no feasible completion.
        Inconsistent public constraints raise instead of producing a world.
        """
        if self.training:
            raise ValueError('Joint sampling requires eval mode')
        if n_samples <= 0:
            raise ValueError('Sample count must be positive')
        memory = self.encode(obs, seats)
        context = self.token_context(memory, cells, owners, public_features)
        b, k, d = context.shape
        demand = self.future_demand(support, owners).repeat_interleave(n_samples, 0)
        memory = memory.repeat_interleave(n_samples, 0)
        context = context.repeat_interleave(n_samples, 0)
        owners = owners.repeat_interleave(n_samples, 0)
        support = support.repeat_interleave(n_samples, 0)
        inventory = inventory.repeat_interleave(n_samples, 0)
        memory_kv = [block.memory(memory) for block in self.decoder]
        result = torch.full((b*n_samples, k), -1, device=obs.device, dtype=torch.long)
        rows = torch.arange(b*n_samples, device=obs.device)
        cache = [(memory.new_empty(b*n_samples, block.heads, k, block.width),
                  memory.new_empty(b*n_samples, block.heads, k, block.width))
                 for block in self.decoder]
        remaining = inventory.clone()
        previous = torch.full_like(rows, N_BELIEF_TYPES)
        valid = torch.ones_like(rows, dtype=torch.bool)
        for i in range(k):
            x = context[:, i:i+1] + self.type_embedding(previous)[:, None]
            for block, mem, state in zip(self.decoder, memory_kv, cache):
                x = block.step(x, mem, state, i)
            logits = self.head(self.decoder_norm(x)).squeeze(1)
            allowed = support[:, i] & self.feasible_types(remaining[rows, owners[:, i]], demand[:, i])
            valid &= allowed.any(-1)
            safe = allowed | (~allowed.any(-1)[:, None] &
                              (torch.arange(12, device=obs.device)[None] == 0))
            u = torch.rand(logits.shape, device=obs.device, generator=generator)
            gumbel = -(-u.clamp(1e-7, 1-1e-7).log()).log()
            chosen = (logits + gumbel).masked_fill(~safe, -torch.inf).argmax(-1)
            result[:, i] = chosen
            remaining[rows, owners[:, i], chosen] -= 1
            previous = chosen
        if not valid.all() or not (remaining == 0).all():
            raise ValueError('Public support and inventory have no complete assignment')
        return result.reshape(b, n_samples, k)
