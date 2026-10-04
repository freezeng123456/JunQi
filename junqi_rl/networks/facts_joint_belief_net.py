"""Learn complete hidden armies from observer facts and public history.

The transformer sees the most recent 64 public events. The constraint solver
sees the complete history, including combat relations for dead pieces. Neural
scores decide between feasible identities; constraints never assign weights.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from junqi_core.board import COMPACT_TO_FLAT
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import Seat
from junqi_rl.belief.joint import IDENTITY_INPUT_SCHEMA, IdentityConstraintInput
from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig
from junqi_rl.networks.causal_decoder import _DecoderBlock


@dataclass
class FactsJointBeliefNetConfig(BeliefNetConfig):
    n_decoder_layer: int = 2
    history_window: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.n_decoder_layer < 1:
            raise ValueError('Joint decoder must contain at least one layer')
        if self.history_window != 64:
            raise ValueError('The facts v1 history window is fixed at 64 events')


def _check_inputs(inputs: IdentityConstraintInput, batch: int) -> None:
    if not isinstance(inputs, IdentityConstraintInput) or inputs.schema != IDENTITY_INPUT_SCHEMA:
        raise ValueError('Facts joint network requires the facts-only identity schema')
    if inputs.cells.shape != (batch, 75):
        raise ValueError('Facts joint network requires 75 hidden identities per observer')
    if len(inputs.public_history) != batch:
        raise ValueError('Public history batch mismatch')


def _compile(inputs, constraints=None):
    if constraints is None:
        from junqi_rl.belief.constraints import compile_constraints
        constraints = [compile_constraints(inputs, row) for row in range(len(inputs.cells))]
    if len(constraints) != len(inputs.cells):
        raise ValueError('One compiled constraint object is required per observer')
    return constraints


def prepare_teacher_masks(inputs, targets, *, constraints=None) -> Tensor:
    """CPU preprocessing; cache these prefix-only masks with dataset samples.

    Each mask depends on public facts and *preceding* targets. Future labels
    never narrow the next-token support. The solver rejects invalid labels.
    """
    labels = targets.detach().cpu().numpy() if isinstance(targets, Tensor) else np.asarray(targets)
    _check_inputs(inputs, len(labels))
    if labels.shape != inputs.cells.shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Joint targets must be integer [batch, 75] identities')
    compiled = _compile(inputs, constraints)
    return torch.from_numpy(np.stack([solver.teacher_masks(row)
                                    for solver, row in zip(compiled, labels)]))


def history_features(inputs: IdentityConstraintInput, window: int = 64) -> np.ndarray:
    """Public-only numeric tokens, oldest to newest with left zero padding."""
    features = np.zeros((len(inputs.cells), window, 24), dtype=np.float32)
    for row, events in enumerate(inputs.public_history):
        observer = int(inputs.observer_seats[row])
        latest = events[-1].step if events else 0
        for offset, event in enumerate(events[-window:], start=max(0, window - len(events))):
            token = features[row, offset]
            token[(int(event.seat) - observer) % 4] = 1
            for col, cell in ((4, event.src), (6, event.dst)):
                x, y = world_to_canonical(cell % 17, cell // 17, Seat(observer))
                token[col:col + 2] = (x / 16, y / 16)
            for col, pid in ((8, event.attacker_pid), (9, event.defender_pid)):
                token[col] = (((pid // 30 - observer) % 4) * 30 + pid % 30 + 1) / 120 if pid >= 0 else 0
            token[10 + int(event.event) - 1] = 1
            token[14:18] = (event.flag_reveal_src, event.flag_reveal_dst,
                            event.flag_captured, event.engineer_revealed)
            for seat in event.seats_died:
                token[18 + (int(seat) - observer) % 4] = 1
            token[22] = np.log1p(latest - event.step) / np.log(4001)
            token[23] = 1
    return features


class _FactsBeliefNet(BeliefNet):
    """Shared public-fact encoder; subclasses choose the identity decoder."""
    identity_encoding = 'known_only'
    uses_global_summary = False

    def __init__(self, cfg: FactsJointBeliefNetConfig | None = None, *, prefix_embedding: bool):
        super().__init__(cfg or FactsJointBeliefNetConfig())
        d = self.cfg.embed_dim
        if prefix_embedding:
            self.type_embedding = nn.Embedding(13, d)
        else:
            # Keep common parameter initialization identical for matched seeds
            # without registering an unused identity-prefix parameter/module.
            torch.empty(13, d).normal_()
        self.owner_embedding = nn.Embedding(4, d)
        self.initial_embedding = nn.Embedding(289, d)
        self.alive_embedding = nn.Embedding(2, d)
        self.known_embedding = nn.Embedding(13, d)
        self.count_projection = nn.Linear(3, d)
        self.domain_projection = nn.Linear(12, d)
        self.history_projection = nn.Linear(24, d)
        self.history_position = nn.Embedding(self.cfg.history_window, d)
        self.decoder = nn.ModuleList(_DecoderBlock(self.cfg) for _ in range(self.cfg.n_decoder_layer))
        self.decoder_norm = nn.LayerNorm(d)
        self.register_buffer('on_board_cells', torch.tensor(COMPACT_TO_FLAT.copy(), dtype=torch.long),
                             persistent=False)
        self.last_sample_metrics = {}

    def encode_facts(self, obs: Tensor, inputs: IdentityConstraintInput):
        _check_inputs(inputs, len(obs))
        if tuple(obs.shape[1:]) != (317, 17, 17):
            raise ValueError('Facts spatial observations must have shape [batch, 317, 17, 17]')
        device = obs.device

        def tensor(array, dtype=torch.long):
            # numpy fact arrays are read-only; copy avoids writable tensor aliases.
            return torch.tensor(np.asarray(array), dtype=dtype, device=device)

        seats = tensor(inputs.observer_seats)
        # Keep all 289 positions until context is gathered; attention itself
        # operates on the compact 129 playable cells.
        board = super().encode(obs, seats, cell_indices=self.on_board_cells)
        reverse = torch.full((289,), -1, dtype=torch.long, device=device)
        reverse[self.on_board_cells] = torch.arange(129, device=device)
        cells = tensor(np.where(inputs.cells >= 0, inputs.cells, inputs.initial_cells))
        initial = tensor(inputs.initial_cells)
        compact = reverse[cells]
        if (compact < 0).any() or (reverse[initial] < 0).any():
            raise ValueError('Facts piece anchors must be playable board cells')
        context = board.gather(1, compact[..., None].expand(-1, -1, board.shape[-1]))
        context = (context + self.owner_embedding(tensor(inputs.owners))
                   + self.initial_embedding(initial)
                   + self.alive_embedding(tensor(inputs.alive))
                   + self.known_embedding(tensor(inputs.known_types + 1))
                   + self.count_projection(torch.log1p(tensor(inputs.public_counts, torch.float32)))
                   + self.domain_projection(tensor(inputs.type_domains, torch.float32)))
        history = tensor(history_features(inputs, self.cfg.history_window), torch.float32)
        tokens = (self.history_projection(history)
                  + self.history_position(torch.arange(self.cfg.history_window, device=device))[None])
        tokens = tokens * history[..., -1:]
        return torch.cat((board, tokens), dim=1), context

class FactsJointBeliefNet(_FactsBeliefNet):
    architecture_id = 'junqi.belief317.facts-autoregressive.v1'
    belief_kind = 'joint'
    distribution_semantics = 'autoregressive neural conditionals under exact public-history prefix constraints'

    def __init__(self, cfg: FactsJointBeliefNetConfig | None = None):
        super().__init__(cfg, prefix_embedding=True)

    @staticmethod
    def prepare_teacher_masks(inputs, targets, *, constraints=None):
        return prepare_teacher_masks(inputs, targets, constraints=constraints)

    def decode(self, memory: Tensor, context: Tensor, targets: Tensor) -> Tensor:
        prefix = F.pad(targets[:, :-1], (1, 0), value=12)
        x = context + self.type_embedding(prefix)
        for block in self.decoder:
            x = block(x, memory)
        return self.head(self.decoder_norm(x))

    def teacher_logits(self, obs: Tensor, *, inputs: IdentityConstraintInput,
                       targets: Tensor, constraints=None, teacher_masks: Tensor | None = None) -> Tensor:
        if targets.dtype != torch.long or tuple(targets.shape) != (len(obs), 75):
            raise ValueError('Joint targets must be int64 [batch, 75] identities')
        if teacher_masks is None:
            teacher_masks = prepare_teacher_masks(inputs, targets, constraints=constraints)
        if teacher_masks.dtype != torch.bool or tuple(teacher_masks.shape) != (len(obs), 75, 12):
            raise ValueError('Teacher masks must be bool [batch, 75, 12]')
        masks = teacher_masks.to(obs.device)
        targets = targets.to(obs.device)
        if not masks.gather(-1, targets[..., None]).all():
            raise ValueError('Targets violate public facts or prior target prefix')
        memory, context = self.encode_facts(obs, inputs)
        return self.decode(memory, context, targets).masked_fill(~masks, -torch.inf)

    def forward(self, obs: Tensor, *, inputs, targets, constraints=None, teacher_masks=None):
        logits = self.teacher_logits(obs, inputs=inputs, targets=targets,
                                     constraints=constraints, teacher_masks=teacher_masks)
        return {'logits': logits, 'log_probs': logits.log_softmax(-1)}

    @torch.no_grad()
    def sample(self, obs: Tensor, *, inputs: IdentityConstraintInput, n_samples: int,
               generator: torch.Generator | None = None, constraints=None,
               overlap_constraints: bool = False) -> Tensor:
        if self.training:
            raise ValueError('Joint sampling requires eval mode')
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError('Sample count must be a positive integer')
        if type(overlap_constraints) is not bool:
            raise ValueError('Constraint overlap must be a boolean')
        _check_inputs(inputs, len(obs))
        began = time.perf_counter()
        compiled = _compile(inputs, constraints)
        memory, context = self.encode_facts(obs, inputs)
        b, k, d = context.shape
        memory = memory.repeat_interleave(n_samples, 0)
        context = context.repeat_interleave(n_samples, 0)
        memory_kv = [block.memory(memory) for block in self.decoder]
        cache = [(memory.new_empty(b * n_samples, block.heads, k, block.width),
                  memory.new_empty(b * n_samples, block.heads, k, block.width))
                 for block in self.decoder]
        prefixes = np.full((b * n_samples, k), -1, dtype=np.int64)
        previous = torch.full((b * n_samples,), 12, dtype=torch.long, device=obs.device)
        constraint_seconds = 0.0
        for index in range(k):
            if overlap_constraints:
                # This token's neural scores depend on the previous label,
                # but not on its feasibility mask. CUDA work can execute while
                # the host solves that exact prefix. No RNG call moves across
                # the boundary, and the finite/mask checks remain after it.
                x = context[:, index:index + 1] + self.type_embedding(previous)[:, None]
                for block, mem, state in zip(self.decoder, memory_kv, cache):
                    x = block.step(x, mem, state, index)
                logits = self.head(self.decoder_norm(x)).squeeze(1)
            start = time.perf_counter()
            allowed = np.stack([compiled[row // n_samples].allowed_next(prefixes[row, :index])
                                for row in range(b * n_samples)])
            constraint_seconds += time.perf_counter() - start
            if not allowed.any(-1).all():
                raise ValueError('Public history has no feasible complete assignment')
            if not overlap_constraints:
                x = context[:, index:index + 1] + self.type_embedding(previous)[:, None]
                for block, mem, state in zip(self.decoder, memory_kv, cache):
                    x = block.step(x, mem, state, index)
                logits = self.head(self.decoder_norm(x)).squeeze(1)
            mask = torch.tensor(allowed, dtype=torch.bool, device=obs.device)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite neural belief scores')
            probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
            previous = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            # This reference adapter synchronizes once per token. The boundary
            # and its measured cost are explicit; no false GPU-throughput claim.
            prefixes[:, index] = previous.cpu().numpy()
        for row, labels in enumerate(prefixes):
            compiled[row // n_samples].validate(labels)
        self.last_sample_metrics = {
            'samples': b * n_samples, 'tokens_per_sample': k,
            'constraint_seconds': constraint_seconds,
            'total_seconds': time.perf_counter() - began,
            'constraint_backend': 'cpu-exact-public-history',
            'neural_device': str(obs.device), 'token_host_syncs': k,
            'overlap_constraints': overlap_constraints,
            'timing_scope': ('constraint CPU work overlaps submitted decoder work; '
                             'do not add it to other GPU phase durations'
                             if overlap_constraints else 'serial decoder and constraint work'),
        }
        return torch.tensor(prefixes, dtype=torch.long, device=obs.device).reshape(b, n_samples, k)
