"""Matched public-fact marginal control, with no identity-prefix features.

The encoder and decoder blocks match the joint network. The causal decoder
only attends to public piece contexts, so every neural prediction is a
function of observer facts alone. Supervision uses public type domains;
true-label prefix masks must never enter the marginal objective.

At inference static marginal scores are renormalized sequentially over exact
feasible prefix supports. This defines a constrained sequential proposal,
not the independent product conditioned on global feasibility.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch import Tensor

from junqi_rl.belief.joint import IdentityConstraintInput
from junqi_rl.networks.facts_joint_belief_net import (
    FactsJointBeliefNetConfig, _FactsBeliefNet, _check_inputs, _compile,
)


FactsMarginalBeliefNetConfig = FactsJointBeliefNetConfig


class FactsMarginalBeliefNet(_FactsBeliefNet):
    architecture_id = 'junqi.belief317.facts-marginal.v1'
    belief_kind = 'marginal'
    distribution_semantics = (
        'static public-only marginal logits sequentially renormalized over exact '
        'feasible prefix supports; not a globally conditioned independent product')

    def __init__(self, cfg: FactsMarginalBeliefNetConfig | None = None):
        super().__init__(cfg, prefix_embedding=False)

    def decode(self, memory: Tensor, context: Tensor) -> Tensor:
        x = context
        for block in self.decoder:
            x = block(x, memory)
        return self.head(self.decoder_norm(x))

    def marginal_logits(self, obs: Tensor, *, inputs: IdentityConstraintInput) -> Tensor:
        memory, context = self.encode_facts(obs, inputs)
        logits = self.decode(memory, context)
        masks = torch.tensor(np.asarray(inputs.type_domains), dtype=torch.bool, device=obs.device)
        if masks.shape != logits.shape or not masks.any(-1).all():
            raise ValueError('Marginal public type domains must be nonempty [batch, 75, 12]')
        return logits.masked_fill(~masks, -torch.inf)

    def forward(self, obs: Tensor, *, inputs: IdentityConstraintInput):
        # Deliberately no targets/teacher_masks arguments at this boundary.
        logits = self.marginal_logits(obs, inputs=inputs)
        return {'logits': logits, 'log_probs': logits.log_softmax(-1)}

    @torch.no_grad()
    def sample(self, obs: Tensor, *, inputs: IdentityConstraintInput, n_samples: int,
               generator: torch.Generator | None = None, constraints=None) -> Tensor:
        if self.training:
            raise ValueError('Marginal sampling requires eval mode')
        if type(n_samples) is not int or n_samples < 1:
            raise ValueError('Sample count must be a positive integer')
        _check_inputs(inputs, len(obs))
        began = time.perf_counter()
        compiled = _compile(inputs, constraints)
        # Decoder runs once, independently of all generated prefixes.
        memory, context = self.encode_facts(obs, inputs)
        logits = self.decode(memory, context)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite neural belief scores')
        logits = logits.repeat_interleave(n_samples, 0)
        b, k = len(obs), logits.shape[1]
        prefixes = np.full((b * n_samples, k), -1, dtype=np.int64)
        constraint_seconds = 0.0
        for index in range(k):
            start = time.perf_counter()
            allowed = np.stack([compiled[row // n_samples].allowed_next(prefixes[row, :index])
                                for row in range(b * n_samples)])
            constraint_seconds += time.perf_counter() - start
            if not allowed.any(-1).all():
                raise ValueError('Public history has no feasible complete assignment')
            mask = torch.tensor(allowed, dtype=torch.bool, device=obs.device)
            probabilities = logits[:, index].masked_fill(~mask, -torch.inf).softmax(-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            prefixes[:, index] = sampled.cpu().numpy()
        for row, labels in enumerate(prefixes):
            compiled[row // n_samples].validate(labels)
        self.last_sample_metrics = {
            'samples': b * n_samples, 'tokens_per_sample': k,
            'constraint_seconds': constraint_seconds,
            'total_seconds': time.perf_counter() - began,
            'constraint_backend': 'cpu-exact-public-history',
            'neural_device': str(obs.device), 'token_host_syncs': k,
        }
        return torch.tensor(prefixes, dtype=torch.long, device=obs.device).reshape(b, n_samples, k)
