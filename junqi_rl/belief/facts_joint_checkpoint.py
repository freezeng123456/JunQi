"""Strict facts/history belief checkpoints and frozen-policy lineage.

Existing joint contracts remain byte-for-byte compatible. The generic loader
dispatches only recognized architectures, never guesses from tensor shapes.
"""
from dataclasses import asdict, fields
import hashlib
from pathlib import Path
import re

import torch

from junqi_core.knowledge import KNOWLEDGE_SCHEMA
from junqi_rl.belief.constraints import CONSTRAINT_SCHEMA
from junqi_rl.belief.joint import IDENTITY_INPUT_SCHEMA
from junqi_rl.checkpoint_compat import _input_layout_hash
from junqi_rl.networks.facts_joint_belief_net import FactsJointBeliefNet, FactsJointBeliefNetConfig
from junqi_rl.networks.facts_marginal_belief_net import FactsMarginalBeliefNet


INPUT_SPEC = {
    'format_version': 1,
    'identity_schema': IDENTITY_INPUT_SCHEMA,
    'knowledge_schema': KNOWLEDGE_SCHEMA,
    'identity_encoding': 'known_only',
    'observation_semantics_version': 5,
    'input_layout_sha256': _input_layout_hash(False, 'known_only'),
    'spatial_channels': 317, 'global_dims': 0, 'board_tokens': 129,
    'hidden_tokens': 75, 'types': 12,
    'sequence': 'relative-owner-then-initial-slot',
    'mode': 'DARK', 'includes_dead_pieces': True,
    'neural_history_window': 64, 'full_history_constraints': True,
    'history_features': 'observer-relative-public-events.v1',
    'constraint_semantics': CONSTRAINT_SCHEMA,
}


MARGINAL_INPUT_SPEC = {
    **INPUT_SPEC,
    'neural_identity_conditioning': 'observer-public-facts-only; no target or sampled identity prefixes',
    'training_support': 'public-type-domains-only; no teacher-prefix masks',
    'sampling_distribution': 'static-marginal-logits-sequential-feasible-prefix-renormalization.v1',
}

_MODEL_CONTRACTS = {
    FactsJointBeliefNet.architecture_id: (FactsJointBeliefNet, INPUT_SPEC),
    FactsMarginalBeliefNet.architecture_id: (FactsMarginalBeliefNet, MARGINAL_INPUT_SPEC),
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _strict_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_strict_equal(actual[k], value)
                                                      for k, value in expected.items())
    return actual == expected


def facts_belief_state(net, *, parent_policy_sha256, **training_state):
    contract = _MODEL_CONTRACTS.get(getattr(net, 'architecture_id', None))
    if contract is None or type(net) is not contract[0]:
        raise ValueError('Expected a supported facts belief model')
    if not isinstance(parent_policy_sha256, str) or not re.fullmatch('[0-9a-f]{64}', parent_policy_sha256):
        raise ValueError('A frozen parent policy SHA-256 is required')
    reserved = {'architecture_id', 'input_spec', 'config', 'net', 'parent_policy_sha256'}
    if reserved.intersection(training_state):
        raise ValueError('Training state cannot overwrite model contract fields')
    state = net.state_dict()
    if any(not torch.isfinite(tensor).all() for tensor in state.values()):
        raise FloatingPointError('Cannot save nonfinite facts belief weights')
    return {**training_state, 'architecture_id': net.architecture_id,
            'input_spec': dict(contract[1]), 'config': asdict(net.cfg),
            'net': state, 'parent_policy_sha256': parent_policy_sha256}


def load_facts_belief(path, *, policy_path, device='cpu', expected_kind=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    contract = _MODEL_CONTRACTS.get(saved.get('architecture_id')) if isinstance(saved, dict) else None
    if contract is None or (expected_kind is not None and contract[0].belief_kind != expected_kind):
        raise ValueError('Facts belief checkpoint architecture mismatch; legacy weights are incompatible')
    model_type, input_spec = contract
    if not _strict_equal(saved.get('input_spec'), input_spec):
        raise ValueError('Facts joint checkpoint input contract mismatch')
    if saved.get('parent_policy_sha256') != _sha256(policy_path):
        raise ValueError('Facts joint checkpoint parent policy mismatch')
    from junqi_rl.training.checkpoint import load_evaluation_policy
    parent = load_evaluation_policy(policy_path, device='cpu')
    if parent.identity_encoding != 'known_only' or parent.uses_global_summary:
        raise ValueError('Facts joint parent policy must use known_only and zero global inputs')
    del parent
    config = saved.get('config')
    if not isinstance(config, dict):
        raise ValueError('Facts joint checkpoint is missing its full model config')
    config_fields = fields(FactsJointBeliefNetConfig)
    if (set(config) != {field.name for field in config_fields}
            or any(type(config[field.name]) is not type(field.default) for field in config_fields)):
        raise ValueError('Facts joint checkpoint requires all config fields with their exact types')
    try:
        cfg = FactsJointBeliefNetConfig(**config)
    except (ValueError, TypeError) as exc:
        raise ValueError('Facts joint checkpoint model config mismatch') from exc
    if not _strict_equal(config, asdict(cfg)):
        raise ValueError('Facts joint checkpoint requires all config fields with their exact types')
    net = model_type(cfg)
    net.load_state_dict(saved['net'], strict=True)
    if any(not torch.isfinite(tensor).all() for tensor in net.state_dict().values()):
        raise FloatingPointError('Nonfinite facts belief checkpoint weights')
    return net.to(device).eval()


def facts_joint_state(net, *, parent_policy_sha256, **training_state):
    """Compatibility entry point that remains strict about the joint architecture."""
    if type(net) is not FactsJointBeliefNet:
        raise ValueError('Expected a facts joint model')
    return facts_belief_state(net, parent_policy_sha256=parent_policy_sha256, **training_state)


def load_facts_joint_belief(path, *, policy_path, device='cpu'):
    return load_facts_belief(path, policy_path=policy_path, device=device, expected_kind='joint')


def load_facts_marginal_belief(path, *, policy_path, device='cpu'):
    return load_facts_belief(path, policy_path=policy_path, device=device, expected_kind='marginal')
