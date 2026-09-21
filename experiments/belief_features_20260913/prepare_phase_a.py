"""Explicit raw-policy warm start with current metadata and a fresh optimizer."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from junqi_rl.checkpoint_compat import validate_policy_checkpoint
from junqi_rl.networks.junqi_net import JunqiNet
from junqi_rl.training.config import load_config
from junqi_rl.training.ppo import PPOTrainer


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(1)
    cfg = load_config(argparse.Namespace(config=str(args.config)))
    assert not cfg.belief.enabled and not cfg.arr.enabled
    original_hash = digest(args.source)
    original = torch.load(args.source, map_location='cpu', weights_only=False)
    original_metadata = dict(original['checkpoint_meta'])
    assert original_metadata['observation_semantics_version'] in (3, 4)
    migrated = dict(original)
    migrated['checkpoint_meta'] = dict(original_metadata, observation_semantics_version=4)
    policy = JunqiNet(cfg.net)
    validate_policy_checkpoint(policy, migrated, source=str(args.source) + ' explicit v4 warm start')
    policy.load_state_dict(original['policy'], strict=True)
    trainer = PPOTrainer(policy, cfg.ppo, device='cpu')
    fresh = trainer.state_dict()
    assert fresh['num_rollout'] == fresh['num_train_step'] == 0
    assert not fresh['optimizer']['state']
    assert all(torch.equal(value, original['policy'][key]) for key, value in fresh['policy'].items())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fresh, args.output)
    reloaded = torch.load(args.output, map_location='cpu', weights_only=False)
    trainer.load_state_dict(reloaded)
    assert all(torch.equal(value, original['policy'][key]) for key, value in trainer.policy.state_dict().items())
    assert digest(args.source) == original_hash
    report = {
        'status': 'initializer_prepared_and_reloaded; no training run',
        'source': str(args.source.resolve()), 'source_sha256': original_hash,
        'initializer': str(args.output.resolve()), 'initializer_sha256': digest(args.output),
        'original_metadata': original_metadata, 'new_metadata': fresh['checkpoint_meta'],
        'all_raw_policy_tensors_exactly_equal': True, 'optimizer_state_entries': 0,
        'rollout_counter': 0, 'gradient_step_counter': 0, 'belief_enabled': False,
        'arrangement_enabled': False, 'ema': False, 'effective_config': asdict(cfg),
        'scope': 'New v4 warm-start experiment; does not reproduce the historical v3 run.',
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'effective_config'}), flush=True)


if __name__ == '__main__':
    main()
