"""Audit an optional fixed-length no-belief policy follow-up and fresh acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.belief_features_20260913.analyze import read, verify_manifest
from experiments.belief_features_20260913.summarize_policy_eval import inspect


def analyze(root):
    folder = root / 'phase_a_B'
    inventory = verify_manifest(folder)
    launch = read(folder / 'launcher_exit.json')
    summary = read(folder / 'summary.json')
    assert launch['exit_code'] == 0 and not launch['deadline_fired']
    assert summary['status'] == 'complete' and (folder / '.done').is_file()
    rollouts = summary['rollouts']
    assert rollouts in (32, 64)
    checkpoint = folder / 'training' / f'ckpt_{rollouts:06d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    initial = torch.load(root / 'source_inputs/phase_a_initializer_v4.pt', map_location='cpu', weights_only=False)
    assert state['num_rollout'] == rollouts and state['num_train_step'] > 0
    assert state['checkpoint_meta']['observation_semantics_version'] == 4
    assert 'belief' not in state and 'arrangement' not in state
    assert not any('ema' in key.lower() for key in state)
    cfg = state['train_cfg']
    assert not cfg['belief']['enabled'] and not cfg['arr']['enabled'] and cfg['random_opponent']
    assert cfg['total_rollouts'] == rollouts
    assert cfg['env']['num_envs'] == 128 and cfg['env']['steps_per_env'] == 512
    assert cfg['ppo']['lr_coef'] == cfg['ppo']['lr_floor'] == cfg['ppo']['lr_ceil'] == 1e-5
    assert state['policy'].keys() == initial['policy'].keys()
    changed = sum(not torch.equal(value, initial['policy'][name]) for name, value in state['policy'].items())
    assert changed > 0 and all(torch.isfinite(value).all() for value in state['policy'].values())
    with checkpoint.open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    raw, left = inspect(folder / 'acceptance_raw')
    candidate, right = inspect(folder / 'acceptance_candidate')
    assert left.keys() == right.keys() and len(left) == 2048
    for key in left:
        assert left[key]['setup_sha256'] == right[key]['setup_sha256']
        assert left[key]['random_seed'] == right[key]['random_seed']
    for report in (raw, candidate):
        p = report['provenance']
        assert p['setup_seed'] == 3031100 and p['random_seed'] == 4031100
        assert not p['ema'] and not p['learned_belief_in_policy']
        assert p['backend'] == 'cuda' and p['dtype'] == 'bfloat16' and p['greedy']
        assert p['effective_checkpoint_metadata']['observation_semantics_version'] == 4
    assert candidate['provenance']['checkpoint_sha256'] == checkpoint_hash
    assert raw['provenance']['checkpoint_sha256'] == '4e75bb094122b119ba15caf94632fcd4e173dbb77ce08d4e6ab9c6bd7716756a'
    for name in ('source_commit', 'runner_sha256', 'games_per_block', 'blocks', 'max_moves'):
        assert raw['provenance'][name] == candidate['provenance'][name]
    seeds = sorted({key[0] for key in left})
    delta = np.array([sum((right[seed, team]['outcome'] == 'win') -
                         (left[seed, team]['outcome'] == 'win') for team in (0, 1)) / 2 for seed in seeds])
    rng = np.random.default_rng(620915)
    bootstrap = delta[rng.integers(len(seeds), size=(5000, len(seeds)))].mean(1)
    return {'status': 'phase_a_training_and_paired_acceptance_verified',
            'verified_files': inventory, 'rollouts': rollouts,
            'environment_steps': rollouts * 128 * 512,
            'policy_gradient_steps': state['num_train_step'], 'changed_policy_tensors': changed,
            'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': checkpoint_hash,
            'no_ema_no_learned_belief': True, 'fixed_final_checkpoint_selection': True,
            'raw': raw, 'candidate': candidate,
            'candidate_minus_raw_win_rate': float(delta.mean()),
            'paired_setup_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'scope': 'Conditional on the two fixed checkpoints and the new 1024 paired setups; '
                     'this does not erase the earlier raw-policy suite with 2045 wins, one loss and two draws.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.root / 'analysis_policy/phase_a.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'rollouts': result['rollouts'],
                      'raw': result['raw']['summary']['metrics'],
                      'candidate': result['candidate']['summary']['metrics']}), flush=True)
