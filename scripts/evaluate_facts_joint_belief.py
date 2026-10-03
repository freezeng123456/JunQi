"""Open the held-out test fold after freezing the validation-selected model.

Reports architecture-specific conditional or marginal NLL and accuracy. This audit never selects
or updates weights and must not be used as a replacement for actual matches.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from types import SimpleNamespace

import torch

from junqi_rl.belief.facts_dataset import concatenate_shards, file_sha256
from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
from scripts.train_facts_joint_belief import (
    _json_write, _validate_manifest, evaluate, metric_names, metric_scope,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--belief', '--joint', dest='belief', required=True,
                        help='Frozen validation-selected facts belief checkpoint')
    parser.add_argument('--belief-kind', choices=('joint', 'marginal'), default=None)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True, help='Audit JSON file')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-threads', type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.num_threads) < 1:
        raise ValueError('Batch size and CPU threads must be positive')
    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)
    net = load_facts_belief(args.belief, policy_path=args.policy, device=device,
                           expected_kind=args.belief_kind)
    kind = net.belief_kind
    nll_key, _ = metric_names(kind)
    saved = torch.load(args.belief, map_location='cpu', weights_only=False)
    if saved.get('artifact_role') != f'validation-selected-facts-{kind}-belief':
        raise ValueError('Final test requires the validation-selected artifact')
    contract = saved.get('training_contract', {})
    parent_hash = file_sha256(args.policy)
    manifest, manifest_hash = _validate_manifest(SimpleNamespace(dataset=args.dataset,
        games=contract['games'], positions_per_game=contract['positions_per_game']), parent_hash)
    if (contract.get('dataset_manifest_sha256') != manifest_hash
            or contract.get('parent_policy_sha256') != parent_hash
            or contract.get('test_fold_used') is not False):
        raise ValueError('Final test dataset/parent/selection contract mismatch')
    epoch = saved.get('selected_epoch')
    metrics = saved.get('epoch_metrics', [])
    if (type(epoch) is not int or epoch < 1 or epoch > len(metrics)
            or min(range(len(metrics)), key=lambda i: metrics[i]['validation'][nll_key]) + 1 != epoch):
        raise ValueError('Selected epoch does not match frozen validation selection')
    protocol = {'schema': f'junqi.facts-{kind}-final-test.v1',
                'joint_checkpoint_sha256': file_sha256(args.belief),
                'parent_policy_sha256': parent_hash, 'dataset_manifest_sha256': manifest_hash,
                'selected_epoch': epoch, 'split': 'test', 'weights_updated': False,
                'selection_performed_on_test': False, 'batch_size': args.batch_size,
                'device_type': device.type}
    if kind == 'marginal':
        protocol['belief_checkpoint_sha256'] = protocol.pop('joint_checkpoint_sha256')
        protocol['belief_kind'] = kind
        protocol['architecture_id'] = net.architecture_id
    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text())
        if existing.get('protocol') != protocol:
            raise ValueError('Existing test audit belongs to different frozen weights or protocol')
        print(json.dumps(existing, indent=2), flush=True)
        return existing
    began = time.perf_counter()
    test = concatenate_shards(args.dataset, 'test')
    report = {'protocol': protocol, 'test': evaluate(net, test, batch_size=args.batch_size, device=device),
              'test_completed_games': manifest['split_games']['test'],
              'validation_at_selection': metrics[epoch - 1]['validation'],
              'test_fold_used': True, 'seconds': time.perf_counter() - began,
              'metric_scope': metric_scope(kind)}
    _json_write(output, report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main()
