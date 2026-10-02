"""V100 gate: exact belief-training resume, feasible samples and legal search.

Requires the acceptance-only 40-complete-game, two-position, 64-move dataset.
The four executed training epochs are disposable engineering acceptance work;
they do not contribute to the formal 2,000-game experiment or its test fold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from junqi_rl.belief.facts_dataset import batch_inputs, file_sha256, iter_shards
from junqi_rl.belief.facts_joint_checkpoint import load_facts_joint_belief
from scripts.train_facts_joint_belief import _json_write, build_parser as training_parser, train


def _tree_digest(root, *, suffixes=None):
    """Read-only identity for source/data files, excluding Python bytecode."""
    root = Path(root)
    entries = []
    for path in sorted(root.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if suffixes is None or path.suffix in suffixes:
            entries.append((path.relative_to(root).as_posix(), file_sha256(path)))
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def _source_digest():
    entries = {}
    suffixes = {'.py', '.cu', '.cpp', '.c', '.h', '.hpp', '.cuh', '.yaml', '.yml'}
    for name in ('junqi_core', 'junqi_rl', 'scripts', 'src', 'configs', 'tests'):
        path = _ROOT / name
        if path.exists():
            entries[name] = _tree_digest(path, suffixes=suffixes)
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def assert_exact(left, right, path='state'):
    """Compare every tensor/scalar recursively; no tolerance for resume drift."""
    if type(left) is not type(right):
        raise AssertionError(f'{path}: state type changed')
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, check_dtype=True)
        return 1
    if isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
        return 1
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: state keys changed')
        return sum(assert_exact(value, right[key], f'{path}.{key}') for key, value in left.items())
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: state length changed')
        return sum(assert_exact(a, b, f'{path}[{index}]') for index, (a, b) in enumerate(zip(left, right)))
    if left != right:
        raise AssertionError(f'{path}: scalar state changed')
    return 0


def _assert_finite(value, path='state'):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise AssertionError(f'{path}: nonfinite tensor')
        return 1
    if isinstance(value, dict):
        return sum(_assert_finite(item, f'{path}.{key}') for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return sum(_assert_finite(item, f'{path}[{i}]') for i, item in enumerate(value))
    if isinstance(value, float) and not np.isfinite(value):
        raise AssertionError(f'{path}: nonfinite scalar')
    return 0


def _hardware():
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA V100 is required; CPU fallback and skipped acceptance are forbidden')
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if 'v100' not in name.lower():
        raise RuntimeError(f'Acceptance requires a V100; found {name}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {'name': name, 'capability': list(torch.cuda.get_device_capability(0)),
            'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
            'cuda_device_count': torch.cuda.device_count()}


def _smoke_protocol(dataset):
    manifest = json.loads((Path(dataset) / 'dataset-manifest.json').read_text())
    protocol = manifest.get('protocol', {})
    expected = {'games': 40, 'positions_per_game': 2, 'max_moves': 64, 'acceptance_only': True}
    if any(type(protocol.get(key)) is not type(value) or protocol.get(key) != value
           for key, value in expected.items()):
        raise ValueError('Gate requires exactly 40 games x 2 positions, max_moves=64, acceptance_only=true')
    if manifest.get('status') != 'complete' or manifest.get('completed_games') != 40:
        raise ValueError('Gate requires all 40 complete games and their supervised shards')
    return expected


def _training_audit(args, artifact_root):
    def run(name, epochs, resume=None):
        argv = ['--policy', str(args.policy), '--dataset', str(args.dataset),
                '--output', str(artifact_root / name), '--device', 'cuda',
                '--games', '40', '--positions-per-game', '2', '--epochs', str(epochs),
                '--batch-size', '8', '--num-threads', '2', '--seed', '7300300']
        if resume is not None:
            argv += ['--resume', str(resume)]
        return train(training_parser().parse_args(argv))

    partial_report = run('partial', 1)
    resumed_report = run('resumed', 2, artifact_root / 'partial' / 'last.pt')
    full_report = run('full', 2)
    partial = torch.load(artifact_root / 'partial' / 'last.pt', map_location='cpu', weights_only=False)
    resumed = torch.load(artifact_root / 'resumed' / 'last.pt', map_location='cpu', weights_only=False)
    full = torch.load(artifact_root / 'full' / 'last.pt', map_location='cpu', weights_only=False)
    comparisons = {key: assert_exact(full[key], resumed[key], key)
                   for key in ('net', 'optimizer', 'best_model_state', 'rng', 'training_contract')}
    for key in ('updates', 'epochs_completed', 'best_epoch', 'best_validation_nll'):
        assert_exact(full[key], resumed[key], key)
    finite_count = sum(_assert_finite(value, name) for name, value in (
        ('partial.net', partial['net']), ('resumed.net', resumed['net']),
        ('full.net', full['net']), ('resumed.optimizer', resumed['optimizer']),
        ('full.optimizer', full['optimizer'])))
    changed = [key for key in full['net'] if not torch.equal(partial['net'][key], full['net'][key])]
    if not changed or resumed['updates'] <= partial['updates']:
        raise AssertionError('Training resume made no finite parameter/optimizer progress')
    selected = artifact_root / 'full' / 'joint-belief.pt'
    net = load_facts_joint_belief(selected, policy_path=args.policy, device='cuda')
    selected_state = torch.load(selected, map_location='cpu', weights_only=False)
    assert_exact(selected_state['net'], full['best_model_state'], 'selected_best_weights')
    report = {'acceptance_only': True, 'executed_epochs': 4, 'unique_target_epochs': 2,
              'minibatch_size': 8, 'model_config': selected_state['config'],
              'updates_before_resume': partial['updates'], 'updates_after_resume': resumed['updates'],
              'resume_matches_uninterrupted_exactly': True, 'compared_tensors_or_arrays': comparisons,
              'finite_tensors_checked': finite_count, 'changed_parameter_tensors_after_resume': len(changed),
              'parameters_changed': True, 'selected_artifact_loads': True,
              'selected_artifact_sha256': file_sha256(selected),
              'best_epoch': full_report['best_epoch'],
              'initial_validation': full_report['initial_validation'],
              'best_validation': full_report['best_validation'],
              'test_fold_used': False,
              'partial_training_seconds': partial_report['training_seconds'],
              'resumed_training_seconds': resumed_report['training_seconds'],
              'full_training_seconds': full_report['training_seconds']}
    return net, report


def _sample_audit(net, dataset):
    from junqi_rl.belief.constraints import compile_constraints
    shard = next(iter_shards(dataset, 'validation'))
    inputs = batch_inputs(shard, np.asarray([0], dtype=np.int64))
    obs = torch.tensor(np.array(shard['spatial'][:1], copy=True), device='cuda', dtype=torch.float32)
    solver = compile_constraints(inputs)
    samples = net.sample(obs, inputs=inputs, n_samples=1, constraints=[solver],
                         generator=torch.Generator(device='cuda').manual_seed(7400300))
    if samples.shape != (1, 1, 75) or samples.device.type != 'cuda':
        raise AssertionError('Actual loaded model returned malformed or non-CUDA samples')
    solver.validate(samples[0, 0].cpu().numpy())
    return {'split': 'validation', 'rows': 1, 'joint_assignments': 1, 'identities_per_assignment': 75,
            'all_assignments_feasible': True, 'sampling': net.last_sample_metrics,
            'constraint_stats': dict(solver.stats)}


def _search_audit(net, policy_path):
    from junqi_core.rules import ShowMode
    from junqi_rl.analysis.facts_joint_search import FactsBlueprint, FactsJointSearch, SearchConfig
    from junqi_rl.env import JunqiEnv, rotate_to_compact_action_id
    from junqi_rl.training.checkpoint import load_evaluation_policy
    policy = load_evaluation_policy(policy_path, device='cuda')
    blueprint = FactsBlueprint(policy, 'cuda')
    search = FactsJointSearch(blueprint, net, SearchConfig(particles=1, depth=1), seed=7500300)
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False, identity_encoding='known_only',
                   record_public_history=True)
    env.reset(seed=7600300)
    legal = env.legal_action_ids()
    action = search.action(env, [])
    report = search.report()
    if action not in legal or report['action_coverage'] != 1.0:
        raise AssertionError('Search selected an illegal action or failed to score every legal action')
    if (report['successful_decisions'] != 1 or report['candidate_actions'] != len(legal)
            or report['sample_budget_failures'] or report['constraint_budget_failures']):
        raise AssertionError('Facts search failed its all-action one-decision contract')
    compact = [rotate_to_compact_action_id(int(aid), env.state.turn) for aid in legal]
    if not torch.isfinite(search.last_q[0, compact]).all():
        raise AssertionError('Not every legal action received a finite searched value')
    return {**report, 'selected_world_action': int(action), 'selected_action_legal': True,
            'all_legal_action_values_finite': True, 'smoke_scope': 'one fresh opening decision'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', '--checkpoint', dest='policy', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Acceptance JSON; outside source dataset')
    args = parser.parse_args(argv)
    artifact_root = args.output.with_name(args.output.stem + '-artifacts')
    if args.output.resolve().is_relative_to(args.dataset.resolve()):
        parser.error('acceptance output must be outside the unchanged source dataset')
    if args.output.resolve() == args.policy.resolve():
        parser.error('acceptance output must not overwrite the parent checkpoint')
    report = {'schema': 'junqi.facts-belief-gpu-training-acceptance.v1', 'status': 'failed',
              'acceptance_only': True}
    phase = 'cuda_preflight'
    started = time.perf_counter()
    try:
        report['hardware'] = _hardware()
        phase = 'immutable_inputs'
        if artifact_root.exists():
            raise ValueError('Acceptance artifacts already exist; use a fresh output JSON name')
        report['dataset_protocol'] = _smoke_protocol(args.dataset)
        report['parent_checkpoint_sha256'] = file_sha256(args.policy)
        report['dataset_tree_sha256'] = _tree_digest(args.dataset)
        report['source_tree_sha256'] = _source_digest()
        phase = 'training_resume'
        net, report['training'] = _training_audit(args, artifact_root)
        phase = 'loaded_model_sample'
        report['sampling'] = _sample_audit(net, args.dataset)
        phase = 'facts_search'
        report['search'] = _search_audit(net, args.policy)
        phase = 'immutable_input_verification'
        torch.cuda.synchronize()
        if (file_sha256(args.policy) != report['parent_checkpoint_sha256']
                or _tree_digest(args.dataset) != report['dataset_tree_sha256']
                or _source_digest() != report['source_tree_sha256']):
            raise AssertionError('Parent checkpoint, dataset or source files changed during acceptance')
        report['source_dataset_and_parent_unchanged'] = True
        report['status'] = 'passed'
    except Exception as error:
        report['failed_phase'] = phase
        report['error'] = f'{type(error).__name__}: {error}'
    report['elapsed_seconds'] = time.perf_counter() - started
    _json_write(args.output, report)
    print(json.dumps({'status': report['status'], 'audit': str(args.output),
                      'failed_phase': report.get('failed_phase')}), flush=True)
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
