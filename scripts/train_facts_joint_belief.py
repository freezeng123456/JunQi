"""Train matched facts-only joint or marginal beliefs from frozen pi0.

Collection defaults to 2,000 complete games and eight positions per game.
This command validates that budget, trains only on the train fold and selects
the best epoch on validation NLL. The held-out test fold remains unopened.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict
import json
import os
import random
import time

import numpy as np
import torch

from junqi_rl.belief.facts_dataset import (
    DATASET_SCHEMA, batch_inputs, concatenate_shards, file_sha256, protocol_sha256,
    split_for_setup, validate_constraint_schema,
)
from junqi_rl.belief.facts_joint_checkpoint import facts_belief_state, load_facts_belief
from junqi_rl.networks.facts_joint_belief_net import FactsJointBeliefNet, FactsJointBeliefNetConfig
from junqi_rl.networks.facts_marginal_belief_net import FactsMarginalBeliefNet
from junqi_rl.training._checkpoint_io import atomic_write
from junqi_rl.training.checkpoint import load_evaluation_policy


TRAINING_SCHEMA = 'junqi.facts-joint-training.v1'
MARGINAL_TRAINING_SCHEMA = 'junqi.facts-marginal-training.v1'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--belief-kind', choices=('joint', 'marginal'), default='joint')
    parser.add_argument('--policy', required=True, help='Frozen facts-only pi0 checkpoint')
    parser.add_argument('--dataset', required=True, help='Completed facts dataset directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--positions-per-game', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10, help='Total target epochs, including resumed epochs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261004, help='Model/minibatch seed, separate from collection')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-threads', type=int, default=2)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--cnn-channels', type=int, default=32)
    parser.add_argument('--cnn-layers', type=int, default=2)
    parser.add_argument('--encoder-layers', type=int, default=2)
    parser.add_argument('--decoder-layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--ff-factor', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--resume', default='', help='last.pt from this exact data/model/optimizer contract')
    return parser


def _json_write(path, payload):
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    atomic_write(path, lambda stream: stream.write(encoded))


def _validate_manifest(args, parent_hash):
    root = Path(args.dataset)
    path = root / 'dataset-manifest.json'
    manifest = json.loads(path.read_text())
    protocol = manifest.get('protocol', {})
    if manifest.get('schema') != DATASET_SCHEMA or manifest.get('status') != 'complete':
        raise ValueError('Training requires a complete facts dataset manifest')
    if (protocol.get('schema') != DATASET_SCHEMA
            or manifest.get('protocol_sha256') != protocol_sha256(protocol)):
        raise ValueError('Dataset protocol schema/hash mismatch')
    validate_constraint_schema(protocol)
    if protocol.get('parent_checkpoint_sha256') != parent_hash:
        raise ValueError('Dataset parent policy mismatch')
    if protocol.get('games') != args.games or protocol.get('positions_per_game') != args.positions_per_game:
        raise ValueError('Dataset complete-game or positions-per-game budget mismatch')
    shards, raw = manifest.get('shards', []), manifest.get('raw_games', [])
    if (manifest.get('completed_games') != args.games
            or manifest.get('preprocessed_games') != args.games
            or len(shards) != args.games or len(raw) != args.games):
        raise ValueError('Dataset has not completed/preprocessed every requested game')
    by_id = {item['game_id']: item for item in raw}
    if len(by_id) != args.games or set(by_id) != set(range(args.games)):
        raise ValueError('Dataset complete-game IDs are missing or duplicated')
    shard_ids = set()
    setup_splits = {}
    counts = {'train': 0, 'validation': 0, 'test': 0}
    rows = dict.fromkeys(counts, 0)
    for item in shards:
        game_id = item['game_id']
        if game_id in shard_ids or game_id not in by_id:
            raise ValueError('Dataset supervised game IDs are missing or duplicated')
        shard_ids.add(game_id)
        original = by_id[game_id]
        setup_hash = item['initial_setup_sha256']
        split = item['split']
        if (split != split_for_setup(setup_hash)
                or original['initial_setup_sha256'] != setup_hash
                or original['split'] != split
                or item['raw_sha256'] != original['sha256']
                or original.get('protocol_sha256') != manifest['protocol_sha256']
                or original.get('parent_checkpoint_sha256') != parent_hash):
            raise ValueError('Dataset splits violate complete-game/setup grouping')
        if setup_hash in setup_splits and setup_splits[setup_hash] != split:
            raise ValueError('Initial layout crosses dataset folds')
        setup_splits[setup_hash] = split
        if (item.get('protocol_sha256') != manifest['protocol_sha256']
                or item.get('parent_checkpoint_sha256') != parent_hash
                or item.get('cpu_replay_validated') is not True
                or item.get('teacher_masks_cached') is not True):
            raise ValueError('Dataset shard provenance or cached-mask contract mismatch')
        if not 1 <= item['rows'] <= args.positions_per_game:
            raise ValueError('Dataset row count exceeds its complete-game sampling contract')
        counts[split] += 1
        rows[split] += item['rows']
    if not all(counts.values()):
        raise ValueError('Train, validation and reserved test folds must all contain complete games')
    if counts != manifest.get('split_games') or rows != manifest.get('split_rows'):
        raise ValueError('Dataset split totals disagree with its game records')
    if sum(rows.values()) != manifest.get('observation_rows'):
        raise ValueError('Dataset observation row total mismatch')
    return manifest, file_sha256(path)


def unknown_targets(shard):
    """Exclude directly known and logically singleton public identities."""
    return (shard['known_types'] < 0) & (shard['type_domains'].sum(-1) > 1)


def _batch(shard, indices, device, *, belief_kind='joint'):
    inputs = batch_inputs(shard, indices)
    obs = torch.as_tensor(np.array(shard['spatial'][indices], copy=True),
                          dtype=torch.float32, device=device)
    targets = torch.as_tensor(np.array(shard['targets'][indices], copy=True),
                              dtype=torch.long, device=device)
    # The marginal path never reads prefix-label masks, even if cached in the
    # shared dataset for joint training. Only public type domains enter logits.
    masks = (torch.as_tensor(np.array(shard['teacher_masks'][indices], copy=True),
                             dtype=torch.bool, device=device) if belief_kind == 'joint' else None)
    unknown = torch.as_tensor((shard['known_types'][indices] < 0)
        & (shard['type_domains'][indices].sum(-1) > 1), dtype=torch.bool, device=device)
    return inputs, obs, targets, masks, unknown


def metric_names(belief_kind):
    if belief_kind == 'marginal':
        return 'marginal_nll_per_unknown_piece', 'marginal_accuracy'
    return 'conditional_nll_per_unknown_piece', 'teacher_forced_accuracy'


def metric_scope(belief_kind):
    if belief_kind == 'marginal':
        return 'Public-only per-piece marginal predictions; not gameplay strength or joint assignment probability'
    return 'Teacher-forced conditional predictions; not gameplay strength or marginal accuracy'


def _supervised_logits(net, obs, *, inputs, targets, masks):
    if getattr(net, 'belief_kind', 'joint') == 'marginal':
        if masks is not None:
            raise ValueError('Marginal supervision must never receive teacher-prefix masks')
        return net.marginal_logits(obs, inputs=inputs)
    return net.teacher_logits(obs, inputs=inputs, targets=targets, teacher_masks=masks)


@torch.no_grad()
def evaluate(net, shard, *, batch_size, device):
    net.eval()
    kind = getattr(net, 'belief_kind', 'joint')
    nll_key, accuracy_key = metric_names(kind)
    total_nll = correct = count = 0.0
    for start in range(0, len(shard['targets']), batch_size):
        indices = np.arange(start, min(start + batch_size, len(shard['targets'])))
        inputs, obs, targets, masks, unknown = _batch(shard, indices, device, belief_kind=kind)
        logits = _supervised_logits(net, obs, inputs=inputs, targets=targets, masks=masks)
        nll = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
        values = nll[unknown]
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite validation NLL')
        total_nll += float(values.sum())
        correct += int(((logits.argmax(-1) == targets) & unknown).sum())
        count += int(unknown.sum())
    if not count:
        raise ValueError('Validation fold contains no unknown identities')
    return {nll_key: total_nll / count,
            accuracy_key: correct / count, 'unknown_targets': int(count),
            'rows': len(shard['targets'])}


def _capture_rng(rng):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'minibatch': rng.bit_generator.state, 'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(saved, rng):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    rng.bit_generator.state = saved['minibatch']
    torch.set_rng_state(saved['torch_cpu'].cpu())
    if saved['torch_cuda']:
        if not torch.cuda.is_available() or len(saved['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed during exact training resume')
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['torch_cuda']])


def train(args):
    kind = args.belief_kind
    model_type = FactsJointBeliefNet if kind == 'joint' else FactsMarginalBeliefNet
    schema = TRAINING_SCHEMA if kind == 'joint' else MARGINAL_TRAINING_SCHEMA
    nll_key, _ = metric_names(kind)
    selected_name = f'{kind}-belief.pt'
    if min(args.games, args.positions_per_game, args.epochs, args.batch_size, args.num_threads) < 1:
        raise ValueError('Training and complete-game budgets must be positive')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA training but no CUDA device is available')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    parent_hash = file_sha256(args.policy)
    policy = load_evaluation_policy(args.policy, device='cpu')
    if policy.identity_encoding != 'known_only' or policy.uses_global_summary:
        raise ValueError('Belief training requires a frozen facts-only zero-global parent policy')
    del policy
    manifest, manifest_hash = _validate_manifest(args, parent_hash)
    cfg = FactsJointBeliefNetConfig(n_encoder_layer=args.encoder_layers,
        n_decoder_layer=args.decoder_layers, embed_dim=args.embed_dim, n_head=args.heads,
        ff_factor=args.ff_factor, cnn_channels=args.cnn_channels, cnn_layers=args.cnn_layers,
        dropout=args.dropout)
    contract = {'schema': schema, 'dataset_manifest_sha256': manifest_hash,
                'dataset_protocol_sha256': manifest['protocol_sha256'],
                'parent_policy_sha256': parent_hash, 'model_config': asdict(cfg),
                'games': args.games, 'positions_per_game': args.positions_per_game,
                'seed': args.seed, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                'optimizer': 'Adam', 'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8,
                'gradient_clip': 0.5, 'device_type': device.type,
                'deterministic_algorithms': True, 'cpu_threads': args.num_threads,
                'loss': 'unknown-only joint conditional cross entropy',
                'unknown_definition': 'known_types < 0 and public type-domain size > 1',
                'selection': 'minimum validation conditional NLL; earliest epoch wins ties',
                'test_fold_used': False}
    if kind == 'marginal':
        contract.update({
            'belief_kind': kind, 'architecture_id': model_type.architecture_id,
            'loss': 'unknown-only public-domain marginal cross entropy',
            'selection': 'minimum validation marginal NLL; earliest epoch wins ties',
            'training_support': 'public type_domains only; no teacher-prefix masks',
            'sampling_distribution': model_type.distribution_semantics,
        })
    output = Path(args.output)
    protocol_path = output / 'training-protocol.json'
    if (protocol_path.exists()
            and protocol_sha256(json.loads(protocol_path.read_text())) != protocol_sha256(contract)):
        raise ValueError('Output directory belongs to a different training contract')
    if not args.resume and (output / 'last.pt').exists():
        raise ValueError('Existing training checkpoint requires explicit --resume')
    saved = None
    if args.resume:
        net = load_facts_belief(args.resume, policy_path=args.policy, device=device, expected_kind=kind)
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (protocol_sha256(saved.get('training_contract')) != protocol_sha256(contract)
                or type(saved.get('resume_schema')) is not int or saved['resume_schema'] != 1):
            raise ValueError('Resume dataset/config/parent/optimizer contract mismatch')
        if args.epochs <= saved['epochs_completed']:
            raise ValueError('Resume must advance the total epoch target')
    else:
        net = model_type(cfg).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # Hash verification and decompression happen once, before all epochs.
    # Only train/validation files are opened; test is reserved for final audit.
    training = concatenate_shards(args.dataset, 'train')
    validation = concatenate_shards(args.dataset, 'validation')
    if not unknown_targets(training).any():
        raise ValueError('Training fold contains no unknown identities')
    start_epoch = updates = 0
    best_epoch = 0
    best_nll = float('inf')
    best_weights = None
    metrics = []
    elapsed_before = 0.0
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch = saved['epochs_completed']
        updates = saved['updates']
        best_epoch, best_nll = saved['best_epoch'], saved['best_validation_nll']
        best_weights = saved['best_model_state']
        metrics = saved['epoch_metrics']
        initial = saved['initial_validation']
        elapsed_before = saved['training_seconds']
        _restore_rng(saved['rng'], rng)
    else:
        initial = evaluate(net, validation, batch_size=args.batch_size, device=device)
    _json_write(protocol_path, contract)
    started = time.perf_counter()
    final_validation = metrics[-1]['validation'] if metrics else initial

    def publish(epoch):
        seconds = elapsed_before + time.perf_counter() - started
        state = facts_belief_state(net, parent_policy_sha256=parent_hash,
            training_contract=contract, resume_schema=1, epochs_completed=epoch,
            updates=updates, optimizer=optimizer.state_dict(), rng=_capture_rng(rng),
            best_epoch=best_epoch, best_validation_nll=best_nll,
            best_model_state=best_weights, epoch_metrics=metrics,
            initial_validation=initial, training_seconds=seconds)
        atomic_write(output / 'last.pt', lambda handle: torch.save(state, handle))
        # The portable inference artifact always carries the selected best
        # model, including when a resumed run moves to a different directory.
        selected = {key: value for key, value in state.items() if key not in (
            'optimizer', 'rng', 'best_model_state', 'resume_schema')}
        selected['net'] = best_weights
        selected['selected_epoch'] = best_epoch
        selected['artifact_role'] = f'validation-selected-facts-{kind}-belief'
        atomic_write(output / selected_name, lambda handle: torch.save(selected, handle))
        report = {'schema': schema, 'training_contract': contract,
                  'epochs_completed': epoch, 'target_epochs': args.epochs,
                  'updates': updates, 'best_epoch': best_epoch,
                  'initial_validation': initial, 'final_validation': final_validation,
                  'best_validation': metrics[best_epoch - 1]['validation'],
                  'epoch_metrics': metrics, 'training_seconds': seconds,
                  'completed_collection_games': manifest['completed_games'],
                  'split_games': manifest['split_games'], 'split_rows': manifest['split_rows'],
                  'test_fold_used': False, 'parameters': net.num_parameters(),
                  'last_checkpoint_sha256': file_sha256(output / 'last.pt'),
                  'selected_checkpoint_sha256': file_sha256(output / selected_name),
                  'finished': epoch == args.epochs,
                  'metric_scope': metric_scope(kind)}
        _json_write(output / 'report.json', report)
        return report

    for epoch in range(start_epoch, args.epochs):
        net.train()
        order = rng.permutation(len(training['targets']))
        total_nll = count = skipped = 0
        epoch_start = time.perf_counter()
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            inputs, obs, targets, masks, unknown = _batch(training, indices, device, belief_kind=kind)
            if not unknown.any():
                skipped += 1
                continue
            logits = _supervised_logits(net, obs, inputs=inputs, targets=targets, masks=masks)
            nll = -logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1)
            loss = nll[unknown].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training NLL')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            if not torch.isfinite(grad):
                raise FloatingPointError('Nonfinite belief gradients')
            optimizer.step()
            updates += 1
            total_nll += float(nll[unknown].detach().sum())
            count += int(unknown.sum())
        final_validation = evaluate(net, validation, batch_size=args.batch_size, device=device)
        item = {'epoch': epoch + 1, 'updates': updates,
                f'train_{nll_key}': total_nll / count,
                'train_unknown_targets': count, 'skipped_fully_known_batches': skipped,
                'validation': final_validation, 'epoch_seconds': time.perf_counter() - epoch_start}
        metrics.append(item)
        if final_validation[nll_key] < best_nll:
            best_nll = final_validation[nll_key]
            best_epoch = epoch + 1
            best_weights = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        report = publish(epoch + 1)
        print(json.dumps(item, allow_nan=False), flush=True)
    return report


def main(argv=None):
    report = train(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return report


if __name__ == '__main__':
    main()
