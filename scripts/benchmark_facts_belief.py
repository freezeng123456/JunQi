"""Bounded validation-only free-sample and saved-state search diagnostics.

No teacher masks or identity targets are passed to model.sample. Simulator truth
is used only to reconstruct the offline public trajectory and score samples
AFTER inference. Development budget selection never opens the test fold.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from junqi_core.knowledge import ObserverKnowledge
from junqi_core.rules import ALL_SEATS, PieceType, ShowMode
from junqi_core.state import GameState
from junqi_rl.analysis.facts_joint_search import FactsBlueprint, FactsBeliefSearch, SearchConfig, public_result, step_world
from junqi_rl.belief.constraints import compile_constraints
from junqi_rl.belief.facts_dataset import batch_inputs, read_npz, validate_trace
from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
from junqi_rl.env import JunqiEnv, unrotate_compact_action_id
from junqi_rl.training.checkpoint import load_evaluation_policy
from scripts.compare_facts_joint import (add_inference_runtime_arguments, inference_source_sha256,
                                       search_runtime_options, selected_belief_provenance, sha256, write_json)


def validation_states(dataset, manifest, count, seed):
    """Select one acting-observer row per shuffled validation game, never by outcome."""
    candidates = [row for row in manifest['shards'] if row['split'] == 'validation']
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    raw_by_id = {row['game_id']: row for row in manifest['raw_games']}
    selected = []
    for item in candidates:
        path = dataset / item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('Saved validation shard hash changed')
        shard = read_npz(path)
        original = raw_by_id[item['game_id']]
        if original['split'] != 'validation' or original['initial_setup_sha256'] != item['initial_setup_sha256']:
            raise ValueError('Raw validation trace grouping mismatch')
        raw_path = dataset / original['path']
        if sha256(raw_path) != original['sha256']:
            raise ValueError('Saved validation trace hash changed')
        trace = read_npz(raw_path)
        validate_trace(trace, original)
        options = np.flatnonzero(shard['observer_seats'] == trace['seats'][shard['ply']])
        if not len(options):
            continue
        row = int(rng.choice(options))
        selected.append((item, original, shard, trace, row))
        if len(selected) == count:
            return selected
    raise ValueError(f'Only {len(selected)} validation games have eligible acting-observer rows; requested {count}')


def restore_public_position(trace, metadata, ply):
    types = trace['initial_piece_types']
    setup = tuple(tuple(PieceType(int(kind)) if kind >= 0 else PieceType.NONE for kind in army)
                  for army in types.reshape(4, 30))
    env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False, identity_encoding='known_only',
                   max_num_moves=metadata['max_moves'], record_public_history=True)
    env._state = GameState.new_game(setup, show_mode=ShowMode.DARK)
    env._beliefs = {seat: None for seat in ALL_SEATS}
    env._knowledge = {seat: ObserverKnowledge(env.state, seat) for seat in ALL_SEATS}
    history = []
    for index in range(ply):
        if int(env.state.turn) != int(trace['seats'][index]):
            raise ValueError('Saved validation trajectory turn mismatch')
        action = unrotate_compact_action_id(int(trace['actions'][index]), env.state.turn)
        result = step_world(env, action)
        actual = (int(result.event), env.state.terminated, env.state.draw,
                  -1 if env.state.winner_team is None else env.state.winner_team)
        recorded = (int(trace['events'][index]), bool(trace['terminated'][index]),
                    bool(trace['draw'][index]), int(trace['winner_team'][index]))
        if actual != recorded:
            raise ValueError('Saved validation trajectory broadcast mismatch')
        history.append(public_result(result))
    if env.state.terminated:
        raise ValueError('Search benchmark requires an active saved state')
    return env, history


def sample_quality(samples, targets, unknown):
    """Offline descriptive metrics; do not treat sampled frequencies as calibrated NLL."""
    labels = np.asarray(samples)
    if labels.ndim != 2 or labels.shape[1] != 75 or not len(labels):
        raise ValueError('Expected complete free identity samples')
    selected = labels[:, unknown]
    if not selected.shape[1]:
        raise ValueError('Saved state contains no uncertain identity')
    counts = (selected[..., None] == np.arange(12)).sum(0)
    frequencies = counts / len(labels)
    entropy = -(frequencies * np.log(np.maximum(frequencies, np.finfo(float).tiny))).sum(-1)
    pair_distances = [np.mean(selected[a] != selected[b]) for a in range(len(labels)) for b in range(a)]
    return dict(samples=len(labels), uncertain_identities=int(unknown.sum()),
                sampled_true_label_frequency=float(np.mean(selected == np.asarray(targets)[unknown])),
                empirical_mode_accuracy=float(np.mean(counts.argmax(-1) == np.asarray(targets)[unknown])),
                mean_empirical_identity_entropy=float(entropy.mean()),
                unique_complete_assignments=len(np.unique(labels, axis=0)),
                mean_pairwise_unknown_hamming=float(np.mean(pair_distances)) if pair_distances else None)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--belief', action='append', required=True, help='Repeat for matched marginal and joint artifacts')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=12, help='One saved acting-observer row per validation game')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--budgets', nargs='+', default=['1x1', '2x1'], help='particles x depth')
    parser.add_argument('--repeats', type=int, default=2, help='Independent search draws per state and budget')
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--proposal-batches', type=int, default=8)
    parser.add_argument('--seed', type=int, default=95000000)
    parser.add_argument('--device', default='cuda')
    add_inference_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if min(args.states, args.samples, args.repeats) < 1:
        raise ValueError('All diagnostic budgets must be positive')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    budgets = [SearchConfig(particles=int(text.split('x')[0]), depth=int(text.split('x')[1]),
                            beta=args.beta, proposal_batches=args.proposal_batches) for text in args.budgets]
    manifest_path = args.dataset / 'dataset-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'junqi.facts-belief-dataset.v1' or manifest.get('status') != 'complete':
        raise ValueError('Diagnostics require a complete dataset manifest')
    if manifest.get('protocol', {}).get('parent_checkpoint_sha256') != sha256(args.policy):
        raise ValueError('Diagnostics dataset parent policy mismatch')
    chosen = validation_states(args.dataset, manifest, args.states, args.seed)
    blueprint = FactsBlueprint(load_evaluation_policy(args.policy, device=device), device)
    models = [(load_facts_belief(path, policy_path=args.policy, device=device), path) for path in args.belief]
    runtimes = [search_runtime_options(args, model) for model, path in models]
    protocol = dict(schema='junqi.facts-belief-validation-benchmark.v1', phase='development', split='validation',
                    policy_sha256=sha256(args.policy), dataset_manifest_sha256=sha256(manifest_path),
                    models=[selected_belief_provenance(path, model, sha256(manifest_path)) for model, path in models],
                    states=args.states, samples=args.samples, repeats=args.repeats, seed=args.seed,
                    search_configs=[asdict(config) for config in budgets], device_type=device.type,
                    inference_runtime=runtimes,
                    torch_version=str(torch.__version__), script_sha256=sha256(__file__),
                    inference_source_sha256=inference_source_sha256(), numpy_version=str(np.__version__),
                    selected_rows=[dict(game_id=item['game_id'], row=row, ply=int(shard['ply'][row]),
                                        setup_sha256=item['initial_setup_sha256'], shard_sha256=item['sha256'])
                                   for item, original, shard, trace, row in chosen],
                    target_input=False, teacher_masks_used=False, test_fold_used=False,
                    warmup_searches_per_model=1)
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        if saved.get('protocol') != protocol or saved.get('complete') is not True:
            raise ValueError('Existing diagnostic output has another/incomplete protocol')
        print(json.dumps(saved, indent=2))
        return saved
    output = dict(protocol=protocol, complete=False, states=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, (item, original, shard, trace, row) in enumerate(chosen):
        env, history = restore_public_position(trace, original, int(shard['ply'][row]))
        obs = torch.as_tensor(np.array(shard['spatial'][row:row + 1], copy=True), device=device)
        actual = blueprint.builder.build(env.state, None, env.state.turn).spatial
        # Archived shards may differ by one float32 ULP after NumPy/runtime changes.
        if not np.allclose(actual, shard['spatial'][row], rtol=0, atol=1e-7):
            raise ValueError('Saved validation observation differs from replayed public root')
        inputs = batch_inputs(shard, np.asarray([row]))
        record = dict(game_id=item['game_id'], row=row, ply=int(shard['ply'][row]), models=[])
        for (model, path), runtime in zip(models, runtimes):
            if index == 0:
                # Exclude first-use kernels/allocations from serial search timing.
                warmup = FactsBeliefSearch(blueprint, model, budgets[-1], seed=args.seed + 200000, **runtime)
                warmup.action(env, history)
                synchronize(device)
            solver = compile_constraints(inputs, inventory_backend=runtime['inventory_backend'])
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            synchronize(device)
            began = time.perf_counter()
            with torch.no_grad():
                sample_kwargs = {'overlap_constraints': True} if runtime['overlap_constraints'] else {}
                samples = model.sample(obs, inputs=inputs, n_samples=args.samples,
                                       generator=generator, constraints=[solver], **sample_kwargs)
            synchronize(device)
            free_seconds = time.perf_counter() - began
            labels = samples[0].cpu().numpy()
            for assignment in labels:
                solver.validate(assignment)
            # Scoring reads truth only after the model call has returned.
            unknown = (shard['known_types'][row] < 0) & (shard['type_domains'][row].sum(-1) > 1)
            sampled = sample_quality(labels, shard['targets'][row], unknown)
            measured = dict(belief_kind=model.belief_kind, checkpoint_sha256=sha256(path),
                            free_sample_quality=sampled, free_sample_seconds=free_seconds,
                            free_sample_metrics=model.last_sample_metrics, constraint_stats=dict(solver.stats), searches=[])
            for config in budgets:
                for repeat in range(args.repeats):
                    search = FactsBeliefSearch(blueprint, model, config,
                        seed=args.seed + 100000 + index * args.repeats + repeat, **runtime)
                    synchronize(device)
                    began = time.perf_counter()
                    action = search.action(env, history)
                    synchronize(device)
                    measured['searches'].append(dict(config=asdict(config), repeat=repeat, action=action,
                        wall_seconds=time.perf_counter() - began, report=search.report()))
            record['models'].append(measured)
        output['states'].append(record)
        write_json(args.output.with_suffix('.progress.json'), output)
        print(json.dumps({'validation_states_completed': index + 1, 'total': args.states}), flush=True)
    output['complete'] = True
    output['metric_scope'] = ('Free constrained samples, scored offline; empirical identity frequencies are not NLL. '
        'Search timings are isolated serial decisions on saved validation positions, not match win rates.')
    write_json(args.output, output)
    print(json.dumps({'complete': True, 'states': len(output['states']), 'output': str(args.output)}), flush=True)
    return output


if __name__ == '__main__':
    main()
