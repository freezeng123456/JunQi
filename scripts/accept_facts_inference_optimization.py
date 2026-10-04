#!/usr/bin/env python3
"""Bounded exact inference comparison with frozen pre-optimization sources.

CPU and CUDA parity are checked on the same device, never against each other.
Hooks capture logits only in the correctness pass; timing and profiler passes
have no such hooks. Default inference stays unchanged regardless of this report.
The CUDA trace shows submitted decoder work and host allowed_next scopes; CPU
constraint duration overlaps GPU work in overlap variants and is not additive.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.rules import PieceType
from junqi_rl.analysis import facts_joint_search as search_module
from junqi_rl.analysis.facts_joint_search import FactsBlueprint, FactsBeliefSearch, SearchConfig
from junqi_rl.belief import constraints as current
from junqi_rl.belief.facts_dataset import batch_inputs, read_npz, write_json
from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
from junqi_rl.belief.joint import build_joint_input
from junqi_rl.networks import facts_joint_belief_net as joint_current
from junqi_rl.training.checkpoint import load_evaluation_policy
from scripts.benchmark_facts_belief import restore_public_position, validation_states
from scripts.benchmark_identity_constraints import workload
from scripts.compare_facts_joint import sha256


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


def constraint_state(solver):
    return dict(stats=dict(solver.stats), matching_cache_stats=dict(solver.matching_cache_stats),
                witnesses=sorted(solver._cache.items()),
                masks=[(key, mask.tolist()) for key, mask in sorted(solver._masks.items())],
                matching=[(key, None if value is None else sorted(value.items()))
                          for key, value in solver._matching_cache.items()])


def run_sample(model, module, obs, inputs, *, samples, seed, backend, overlap,
               capture=False, profile_scopes=False):
    synchronize(obs.device)
    start = time.perf_counter()
    kwargs = {'inventory_backend': backend} if module is current else {}
    solver = module.compile_constraints(inputs, max_search_nodes=100_000, **kwargs)
    if profile_scopes:
        allowed_next = solver.allowed_next

        def scoped(prefix):
            with torch.profiler.record_function('cpu_exact_allowed_next'):
                return allowed_next(prefix)

        solver.allowed_next = scoped
    generator = torch.Generator(device=obs.device).manual_seed(seed)
    logits = []
    hook = (model.head.register_forward_hook(lambda _m, _i, out: logits.append(out.detach().clone()))
            if capture else None)
    try:
        sample_kwargs = {'overlap_constraints': overlap} if isinstance(model, joint_current.FactsJointBeliefNet) else {}
        labels = model.sample(obs, inputs=inputs, n_samples=samples, constraints=[solver],
                              generator=generator, **sample_kwargs)
    finally:
        if hook is not None:
            hook.remove()
    synchronize(obs.device)
    seconds = time.perf_counter() - start
    return dict(labels=labels.detach().cpu(), rng=generator.get_state().cpu(),
                logits=torch.cat(logits).cpu() if capture else None,
                constraints=constraint_state(solver) if capture else None,
                seconds=seconds, sample_metrics=dict(model.last_sample_metrics))


def compare_outputs(reference, candidate):
    for key in ('labels', 'rng', 'logits'):
        if not torch.equal(reference[key], candidate[key]):
            raise AssertionError(f'Exact device-local {key} parity failed')
    if reference['constraints'] != candidate['constraints']:
        raise AssertionError('Masks, witnesses, nodes, or matching-cache order differ')


def run_action(blueprint, model, module, state, *, samples, seed, backend, overlap):
    """Reference compiler substitution is local to this sequential acceptance run."""
    original = search_module.compile_constraints
    try:
        search_module.compile_constraints = module.compile_constraints
        search = FactsBeliefSearch(blueprint, model, SearchConfig(particles=samples, depth=1),
                                  seed=seed, capture_diagnostics=True,
                                  inventory_backend=backend, overlap_constraints=overlap)
        action = search.action(state['env'], state['history'])
    finally:
        search_module.compile_constraints = original
    decision = dict(search.last_decision)
    particles = dict(decision['particles'])
    particles.pop('constraint_implementation', None)
    particles.pop('inventory_backend', None)
    decision['particles'] = particles
    return dict(action=action, decision=decision, rng=search.generator.get_state().cpu())


def bounded_prefix_checks(reference_module, inputs, targets):
    results = []
    for budget in (1, 2, 5, 100_000):
        reference = reference_module.compile_constraints(inputs, max_search_nodes=budget)
        candidate = current.compile_constraints(inputs, max_search_nodes=budget,
                                                 inventory_backend='incremental')
        checked, failures = 0, []
        for index in range(75):
            outcomes = []
            for solver in (reference, candidate):
                try:
                    outcomes.append(('mask', solver.allowed_next(targets[:index]).tolist()))
                except (reference_module.ConstraintBudgetExceeded, current.ConstraintBudgetExceeded) as error:
                    outcomes.append(('budget', str(error)))
            if outcomes[0] != outcomes[1] or constraint_state(reference) != constraint_state(candidate):
                raise AssertionError(f'Prefix/budget/cache parity failed at prefix {index}, budget {budget}')
            checked += 1
            if outcomes[0][0] == 'budget':
                failures.append(index)
                # Retry the same query with a larger budget. Interrupted caches
                # and the accumulated node statistics must still be identical.
                reference.max_search_nodes = candidate.max_search_nodes = 100_000
                first, second = reference.allowed_next(targets[:index]), candidate.allowed_next(targets[:index])
                np.testing.assert_array_equal(first, second)
                if constraint_state(reference) != constraint_state(candidate):
                    raise AssertionError('Parity failed after budget retry')
        results.append(dict(budget=budget, prefixes=checked, budget_failure_prefixes=failures))
    return results


def cpu_profile(module, inputs, targets, output, backend=None):
    profile = cProfile.Profile()
    with profile:
        workload(module, inputs, targets, seed=713, max_search_nodes=100_000,
                 compiler_options={'inventory_backend': backend} if backend else None)
    profile.dump_stats(str(output.with_suffix('.prof')))
    with output.with_suffix('.txt').open('w') as handle:
        pstats.Stats(profile, stream=handle).sort_stats('tottime').print_stats(35)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--belief', type=Path, action='append', required=True)
    parser.add_argument('--reference-constraints', type=Path, required=True)
    parser.add_argument('--reference-joint-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=3)
    parser.add_argument('--samples', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261004)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--include-q12', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Also write isolated reference/candidate torch traces')
    parser.add_argument('--search-actions', action='store_true', help='Also compare full root Q/probabilities/actions and final RNG')
    args = parser.parse_args(argv)
    if min([args.states, args.repeats] + args.samples) < 1 or args.seed < 0:
        parser.error('Positive state/sample/repeat counts and nonnegative seed required')
    if args.output.exists():
        raise ValueError('Acceptance output already exists; preserve prior evidence')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA acceptance must run on CUDA; no CPU fallback')
    reference_constraints = load_source(args.reference_constraints, 'junqi_rl.belief._optimization_reference')
    reference_joint = load_source(args.reference_joint_source, 'junqi_rl.networks._optimization_reference')
    if reference_constraints.CONSTRAINT_SCHEMA != current.CONSTRAINT_SCHEMA:
        raise ValueError('Reference must implement the same exact constraint schema')
    manifest_path = args.dataset / 'dataset-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or manifest.get('schema') != 'junqi.facts-belief-dataset.v1':
        raise ValueError('Expected a complete facts dataset')
    if manifest['protocol']['parent_checkpoint_sha256'] != sha256(args.policy):
        raise ValueError('Dataset parent policy differs')
    blueprint = FactsBlueprint(load_evaluation_policy(args.policy, device=device), device)
    chosen = validation_states(args.dataset, manifest, args.states, args.seed)
    states = []
    for item, metadata, shard, trace, row in chosen:
        env, history = restore_public_position(trace, metadata, int(shard['ply'][row]))
        actual = blueprint.builder.build(env.state, None, env.state.turn).spatial
        if not np.allclose(actual, shard['spatial'][row], rtol=0, atol=1e-7):
            raise ValueError('Validation replay observation mismatch')
        states.append(dict(name=f"game-{item['game_id']}-row-{row}", inputs=batch_inputs(shard, [row]),
                           obs=torch.as_tensor(np.array(shard['spatial'][row:row + 1], copy=True), device=device),
                           targets=shard['targets'][row].tolist(), public_events=len(batch_inputs(shard, [row]).public_history[0]),
                           source_sha256=item['sha256'], env=env, history=history))
    if args.include_q12:
        from tests.test_facts_joint_search import q12_fixture
        fixture = ROOT / 'tests/fixtures/facts_q12_public_trace.json'
        for ply in (161, 200):
            env, history = q12_fixture(ply)
            inputs = build_joint_input([env.knowledge_snapshot(env.state.turn)])
            states.append(dict(name=f'q12-ply-{ply}', inputs=inputs,
                               obs=torch.as_tensor(blueprint.builder.build(env.state, None, env.state.turn).spatial[None], device=device),
                               targets=[TYPE_TO_INDEX[PieceType(int(env.state.piece_type_arr[pid]))]
                                        for pid in inputs.piece_ids[0]],
                               public_events=ply, source_sha256=sha256(fixture), env=env, history=history))
    models = [(load_facts_belief(path, policy_path=args.policy, device=device), path) for path in args.belief]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='junqi.facts-inference-optimization-acceptance.v1', status='running',
                  device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                  python=platform.python_version(), torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                  policy_sha256=sha256(args.policy), dataset_manifest_sha256=sha256(manifest_path),
                  reference_constraints_sha256=sha256(args.reference_constraints),
                  reference_joint_sha256=sha256(args.reference_joint_source),
                  candidate_constraints_sha256=sha256(current.__file__),
                  candidate_joint_sha256=sha256(joint_current.__file__),
                  script_sha256=sha256(__file__), samples=args.samples, repeats=args.repeats, seed=args.seed,
                  full_search_action_parity=args.search_actions,
                  scope='isolated device-local sampler parity and serial sampling timings; not match wins or worker throughput',
                  target_input=False, teacher_masks_input=False, test_fold_used=False,
                  production_defaults_changed=False, states=[])
    for state_index, state in enumerate(states):
        record = dict(name=state['name'], source_sha256=state['source_sha256'], public_events=state['public_events'],
                      bounded_prefix_checks=bounded_prefix_checks(reference_constraints, state['inputs'], state['targets']), models=[])
        if state_index == 0:
            cpu_profile(reference_constraints, state['inputs'], state['targets'],
                        args.output.parent / (args.output.stem + '-profile-constraints-reference'))
            cpu_profile(current, state['inputs'], state['targets'],
                        args.output.parent / (args.output.stem + '-profile-constraints-incremental'), 'incremental')
        for model_index, (model, path) in enumerate(models):
            if model.belief_kind == 'joint':
                reference_model = reference_joint.FactsJointBeliefNet(
                    reference_joint.FactsJointBeliefNetConfig(**asdict(model.cfg))).to(device).eval()
                reference_model.load_state_dict(model.state_dict(), strict=True)
                variants = [('reference', reference_model, reference_constraints, 'scan', False),
                            ('incremental', model, current, 'incremental', False),
                            ('overlap', model, current, 'scan', True),
                            ('combined', model, current, 'incremental', True)]
            else:
                variants = [('reference', model, reference_constraints, 'scan', False),
                            ('incremental', model, current, 'incremental', False)]
            measured = dict(belief_kind=model.belief_kind, checkpoint_sha256=sha256(path), sample_budgets=[])
            for samples in args.samples:
                seed = args.seed + state_index * 10000 + model_index * 100 + samples
                runs = {}
                baseline = None
                for name, selected_model, module, backend, overlap in variants:
                    result = run_sample(selected_model, module, state['obs'], state['inputs'], samples=samples,
                                        seed=seed, backend=backend, overlap=overlap, capture=True)
                    if baseline is None:
                        baseline = result
                    else:
                        compare_outputs(baseline, result)
                    runs[name] = dict(seconds=[], parity='exact',
                                      labels_sha256=hashlib.sha256(result['labels'].numpy().tobytes()).hexdigest(),
                                      rng_sha256=hashlib.sha256(result['rng'].numpy().tobytes()).hexdigest(),
                                      constraints_sha256=digest(result['constraints']))
                if args.search_actions:
                    reference_action = None
                    for name, selected_model, module, backend, overlap in variants:
                        action_result = run_action(blueprint, selected_model, module, state, samples=samples,
                                                   seed=seed, backend=backend, overlap=overlap)
                        if reference_action is None:
                            reference_action = action_result
                        elif (action_result['action'] != reference_action['action']
                              or action_result['decision'] != reference_action['decision']
                              or not torch.equal(action_result['rng'], reference_action['rng'])):
                            raise AssertionError('Full search particles/Q/probabilities/action/RNG parity failed')
                        runs[name]['search_action'] = action_result['action']
                        runs[name]['search_decision_sha256'] = digest(action_result['decision'])
                        runs[name]['search_rng_sha256'] = hashlib.sha256(action_result['rng'].numpy().tobytes()).hexdigest()
                # An explicit warmup per variant/budget precedes uninstrumented timing.
                for name, selected_model, module, backend, overlap in variants:
                    run_sample(selected_model, module, state['obs'], state['inputs'], samples=samples,
                               seed=seed + 100000, backend=backend, overlap=overlap)
                for repetition in range(args.repeats):
                    order = variants if repetition % 2 == 0 else list(reversed(variants))
                    for name, selected_model, module, backend, overlap in order:
                        result = run_sample(selected_model, module, state['obs'], state['inputs'], samples=samples,
                                            seed=seed + repetition, backend=backend, overlap=overlap)
                        runs[name]['seconds'].append(result['seconds'])
                        runs[name]['last_sample_metrics'] = result['sample_metrics']
                reference_median = statistics.median(runs['reference']['seconds'])
                for name, run in runs.items():
                    run['p50_seconds'] = statistics.median(run['seconds'])
                    run['p95_seconds'] = float(np.percentile(run['seconds'], 95))
                    run['speedup_reference_median'] = reference_median / run['p50_seconds']
                measured['sample_budgets'].append(dict(samples=samples, variants=runs))
                if args.profile and state_index == model_index == 0 and samples == args.samples[-1]:
                    for name, selected_model, module, backend, overlap in (variants[0], variants[-1]):
                        activities = [torch.profiler.ProfilerActivity.CPU]
                        if device.type == 'cuda':
                            activities.append(torch.profiler.ProfilerActivity.CUDA)
                        with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                            run_sample(selected_model, module, state['obs'], state['inputs'], samples=samples,
                                       seed=seed, backend=backend, overlap=overlap, profile_scopes=True)
                        prof.export_chrome_trace(str(args.output.parent / f'{args.output.stem}-timeline-{name}.json'))
                        (args.output.parent / f'{args.output.stem}-timeline-{name}.txt').write_text(
                            prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=35))
            record['models'].append(measured)
        report['states'].append(record)
        write_json(args.output.with_suffix('.progress.json'), report)
        print(json.dumps(dict(states_completed=state_index + 1, states_total=len(states), exact_parity=True)), flush=True)
    report['status'] = 'passed'
    report['interpretation'] = ('Passed means exact samples, neural logits, RNG, masks, witnesses and budget/cache behavior '
                                'on recorded cases. Timings are estimates under this device/process setup; '
                                'production promotion requires replicated serial and 8-worker measurements. '
                                'No source default, policy weights, or belief weights were changed by this command.')
    write_json(args.output, report)
    print(json.dumps(dict(status='passed', states=len(states), output=str(args.output))), flush=True)
    return report


if __name__ == '__main__':
    main()
