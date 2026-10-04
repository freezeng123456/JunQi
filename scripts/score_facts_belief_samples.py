#!/usr/bin/env python3
"""Comparable validation-only proposal likelihood and accepted-sample Brier.

The chain likelihood describes the neural proposal before complete public replay
rejection. It is not the normalized likelihood of the accepted distribution.
Marginal neural scores see public facts only; true prefixes enter its offline
support calculation, never its encoder or decoder. Free sampling finishes before
the scorer reads hidden targets. Games, rather than identities, are the sampling
units for aggregate intervals.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.rules import PieceType
from junqi_rl.analysis.facts_joint_search import FactsBlueprint, FactsBeliefSearch, SearchConfig
from junqi_rl.belief.constraints import compile_constraints
from junqi_rl.belief.facts_dataset import batch_inputs
from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
from junqi_rl.training.checkpoint import load_evaluation_policy
from scripts.benchmark_facts_belief import restore_public_position, synchronize
from scripts.compare_facts_joint import (
    inference_source_sha256, selected_belief_provenance, sha256, write_json,
)


@torch.no_grad()
def proposal_chain_score(model, obs, inputs, targets, *, max_constraint_nodes=100_000):
    """Score one true chain with the same decoder and causal masks as sample()."""
    labels = np.asarray(targets)
    if len(obs) != 1 or labels.shape != (75,) or labels.dtype.kind not in 'iu':
        raise ValueError('Proposal scoring requires one 75-identity integer target chain')
    if model.training or model.belief_kind not in ('marginal', 'joint'):
        raise ValueError('Proposal scoring requires an evaluated facts marginal or joint network')
    solver = compile_constraints(inputs, max_search_nodes=max_constraint_nodes)
    solver.validate(labels)
    memory, context = model.encode_facts(obs, inputs)
    if model.belief_kind == 'marginal':
        static = model.decode(memory, context)
        if not torch.isfinite(static).all():
            raise FloatingPointError('Nonfinite neural proposal logits')
    else:
        memory_kv = [block.memory(memory) for block in model.decoder]
        caches = [(memory.new_empty(1, block.heads, 75, block.width),
                   memory.new_empty(1, block.heads, 75, block.width)) for block in model.decoder]
        previous = torch.full((1,), 12, dtype=torch.long, device=obs.device)
    token_log_probs, supports = [], []
    for index, label in enumerate(labels):
        allowed = solver.allowed_next(labels[:index])
        if not allowed.any() or not allowed[int(label)]:
            raise ValueError('Truth is outside the exact causal proposal support')
        if model.belief_kind == 'marginal':
            logits = static[:, index]
        else:
            x = context[:, index:index + 1] + model.type_embedding(previous)[:, None]
            for block, mem, cache in zip(model.decoder, memory_kv, caches):
                x = block.step(x, mem, cache, index)
            logits = model.head(model.decoder_norm(x)).squeeze(1)
            previous.fill_(int(label))
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite neural proposal logits')
        mask = torch.as_tensor(allowed.copy(), dtype=torch.bool, device=obs.device)
        log_probs = logits.masked_fill(~mask, -torch.inf).log_softmax(-1)
        token_log_probs.append(float(log_probs[0, int(label)].cpu()))
        supports.append(np.flatnonzero(allowed).tolist())
    unknown = ((inputs.known_types[0] < 0) & (inputs.type_domains[0].sum(-1) > 1))
    logp = np.asarray(token_log_probs, dtype=np.float64)
    return dict(proposal_chain_log_likelihood=float(logp.sum()),
                proposal_chain_nll_per_identity=float(-logp.mean()),
                proposal_chain_nll_per_unknown_identity=(float(-logp[unknown].mean())
                                                         if unknown.any() else None),
                unknown_identities=int(unknown.sum()), token_log_probabilities=token_log_probs,
                causal_allowed_labels=supports, truth_in_support=True,
                constraint_stats=dict(solver.stats),
                likelihood_scope='neural proposal before full public replay rejection',
                accepted_distribution_exact_likelihood=False)


def empirical_brier(samples, targets, unknown, *, bootstrap_repeats=400, seed=0):
    """Brier from accepted free samples, with assignment-level MC resampling.

    The unbiased U-statistic corrects the finite-sample plug-in upward bias.
    Its realised estimate can be negative; it is retained without clipping.
    """
    labels, truth, mask = np.asarray(samples), np.asarray(targets), np.asarray(unknown)
    if (labels.ndim != 2 or labels.shape[1] != 75 or len(labels) < 2
            or labels.dtype.kind not in 'iu' or truth.shape != (75,)
            or truth.dtype.kind not in 'iu' or mask.shape != (75,) or mask.dtype != bool):
        raise ValueError('Brier requires at least two complete samples, integer truth, and a boolean mask')
    if (labels < 0).any() or (labels >= 12).any() or (truth < 0).any() or (truth >= 12).any():
        raise ValueError('Identity labels must be between 0 and 11')
    if type(bootstrap_repeats) is not int or bootstrap_repeats < 2:
        raise ValueError('Monte Carlo bootstrap repeats must be at least two')
    n = len(labels)
    frequencies = (labels[..., None] == np.arange(12)).mean(0)
    result = dict(samples=n, unknown_identities=int(mask.sum()),
                  empirical_identity_probabilities=frequencies.tolist(),
                  unique_complete_assignments=len(np.unique(labels, axis=0)),
                  complete_assignment_duplicate_fraction=1 - len(np.unique(labels, axis=0)) / n,
                  brier_scope='sum over 12 labels, mean over uncertain identities',
                  monte_carlo_unit='complete accepted identity assignment',
                  mc_bootstrap_repeats=bootstrap_repeats)
    if not mask.any():
        return dict(result, empirical_brier=None, unbiased_brier=None,
                    plugin_bias_estimate=None, mc_standard_error=None, mc_interval95=None)
    one_hot_truth = np.eye(12)[truth[mask]]
    p = frequencies[mask]
    plugin = float(np.square(p - one_hot_truth).sum(-1).mean())
    bias = float((p * (1 - p)).sum(-1).mean() / (n - 1))
    selected = (labels[:, mask, None] == np.arange(12)).astype(np.float64)
    rng = np.random.default_rng(seed)
    replicates = np.empty(bootstrap_repeats, dtype=np.float64)
    for repeat in range(bootstrap_repeats):
        boot_p = selected[rng.integers(0, n, n)].mean(0)
        replicates[repeat] = np.square(boot_p - one_hot_truth).sum(-1).mean()
    return dict(result, empirical_brier=plugin, unbiased_brier=plugin - bias,
                plugin_bias_estimate=bias, mc_standard_error=float(replicates.std(ddof=1)),
                mc_interval95=np.quantile(replicates, [.025, .975]).tolist(),
                mc_interval_scope='conditional sample bootstrap; not across-game uncertainty')


def aggregate_scores(records, *, seed=0, repeats=2000):
    """Aggregate one state per game, preserving model seed and stage."""
    groups = {}
    for state in records:
        for model in state['models']:
            groups.setdefault(model['checkpoint_sha256'], []).append((state, model))
    summaries = []
    for digest, entries in sorted(groups.items()):
        if len({state['game_id'] for state, _ in entries}) != len(entries):
            raise ValueError('Aggregate scores require one independently selected state per game')
        summary = dict(checkpoint_sha256=digest, belief_kind=entries[0][1]['belief_kind'],
                       training_seed=entries[0][1]['training_seed'], games=len(entries), stages={})
        for stage in ('all', 'early', 'middle', 'late'):
            rows = [model for state, model in entries if stage == 'all' or state['stage'] == stage]
            metrics = {}
            for field, section in (('proposal_chain_nll_per_unknown_identity', 'proposal_score'),
                                   ('empirical_brier', 'free_sample_score'),
                                   ('unbiased_brier', 'free_sample_score')):
                values = np.asarray([row[section][field] for row in rows
                                     if row[section][field] is not None], dtype=np.float64)
                if not len(values):
                    metrics[field] = dict(games=0, mean=None, game_bootstrap_ci95=None)
                    continue
                rng = np.random.default_rng(seed)
                means = values[rng.integers(0, len(values), (repeats, len(values)))].mean(-1)
                metrics[field] = dict(games=len(values), mean=float(values.mean()),
                    game_bootstrap_ci95=(np.quantile(means, [.025, .975]).tolist()
                                         if len(values) > 1 else None))
            summary['stages'][stage] = metrics
        summaries.append(summary)
    return summaries


def paired_score_differences(records, *, seed=0, repeats=2000):
    """Compare matched M/J seeds on exactly the same independent source games."""
    paired = {}
    for state in records:
        by_seed = {}
        for model in state['models']:
            kinds = by_seed.setdefault(model['training_seed'], {})
            if model['belief_kind'] in kinds:
                raise ValueError('Matched score differences need one model per kind and training seed')
            kinds[model['belief_kind']] = model
        for training_seed, kinds in by_seed.items():
            if set(kinds) == {'marginal', 'joint'}:
                paired.setdefault(training_seed, []).append((state, kinds))
    summaries = []
    for training_seed, entries in sorted(paired.items()):
        if len({state['game_id'] for state, kinds in entries}) != len(entries):
            raise ValueError('Paired belief comparisons need distinct source games')
        result = dict(training_seed=training_seed, difference='joint minus marginal; lower is better', stages={})
        for stage in ('all', 'early', 'middle', 'late'):
            metrics = {}
            for field, section in (('proposal_chain_nll_per_unknown_identity', 'proposal_score'),
                                   ('empirical_brier', 'free_sample_score'),
                                   ('unbiased_brier', 'free_sample_score')):
                values = np.asarray([kinds['joint'][section][field] - kinds['marginal'][section][field]
                    for state, kinds in entries if (stage == 'all' or state['stage'] == stage)
                    and kinds['joint'][section][field] is not None
                    and kinds['marginal'][section][field] is not None], dtype=np.float64)
                rng = np.random.default_rng(seed)
                ci = (np.quantile(values[rng.integers(0, len(values), (repeats, len(values)))].mean(-1),
                                  [.025, .975]).tolist() if len(values) > 1 else None)
                metrics[field] = dict(games=len(values), mean=float(values.mean()) if len(values) else None,
                                      paired_game_bootstrap_ci95=ci)
            result['stages'][stage] = metrics
        summaries.append(result)
    return summaries


def diagnostic_setup(args):
    """Shared frozen model/data setup for scoring and value calibration."""
    from scripts.diagnose_facts_belief_search import load_selected_validation_states, selection_metadata
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    manifest_path = args.dataset / 'dataset-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('status') != 'complete'
            or manifest.get('protocol', {}).get('parent_checkpoint_sha256') != sha256(args.policy)):
        raise ValueError('Diagnostics require the complete frozen-policy dataset')
    selected = load_selected_validation_states(args.dataset, args.selection)
    selection_states = len(selected)
    state_indices = getattr(args, 'state_indices', None)
    if state_indices is not None:
        if (args.states is not None or not state_indices or len(set(state_indices)) != len(state_indices)
                or any(index < 0 or index >= len(selected) for index in state_indices)):
            raise ValueError('--state-indices must be unique valid indices and cannot combine with --states')
        selected = [selected[index] for index in state_indices]
    if args.states is not None:
        if not 1 <= args.states <= len(selected):
            raise ValueError('--states must select a nonempty prefix of the frozen state manifest')
        selected = selected[:args.states]
    blueprint = FactsBlueprint(load_evaluation_policy(args.policy, device=device), device)
    models = []
    for path in args.belief:
        model = load_facts_belief(path, policy_path=args.policy, device=device)
        provenance = selected_belief_provenance(path, model, sha256(manifest_path))
        models.append((model, path, provenance))
    if len({row[2]['checkpoint_sha256'] for row in models}) != len(models):
        raise ValueError('Duplicate belief artifacts would duplicate diagnostic measurements')
    protocol = dict(phase='development', split='validation', policy_sha256=sha256(args.policy),
                    dataset_manifest_sha256=sha256(manifest_path), selection_sha256=sha256(args.selection),
                    models=[row[2] for row in models], states=len(selected), seed=args.seed,
                    selection_states=selection_states, state_indices=state_indices,
                    selected_rows=[selection_metadata(*row) for row in selected],
                    device_type=device.type, torch_version=str(torch.__version__),
                    numpy_version=str(np.__version__), inference_source_sha256=inference_source_sha256(),
                    diagnostic_helpers_sha256={name: sha256(Path(__file__).parent / name) for name in (
                        'score_facts_belief_samples.py', 'diagnose_facts_belief_search.py',
                        'benchmark_facts_belief.py', 'compare_facts_joint.py')},
                    observation_channels=317, observation_global_dims=0, identity_encoding='known_only',
                    test_fold_used=False, weights_updated=False, new_training_games=0)
    return device, selected, blueprint, models, protocol


def verify_frozen_diagnostic_artifacts(args, models, protocol):
    if (sha256(args.policy) != protocol['policy_sha256']
            or sha256(args.selection) != protocol['selection_sha256']
            or sha256(args.dataset / 'dataset-manifest.json') != protocol['dataset_manifest_sha256']
            or any(sha256(path) != provenance['checkpoint_sha256'] for model, path, provenance in models)
            or any(sha256(Path(__file__).parent / name) != expected
                   for name, expected in protocol['diagnostic_helpers_sha256'].items())
            or inference_source_sha256() != protocol['inference_source_sha256']):
        raise RuntimeError('Frozen model, data, selection or diagnostic source changed during the run')


def open_diagnostic_output(path, protocol):
    """Reuse only complete outputs or an exact, durably completed state prefix."""
    path = Path(path)
    candidate = path if path.exists() else path.with_suffix('.progress.json')
    if not candidate.exists():
        return dict(protocol=protocol, complete=False, states=[])
    saved = json.loads(candidate.read_text())
    if saved.get('protocol') != protocol or not isinstance(saved.get('states'), list):
        raise ValueError('Existing diagnostic output has another protocol')
    expected = protocol['selected_rows'][:len(saved['states'])]
    if len(saved['states']) > protocol['states'] or any(
            any(state.get(key) != value for key, value in row.items())
            for state, row in zip(saved['states'], expected)):
        raise ValueError('Saved diagnostic progress is not the frozen state prefix')
    if path.exists() and (saved.get('complete') is not True or len(saved['states']) != protocol['states']):
        raise ValueError('Final diagnostic output must cover the complete frozen budget')
    return saved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--belief', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=None, help='Smoke-test prefix of shared frozen selection')
    parser.add_argument('--state-indices', nargs='+', type=int, default=None,
                        help='Unique frozen selection indices for disjoint bounded workers')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--proposal-batches', type=int, default=8)
    parser.add_argument('--max-constraint-nodes', type=int, default=100_000)
    parser.add_argument('--mc-bootstrap-repeats', type=int, default=400)
    parser.add_argument('--seed', type=int, default=96100400)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.samples < 2 or args.mc_bootstrap_repeats < 2:
        raise ValueError('At least two free samples and MC bootstrap repeats are required')
    config = SearchConfig(particles=args.samples, proposal_batches=args.proposal_batches,
                          max_constraint_nodes=args.max_constraint_nodes)
    device, selected, blueprint, models, protocol = diagnostic_setup(args)
    protocol.update(schema='junqi.facts-belief-comparable-validation.v1', samples=args.samples,
                    proposal_batches=args.proposal_batches, max_constraint_nodes=args.max_constraint_nodes,
                    mc_bootstrap_repeats=args.mc_bootstrap_repeats, script_sha256=sha256(__file__),
                    likelihood_scope='proposal before public replay rejection',
                    free_sample_scope='accepted after public replay and root information checks')
    from scripts.diagnose_facts_belief_search import selection_metadata
    output = open_diagnostic_output(args.output, protocol)
    if output['complete']:
        return output
    began = time.perf_counter()
    for index, selected_row in enumerate(selected):
        if index < len(output['states']):
            continue
        item, raw, shard, trace, row = selected_row
        env, history = restore_public_position(trace, raw, int(shard['ply'][row]))
        log_pi, _, legal, obs = blueprint.distribution([env])
        obs, legal = obs.clone(), legal.clone()
        if not np.allclose(obs[0].cpu().numpy(), shard['spatial'][row], rtol=0, atol=1e-7):
            raise ValueError('Saved observation differs from replayed public root')
        inputs = batch_inputs(shard, np.asarray([row]))
        record = dict(selection_metadata(*selected_row), models=[])
        for model, path, provenance in models:
            sampling_seed = args.seed + item['game_id'] * 1009
            search = FactsBeliefSearch(blueprint, model, config, seed=sampling_seed)
            synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                worlds = search.particles(env, history, obs, legal)
            synchronize(device)
            sampling_seconds = time.perf_counter() - started
            labels = np.asarray([[TYPE_TO_INDEX[PieceType(int(world.state.piece_type_arr[int(pid)]))]
                                  for pid in inputs.piece_ids[0]] for world in worlds], dtype=np.int64)
            # Targets are read only after all free samples are accepted.
            truth = np.array(shard['targets'][row], copy=True)
            unknown = (inputs.known_types[0] < 0) & (inputs.type_domains[0].sum(-1) > 1)
            scored = empirical_brier(labels, truth, unknown,
                bootstrap_repeats=args.mc_bootstrap_repeats, seed=sampling_seed + 1)
            proposal = proposal_chain_score(model, obs, inputs, truth,
                                           max_constraint_nodes=args.max_constraint_nodes)
            record['models'].append(dict(checkpoint_sha256=provenance['checkpoint_sha256'],
                belief_kind=model.belief_kind, training_seed=provenance['training_seed'],
                free_samples=labels.tolist(), sampling_seed=sampling_seed,
                free_sample_seconds=sampling_seconds, sampling_report=search.report(),
                free_sample_score=scored, proposal_score=proposal))
        output['states'].append(record)
        write_json(args.output.with_suffix('.progress.json'), output)
        print(json.dumps(dict(phase='belief_scoring', states_completed=index + 1,
                             target_states=len(selected))), flush=True)
    output.update(complete=True, seconds=time.perf_counter() - began,
                  aggregate=aggregate_scores(output['states'], seed=args.seed),
                  paired_joint_minus_marginal=paired_score_differences(output['states'], seed=args.seed),
                  aggregate_interval_scope='whole validation games; each trained belief seed separate')
    verify_frozen_diagnostic_artifacts(args, models, protocol)
    write_json(args.output, output)
    return output


if __name__ == '__main__':
    main()
