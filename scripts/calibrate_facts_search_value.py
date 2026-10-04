#!/usr/bin/env python3
"""Calibrate frozen depth-one leaf values against independent pi0 continuations.

Accepted neural worlds and publicly selected root actions define the leaves.
The critic sees the next acting player's legitimate known_only 317/0 input.
Every continuation uses the same frozen policy for all players, an independent
counter-based random stream, and a declared completed-continuation budget.
Terminal outcomes and optional cutoff draws are reported separately. These
simulated continuations are neither new training games nor formal match games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from junqi_core.identity import TYPE_TO_INDEX
from junqi_core.rules import PieceType
from junqi_rl.analysis.facts_joint_search import (
    FactsBeliefSearch, SearchConfig, _finished_value, clone_world, step_world,
)
from junqi_rl.analysis.random_eval import WEIGHTED_SAMPLING_VERSION, _sample_weights
from junqi_rl.belief.joint import build_joint_input
from junqi_rl.env import unrotate_compact_action_id
from scripts.benchmark_facts_belief import restore_public_position, synchronize
from scripts.compare_facts_joint import sha256, write_json
from scripts.score_facts_belief_samples import (
    diagnostic_setup, open_diagnostic_output, verify_frozen_diagnostic_artifacts,
)


def select_root_actions(log_policy, legal, count, *, seed):
    """Highest pi0 action then seeded legal alternatives, without hidden labels."""
    logp, support = np.asarray(log_policy), np.asarray(legal)
    if (logp.ndim != 1 or support.shape != logp.shape or support.dtype != bool
            or not support.any() or not np.isfinite(logp[support]).all()
            or type(count) is not int or count < 1):
        raise ValueError('Action calibration needs finite legal pi0 scores and a positive budget')
    ids = np.flatnonzero(support)
    top = int(ids[np.argmax(logp[ids])])
    alternatives = ids[ids != top]
    np.random.default_rng(seed).shuffle(alternatives)
    return [top] + alternatives[:max(0, min(count, len(ids)) - 1)].tolist()


def root_team_leaf_value(blueprint, world, root_seat):
    """Match production depth-one terminal handling and next-actor sign exactly."""
    finished = _finished_value(world, root_seat)
    if finished is not None:
        return dict(value=finished, source=('terminal_team_reward' if world.state.terminated
                                           else 'declared_total_move_limit_draw'),
                    next_actor=int(world.state.turn), root_team=int(root_seat) & 1,
                    sign=None, next_actor_value=None)
    actor_value = float(blueprint.values([world])[0].detach().cpu())
    sign = 1 if (int(world.state.turn) & 1) == (int(root_seat) & 1) else -1
    return dict(value=sign * actor_value, source='frozen_next_actor_value_head',
                next_actor=int(world.state.turn), root_team=int(root_seat) & 1,
                sign=sign, next_actor_value=actor_value)


def value_training_target_contract(policy_path):
    """Read the actual frozen parent's reward and time-discount provenance."""
    saved = torch.load(policy_path, map_location='cpu', weights_only=False)
    train = saved.get('train_cfg') if isinstance(saved, dict) else None
    ppo = train.get('ppo') if isinstance(train, dict) else None
    if not isinstance(ppo, dict) or any(key not in ppo for key in ('gamma', 'gae_lambda', 'td_lambda')):
        raise ValueError('Value calibration requires explicit parent training discount/trace provenance')
    gamma, gae, td = (ppo[key] for key in ('gamma', 'gae_lambda', 'td_lambda'))
    if any(type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1
           for value in (gamma, gae, td)):
        raise ValueError('Parent discount and trace parameters must be finite in [0,1]')
    if train.get('reward_shaping') is not False or train.get('random_opponent') is not False:
        raise ValueError('This calibration requires unshaped four-player self-play terminal rewards')
    root = Path(__file__).resolve().parents[1]
    sources = ('junqi_rl/training/rollout_gpu.py', 'junqi_rl/training/gpu_collector.py',
               'junqi_rl/training/ppo.py')
    return dict(gamma=float(gamma), gae_lambda=float(gae), td_lambda=float(td),
        reward_shaping=False, random_opponent=False, reward='terminal acting-team +/-1; draw/nonterminal zero',
        discount_time_unit='one environment ply, including every acting seat',
        value_update='TD(lambda) with actual next-actor team sign; lambda is bootstrap trace, not reward discount',
        independent_calibration_target='discounted terminal Monte Carlo return, the value fixed-point target',
        exact_finite_training_label=False,
        terminal_discount_exponent='max(continuation_moves-1,0); immediate terminal step has exponent zero',
        training_implementation_sha256={path: sha256(root / path) for path in sources})


def discounted_terminal_return(outcome, moves, gamma):
    if (not np.isfinite(outcome) or type(moves) is not int or moves < 0
            or type(gamma) not in (int, float) or not np.isfinite(gamma) or not 0 <= gamma <= 1):
        raise ValueError('Terminal return requires a finite outcome, nonnegative ply count and gamma in [0,1]')
    return float(outcome * gamma ** max(0, moves - 1))


@torch.no_grad()
def pi0_continuations(blueprint, leaf, root_seat, seeds, *, max_continuation_moves=0, gamma=1.0):
    """Finish every fixed seeded clone; lane RNG is independent of other lanes.

    A positive max_continuation_moves is an explicitly truncated calibration
    target assigning zero to unfinished worlds. Default zero uses the original
    full game's declared move limit. Short caps are labelled acceptance-only.
    """
    seeds = np.asarray(seeds)
    if (seeds.ndim != 1 or not len(seeds) or seeds.dtype.kind not in 'iu'
            or (seeds < 0).any() or len(set(seeds.tolist())) != len(seeds)
            or type(max_continuation_moves) is not int or max_continuation_moves < 0):
        raise ValueError('Continuations require distinct nonnegative integer seeds and a valid move cap')
    discounted_terminal_return(0.0, 0, gamma)  # Validate the reward discount before any inference.
    worlds = [clone_world(leaf) for _ in seeds]
    records = [None] * len(worlds)
    digests = [hashlib.sha256() for _ in worlds]
    active = list(range(len(worlds)))
    initial_ply = leaf.state.move_counter
    steps = 0
    while active:
        pending = []
        for index in active:
            world = worlds[index]
            finished = _finished_value(world, root_seat)
            capped = bool(max_continuation_moves and steps >= max_continuation_moves)
            if finished is None and not capped:
                pending.append(index)
                continue
            terminal = bool(world.state.terminated)
            draw = bool(world.state.draw) if terminal else True
            reason = ('team_victory' if terminal and not draw else
                      'terminal_draw' if terminal else
                      'continuation_move_cap_draw' if capped else 'original_total_move_limit_draw')
            moves = world.state.move_counter - initial_ply
            outcome = float(finished if finished is not None else 0.0)
            records[index] = dict(seed=int(seeds[index]), completed=True, terminal=terminal,
                draw=draw, root_team_return=outcome,
                root_team_discounted_return=discounted_terminal_return(outcome, moves, gamma),
                moves=moves, end_ply=world.state.move_counter,
                winner_team=(None if world.state.winner_team is None else int(world.state.winner_team)),
                termination_reason=reason, action_history_sha256=digests[index].hexdigest())
        active = pending
        if not active:
            break
        batch = [worlds[index] for index in active]
        logp, _, legal, _ = blueprint.distribution(batch)
        stream_seeds = torch.as_tensor(seeds[active].astype(np.int64), device=blueprint.device)
        actions = _sample_weights(logp.exp(), legal, stream_seeds, steps, stream=47)
        for index, compact_id in zip(active, actions.cpu().tolist()):
            world = worlds[index]
            action = unrotate_compact_action_id(int(compact_id), world.state.turn)
            digests[index].update(np.asarray([int(world.state.turn), action], dtype='<i8').tobytes())
            step_world(world, action)
        steps += 1
    if any(row is None or row['completed'] is not True for row in records):
        raise RuntimeError('Continuation budget includes an unfinished clone')
    return records


def summarize_calibration(states, *, bootstrap_repeats=2000, seed=0):
    """Model-specific error and calibration bins; intervals resample source games."""
    groups = {}
    for state in states:
        for model in state['models']:
            groups.setdefault(model['checkpoint_sha256'], []).append((state, model))
    output = []
    for digest, rows in sorted(groups.items()):
        if len({state['game_id'] for state, model in rows}) != len(rows):
            raise ValueError('Value calibration requires distinct source games')
        leaves = [leaf for state, model in rows for leaf in model['leaves']]
        neural = [leaf for leaf in leaves if leaf['prediction']['source'] == 'frozen_next_actor_value_head']
        errors_by_game = []
        for state, model in rows:
            errors = [leaf['prediction']['value'] - leaf['continuation_discounted_mean_return']
                      for leaf in model['leaves']
                      if leaf['prediction']['source'] == 'frozen_next_actor_value_head']
            noise = [leaf['continuation_discounted_sample_variance'] / len(leaf['continuations'])
                     for leaf in model['leaves']
                     if leaf['prediction']['source'] == 'frozen_next_actor_value_head'
                     and leaf['continuation_discounted_sample_variance'] is not None]
            if errors:
                errors_by_game.append([np.mean(errors), np.mean(np.square(errors)), np.mean(np.abs(errors)),
                                       np.mean(np.square(errors)) - np.mean(noise)
                                       if len(noise) == len(errors) else np.nan])
        stats = dict(bias=None, rmse=None, mae=None, game_bootstrap_ci95=None,
                     mse_mc_noise_corrected=None)
        if errors_by_game:
            errors = np.asarray(errors_by_game)
            mean = errors.mean(0)
            stats.update(bias=float(mean[0]), rmse=float(np.sqrt(mean[1])), mae=float(mean[2]),
                         mse_mc_noise_corrected=float(mean[3]) if np.isfinite(mean[3]) else None)
            if len(errors) > 1:
                rng = np.random.default_rng(seed)
                boot = errors[rng.integers(0, len(errors), (bootstrap_repeats, len(errors)))].mean(1)
                boot[:, 1] = np.sqrt(boot[:, 1])
                bounds = np.quantile(boot, [.025, .975], axis=0)
                stats['game_bootstrap_ci95'] = dict(
                    bias=bounds[:, 0].tolist(), rmse=bounds[:, 1].tolist(), mae=bounds[:, 2].tolist())
        bins = []
        for lower, upper in zip(np.linspace(-1, 1, 6)[:-1], np.linspace(-1, 1, 6)[1:]):
            entries = [leaf for leaf in neural if lower <= leaf['prediction']['value']
                       and (leaf['prediction']['value'] < upper or upper == 1)]
            bins.append(dict(lower=float(lower), upper=float(upper), leaves=len(entries),
                mean_prediction=(float(np.mean([leaf['prediction']['value'] for leaf in entries]))
                                 if entries else None),
                mean_continuation_discounted_return=(float(np.mean([
                    leaf['continuation_discounted_mean_return'] for leaf in entries])) if entries else None)))
        complete = [r for leaf in leaves for r in leaf['continuations']]
        output.append(dict(checkpoint_sha256=digest, belief_kind=rows[0][1]['belief_kind'],
            training_seed=rows[0][1]['training_seed'], source_games=len(rows),
            neural_value_leaves=len(neural), terminal_or_cutoff_leaves=len(leaves) - len(neural),
            completed_continuations=len(complete), terminal_continuations=sum(r['terminal'] for r in complete),
            cutoff_draw_continuations=sum(not r['terminal'] for r in complete),
            errors=stats, bins=bins,
            metric_scope='errors against finite independent discounted pi0 mean, averaged within source game',
            corrected_mse_scope='subtract estimated finite-continuation mean variance; not clipped if negative',
            interval_scope='whole source games; one fixed trained belief artifact'))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--belief', action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--states', type=int, default=None)
    parser.add_argument('--state-indices', nargs='+', type=int, default=None)
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--root-actions', type=int, default=1)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--continuation-max-moves', type=int, default=0,
                        help='0 finishes the original game budget; positive caps define a truncated draw target')
    parser.add_argument('--proposal-batches', type=int, default=8)
    parser.add_argument('--max-constraint-nodes', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=97100400)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if min(args.particles, args.root_actions, args.rollouts) < 1 or args.continuation_max_moves < 0:
        raise ValueError('World, action and completed-continuation budgets must be positive; cap nonnegative')
    config = SearchConfig(particles=args.particles, proposal_batches=args.proposal_batches,
                          max_constraint_nodes=args.max_constraint_nodes)
    device, selected, blueprint, models, protocol = diagnostic_setup(args)
    target_contract = value_training_target_contract(args.policy)
    protocol.update(schema='junqi.facts-search-value-calibration.v1', particles=args.particles,
        root_actions_budget=args.root_actions, continuations_per_leaf=args.rollouts,
        continuation_max_moves=args.continuation_max_moves, proposal_batches=args.proposal_batches,
        max_constraint_nodes=args.max_constraint_nodes, script_sha256=sha256(__file__),
        action_selection='highest pi0 probability then seeded legal alternatives',
        continuation_sampling=WEIGHTED_SAMPLING_VERSION, continuation_stream=47,
        continuation_seed_protocol='SeedSequence(base,game_id,canonical_action,particle,repeat)',
        value_training_target_contract=target_contract,
        acceptance_only=args.continuation_max_moves > 0,
        target_scope=('truncated zero-on-cutoff pi0 continuation return' if args.continuation_max_moves
                      else 'discounted pi0 terminal return within original completed-game move limit'))
    output = open_diagnostic_output(args.output, protocol)
    if output['complete']:
        return output
    from scripts.diagnose_facts_belief_search import selection_metadata
    began = time.perf_counter()
    for state_index, selected_row in enumerate(selected):
        if state_index < len(output['states']):
            continue
        item, raw, shard, trace, row = selected_row
        env, history = restore_public_position(trace, raw, int(shard['ply'][row]))
        root_seat = env.state.turn
        logp, _, legal, obs = blueprint.distribution([env])
        logp, legal, obs = logp.clone(), legal.clone(), obs.clone()
        if not np.allclose(obs[0].cpu().numpy(), shard['spatial'][row], rtol=0, atol=1e-7):
            raise ValueError('Saved observation differs from replayed public root')
        actions = select_root_actions(logp[0].cpu().numpy(), legal[0].cpu().numpy(), args.root_actions,
                                      seed=args.seed + item['game_id'])
        inputs = build_joint_input([env.knowledge_snapshot(root_seat)])
        record = dict(selection_metadata(*selected_row), root_actions=actions, models=[])
        for model, path, provenance in models:
            belief_seed = args.seed + 10_000_000 + item['game_id'] * 1009
            search = FactsBeliefSearch(blueprint, model, config, seed=belief_seed)
            synchronize(device)
            worlds = search.particles(env, history, obs, legal)
            labels = [[TYPE_TO_INDEX[PieceType(int(world.state.piece_type_arr[int(pid)]))]
                       for pid in inputs.piece_ids[0]] for world in worlds]
            measured = dict(checkpoint_sha256=provenance['checkpoint_sha256'],
                belief_kind=model.belief_kind, training_seed=provenance['training_seed'],
                belief_seed=belief_seed, sampled_assignments=labels, sampling_report=search.report(), leaves=[])
            for canonical_id in actions:
                for particle, world in enumerate(worlds):
                    leaf = clone_world(world)
                    action = unrotate_compact_action_id(canonical_id, root_seat)
                    step_world(leaf, action)
                    prediction = root_team_leaf_value(blueprint, leaf, root_seat)
                    seeds = np.asarray([np.random.SeedSequence(
                        [args.seed, item['game_id'], canonical_id, particle, repeat]).generate_state(1)[0]
                        for repeat in range(args.rollouts)], dtype=np.int64)
                    synchronize(device)
                    started = time.perf_counter()
                    returns = pi0_continuations(blueprint, leaf, root_seat, seeds,
                        max_continuation_moves=args.continuation_max_moves, gamma=target_contract['gamma'])
                    synchronize(device)
                    values = np.asarray([result['root_team_return'] for result in returns], dtype=np.float64)
                    variance = float(values.var(ddof=1)) if len(values) > 1 else None
                    discounted = np.asarray([result['root_team_discounted_return'] for result in returns])
                    discounted_variance = float(discounted.var(ddof=1)) if len(discounted) > 1 else None
                    measured['leaves'].append(dict(canonical_action=canonical_id, world_action=action,
                        particle=particle, leaf_ply=leaf.state.move_counter, prediction=prediction,
                        continuations=returns, continuation_mean_return=float(values.mean()),
                        continuation_sample_variance=variance,
                        continuation_discounted_mean_return=float(discounted.mean()),
                        continuation_discounted_sample_variance=discounted_variance,
                        continuation_discounted_mc_standard_error=(
                            float(np.sqrt(discounted_variance / len(discounted)))
                            if discounted_variance is not None else None),
                        continuation_mc_standard_error=(float(np.sqrt(variance / len(values)))
                                                        if variance is not None else None),
                        predicted_minus_continuation_return=float(prediction['value'] - values.mean()),
                        predicted_minus_discounted_continuation_return=float(
                            prediction['value'] - discounted.mean()),
                        continuation_seconds=time.perf_counter() - started))
            record['models'].append(measured)
        output['states'].append(record)
        write_json(args.output.with_suffix('.progress.json'), output)
        print(json.dumps(dict(phase='value_calibration', source_states_completed=state_index + 1,
                             target_source_states=len(selected))), flush=True)
    summary = summarize_calibration(output['states'], seed=args.seed)
    output.update(complete=True, seconds=time.perf_counter() - began, aggregate=summary,
        completed_continuations=sum(row['completed_continuations'] for row in summary),
        terminal_continuations=sum(row['terminal_continuations'] for row in summary),
        cutoff_draw_continuations=sum(row['cutoff_draw_continuations'] for row in summary),
        new_training_games=0, formal_match_games=0)
    verify_frozen_diagnostic_artifacts(args, models, protocol)
    if sha256(__file__) != protocol['script_sha256']:
        raise RuntimeError('Frozen calibration source changed during the run')
    write_json(args.output, output)
    return output


if __name__ == '__main__':
    main()
