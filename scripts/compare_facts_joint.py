#!/usr/bin/env python3
"""Paired fresh-layout matches between facts policies and neural belief searches.

Only completed games enter the result. Each layout is played twice, swapping
the candidate's team. Confidence intervals resample whole layout pairs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from junqi_rl.training._checkpoint_io import atomic_write


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inference_source_sha256():
    """Freeze transitive Python rules/features/model/search code for safe resume."""
    root = Path(__file__).resolve().parents[1]
    files = sorted(path for package in ('junqi_core', 'junqi_rl') for path in (root / package).rglob('*.py'))
    manifest = {str(path.relative_to(root)): sha256(path) for path in files}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def paired_statistics(games, *, seed=0, repeats=10000):
    """No independence assumption between the two team assignments."""
    by_pair = {}
    for game in games:
        if type(game['candidate_team']) is not int or game['candidate_team'] not in (0, 1):
            raise ValueError('Candidate team must be 0 or 1')
        if game['outcome'] not in ('win', 'loss', 'draw'):
            raise ValueError('Only completed games belong in the comparison')
        pair = by_pair.setdefault(game['pair'], {})
        if game['candidate_team'] in pair:
            raise ValueError('Duplicate layout/team result')
        pair[game['candidate_team']] = game
    paired = [p for p in by_pair.values() if set(p) == {0, 1}]
    if any(p[0]['setup_sha256'] != p[1]['setup_sha256'] for p in paired):
        raise ValueError('A layout pair must use the same initial setup')
    used = [g for p in paired for g in p.values()]
    wins = sum(g['outcome'] == 'win' for g in used)
    losses = sum(g['outcome'] == 'loss' for g in used)
    draws = len(used) - wins - losses
    result = dict(completed_games=len(games), complete_pairs=len(paired),
                  unpaired_completed_games=len(games) - len(used),
                  paired_games=len(used), wins=wins, losses=losses, draws=draws,
                  win_rate=wins / len(used) if used else None,
                  score=(wins + 0.5 * draws) / len(used) if used else None,
                  score_ci95=None, win_rate_ci95=None,
                  interval_method='percentile bootstrap over complete layout pairs')
    if len(paired) >= 2:
        scores = np.asarray([np.mean([{'win': 1., 'draw': .5, 'loss': 0.}[g['outcome']]
                                      for g in p.values()]) for p in paired])
        win_rates = np.asarray([np.mean([g['outcome'] == 'win' for g in p.values()])
                                for p in paired])
        choices = np.random.default_rng(seed).integers(0, len(paired), (repeats, len(paired)))
        result['score_ci95'] = np.quantile(scores[choices].mean(1), [.025, .975]).tolist()
        result['win_rate_ci95'] = np.quantile(win_rates[choices].mean(1), [.025, .975]).tolist()
    return result


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, indent=2) + '\n').encode()))


def bind_protocol(out, protocol):
    """One shared protocol, serialized even when workers start simultaneously."""
    with (out / '.protocol.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = out / 'protocol.json'
        if path.exists():
            if json.loads(path.read_text()) != protocol:
                raise ValueError('Comparison directory belongs to another protocol')
        else:
            write_json(path, protocol)
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def validated_games(out, protocol, digest):
    games = []
    for path in sorted(out.glob('game-*.json')):
        game = json.loads(path.read_text())
        pair, team = game['pair'], game['candidate_team']
        if (type(pair) is not int or not 0 <= pair < protocol['pairs']
                or type(team) is not int or team not in (0, 1)
                or path.name != f'game-{pair:06d}-{team}.json'
                or game.get('protocol_sha256') != digest
                or game['setup_seed'] != protocol['seed'] + pair
                or game['setup_sha256'] != setup_sha256(protocol['seed'] + pair)):
            raise ValueError('Completed match provenance/protocol mismatch')
        if protocol.get('schema') == 'junqi.facts-belief-comparison.v2':
            validate_side_metrics(game, protocol)
        games.append(game)
    paired_statistics(games, repeats=1)
    return games


def collect_layout_hashes(value):
    """Collect observed layouts, including planned layouts in comparison protocols."""
    result = set()
    def visit(item):
        if isinstance(item, dict):
            if (str(item.get('schema', '')).startswith(('junqi.facts-joint-comparison.',
                    'junqi.facts-belief-comparison.')) and 'seed' in item and 'pairs' in item):
                result.update(setup_sha256(item['seed'] + pair) for pair in range(item['pairs']))
            for key, child in item.items():
                if key in ('initial_setup_sha256', 'setup_sha256', 'layout_sha256', 'setup_hash', 'layout_hash'):
                    if not isinstance(child, str) or len(child) != 64:
                        raise ValueError('Malformed excluded layout digest')
                    result.add(child)
                else:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return result


def excluded_layouts(manifest, paths):
    forbidden = collect_layout_hashes(manifest)
    if not forbidden:
        raise ValueError('Dataset has no layout hashes; cannot verify held-out match layouts')
    evidence = []
    for filename in paths:
        path = Path(filename)
        if path.is_dir():
            # The protocol also excludes unfinished planned pairs from development.
            path = path / 'protocol.json'
        content = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                   if path.suffix == '.jsonl' else json.loads(path.read_text()))
        hashes = collect_layout_hashes(content)
        if not hashes:
            raise ValueError(f'Exclusion artifact has no layout hashes: {path}')
        evidence.append(dict(sha256=sha256(path), layout_count=len(hashes)))
        forbidden.update(hashes)
    return forbidden, sorted(evidence, key=lambda item: item['sha256'])


def validate_side_metrics(game, protocol):
    """Symmetric provenance and complete search coverage are mandatory for both teams."""
    if game.get('sides_sha256') != hashlib.sha256(json.dumps(protocol['sides'], sort_keys=True).encode()).hexdigest():
        raise ValueError('Completed match side provenance mismatch')
    moves = 0
    for role in ('candidate', 'opponent'):
        report = game.get('side_metrics', {}).get(role)
        if not isinstance(report, dict) or type(report.get('decisions')) is not int or report['decisions'] < 0:
            raise ValueError('Completed match has invalid side decision counts')
        moves += report['decisions']
        if protocol['sides'][role]['kind'] == 'belief_search':
            if (report.get('successful_decisions') != report['decisions']
                    or report.get('sample_budget_failures') != 0
                    or report.get('constraint_budget_failures') != 0
                    or report.get('candidate_actions') != report.get('legal_actions')
                    or report.get('config') != protocol['search_config']):
                raise ValueError('Completed match search coverage or no-fallback contract failed')
    if moves != game.get('plies') or type(game.get('plies')) is not int or game['plies'] <= 0:
        raise ValueError('Completed match side counts must equal all game plies')


def selected_belief_provenance(path, net, manifest_hash):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    kind = net.belief_kind
    if (saved.get('artifact_role') != f'validation-selected-facts-{kind}-belief'
            or saved.get('training_contract', {}).get('dataset_manifest_sha256') != manifest_hash
            or saved.get('training_contract', {}).get('test_fold_used') is not False):
        raise ValueError('Match dataset must bind a validation-selected belief checkpoint')
    metric = ('conditional_nll_per_unknown_piece' if kind == 'joint' else 'marginal_nll_per_unknown_piece')
    epochs = saved.get('epoch_metrics', [])
    selected = saved.get('selected_epoch')
    if (type(selected) is not int or not epochs or not 1 <= selected <= len(epochs)
            or selected != min(range(len(epochs)), key=lambda i: epochs[i]['validation'][metric]) + 1):
        raise ValueError('Belief checkpoint does not match validation selection')
    contract = saved['training_contract']
    return dict(kind='belief_search', belief_kind=kind, checkpoint_sha256=sha256(path),
                architecture_id=net.architecture_id, model_config=saved['config'],
                selected_epoch=selected, training_seed=contract['seed'],
                training_contract_sha256=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                completed_training_games=contract['games'], observation_positions_per_game=contract['positions_per_game'],
                distribution_semantics=net.distribution_semantics)


@lru_cache(maxsize=4096)
def setup_sha256(seed):
    import random
    from junqi_core.setup import generate_random_setup
    from junqi_core.state import GameState
    state = GameState.new_game(generate_random_setup(random.Random(seed)))
    return hashlib.sha256(state.piece_type_arr.tobytes()).hexdigest()


def main(argv=None):
    from junqi_core.rules import ShowMode, MAX_NUM_MOVES
    from junqi_core.state import classify_termination
    from junqi_rl.analysis.facts_joint_search import (
        FactsBlueprint, FactsJointSearch, SearchConfig, public_result, step_world)
    from junqi_rl.belief.facts_joint_checkpoint import load_facts_belief
    from junqi_rl.env import JunqiEnv, unrotate_compact_action_id
    from junqi_rl.training.checkpoint import load_evaluation_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--belief', '--joint', dest='belief', required=True,
                        help='Candidate validation-selected marginal or joint belief checkpoint')
    parser.add_argument('--opponent-belief', help='Optional selected belief for the opponent; otherwise pi0 directly')
    parser.add_argument('--exclude-layouts', action='append', default=[],
                        help='Prior comparison directory/protocol/result JSON, repeatable; all planned pairs excluded')
    parser.add_argument('--phase', choices=('development', 'formal'), default='formal')
    parser.add_argument('--dataset', required=True, help='Dataset directory to exclude all training/validation/test layouts')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pairs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=93000000)
    parser.add_argument('--particles', type=int, default=1)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--proposal-batches', type=int, default=8)
    parser.add_argument('--beta', type=float, default=.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--wall-seconds', type=float, default=7200)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.pairs < 1 or args.wall_seconds <= 0 or not 0 <= args.worker_index < args.workers:
        raise ValueError('Invalid game/worker/time budget')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy = load_evaluation_policy(args.policy, device=args.device)
    belief = load_facts_belief(args.belief, policy_path=args.policy, device=args.device)
    opponent_belief = (load_facts_belief(args.opponent_belief, policy_path=args.policy, device=args.device)
                       if args.opponent_belief else None)
    blueprint = FactsBlueprint(policy, args.device)
    config = SearchConfig(particles=args.particles, depth=args.depth, beta=args.beta,
                          proposal_batches=args.proposal_batches)
    dataset = Path(args.dataset)
    manifest_path = dataset / 'dataset-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'junqi.facts-belief-dataset.v1' or manifest.get('status') != 'complete':
        raise ValueError('Comparison requires the complete training dataset manifest')
    manifest_hash = sha256(manifest_path)
    sides = {'candidate': selected_belief_provenance(args.belief, belief, manifest_hash),
             'opponent': (selected_belief_provenance(args.opponent_belief, opponent_belief, manifest_hash)
                          if opponent_belief is not None else {'kind': 'direct_policy'})}
    for side in sides.values():
        side['policy_sha256'] = sha256(args.policy)
    sides_hash = hashlib.sha256(json.dumps(sides, sort_keys=True).encode()).hexdigest()
    forbidden, exclusion_evidence = excluded_layouts(manifest, args.exclude_layouts)
    requested = [setup_sha256(args.seed + pair) for pair in range(args.pairs)]
    if len(set(requested)) != len(requested) or forbidden.intersection(requested):
        raise ValueError('Held-out evaluation layouts duplicate or overlap excluded dataset/prior layouts')
    protocol = dict(schema='junqi.facts-belief-comparison.v2',
                    policy_sha256=sha256(args.policy), joint_sha256=sha256(args.belief),
                    dataset_manifest_sha256=manifest_hash, sides=sides, phase=args.phase,
                    seed=args.seed, pairs=args.pairs, max_moves=MAX_NUM_MOVES,
                    search_config=asdict(config), workers=args.workers,
                    baseline=sides['opponent'], candidate=sides['candidate'],
                    team_assignments=[0, 1], action_selection='sample distribution',
                    torch_version=str(torch.__version__), device_type=torch.device(args.device).type,
                    deterministic_algorithms=True, comparison_script_sha256=sha256(__file__),
                    inference_source_sha256=inference_source_sha256(), numpy_version=str(np.__version__),
                    exclusion_artifacts=exclusion_evidence,
                    forbidden_layout_sha256=hashlib.sha256(json.dumps(sorted(forbidden)).encode()).hexdigest(),
                    forbidden_layout_count=len(forbidden))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    digest = bind_protocol(out, protocol)
    existing = validated_games(out, protocol, digest)
    if existing and not args.resume and any(g['pair'] % args.workers == args.worker_index for g in existing):
        raise ValueError('Existing completed games require --resume')
    started = time.perf_counter()
    deadline = started + args.wall_seconds

    def save():
        games = validated_games(out, protocol, digest)
        games.sort(key=lambda g: (g['pair'], g['candidate_team']))
        result = dict(protocol=protocol, statistics=paired_statistics(games, seed=args.seed),
                      finished=len(games) == args.pairs * 2, games=games)
        # Each worker writes its own snapshot; aggregation is independent and atomic.
        write_json(out / f'comparison-worker-{args.worker_index}.json', result)
        return result

    for pair in range(args.worker_index, args.pairs, args.workers):
        for team in (0, 1):
            destination = out / f'game-{pair:06d}-{team}.json'
            if destination.exists():
                if not args.resume:
                    raise ValueError('Refusing to overwrite completed match')
                validated_games(out, protocol, digest)
                continue
            if time.perf_counter() >= deadline:
                print(json.dumps(save()), flush=True)
                return
            env = JunqiEnv(show_mode=ShowMode.DARK, include_global=False,
                           identity_encoding='known_only', record_public_history=True)
            env.reset(args.seed + pair)
            setup_hash = hashlib.sha256(env.state.piece_type_arr.tobytes()).hexdigest()
            if setup_hash in forbidden:
                raise ValueError('Held-out evaluation layout overlaps belief dataset')
            game_seed = args.seed + pair * 2 + team
            searches = {'candidate': FactsJointSearch(blueprint, belief, config, seed=game_seed),
                        'opponent': (FactsJointSearch(blueprint, opponent_belief, config, seed=game_seed + 100000)
                                     if opponent_belief is not None else None)}
            base_rng = torch.Generator(device=args.device).manual_seed(game_seed + 100000)
            direct_decisions = 0

            def side_reports():
                return {role: (search.report() if search is not None else dict(decisions=direct_decisions))
                        for role, search in searches.items()}
            history = []
            last = None
            game_started = time.perf_counter()
            try:
                while not env.state.terminated:
                    if len(history) >= MAX_NUM_MOVES:
                        raise RuntimeError('Rules failed to terminate at their own move limit')
                    if time.perf_counter() >= deadline:
                        write_json(out / f'interrupted-worker-{args.worker_index}.json',
                                   dict(pair=pair, candidate_team=team, plies=len(history),
                                        reason='wall budget; unfinished game excluded'))
                        save()
                        return
                    seat = env.state.turn
                    role = 'candidate' if int(seat) % 2 == team else 'opponent'
                    if searches[role] is not None:
                        action = searches[role].action(env, history)
                    else:
                        direct_decisions += 1
                        log_pi, _, _, _ = blueprint.distribution([env])
                        aid = int(torch.multinomial(log_pi.exp(), 1, generator=base_rng)[0, 0])
                        action = unrotate_compact_action_id(aid, seat)
                    last = step_world(env, action)
                    history.append(public_result(last))
            except Exception as exc:
                write_json(out / f'failure-worker-{args.worker_index}.json',
                           dict(pair=pair, candidate_team=team, plies=len(history),
                                protocol_sha256=digest, setup_seed=args.seed + pair,
                                setup_sha256=setup_hash, observer=int(env.state.turn),
                                public_history=history,
                                error=f'{type(exc).__name__}: {exc}', side_metrics=side_reports()))
                raise
            outcome = 'draw' if env.state.draw else ('win' if env.state.winner_team == team else 'loss')
            game = dict(pair=pair, setup_seed=args.seed + pair, setup_sha256=setup_hash,
                        protocol_sha256=digest,
                        candidate_team=team, outcome=outcome, plies=len(history),
                        reason=classify_termination(env.state, last),
                        seconds=time.perf_counter() - game_started, search_metrics=searches['candidate'].report(),
                        sides_sha256=sides_hash, side_metrics=side_reports(),
                        action_history_sha256=hashlib.sha256(repr(history).encode()).hexdigest())
            write_json(destination, game)
            save()
            print(json.dumps({k: v for k, v in game.items() if k not in ('search_metrics', 'side_metrics')}), flush=True)
    print(json.dumps(save()), flush=True)


if __name__ == '__main__':
    main()
