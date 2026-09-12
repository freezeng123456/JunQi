"""Verify paired raw-policy evaluation artifacts and retain every failed game."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.belief_features_20260913.analyze import read, verify_manifest


def inspect(folder):
    inventory = verify_manifest(folder)
    summary, provenance = read(folder / 'summary.json'), read(folder / 'provenance.json')
    assert summary['status'] == 'complete' and (folder / '.done').is_file()
    records = [json.loads(row) for row in (folder / 'games.jsonl').read_text().splitlines()]
    assert len(records) == summary['requested_total_games'] == summary['recovered_games']
    assert len(records) == provenance['games_per_block'] * provenance['blocks']
    blocks = [read(path) for path in sorted(folder.glob('block_*.json'))]
    assert len(blocks) == provenance['blocks']
    assert records == [row for block in blocks for row in block['records']]
    keyed = {(row['setup_seed'], row['first_team']): row for row in records}
    assert len(keyed) == len(records)
    counts = Counter(row['outcome'] for row in records)
    assert set(counts) <= {'win', 'loss', 'draw', 'ongoing'}
    for outcome in ('win', 'loss', 'draw', 'ongoing'):
        metric = {'win': 'wins', 'loss': 'losses', 'draw': 'draws', 'ongoing': 'ongoing'}[outcome]
        assert counts[outcome] == summary['metrics']['eval/' + metric]
    assert summary['metrics']['eval/win_rate'] == counts['win'] / len(records)
    seeds = sorted({row['setup_seed'] for row in records})
    assert seeds == list(range(provenance['setup_seed'], provenance['setup_seed'] + len(seeds)))
    for seed in seeds:
        pair = [keyed[(seed, team)] for team in (0, 1)]
        assert pair[0]['setup_sha256'] == pair[1]['setup_sha256']
        assert pair[0]['random_seed'] == pair[1]['random_seed']
    return {
        'verified_files': inventory, 'summary': summary, 'provenance': provenance,
        'both_teams_won_pairs': sum(all(keyed[(seed, team)]['outcome'] == 'win' for team in (0, 1)) for seed in seeds),
        'independent_setup_pairs': len(seeds),
        'nonwins': [row for row in records if row['outcome'] != 'win'],
    }, keyed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prefix', choices=['current_cpu', 'current_gpu'], required=True)
    args = parser.parse_args()
    results, keyed = {}, {}
    for role in 'AB':
        results[role], keyed[role] = inspect(args.root / f'{args.prefix}_{role}')
    checkpoints = {role: Path(results[role]['provenance']['checkpoint']).name for role in 'AB'}
    assert set(checkpoints.values()) == {'frozen_policy_v4.pt', 'ckpt_001000.pt'}
    guarded_role = next(role for role in 'AB' if checkpoints[role] == 'frozen_policy_v4.pt')
    raw_role = next(role for role in 'AB' if checkpoints[role] == 'ckpt_001000.pt')
    for role in 'AB':
        provenance = results[role]['provenance']
        assert provenance['effective_checkpoint_metadata']['observation_semantics_version'] == 4
        assert provenance['ema'] is False and provenance['learned_belief_in_policy'] is False
    left, right = keyed['A'], keyed['B']
    assert left.keys() == right.keys()
    for key in left:
        assert left[key]['setup_sha256'] == right[key]['setup_sha256']
        assert left[key]['random_seed'] == right[key]['random_seed']
    for name in ('source_commit', 'runner_sha256', 'backend', 'dtype', 'greedy',
                 'games_per_block', 'blocks', 'setup_seed', 'random_seed', 'max_moves'):
        assert results['A']['provenance'][name] == results['B']['provenance'][name], name
    left, right = keyed[guarded_role], keyed[raw_role]
    seeds = sorted({key[0] for key in left})
    deltas = np.array([sum((left[(seed, team)]['outcome'] == 'win') -
                          (right[(seed, team)]['outcome'] == 'win') for team in (0, 1)) / 2 for seed in seeds])
    rng = np.random.default_rng(620914)
    bootstrap = deltas[rng.integers(len(seeds), size=(5000, len(seeds)))].mean(1)
    comparison = {
        'status': 'all_paired_policy_results_verified', 'runs': results,
        'labels': {guarded_role: 'guarded_s501_v4', raw_role: 'raw_001000_evaluated_on_v4'},
        'roles_by_policy': {'guarded': guarded_role, 'raw': raw_role},
        'same_setup_and_random_seeds_and_paired_teams_verified': True,
        'guarded_minus_raw_win_rate': float(deltas.mean()),
        'paired_setup_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
        'pair_bootstrap_scope': 'Conditional on these two fixed policy checkpoints; both team outcomes resampled together.',
        'finite_perfect_suite_is_not_proof_of_population_win_probability_one': True,
        'no_learned_belief_or_ema_in_either_evaluation': True,
    }
    output = args.root / 'analysis_policy'
    output.mkdir(exist_ok=True)
    (output / f'{args.prefix}.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps({role: results[role]['summary']['metrics'] for role in 'AB'}), flush=True)


if __name__ == '__main__':
    main()
