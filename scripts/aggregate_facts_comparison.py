"""Validate and aggregate completed paired matches after all workers finish."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_facts_joint import paired_statistics, validated_games, write_json


def side_cost_summary(games):
    """Sum cost over completed games for each role without mixing shared priors."""
    answer = {}
    keys = ('decisions', 'successful_decisions', 'proposals', 'accepted_proposals', 'rejected_proposals',
            'sample_budget_failures', 'constraint_budget_failures', 'legal_actions', 'candidate_actions',
            'decision_seconds_total', 'sampling_seconds', 'constraint_seconds', 'simulation_seconds',
            'materialization_seconds', 'unique_accepted_assignments', 'rollout_world_steps')
    for role in ('candidate', 'opponent'):
        reports = [game['side_metrics'][role] for game in games if 'side_metrics' in game]
        if not reports:
            continue
        totals = {key: sum(item.get(key, 0) for item in reports) for key in keys}
        totals['mean_decision_seconds'] = totals['decision_seconds_total'] / max(1, totals['decisions'])
        totals['action_coverage'] = totals['candidate_actions'] / max(1, totals['legal_actions'])
        for queries in ('blueprint_queries', 'constraint_queries'):
            names = {name for item in reports for name in item.get(queries, {})}
            totals[queries] = {name: sum(item.get(queries, {}).get(name, 0) for item in reports) for name in names}
        answer[role] = totals
    return answer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args(argv)
    protocol = json.loads((args.output / 'protocol.json').read_text())
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    games = validated_games(args.output, protocol, digest)
    stats = paired_statistics(games, seed=protocol['seed'])
    report = dict(protocol=protocol, protocol_sha256=digest, statistics=stats,
                  finished=stats['complete_pairs'] == protocol['pairs'], games=games)
    if protocol.get('schema') == 'junqi.facts-belief-comparison.v2':
        report['side_costs'] = side_cost_summary(games)
    write_json(args.output / 'comparison.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'games'}, indent=2))
    if args.require_complete and not report['finished']:
        raise RuntimeError('Requested layout pairs have not all finished; unfinished work excluded')
    return report


if __name__ == '__main__':
    main()
