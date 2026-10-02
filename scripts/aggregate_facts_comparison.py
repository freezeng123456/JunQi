"""Validate and aggregate completed paired matches after all workers finish."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compare_facts_joint import paired_statistics, validated_games, write_json


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
    write_json(args.output / 'comparison.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'games'}, indent=2))
    if args.require_complete and not report['finished']:
        raise RuntimeError('Requested layout pairs have not all finished; unfinished work excluded')
    return report


if __name__ == '__main__':
    main()
