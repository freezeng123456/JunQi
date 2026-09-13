"""Durable paired evaluation of raw policy weights under observation semantics v4."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import torch

from junqi_rl.analysis.protocol import merge_evaluations
from junqi_rl.analysis.random_eval import evaluate_paired_vs_random
from junqi_rl.checkpoint_compat import validate_policy_checkpoint
from junqi_rl.training.checkpoint import load_evaluation_policy


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-semantics-3-to-4', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--games-per-block', type=int, default=16)
    parser.add_argument('--blocks', type=int, default=32)
    parser.add_argument('--setup-seed', type=int, default=3_001_100)
    parser.add_argument('--random-seed', type=int, default=4_001_100)
    parser.add_argument('--max-moves', type=int, default=4000)
    parser.add_argument('--seconds', type=int, default=3600)
    args = parser.parse_args()
    assert args.games_per_block > 0 and args.games_per_block % 2 == 0
    assert args.blocks > 0 and args.seconds > 0
    assert args.device == 'cuda' or args.dtype == 'float32'
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2 if args.device == 'cpu' else 1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(901)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    original_metadata = dict(checkpoint['checkpoint_meta'])
    migration = None
    if args.allow_semantics_3_to_4:
        assert original_metadata['observation_semantics_version'] == 3
        checkpoint = dict(checkpoint)
        checkpoint['checkpoint_meta'] = dict(original_metadata, observation_semantics_version=4)
        migration = 'Evaluate unchanged v3 raw weights on current v4 inputs; not an original-run reproduction.'
    else:
        assert original_metadata['observation_semantics_version'] == 4
    assert not args.allow_semantics_3_to_4, 'This experiment uses current v4 only'
    policy = load_evaluation_policy(args.checkpoint, device=args.device)
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    provenance = {
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': digest(args.checkpoint),
        'source_commit': code_commit, 'runner_sha256': digest(Path(__file__)),
        'original_checkpoint_metadata': original_metadata,
        'effective_checkpoint_metadata': checkpoint['checkpoint_meta'],
        'migration': migration, 'policy_weights': 'raw policy', 'ema': False,
        'learned_belief_in_policy': False, 'observation': 'current public rules v4',
        'network': asdict(policy.cfg),
        'backend': args.device, 'dtype': args.dtype, 'greedy': True,
        'games_per_block': args.games_per_block, 'blocks': args.blocks,
        'setup_seed': args.setup_seed, 'random_seed': args.random_seed,
        'max_moves': args.max_moves, 'budget_seconds': args.seconds,
        'pairing': 'same setup and random seeds for teams 0 and 1 and both checkpoints',
        'cpu_gpu_random_streams_identical': False,
    }
    save(args.output / 'provenance.json', provenance)
    del checkpoint
    metrics, records = [], []
    started = time.monotonic()

    def interrupted(signum, _frame):
        raise TimeoutError(f'Evaluation stopped by signal {signum}')

    for sig in (signal.SIGALRM, signal.SIGTERM):
        signal.signal(sig, interrupted)
    signal.alarm(args.seconds)
    status, error = 'running', None
    try:
        for block in range(args.blocks):
            block_records = []
            offset = block * args.games_per_block // 2
            stats = evaluate_paired_vs_random(
                policy, num_games=args.games_per_block,
                num_envs=min(64, args.games_per_block // 2),
                use_gpu=args.device == 'cuda', device=args.device,
                seed=args.random_seed + offset, setup_seed=args.setup_seed + offset,
                max_moves=args.max_moves, greedy=True, game_records=block_records,
                autocast_dtype=torch.bfloat16 if args.dtype == 'bfloat16' else None,
            )
            assert len(block_records) == args.games_per_block
            for row in block_records:
                row['block'] = block
            paired = {}
            for row in block_records:
                key = row['setup_seed']
                paired.setdefault(key, []).append(row)
            assert len(paired) == args.games_per_block // 2
            for pair in paired.values():
                assert {row['first_team'] for row in pair} == {0, 1}
                assert len({row['setup_sha256'] for row in pair}) == 1
            save(args.output / f'block_{block:03d}.json', {'metrics': stats, 'records': block_records})
            metrics.append(stats)
            records.extend(block_records)
            print(json.dumps({'block': block, 'elapsed_seconds': time.monotonic() - started,
                              'cumulative': merge_evaluations(*metrics)}), flush=True)
            save(args.output / 'progress.json', {'completed_blocks': len(metrics), 'metrics': merge_evaluations(*metrics)})
        status = 'complete'
    except TimeoutError as exc:
        status, error = 'partial', str(exc)
    except BaseException as exc:
        status, error = 'failed', repr(exc)
        raise
    finally:
        signal.alarm(0)
        team_counts = {str(team): dict(Counter(r['outcome'] for r in records if r['first_team'] == team)) for team in (0, 1)}
        summary = {'status': status, 'error': error, 'completed_blocks': len(metrics),
                   'requested_total_games': args.blocks * args.games_per_block,
                   'recovered_games': len(records), 'team_counts': team_counts,
                   'metrics': merge_evaluations(*metrics) if metrics else {},
                   'elapsed_seconds': time.monotonic() - started}
        save(args.output / 'summary.json', summary)
        (args.output / 'games.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in records))
        (args.output / ('.done' if status == 'complete' else '.partial')).write_text(status + '\n')
        manifest = ''.join(digest(p) + '  ' + p.name + '\n' for p in sorted(args.output.iterdir()) if p.is_file())
        (args.output / 'artifacts.sha256').write_text(manifest)
        print(json.dumps(summary), flush=True)
    return 0 if status == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(run())
