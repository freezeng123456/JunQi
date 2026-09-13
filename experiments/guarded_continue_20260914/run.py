"""One bounded lease job with serial stages and durable completion evidence."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/tmp/junqi_guarded_20260914')
RESULT = ROOT / 'results'
CODE = Path(__file__).resolve().parents[2]
PY = sys.executable
CHILD = None


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stop(_sig, _frame):
    raise TimeoutError('Whole experiment reached its one-shot wall budget')


def stage(name, args, seconds, cpu=False):
    global CHILD
    save(RESULT / 'status.json', {'status': 'running', 'stage': name})
    started = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    code = None
    try:
        with (RESULT / (name + '.log')).open('x') as log:
            CHILD = subprocess.Popen([PY, *args], cwd=CODE, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            code = CHILD.wait(timeout=seconds)
        if code:
            raise RuntimeError(f'{name} exited {code}')
    finally:
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait(timeout=30)
        CHILD = None
        save(RESULT / (name + '.receipt.json'), {'stage': name, 'exit_code': code,
            'elapsed_seconds': time.monotonic() - started, 'budget_seconds': seconds})


def evaluate(name, checkpoint, blocks, setup, random, cpu=False):
    stage(name, ['-m', 'experiments.guarded_continue_20260914.evaluate',
        '--checkpoint', str(checkpoint), '--output', str(RESULT / name),
        '--device', 'cpu' if cpu else 'cuda', '--dtype', 'float32' if cpu else 'bfloat16',
        '--games-per-block', '16' if cpu else '128', '--blocks', str(blocks),
        '--setup-seed', str(setup), '--random-seed', str(random),
        '--seconds', '2300' if cpu else '550'], 2400 if cpu else 600, cpu)


def main():
    RESULT.mkdir(exist_ok=False)
    # Calculate the lease allowance once; SIGALRM is a one-shot guard, not a clock poll.
    started = datetime.now().astimezone()
    cutoff = datetime.fromisoformat('2026-09-16T09:00:00+08:00')
    budget = min(4 * 3600, int((cutoff - started).total_seconds()))
    if budget < 3 * 3600:
        raise RuntimeError('Insufficient time for the frozen training/evaluation/recovery budget')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(budget)
    base = ROOT / 'guarded_s501_raw_policy_v4_2048wins.pt'
    cfg = CODE / 'configs/guarded_continue_20260914.yaml'
    save(RESULT / 'provenance.json', {'started': started.isoformat(), 'budget_seconds': budget,
        'lease_recovery': '2026-09-16T11:00:00+08:00', 'latest_compute_cutoff': cutoff.isoformat(),
        'python': PY, 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
        'baseline_sha256': digest(base), 'config_sha256': digest(cfg), 'ema': False,
        'learned_belief': False, 'canonical_root': str(ROOT), 'host': '28.38.182.102:36000'})
    status, error = 'complete', None
    try:
        stage('native_smoke', ['-m', 'pytest', '-q', 'tests/test_gpu_player_information.py',
            'tests/test_gpu_combat_memory_parity.py', 'tests/test_gpu_evaluation_fixed_games.py'], 600)
        stage('initialize', ['-m', 'experiments.guarded_continue_20260914.initialize', '--source', str(base),
            '--config', str(cfg), '--output', str(RESULT / 'initializer.pt')], 300)
        evaluate('baseline_monitor', base, 4, 3091400, 4091400)
        stage('train', ['scripts/train.py', '--config', str(cfg), '--resume', str(RESULT / 'initializer.pt')], 3600)
        stage('checkpoint_audit', ['-m', 'experiments.guarded_continue_20260914.audit', '--root', str(ROOT)], 300, True)
        terminal = RESULT / 'candidate_raw_policy_v4.pt'
        evaluate('baseline_gpu', base, 16, 3191400, 4191400)
        evaluate('candidate_gpu', terminal, 16, 3191400, 4191400)
        stage('h2h', ['-m', 'experiments.guarded_continue_20260914.audit', '--root', str(ROOT), '--h2h'], 900)
        evaluate('baseline_cpu', base, 32, 3291400, 4291400, True)
        evaluate('candidate_cpu', terminal, 32, 3291400, 4291400, True)
        audit = json.loads((RESULT / 'checkpoint_audit.json').read_text())
        train_log = (RESULT / 'train.log').read_text()
        assert 'Evaluation failed:' not in train_log
        expected = list(range(16, audit['rollouts'] + 1, 16))
        for rollout in expected:
            for kind, count in [('random', 512), ('h2h', 128)]:
                rows = (RESULT / 'train' / f'eval_{rollout:06d}_{kind}.jsonl').read_text().splitlines()
                assert len(rows) == count
        save(RESULT / 'training_coverage.json', {'rollouts': audit['rollouts'], 'expected_max_rollouts': 128,
            'training_status': 'complete' if audit['rollouts'] == 128 else 'stopped_early',
            'evaluation_rollouts': expected, 'all_monitor_records_present': True})
    except BaseException as exc:
        status, error = 'failed_or_partial', repr(exc)
        (RESULT / 'failure.txt').write_text(traceback.format_exc())
    finally:
        signal.alarm(0)
        save(RESULT / 'status.json', {'status': status, 'error': error})
        (RESULT / ('.done' if status == 'complete' else '.partial')).write_text(status + '\n')
        paths = sorted(p for p in RESULT.rglob('*') if p.is_file() and not p.is_symlink())
        (RESULT / 'artifacts.sha256').write_text(''.join(digest(p) + '  ' + str(p.relative_to(RESULT)) + '\n' for p in paths))
        with tarfile.open(ROOT / 'results.tar.gz', 'x:gz') as tar:
            tar.add(RESULT, arcname='results')
        save(ROOT / 'archive.json', {'status': status, 'sha256': digest(ROOT / 'results.tar.gz'),
            'files': len(paths), 'error': error})
    return 0 if status == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
