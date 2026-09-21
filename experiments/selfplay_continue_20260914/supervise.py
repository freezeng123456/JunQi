"""Lease-bounded self-play with two-hour durable private checkpoint backups.

Token is read once from stdin and retained only in the detached controller's RAM.
No credential is inherited by training subprocesses or written to disk.
"""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time

ROOT = Path('/tmp/junqi_selfplay_20260914')
RESULT = ROOT / 'results'
CODE = Path(__file__).resolve().parents[2]
PY = '/jizhicfs/yuyechen/miniconda3/envs/cl/bin/python'
REPO = 'a1390892757/junqi-guarded-continue-20260914'
CHILD = None
DEADLINE = threading.Event()
ENV = dict(os.environ, PYTHONPATH=f'{CODE}:{ROOT}/runtime', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', HF_HUB_DISABLE_XET='1',
           HF_HUB_DISABLE_PROGRESS_BARS='1', HF_HUB_DISABLE_TELEMETRY='1')


def save(p, value):
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def stop_training(hard=False):
    DEADLINE.set()
    child = CHILD
    if child is not None and child.poll() is None:
        try: os.killpg(child.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except ProcessLookupError: pass


def stage(name, args, seconds=600, cpu=False):
    global CHILD
    if DEADLINE.is_set():
        raise RuntimeError('lease compute cutoff reached')
    env = dict(ENV)
    if cpu: env['CUDA_VISIBLE_DEVICES'] = ''
    save(RESULT / 'status.json', dict(status='running', stage=name))
    start = time.monotonic()
    with (RESULT / f'{name}.log').open('x') as out:
        CHILD = subprocess.Popen([PY, *args], cwd=CODE, env=env, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = CHILD.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try: CHILD.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait()
            raise
        finally:
            save(RESULT / f'{name}.receipt.json', dict(exit_code=CHILD.poll(), elapsed_seconds=time.monotonic()-start))
            CHILD = None
    if code: raise RuntimeError(f'{name} exited {code}')


def snapshot(api, checkpoint, serial):
    folder = RESULT / 'snapshots' / serial
    env = dict(ENV, CUDA_VISIBLE_DEVICES='')
    with (RESULT / f'snapshot_{serial}.log').open('x') as log:
        subprocess.run([PY, '-m', 'experiments.selfplay_continue_20260914.snapshot',
                        '--checkpoint', str(checkpoint), '--baseline', str(ROOT/'baseline.pt'),
                        '--output', str(folder)], cwd=CODE, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=600)
    for name in ['provenance.json', 'status.json', 'initializer.json']:
        if (RESULT/name).exists(): shutil.copy2(RESULT/name, folder/name)
    for name in ['train.log', 'smoke_train.log']:
        if (RESULT/name).exists(): shutil.copy2(RESULT/name, folder/name)
    for src in (RESULT/'train').glob('eval_*.jsonl'):
        shutil.copy2(src, folder/src.name)
    shutil.copy2(ROOT/'source.bundle', folder/'source.bundle')
    files = {p.name:digest(p) for p in folder.iterdir() if p.is_file()}
    (folder/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(files.items())))
    prefix = f'selfplay_20260914/snapshots/{serial}'
    commit = api.upload_folder(repo_id=REPO, folder_path=str(folder), path_in_repo=prefix,
                               commit_message=f'Preserve unpromoted self-play {serial}')
    remote = api.model_info(REPO, revision=commit.oid, files_metadata=True)
    assert remote.private
    lookup = {x.rfilename:x for x in remote.siblings}
    for name, expected in files.items():
        item = lookup[f'{prefix}/{name}']
        assert item.size == (folder/name).stat().st_size
        if item.lfs is not None:
            assert item.lfs.sha256 == expected
        else:
            content = (folder/name).read_bytes()
            assert item.blob_id == hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
    record = dict(status='private_checkpoint_hash_verified', commit=commit.oid,
                  repo=REPO, prefix=prefix, checkpoint=str(checkpoint),
                  checkpoint_sha256=files[checkpoint.name], audit=json.loads((folder/'audit.json').read_text()))
    save(RESULT/f'backup_{serial}.json', record)
    save(RESULT/'backup.latest.json', record)
    print(f'Backup verified: {serial} {commit.oid}', flush=True)


def main(token):
    global CHILD
    RESULT.mkdir(exist_ok=False)
    started = datetime.now().astimezone()
    soft = datetime.fromisoformat('2026-09-16T07:00:00+08:00')
    hard = datetime.fromisoformat('2026-09-16T07:10:00+08:00')
    allowance = (soft-started).total_seconds()
    assert allowance > 3600, 'insufficient lease time'
    # One calculation and two one-shot timers; no recurring clock queries.
    timers = [threading.Timer(allowance, stop_training),
              threading.Timer((hard-started).total_seconds(), stop_training, kwargs={'hard':True})]
    for timer in timers:
        timer.daemon = True
        timer.start()
    save(RESULT/'provenance.json', dict(started=started.isoformat(), soft_cutoff=soft.isoformat(),
         hard_cutoff=hard.isoformat(), reclaim='2026-09-16T11:00:00+08:00',
         source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
         baseline_sha256=digest(ROOT/'baseline.pt'), source_bundle_sha256=digest(ROOT/'source.bundle'),
         root=str(ROOT), backup_seconds=7200, python=PY, gpu='one NVIDIA H20',
         max_rollouts=20000, ema=False, belief=False, arrangement=False))
    cfg = str(CODE/'configs/selfplay_continue_20260914.yaml')
    api = None
    status = 'stage_complete_needs_evaluation'
    error = None
    try:
        for key in ['HF_HUB_DISABLE_XET','HF_HUB_DISABLE_PROGRESS_BARS','HF_HUB_DISABLE_TELEMETRY']:
            os.environ[key] = '1'
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        assert api.whoami()['name'] == 'a1390892757'
        assert api.model_info(REPO).private
        stage('selfplay_tests', ['-m','pytest','-q','tests/test_collect_autocast_bf16.py',
              'tests/test_random_opponent_invariants.py::test_self_play_keeps_all_advantages'])
        stage('initialize', ['-m','experiments.guarded_continue_20260914.initialize',
              '--source',str(ROOT/'baseline.pt'),'--config',cfg,'--output',str(RESULT/'initializer.pt')], cpu=True)
        stage('smoke_train', ['scripts/train.py','--config',cfg,'--resume',str(RESULT/'initializer.pt'),
              '--total_rollouts','4','--save_every','4','--eval_every','4','--log_every','1',
              '--save_dir',str(RESULT/'smoke')], seconds=1200)
        assert 'Evaluation failed:' not in (RESULT/'smoke_train.log').read_text()
        snapshot(api, (RESULT/'smoke/ckpt_latest.pt').resolve(strict=True), 'smoke_r000004')
        if DEADLINE.is_set(): raise RuntimeError('lease cutoff before long stage')
        with (RESULT/'train.log').open('x') as out:
            CHILD = subprocess.Popen([PY,'scripts/train.py','--config',cfg,'--resume',
                       str(RESULT/'smoke/ckpt_latest.pt')],cwd=CODE,env=ENV,stdout=out,
                       stderr=subprocess.STDOUT,start_new_session=True)
            save(RESULT/'status.json', dict(status='training',pid=CHILD.pid,
                 supervisor_pid=os.getpid(), resumed_rollout=4, max_rollouts=20000))
            serial = 0
            while True:
                try:
                    code = CHILD.wait(timeout=7200)
                    break
                except subprocess.TimeoutExpired:
                    serial += 1
                    ckpt = (RESULT/'train/ckpt_latest.pt').resolve(strict=True)
                    try:
                        snapshot(api, ckpt, f'backup_{serial:03d}_{ckpt.stem}')
                        text = (RESULT/'train.log').read_text()
                        # Guard implementation/numerical failures; do not silently train through them.
                        recent = text.splitlines()[-200:]
                        if 'Evaluation failed:' in text or any(re.search(r'nan_skip=[1-9]|grad_skip=[1-9]|loss_[pv]=[+-]?(nan|inf)',x) for x in recent):
                            stop_training()
                            raise RuntimeError('training/evaluation guard requires diagnosis')
                    except Exception as exc:
                        save(RESULT/'backup.error.json',dict(type=type(exc).__name__, message=str(exc).replace(token,'[redacted]')))
                        if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
                            stop_training()
                        # A failed backup may be transient; preserve the active experiment.
                        if DEADLINE.is_set(): raise
            save(RESULT/'train.receipt.json', dict(exit_code=code, deadline_reached=DEADLINE.is_set()))
            if code: raise RuntimeError(f'train exited {code}')
            CHILD = None
        if 'EARLY STOP:' in (RESULT/'train.log').read_text():
            status = 'guard_stopped_needs_diagnosis'
        elif DEADLINE.is_set():
            status = 'lease_training_stopped_needs_final_evaluation'
    except BaseException as exc:
        status, error = 'partial_needs_attention', f'{type(exc).__name__}: {str(exc).replace(token,"[redacted]")}'
        print(error, flush=True)
    finally:
        if CHILD is not None and CHILD.poll() is None:
            stop_training()
            try: CHILD.wait(timeout=180)
            except subprocess.TimeoutExpired:
                stop_training(hard=True)
                CHILD.wait()
        CHILD = None
        for timer in timers: timer.cancel()
        save(RESULT/'status.json',dict(status=status,error=error))
        # Final raw parameters are archived/uploaded even after an early stop.
        latest = RESULT/'train/ckpt_latest.pt'
        if not latest.exists(): latest = RESULT/'smoke/ckpt_latest.pt'
        if api is not None and latest.exists():
            try: snapshot(api, latest.resolve(strict=True), 'terminal')
            except Exception as exc: save(RESULT/'final_backup.error.json',dict(type=type(exc).__name__, message=str(exc).replace(token,'[redacted]')))
        (RESULT/'.stage_finished').write_text(status+'\n')
        paths = sorted(p for p in RESULT.rglob('*') if p.is_file() and not p.is_symlink())
        (RESULT/'artifacts.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(RESULT)}\n' for p in paths))
        with tarfile.open(ROOT/'results.tar.gz','x:gz') as tar: tar.add(RESULT,arcname='results')
        archive_sha = digest(ROOT/'results.tar.gz')
        save(ROOT/'archive.json',dict(status=status,sha256=archive_sha,files=len(paths)))
        if api is not None:
            try:
                commit=api.upload_file(repo_id=REPO,path_or_fileobj=str(ROOT/'results.tar.gz'),
                    path_in_repo='selfplay_20260914/results.tar.gz',commit_message='Archive complete self-play stage evidence')
                info=api.model_info(REPO,revision=commit.oid,files_metadata=True)
                assert next(x for x in info.siblings if x.rfilename=='selfplay_20260914/results.tar.gz').lfs.sha256==archive_sha
                save(ROOT/'archive_upload.json',dict(commit=commit.oid,sha256=archive_sha,verified=True))
            except Exception as exc: save(ROOT/'archive_upload.error.json',dict(type=type(exc).__name__,message=str(exc).replace(token,'[redacted]')))


if __name__ == '__main__':
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not RESULT.exists(), 'inspect existing state before any restart'
    token = sys.stdin.readline().rstrip('\r\n')
    assert token.startswith('hf_')
    pid = os.fork()
    if pid:
        print(f'Detached controller PID {pid}', flush=True)
        raise SystemExit(0)
    os.setsid()
    with open(os.devnull,'rb') as null, (ROOT/'controller.log').open('xb',buffering=0) as log:
        os.dup2(null.fileno(),0)
        os.dup2(log.fileno(),1)
        os.dup2(log.fileno(),2)
    main(token)
