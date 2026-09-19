import os,sys,json,subprocess,signal,time,shutil,threading
from datetime import datetime
from pathlib import Path
from common import *
def run(arm,token):
 from huggingface_hub import HfApi
 api=HfApi(token=token);assert api.model_info(REPO).private
 out=ROOT/arm;out.mkdir(exist_ok=False)
 deadline=datetime.fromisoformat('2026-09-19T08:30:00+08:00' if arm in ('A','C') else '2026-09-19T12:30:00+08:00').timestamp()
 env=dict(os.environ,PYTHONPATH=str(ROOT/'code'),CUDA_VISIBLE_DEVICES='1' if arm=='C' else '0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 published=set();child=None
 def backup(path,label):
  folder=out/'snapshots'/label
  if not folder.exists():
   subprocess.run([PYTHON,'-m','experiments.selfplay_continue_20260914.snapshot','--checkpoint',str(path),'--baseline',str(ROOT/'start.pt'),'--output',str(folder)],cwd=ROOT/'code',env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True,timeout=600)
   for p in [out/'config.yaml',ROOT/'PROTOCOL.md',ROOT/'provenance.json',out/'train.log',out/'smoke.log']:
    if p.exists():shutil.copy2(p,folder/p.name)
   for p in (out/'train').glob('eval*.jsonl'):shutil.copy2(p,folder/p.name)
  receipt=publish(api,folder,f'{PREFIX}/{arm}/snapshots/{label}');save(out/f'uploaded_{label}.json',receipt);published.add(path.name)
 def launch(args,log):
  return subprocess.Popen([PYTHON,*args],cwd=ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 def wait(child,cutoff):
  next_backup=time.time()+7200
  while child.poll() is None:
   remaining=cutoff-time.time()
   if remaining<=0:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=180)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return 'deadline'
   try:child.wait(timeout=min(remaining,60))
   except subprocess.TimeoutExpired:pass
   if time.time()>=next_backup:
    files=sorted((out/'train').glob('ckpt_[0-9]*.pt'))
    if files:
     try:backup(files[-1],files[-1].stem)
     except Exception as e:save(out/'backup_error.json',{'error':type(e).__name__})
    next_backup=time.time()+7200
  return 'complete' if child.returncode==0 else 'failed'
 try:
  import yaml
  cfg=json.loads((ROOT/'base_config.json').read_text())
  cfg.update(save_dir=str(out/'train'),resume=str(ROOT/'start.pt'),total_rollouts=24096,seed=9192,eval_every=512,eval_num_games=256,eval_baseline_games=256,eval_setup_seed=7191900,eval_game_seed=8191900,eval_baseline_ckpt=str(ROOT/'start.pt'),save_every=128,log_every=16)
  cfg['seed']=9193 if arm=='C' else 9192
  cfg['env']['seed']=cfg['seed']
  alpha=.006 if arm=='B' else .002
  for k in ['temperature_coef','temperature_ceil','temperature_floor']:cfg['ppo'][k]=alpha
  (out/'config.yaml').write_text(yaml.safe_dump(cfg))
  save(out/'status.json',{'status':'smoke','arm':arm,'alpha':alpha})
  with (out/'smoke.log').open('w') as log:
   child=launch(['scripts/train.py','--config',str(out/'config.yaml'),'--total_rollouts','20004','--save_every','4','--eval_every','4','--save_dir',str(out/'smoke')],log)
   smoke_status=wait(child,min(deadline,time.time()+1500))
  assert smoke_status=='complete','smoke failed'
  assert 'Evaluation failed:' not in (out/'smoke.log').read_text()
  backup(out/'smoke/ckpt_020004.pt','smoke')
  # Smoke weights are discarded; both directions start from identical optimizer and policy.
  with (out/'train.log').open('w') as log:
   child=launch(['scripts/train.py','--config',str(out/'config.yaml')],log)
   save(out/'status.json',{'status':'training','pid':child.pid,'arm':arm,'target':24096,'cutoff':deadline})
   status=wait(child,deadline)
  text=(out/'train.log').read_text();assert 'Evaluation failed:' not in text,'evaluation failure'
  files=sorted((out/'train').glob('ckpt_[0-9]*.pt'));assert files
  backup(files[-1],'matched_terminal')
  save(out/'matched.json',{'status':status,'checkpoint':files[-1].name,'rollout':int(files[-1].stem.split('_')[1])})
  publish(api,out/'snapshots/matched_terminal',f'{PREFIX}/{arm}/matched')
  if arm=='B' and status=='complete' and time.time()<deadline-1800:
   with (out/'extension.log').open('w') as log:
    child=launch(['scripts/train.py','--config',str(out/'config.yaml'),'--resume',str(files[-1]),'--total_rollouts','26144','--save_dir',str(out/'extension')],log)
    status=wait(child,deadline)
   ext=sorted((out/'extension').glob('ckpt_[0-9]*.pt'))
   if ext:backup(ext[-1],'extended_terminal')
  save(out/'status.json',{'status':'finished','outcome':status,'arm':arm})
 except BaseException as e:
  save(out/'status.json',{'status':'failed','error':type(e).__name__+': '+str(e).replace(token,'[redacted]')})
 finally:
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=180)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  # Save all numbered checkpoints, logs and configs before lease expiry.
  try:
   receipt=publish(api,out,f'{PREFIX}/{arm}/complete');save(ROOT/f'{arm}_complete_upload.json',receipt)
  except Exception as e:save(ROOT/f'{arm}_upload_failure.json',{'error':type(e).__name__})
if __name__=='__main__':
 token=sys.stdin.readline().strip();assert token.startswith('hf_')
 run(sys.argv[1],token)
