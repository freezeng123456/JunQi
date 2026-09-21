"""Finite lease-bounded feature ablation; no app automation is required."""
import sys,os,json,time,signal,subprocess,shutil,re
from datetime import datetime
from pathlib import Path
from common import *
TRAIN_END=datetime.fromisoformat('2026-09-21T19:30:00+08:00').timestamp()
EVAL_END=datetime.fromisoformat('2026-09-21T22:30:00+08:00').timestamp()
ENV=dict(os.environ,PYTHONPATH=str(ROOT/'code')+':'+str(ROOT/'runtime'),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DOWNLOAD_TIMEOUT='60',HF_HUB_ETAG_TIMEOUT='60',JUNQI_START_CHECKPOINT=str(ROOT/'start.pt'))
child=None
BAD=re.compile(r'loss_[pv]=[+-]?(?:nan|inf)|(?:nan_skip|grad_skip)=[1-9]|Evaluation failed:|Traceback \(most recent',re.I)

def gpu_processes():
 return [x for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if x.strip()]

def stop():
 global child
 if child is not None and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=180)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()

def main(creds,arm):
 global child
 import yaml
 from huggingface_hub import HfApi,hf_hub_download
 assert arm in ('N','M','R')
 api=HfApi(token=creds['hf']);assert api.model_info(REPO).private
 out=ROOT/arm;out.mkdir(exist_ok=False)
 def status(**data):save(ROOT/'status.json',dict(time=time.time(),arm=arm,**data))
 def upload(folder,label,prefix=None):
  try:
   receipt=publish(api,folder,prefix or PREFIX+'/'+arm+'/'+label);save(out/(label+'_upload.json'),receipt);return True
  except Exception as e:
   save(out/(label+'_upload_failure.json'),{'type':type(e).__name__,'time':time.time()});return False
 def run(args,log,deadline,phase,backup=None):
  global child
  if time.time()>=deadline:return 'deadline'
  with log.open('w') as f:
   child=subprocess.Popen([PYTHON,*map(str,args)],cwd=ROOT/'code',env=ENV,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   status(phase=phase,pid=child.pid,deadline=deadline);next_backup=time.time()+7200
   while child.poll() is None:
    if time.time()>=deadline:stop();return 'deadline'
    try:child.wait(timeout=min(30,max(.1,deadline-time.time())))
    except subprocess.TimeoutExpired:pass
    if len(gpu_processes())>1:stop();raise RuntimeError('another GPU computation appeared; our child stopped to avoid overlap')
    if BAD.search(log.read_text()[-16000:]):stop();raise RuntimeError(phase+' runtime/numerical failure')
    if backup and time.time()>=next_backup:
     backup();next_backup=time.time()+7200
  assert child.returncode==0,f'{phase} exited {child.returncode}'
  assert not BAD.search(log.read_text()),phase+' runtime/numerical failure'
  return 'complete'
 def latest():
  paths=sorted((out/'train').glob('ckpt_[0-9]*.pt'));return paths[-1] if paths else out/'initial.pt'
 def snapshot(p,label):
  folder=out/'snapshots'/label
  if not (folder/'audit.json').exists():
   if folder.exists():shutil.rmtree(folder)
   subprocess.run([PYTHON,'-m','experiments.feature_focus_20260921.snapshot',str(p),str(out/'initial.pt'),str(folder)],cwd=ROOT/'code',env=dict(ENV,CUDA_VISIBLE_DEVICES=''),check=True,timeout=600)
  for pattern in ('*.yaml','*.log','migration.json','initial.pt.json','smoke_verified.json'):
   for q in out.glob(pattern):shutil.copy2(q,folder/q.name)
  for q in (out/'train').glob('*.jsonl'):shutil.copy2(q,folder/q.name)
  for n in ('PROTOCOL.md','provenance.json'):shutil.copy2(ROOT/n,folder/n)
  upload(folder,label)
 def config(block,resume,target):
  cfg=json.loads((ROOT/'base_config.json').read_text())
  cfg['net'].update(tactical_features=arm in ('M','R'),relational_features=arm=='R',depth=4);cfg['ppo']['net']=cfg['net'].copy()
  cfg.update(save_dir=str(out/'train'),resume=str(resume),total_rollouts=target,seed=9210+block,save_every=128,log_every=16,eval_every=1024,eval_num_games=256,eval_baseline_games=256,eval_setup_seed=31921000,eval_game_seed=32921000,eval_baseline_ckpt=str(ROOT/'start.pt'),early_stop_win_rate=0.)
  cfg['env']['seed']=cfg['seed']
  for k in ('lr_coef','lr_ceil','lr_floor'):cfg['ppo'][k]=1e-5
  for k in ('temperature_coef','temperature_ceil','temperature_floor'):cfg['ppo'][k]=.006
  cfg['ppo']['lr_decay']=0.;cfg['ppo']['temperature_decay']=0.
  p=out/f'block_{block}.yaml';p.write_text(yaml.safe_dump(cfg));return p
 failure=None
 try:
  provenance=json.loads((ROOT/'provenance.json').read_text());assert sha(ROOT/'start.pt')==provenance['start_sha256']
  assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'code',text=True).strip()==provenance['source_commit']
  while gpu_processes():
   status(phase='waiting_gpu_available',train_cutoff=TRAIN_END)
   if time.time()>=TRAIN_END:raise TimeoutError('GPU remained occupied until cutoff')
   time.sleep(60)
  if arm=='N':
   shutil.copy2(ROOT/'start.pt',out/'initial.pt');save(out/'initial.pt.json',{'kind':'N','source_sha256':sha(ROOT/'start.pt'),'unchanged':True})
  else:
   with (out/'migration.log').open('w') as f:
    subprocess.run([PYTHON,'-m','experiments.feature_focus_20260921.migrate',str(ROOT/'start.pt'),str(out/'initial.pt'),arm],cwd=ROOT/'code',env=dict(ENV,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
  shutil.copy2(out/'initial.pt.json',out/'migration.json')
  run(['-m','experiments.explore_20260921.gpu_identity',ROOT/'start.pt',out/'initial.pt'],out/'gpu_identity.log',min(TRAIN_END,time.time()+600),'gpu_identity')
  cfg=config(0,out/'initial.pt',32292)
  assert run(['scripts/train.py','--config',cfg,'--save_dir',out/'smoke','--save_every',4,'--eval_every',4],out/'smoke.log',min(TRAIN_END,time.time()+1800),'smoke')=='complete'
  snapshot(out/'smoke/ckpt_032292.pt','smoke')
  smoke=json.loads((out/'snapshots/smoke/audit.json').read_text())
  if arm!='N':assert any(smoke['added_module_tensor_changes'].values()),'new module did not update'
  if arm=='R':assert any(v for k,v in smoke['added_module_tensor_changes'].items() if k.startswith('relational_head.'))
  save(out/'smoke_verified.json',{'strict_reload':True,'smoke_discarded':True,'new_tensors_changed':smoke['added_module_tensor_changes']})
  for block in range(1,4):
   if time.time()>=TRAIN_END:break
   cfg=config(block,latest(),32288+2048*block)
   result=run(['scripts/train.py','--config',cfg],out/f'block_{block}.log',TRAIN_END,f'train_{block}',lambda:snapshot(latest(),'backup_'+latest().stem))
   snapshot(latest(),'stage_'+str(block))
   if block==2 and latest().name=='ckpt_036384.pt':snapshot(latest(),'primary')
   if result=='deadline':break
 except BaseException as e:
  failure={'type':type(e).__name__,'message':str(e)}
  for v in creds.values():failure['message']=failure['message'].replace(v,'[redacted]')
  save(out/'training_failure.json',failure)
 finally:stop()
 try:snapshot(latest(),'terminal')
 except Exception as e:save(out/'terminal_failure.json',{'type':type(e).__name__})
 comp=out/'comparison';comp.mkdir(exist_ok=True);results={}
 def download(other,label,number):
  while time.time()<EVAL_END-900:
   try:return Path(hf_hub_download(REPO,PREFIX+'/'+other+'/'+label+'/ckpt_'+f'{number:06d}'+'.pt',token=creds['hf'],local_dir=ROOT/'download'))
   except Exception:status(phase='waiting_'+other+'_'+label);time.sleep(60)
  raise TimeoutError(other+' '+label+' unavailable before evaluation cutoff')
 def evaluate(label,a,b,games,setup,seed,knockout=None):
  if time.time()>=EVAL_END:return
  args=[ROOT/'evaluate.py',a,b,comp/label,games,setup,seed]+([knockout] if knockout else [])
  outcome=run(args,comp/(label+'.log'),EVAL_END,'eval_'+label)
  p=comp/label/'summary.json'
  if p.exists():results[label]=json.loads(p.read_text())
  save(comp/'summary.json',{'status':'running','matches':results});upload(comp,'comparison');return outcome
 primary=out/'train/ckpt_036384.pt'
 try:
  assert primary.exists(),'primary target not reached'
  save(comp/'selection.json',{'primary_rule':'fixed 4096 new rollouts, independent of scores','rollouts':36384,'candidate_sha256':sha(primary),'start_sha256':sha(ROOT/'start.pt'),'training_failure':failure})
  evaluate(arm+'_vs_start',primary,ROOT/'start.pt',2048,51921000,52921000)
  if arm!='N':
   control=download('N','primary',36384)
   evaluate(arm+'_vs_N',primary,control,2048,51921000,52921000)
  if arm=='R':evaluate('R_vs_M',primary,download('M','primary',36384),2048,51921000,52921000)
  evaluate(arm+'_vs_random',primary,'random',512,53921000,54921000)
  extra=out/'train/ckpt_038432.pt'
  if extra.exists() and time.time()<EVAL_END-2700:
   control_extra=ROOT/'start.pt' if arm=='N' else download('N','stage_3',38432)
   evaluate(arm+'_extra_vs_'+('start' if arm=='N' else 'N_extra'),extra,control_extra,2048,55921000,56921000)
  if arm=='R' and time.time()<EVAL_END-2700:
   evaluate('R_diag_full_vs_N',primary,control,512,57921000,58921000)
   for group in ('threat','location','flags','information','resources'):
    if time.time()>=EVAL_END-300:break
    evaluate('R_without_'+group+'_vs_N',primary,control,512,57921000,58921000,group)
 except BaseException as e:
  message=str(e)
  for value in creds.values():message=message.replace(value,'[redacted]')
  save(comp/'failure.json',{'type':type(e).__name__,'message':message[:300]})
 finally:stop()
 expected={arm+'_vs_start',arm+'_vs_random'}|({arm+'_vs_N'} if arm!='N' else set())|({'R_vs_M'} if arm=='R' else set())
 save(comp/'summary.json',{'status':'complete' if expected<=set(results) else 'partial','matches':results,'training_failure':failure,'arm':arm})
 lines=['# JunQi '+arm+' 特征消融结果','','N原结构；M修正材料/不确定性；R=M+公开关系特征。固定LR1e-5、4层。主要预算4096新增轮，extra6144单列。单训练种子、多重探索比较；屏蔽诊断不等于重训消融。','','|对局|胜/负/和|纯胜率|得分率|配对95%区间|','|---|---|---|---|---|']
 for label,r in results.items():
  c=r['counts'];lo,hi=r['pair_bootstrap_ci95'];lines.append(f"|{label}|{c.get('win',0)}/{c.get('loss',0)}/{c.get('draw',0)}|{r['win_fraction']:.4%}|{r['score']:.4%}|{lo:.4%}–{hi:.4%}|")
 (comp/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
 for n in ('PROTOCOL.md','provenance.json','FEATURE_PLAN.zh.md'):shutil.copy2(ROOT/n,comp/n)
 status(phase='publishing',matches=len(results));upload(comp,'comparison')
 # Retry any previously failed immutable checkpoint uploads before final history.
 for folder in (out/'snapshots').glob('*'):
  if folder.is_dir() and not (out/(folder.name+'_upload.json')).exists():upload(folder,folder.name)
 try:
  from publish_github import main as github
  github(creds,arm)
 except Exception as e:save(out/'github_upload_failure.json',{'type':type(e).__name__})
 try:save(ROOT/(arm+'_complete_upload.json'),publish(api,out,PREFIX+'/'+arm+'/complete'))
 except Exception as e:save(ROOT/(arm+'_complete_upload_failure.json'),{'type':type(e).__name__})
 status(phase='remote_complete' if expected<=set(results) else 'remote_partial',matches=len(results),local_recovery_required=True)

if __name__=='__main__':
 credentials=json.loads(sys.stdin.readline())
 try:main(credentials,sys.argv[1])
 except BaseException as e:
  stop();save(ROOT/'status.json',{'phase':'failed','error_type':type(e).__name__,'time':time.time()});raise SystemExit(1)
