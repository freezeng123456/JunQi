import sys,os,time,json,hashlib,signal
from pathlib import Path
from datetime import datetime
from common import *
sys.path.insert(0,str(ROOT/'code'))
def evaluate(first,second,out,games=2048,setup=9191900,seed=10191900):
 import torch,numpy as np
 from junqi_rl.training.checkpoint import load_evaluation_policy
 from junqi_rl.analysis.random_eval import evaluate_paired_head_to_head,evaluate_paired_vs_random
 torch.set_num_threads(1);torch.manual_seed(919)
 a=load_evaluation_policy(first,device='cuda');b=load_evaluation_policy(second,device='cuda') if second else None
 rows=[]
 # Immutable inputs before opening scores.
 out.mkdir(parents=True,exist_ok=False)
 save(out/'inputs.json',{'first':str(first),'first_sha256':sha(first),'second':str(second),'second_sha256':sha(second) if second else None,'games':games,'setup_seed':setup,'game_seed':seed})
 for offset in range(0,games,128):
  batch=[]
  kw=dict(num_games=min(128,games-offset),num_envs=64,device='cuda',seed=seed+offset//2,setup_seed=setup+offset//2,max_moves=4000,greedy=True,game_records=batch,autocast_dtype=torch.bfloat16)
  if b is not None:evaluate_paired_head_to_head(a,b,**kw)
  else:evaluate_paired_vs_random(a,use_gpu=True,**kw)
  for r in batch:r['block']=offset//128
  rows+=batch
  (out/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 assert len(rows)==games
 from collections import Counter,defaultdict
 counts=Counter(r['outcome'] for r in rows);assert set(counts)<= {'win','loss','draw'}
 pairs=defaultdict(list)
 for r in rows:pairs[(r['setup_seed'],r['random_seed'])].append(r)
 assert all(len(v)==2 and {r['first_team'] for r in v}=={0,1} and len({r['setup_sha256'] for r in v})==1 for v in pairs.values())
 vals=np.array([sum({'win':1,'loss':0,'draw':.5}[r['outcome']] for r in v)/2 for v in pairs.values()]);rng=np.random.default_rng(919)
 boot=np.array([rng.choice(vals,size=len(vals),replace=True).mean() for _ in range(10000)])
 result={'status':'complete','games':games,'counts':dict(counts),'win_fraction':counts['win']/games,'score':float(vals.mean()),'pair_bootstrap_ci95':np.quantile(boot,[.025,.975]).tolist(),'pairs':len(vals)}
 save(out/'summary.json',result);print(out.name,json.dumps(result),flush=True)
 del a,b;torch.cuda.empty_cache();return result

def main(token):
 from huggingface_hub import HfApi,snapshot_download
 api=HfApi(token=token)
 out=ROOT/'comparison';out.mkdir(exist_ok=False)
 limit=datetime.fromisoformat('2026-09-19T14:20:00+08:00').timestamp()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('evaluation cutoff')))
 signal.alarm(max(1,int(limit-time.time())))
 summary={}
 try:
  # GPU1 belongs to C until its compute process has exited, regardless of upload state.
  while True:
   s=ROOT/'C/status.json'
   if s.exists() and json.loads(s.read_text()).get('status') in ('finished','failed'):
    import subprocess
    uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
    if uuid not in busy:break
   if time.time()>datetime.fromisoformat('2026-09-19T09:00:00+08:00').timestamp():raise RuntimeError('C GPU release not confirmed')
   time.sleep(60)
  summary['start_vs_original']=evaluate(ROOT/'start.pt',ROOT/'original.pt',out/'start_vs_original')
  # Wait for immutable matched A/B publication; B may still train its extension.
  while True:
   info=api.model_info(REPO);files=api.list_repo_files(REPO,revision=info.sha)
   ready=all(f'{PREFIX}/{a}/matched/audit.json' in files for a in ('A','B'))
   if ready:break
   if time.time()>datetime.fromisoformat('2026-09-19T12:50:00+08:00').timestamp():raise RuntimeError('matched artifacts unavailable')
   time.sleep(120)
  revision=info.sha
  snapshot_download(REPO,revision=revision,allow_patterns=[f'{PREFIX}/A/complete/train/ckpt_*.pt',f'{PREFIX}/A/matched/*',f'{PREFIX}/B/matched/*'],local_dir=str(ROOT/'download'),token=token)
  ar=ROOT/'download'/PREFIX/'A/matched';br=ROOT/'B/train'
  aa=json.loads((ar/'audit.json').read_text());bb=json.loads((ROOT/'download'/PREFIX/'B/matched/audit.json').read_text())
  target=min(aa['rollouts'],bb['rollouts'],24096)
  if target==24096:
   a=ar/'ckpt_024096.pt';b=br/'ckpt_024096.pt'
  else:
   ad=ROOT/'download'/PREFIX/'A/complete/train';an={p.name for p in ad.glob('ckpt_[0-9]*.pt')};bn={p.name for p in br.glob('ckpt_[0-9]*.pt')};name=sorted(an&bn)[-1];a=ad/name;b=br/name;target=int(a.stem.split('_')[1])
  save(out/'selection.json',{'rule':'fixed equal budget; no heldout selection','total_rollouts':target,'new_rollouts':target-20000,'A_sha256':sha(a),'B_sha256':sha(b),'hf_input_revision':revision})
  summary['A_vs_B']=evaluate(a,b,out/'A_vs_B')
  for label,p in [('A',a),('B',b)]:
   summary[label+'_vs_start']=evaluate(p,ROOT/'start.pt',out/(label+'_vs_start'))
   summary[label+'_vs_random']=evaluate(p,None,out/(label+'_vs_random'),512,11191900,12191900)
  c=ROOT/'C/train'/f'ckpt_{target:06d}.pt'
  if c.exists():
   summary['C_vs_start']=evaluate(c,ROOT/'start.pt',out/'C_vs_start')
   summary['C_vs_B']=evaluate(c,b,out/'C_vs_B')
  save(out/'summary.json',{'status':'primary_complete','matches':summary})
  publish(api,out,f'{PREFIX}/comparison')
  # Separate extra-budget B from the primary equal-budget experiment.
  while time.time()<datetime.fromisoformat('2026-09-19T12:40:00+08:00').timestamp():
   s=ROOT/'B/status.json'
   if s.exists() and json.loads(s.read_text()).get('status') in ('finished','failed'):break
   time.sleep(60)
  ext=sorted((ROOT/'B/extension').glob('ckpt_[0-9]*.pt'))
  if ext and time.time()<limit-1800:summary['B_extended_vs_A']=evaluate(ext[-1],a,out/'B_extended_vs_A',2048,13191900,14191900)
  save(out/'summary.json',{'status':'complete','matches':summary})
 except BaseException as e:save(out/'summary.json',{'status':'partial','error':type(e).__name__+': '+str(e).replace(token,'[redacted]'),'matches':summary})
 finally:
  signal.alarm(0)
  lines=['# JunQi 双方向实验比较','', '结果状态见 summary.json；A 与 B 比较固定相同训练预算，C 为 A 的异种子复现。','', '|对局|局数|胜/负/和|纯胜率|得分率|配对95%区间|','|---|---:|---|---:|---:|---|']
  for name,r in summary.items():
   c=r['counts'];lo,hi=r['pair_bootstrap_ci95'];lines.append(f"|{name}|{r['games']}|{c.get('win',0)}/{c.get('loss',0)}/{c.get('draw',0)}|{r['win_fraction']:.2%}|{r['score']:.2%}|{lo:.2%}–{hi:.2%}|")
  lines+=['','得分率为胜加半个和局。历史训练监控不参与独立样本统计；额外预算 B 单列，不用于宣称方向优势。','未自动替换原推荐模型；代码、检查点和源哈希位于同仓库 two_directions_20260919。']
  (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
  for name in ['PROTOCOL.md','provenance.json']:__import__('shutil').copy2(ROOT/name,out/name)
  receipt=publish(api,out,f'{PREFIX}/comparison');save(ROOT/'comparison_upload.json',receipt)
if __name__=='__main__':main(sys.stdin.readline().strip())
