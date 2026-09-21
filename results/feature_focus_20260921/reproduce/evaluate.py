import sys,os,time,json,hashlib,signal
from pathlib import Path
from datetime import datetime
from common import *
sys.path.insert(0,str(ROOT/'code'))
def evaluate(first,second,out,games=2048,setup=9191900,seed=10191900,knockout=None):
 import torch,numpy as np
 from junqi_rl.training.checkpoint import load_evaluation_policy
 from junqi_rl.analysis.random_eval import evaluate_paired_head_to_head,evaluate_paired_vs_random
 torch.set_num_threads(1);torch.manual_seed(919)
 a=load_evaluation_policy(first,device='cuda');b=load_evaluation_policy(second,device='cuda') if second else None
 if knockout:
  assert a.relational_head is not None
  slices={'threat':slice(0,4),'location':slice(4,7),'flags':slice(7,12),'information':slice(12,13)}
  with torch.no_grad():
   if knockout in slices:a.relational_head.weights[:,slices[knockout]]=0
   if knockout=='information':a.relational_head.special_weights[2]=0
   if knockout=='resources':a.relational_head.special_weights[:2]=0
  assert knockout in set(slices)|{'resources'}
 rows=[]
 # Immutable inputs before opening scores.
 out.mkdir(parents=True,exist_ok=False)
 save(out/'inputs.json',{'feature_knockout':knockout,'first':str(first),'first_sha256':sha(first),'second':str(second),'second_sha256':sha(second) if second else None,'games':games,'setup_seed':setup,'game_seed':seed})
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

if __name__=='__main__':
 evaluate(Path(sys.argv[1]),Path(sys.argv[2]) if sys.argv[2]!='random' else None,Path(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),sys.argv[7] if len(sys.argv)>7 else None)
