"""Strict audit and inference export for additive architecture experiments."""
import argparse,hashlib,json,shutil
from pathlib import Path
from dataclasses import asdict
import torch
from junqi_rl.training.checkpoint import load_evaluation_policy

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('baseline');ap.add_argument('output');a=ap.parse_args()
 torch.set_num_threads(1);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 p=Path(a.checkpoint);copy=out/p.name;before=sha(p);shutil.copy2(p,copy);assert before==sha(copy)==sha(p)
 state=torch.load(copy,map_location='cpu',weights_only=False);base=torch.load(a.baseline,map_location='cpu',weights_only=False);count=[0]
 def check(x):
  if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();count[0]+=1
  elif isinstance(x,dict):
   for v in x.values():check(v)
  elif isinstance(x,(tuple,list)):
   for v in x:check(v)
 check(state);model=load_evaluation_policy(copy,device='cpu')
 cfg=state['train_cfg'];assert not cfg['random_opponent'] and not cfg['reward_shaping'];assert not cfg['arr']['enabled'] and not cfg['belief']['enabled']
 changed=sum(not torch.equal(v,base['policy'][k]) for k,v in state['policy'].items() if k in base['policy']);assert changed>0
 export={'policy':state['policy'],'checkpoint_meta':state['checkpoint_meta'],'train_cfg':{'net':asdict(model.cfg)},'inference_export':{'source_sha256':before,'promoted':False}}
 torch.save(export,out/'candidate_raw_policy.pt');restored=load_evaluation_policy(out/'candidate_raw_policy.pt',device='cpu')
 assert all(torch.equal(v,restored.state_dict()[k]) for k,v in state['policy'].items())
 report={'status':'finite_strict_reload_verified','sha256':before,'rollouts':state['num_rollout'],'optimizer_updates':state['num_train_step'],'finite_tensors':count[0],'policy_tensors':len(state['policy']),'parameters':sum(p.numel() for p in model.parameters()),'changed_inherited_tensors':changed,'config':cfg}
 (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='config'}))
if __name__=='__main__':main()
