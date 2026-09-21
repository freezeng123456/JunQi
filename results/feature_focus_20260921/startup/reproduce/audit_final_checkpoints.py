from pathlib import Path
import sys,json,hashlib
base=Path(__file__).resolve().parent;sys.path.insert(0,str(base.parent.parent/'work/JunQi-feature-focus-20260921'))
import torch
from junqi_rl.training.checkpoint import load_evaluation_policy
torch.set_num_threads(1);root=base/'recovered/feature_focus_20260921';proof={}
for arm in ('N','M','R'):
 for label in ('primary','terminal'):
  paths=list((root/arm/label).glob('ckpt_[0-9]*.pt'));assert len(paths)==1
  p=paths[0];state=torch.load(p,map_location='cpu',weights_only=False);model=load_evaluation_policy(p,device='cpu');count=[0]
  def check(x):
   if isinstance(x,torch.Tensor):assert torch.isfinite(x).all();count[0]+=1
   elif isinstance(x,dict):
    for v in x.values():check(v)
   elif isinstance(x,(list,tuple)):
    for v in x:check(v)
  check(state);assert all(torch.equal(v,model.state_dict()[k]) for k,v in state['policy'].items())
  proof[arm+'_'+label]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rollouts':state['num_rollout'],'optimizer_updates':state['num_train_step'],'finite_tensors':count[0],'parameters':sum(p.numel() for p in model.parameters()),'strict_reload':True}
  if label=='primary':
   selection=json.loads((root/arm/'comparison/selection.json').read_text());assert proof[arm+'_'+label]['sha256']==selection['candidate_sha256'];assert state['num_rollout']==selection['rollouts']==36384
(base/'final_checkpoints_verified.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
