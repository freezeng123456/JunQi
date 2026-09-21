"""Explicit identity-preserving policy growth, optimizer moments mapped by name."""
from pathlib import Path
import copy,json,hashlib,argparse
from dataclasses import asdict
import torch
from junqi_rl.networks.junqi_net import JunqiNet,JunqiNetConfig
from junqi_rl.training.checkpoint import load_evaluation_policy
from junqi_rl.checkpoint_compat import current_checkpoint_metadata

def migrate(source,output,kind):
    torch.set_num_threads(1);torch.manual_seed(921)
    state=torch.load(source,map_location='cpu',weights_only=False)
    old=load_evaluation_policy(source,device='cpu');cfg=asdict(old.cfg)
    if kind in ('M','R'):cfg['tactical_features']=True;cfg['relational_features']=kind=='R'
    else:raise ValueError(kind)
    new=JunqiNet(JunqiNetConfig(**cfg)).eval();weights=new.state_dict()
    for k,v in state['policy'].items():assert weights[k].shape==v.shape;weights[k]=v.clone()
    new.load_state_dict(weights)
    # The legacy trainer uses one AdamW group in named_parameters order.
    old_names=[n for n,_ in old.named_parameters()];new_names=[n for n,_ in new.named_parameters()]
    opt=copy.deepcopy(state['optimizer']);assert len(opt['param_groups'])==1
    old_ids=opt['param_groups'][0]['params'];assert len(old_ids)==len(old_names)
    mapping=dict(zip(old_names,old_ids));new_state={}
    for i,n in enumerate(new_names):
        if n in mapping and mapping[n] in opt['state']:new_state[i]=copy.deepcopy(opt['state'][mapping[n]])
    opt['state']=new_state;opt['param_groups'][0]['params']=list(range(len(new_names)))
    state=copy.deepcopy(state);state['policy']=new.state_dict();state['optimizer']=opt
    state['train_cfg']['net']=cfg;state['train_cfg']['ppo']['net']=cfg
    state['cfg'].net=new.cfg;state['checkpoint_meta']=current_checkpoint_metadata(new)
    state['migration']={'kind':kind,'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),'shared_policy_tensors':len(old_names),'inherited_optimizer_states':len(new_state),'added_parameters':len(new_names)-len(old_names)}
    Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save(state,output)
    restored=load_evaluation_policy(output,device='cpu')
    for k,v in old.state_dict().items():assert torch.equal(restored.state_dict()[k],v),k
    optimizer=torch.optim.AdamW(restored.parameters());optimizer.load_state_dict(opt)
    assert len(optimizer.state)==len(old_names)
    Path(str(output)+'.json').write_text(json.dumps(state['migration'],indent=2)+'\n')
    return old,restored,state

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source');ap.add_argument('output');ap.add_argument('kind',choices=['M','R']);a=ap.parse_args()
    _,_,state=migrate(a.source,a.output,a.kind);print(json.dumps(state['migration']))
