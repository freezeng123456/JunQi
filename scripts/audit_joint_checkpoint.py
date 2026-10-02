"""Verify joint checkpoint loading and one resumed optimizer update."""
import argparse
import json
from pathlib import Path
import torch
from junqi_rl.belief.joint_checkpoint import load_joint_belief


def main():
    p=argparse.ArgumentParser(); p.add_argument('--policy',required=True)
    p.add_argument('--initial',required=True); p.add_argument('--resumed',required=True)
    p.add_argument('--output',required=True); a=p.parse_args()
    old=torch.load(a.initial,map_location='cpu',weights_only=False)
    new=torch.load(a.resumed,map_location='cpu',weights_only=False)
    net=load_joint_belief(a.resumed,policy_path=a.policy)
    assert new['updates']>old['updates']
    assert new['optimizer']['state']
    assert all(torch.isfinite(v).all() for v in net.state_dict().values())
    changed=sum(not torch.equal(v,old['net'][k]) for k,v in new['net'].items())
    old_step=max(float(v['step']) for v in old['optimizer']['state'].values())
    new_step=max(float(v['step']) for v in new['optimizer']['state'].values())
    assert changed>0 and new_step>old_step
    report={'status':'PASS','initial_updates':old['updates'],'resumed_updates':new['updates'],
            'changed_network_tensors':changed,'initial_optimizer_step':old_step,
            'resumed_optimizer_step':new_step,'parameters':net.num_parameters(),
            'input_spec':new['input_spec'],'parent_policy_sha256':new['parent_policy_sha256']}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()
