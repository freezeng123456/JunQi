#!/usr/bin/env python3
"""Audit completed reference acceptance artifacts on an allocated compute node."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from junqi_rl.training.distributed_checkpoint import digest


def equal(a,b,path='state'):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,atol=0,rtol=0,msg=path)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert set(a)==set(b),(path,set(a)^set(b))
        for k in a:
            if k!='lineage_id':equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+f'[{i}]')
    else:assert a==b,path


def load(path):return torch.load(path,map_location='cpu',weights_only=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--world',type=int,required=True)
    args=p.parse_args();root=Path(args.root)
    a=load(root/'continuous/ckpt_000003.pt');b=load(root/'restarted/ckpt_000003.pt')
    for key in ('policy','optimizer','ema','arrangement','num_rollout','num_train_step'):
        equal(a[key],b[key],key)
    assert a['distributed_consensus']['identical'] and b['distributed_consensus']['identical']
    assert a['distributed_consensus']['world_size']==args.world
    assert a['num_train_step']>0 and a['arrangement']['trainer']['total_batches']>0
    hashes={}
    per_rank=[]
    for state,folder in ((a,'continuous'),(b,'restarted')):
        rt=state['runtime_state'];assert rt['world_size']==args.world
        for item in rt['shards']:
            path=root/folder/item['file'];assert digest(path)==item['sha256'];hashes[str(path)]=digest(path)
        records=[json.loads(x) for x in (root/folder/'training_games.jsonl').read_text().splitlines()]
        assert len(records)==rt['games']['completed_games']>0
        assert len({(r['lineage_id'],r['game_id']) for r in records})==len(records)
        assert all('first_policy' in r for r in records)
    for left,right in zip(a['runtime_state']['shards'],b['runtime_state']['shards']):
        x=load(root/'continuous'/left['file']);y=load(root/'restarted'/right['file'])
        equal(x,y,'rank'+str(left['rank']))
        assert x['rng']['cuda_scope']=='rank' and len(x['rng']['torch_cuda'])==1
        per_rank.append(x['games']['completed_games'])
    result=dict(passed=True,world_size=args.world,completed_games=sum(per_rank),
        per_rank_completed_games=per_rank,move_updates=a['num_train_step'],
        setup_updates=a['arrangement']['trainer']['total_batches'],exact_restart=True,hashes=hashes)
    (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='hashes'}))

if __name__=='__main__':main()
