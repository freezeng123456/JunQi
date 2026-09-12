"""Audit immutable partitions and the public feature distributions, one shard at a time."""
from __future__ import annotations
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import torch
from junqi_core.observation import CHANNEL_LAYOUT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    result={}; games={}
    for path in sorted(args.data.glob('*.pt.gz')):
        metadata=json.loads(path.with_suffix('.json').read_text())
        with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        assert digest==metadata['sha256']
        with gzip.open(path,'rb') as f: data=torch.load(f,map_location='cpu',weights_only=False)
        labels=data['labels'].long(); mask=labels>=0
        obs=data['spatial']
        q=(obs[:,CHANNEL_LAYOUT['belief_left_side']]+obs[:,CHANNEL_LAYOUT['belief_right_side']]).flatten(-2).transpose(-1,-2).float()
        q=q/q.sum(-1,keepdim=True).clamp_min(1e-12)
        nll=-q.gather(-1,labels.clamp_min(0)[...,None]).squeeze(-1).clamp_min(1e-12).log()
        sizes=(q>0).sum(-1)
        stages={}
        steps=data['step'].long()
        for name,lo,hi in [('opening',0,32),('middle',33,256),('late',257,100000)]:
            selected=mask & ((steps>=lo)&(steps<=hi))[:,None]
            if selected.any():
                stages[name]={'labels':int(selected.sum()),'rule_nll':float(nll[selected].mean()),
                    'mean_allowed_types':float(sizes[selected].float().mean())}
        extra={}
        for name in ['temporal','relational']:
            values=data[name].flatten(-2).transpose(-1,-2)[mask].float()
            extra[name]={'mean_abs_by_channel':values.abs().mean(0).tolist(),
                'std_by_channel':values.std(0).tolist(),
                'nonzero_by_channel':(values!=0).float().mean(0).tolist(),
                'all_finite':bool(values.isfinite().all())}
        name=path.name.split('.')[0]
        games[name]=set(data['game_id'].tolist())
        result[name]={'sha256':digest,'rows':len(labels),'games':len(games[name]),
            'labels':int(mask.sum()),'class_counts':torch.bincount(labels[mask],minlength=12).tolist(),
            'rule_nll':float(nll[mask].mean()),'support_size_counts':torch.bincount(sizes[mask],minlength=13).tolist(),
            'stages':stages,'features':extra,'generation':data['meta']}
        print(json.dumps({'shard':name,'rows':len(labels),'labels':int(mask.sum()),'rule_nll':result[name]['rule_nll']}),flush=True)
        del data,labels,mask,obs,q,nll,sizes,values
        gc.collect()
    overlaps={f'{a}/{b}':sorted(games[a]&games[b]) for i,a in enumerate(games) for b in list(games)[i+1:] if games[a]&games[b]}
    assert not overlaps,overlaps
    args.output.write_text(json.dumps({'all_game_partitions_disjoint':True,'shards':result},indent=2)+'\n')


if __name__=='__main__': main()
