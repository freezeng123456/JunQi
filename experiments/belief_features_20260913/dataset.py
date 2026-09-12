"""Frozen-policy, whole-game-partitioned data with both public feature arms."""
from __future__ import annotations
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch

from junqi_core.board import COMPACT_TO_FLAT
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_rl.belief.inference import rule_only_belief_observations
from junqi_rl.belief.sampling import current_hidden_labels
from junqi_rl.checkpoint_compat import validate_policy_checkpoint, OBSERVATION_SEMANTICS_VERSION
from junqi_rl.gpu_rollout import GpuRollout, _CudaArrayInterfaceView
from junqi_rl.networks.junqi_net import JunqiNet
from experiments.belief_features_20260913.features import PublicRelations, PublicTrajectory

SNAPSHOTS=(0,8,16,32,64,96,128,192,256,384,512,768,1024,1280)
COHORTS={
 'A': [('train_a0',180100,256,'policy'),('train_a1',182100,256,'policy'),
       ('validation',280100,128,'policy'),('test',380100,128,'policy')],
 'B': [('train_b0',184100,256,'policy'),('train_b1',186100,256,'policy'),
       ('ood_attack',480100,128,'attack'),('ood_random',580100,128,'random')],
}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def write_json(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def public_views(world):
    n=world.num_envs
    return tuple(torch.as_tensor(_CudaArrayInterfaceView(getattr(world.state,f'd_{name}_ptr'),
        (n,120),dtype),device='cuda') for name,dtype in
        (('pos_x','|i1'),('pos_y','|i1'),('alive','|b1')))


def canonicalize_labels(labels):
    value=torch.as_tensor(labels,device='cuda').reshape(-1,4,17,17)
    return torch.stack([torch.rot90(value[:,s],k,(-2,-1))
                        for s,k in enumerate((0,1,2,-1))],1).flatten(-2)


@torch.inference_mode()
def snapshot(world,tracker,relations,seed,step,storage):
    n=world.num_envs
    spatial,global_=world.build_all_seat_observations_torch()
    rules=world.rule_beliefs_torch()
    spatial=rule_only_belief_observations(spatial,rules).clone()
    # Freeze every model input before any hidden labels are fetched.
    temporal=tracker.all_observers().clone()
    flat=spatial.flatten(0,1)
    relational=torch.cat([relations(flat[i:i+32]) for i in range(0,len(flat),32)]).reshape(n,4,32,17,17)
    frozen={name:value.to(dtype=torch.float16).cpu() for name,value in
            (('spatial',spatial),('temporal',temporal),('relational',relational))}
    terminated=world.terminated_torch().clone()
    # Simulator truth is used only below, as a supervised target.
    soa=world.state.copy_to_host()
    labels,_=current_hidden_labels(soa,rules.cpu().numpy(),np.arange(n))
    labels=canonicalize_labels(labels)
    labels[terminated]=-1
    mask=labels>=0
    priors=(spatial[:,:,CHANNEL_LAYOUT['belief_left_side']]
            +spatial[:,:,CHANNEL_LAYOUT['belief_right_side']]).flatten(-2).transpose(-1,-2)
    allowed=priors.gather(-1,labels.clamp_min(0)[...,None]).squeeze(-1)>0
    if (mask & ~allowed).any():
        raise RuntimeError('Simulator label contradicts public deductive support')
    keep=mask.any(-1).flatten()
    rows=int(keep.sum())
    for key,value in frozen.items(): storage[key].append(value.flatten(0,1)[keep.cpu()].contiguous())
    storage['labels'].append(labels.flatten(0,1)[keep].to(torch.int8).cpu())
    storage['seat'].append(torch.arange(4,device='cuda').repeat(n)[keep].to(torch.int8).cpu())
    storage['game_id'].append((torch.arange(n,device='cuda')+seed).repeat_interleave(4)[keep].cpu())
    storage['step'].append(torch.full((rows,),step,dtype=torch.int16))
    return dict(step=step,observer_rows=rows,labels=int(mask.sum()),ongoing_games=int((~terminated).sum()))


@torch.inference_mode()
def generate_cohort(path,policy,cohort,*,snapshots=SNAPSHOTS):
    name,seed,n,behavior=cohort
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    world=GpuRollout(n,max_num_moves=max(snapshots)+1)
    world.reset(seed_base=seed)
    tracker=PublicTrajectory(*public_views(world))
    relations=PublicRelations().cuda()
    storage={k:[] for k in ('spatial','temporal','relational','labels','seat','game_id','step')}
    facts=[]; started=time.monotonic(); completed=0
    board=torch.tensor(COMPACT_TO_FLAT,device='cuda')
    for step in range(max(snapshots)+1):
        if step in snapshots:
            fact=snapshot(world,tracker,relations,seed,step,storage)
            facts.append(fact)
            print(json.dumps(dict(event='snapshot',cohort=name,**fact)),flush=True)
        if step==max(snapshots): break
        acting=world.turn_torch().clone()
        acting=torch.where(world.terminated_torch(),0,acting)
        sp,gl=world.build_acting_seat_observation_torch(acting)
        legal=world.legal_mask_canonical_torch_device(acting)
        if behavior=='policy':
            with torch.autocast('cuda',dtype=torch.bfloat16):
                action,_,_=policy.act(sp,gl,legal)
        else:
            scores=torch.rand(legal.shape,device='cuda')
            if behavior=='attack':
                enemy=(sp[:,CHANNEL_LAYOUT['piece_left_side_enemy']]
                      +sp[:,CHANNEL_LAYOUT['piece_right_side_enemy']]).flatten(1).index_select(1,board)>0
                scores += enemy[:,None].expand(-1,129,-1).reshape(n,-1)*2
            action=scores.masked_fill(~legal,-1e9).argmax(-1).to(torch.int32)
        result=world.step_device_torch(action.to(torch.int32),acting)
        # No mid-cohort reset: game IDs and public histories stay independent.
        world.update_beliefs_device(result,acting)
        tracker.update(*public_views(world))
        completed=step+1
    data={k:torch.cat(v) for k,v in storage.items()}
    data['meta']=dict(name=name,seed=seed,num_games=n,behavior=behavior,snapshots=facts,
                      completed_steps=completed,seconds=time.monotonic()-started,
                      rows=len(data['labels']),valid_labels=int((data['labels']>=0).sum()),
                      game_ids=sorted(data['game_id'].unique().tolist()))
    if not torch.isfinite(data['spatial']).all() or not torch.isfinite(data['temporal']).all() or not torch.isfinite(data['relational']).all():
        raise RuntimeError('Nonfinite dataset inputs')
    path=Path(path); tmp=Path(str(path)+'.tmp')
    with gzip.open(tmp,'wb',compresslevel=1) as f: torch.save(data,f)
    tmp.replace(path)
    summary={**data['meta'],'file':path.name,'sha256':sha256(path),'bytes':path.stat().st_size}
    write_json(path.with_suffix('.json'),summary)
    print(json.dumps(dict(event='cohort_complete',**summary)),flush=True)
    del data,storage,world,tracker,relations
    gc.collect(); torch.cuda.empty_cache()
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--role',choices=['A','B'],required=True)
    p.add_argument('--policy',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    status=root/f'generation_{args.role}.json'
    write_json(status,dict(status='running',role=args.role))
    torch.set_num_threads(1)
    state=torch.load(args.policy,map_location='cpu',weights_only=False)
    if state.get('checkpoint_meta',{}).get('observation_semantics_version')!=OBSERVATION_SEMANTICS_VERSION:
        raise ValueError('Frozen policy must already use current observation semantics')
    policy=JunqiNet(state['cfg'].net).cuda()
    validate_policy_checkpoint(policy,state,source=args.policy)
    policy.load_state_dict(state['policy']); policy.eval(); policy.requires_grad_(False)
    meta=dict(role=args.role,policy_sha256=sha256(args.policy),code_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        observation_semantics=OBSERVATION_SEMANTICS_VERSION)
    del state
    cohorts=COHORTS[args.role] if not args.smoke else [(f'smoke_{args.role}',990100,4,'policy')]
    results=[]
    try:
        for cohort in cohorts:
            path=root/(cohort[0]+'.pt.gz')
            if path.exists(): raise FileExistsError(path)
            results.append(generate_cohort(path,policy,cohort,snapshots=(0,4,8,16) if args.smoke else SNAPSHOTS))
        write_json(status,dict(status='completed',**meta,expected_cohorts=len(cohorts),cohorts=results))
        (root/f'generation_{args.role}.done').touch()
    except BaseException as e:
        write_json(status,dict(status='failed',**meta,error=repr(e),cohorts=results))
        raise

if __name__=='__main__': main()
