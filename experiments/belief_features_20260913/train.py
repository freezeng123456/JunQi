"""Paired, no-EMA supervised belief experiments on immutable shared data."""
from __future__ import annotations
import argparse
import contextlib
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_rl.networks.belief_net import BeliefNetConfig
from experiments.belief_features_20260913.dataset import sha256, write_json
from experiments.belief_features_20260913.features import FeatureBeliefNet

TRAIN_FILES=('train_a0','train_a1','train_b0','train_b1')
EVAL_FILES=('validation','test','ood_attack','ood_random')
SEEDS=(601,602,603)


def read_shard(path):
    with gzip.open(path,'rb') as f: return torch.load(f,map_location='cpu',weights_only=False)


def rule_prior(obs):
    q=(obs[:,CHANNEL_LAYOUT['belief_left_side']]+obs[:,CHANNEL_LAYOUT['belief_right_side']])
    q=q.flatten(-2).transpose(-1,-2).float()
    return q/q.sum(-1,keepdim=True).clamp_min(1e-12)


def masked_log_probs(logits,obs):
    return logits.float().masked_fill(rule_prior(obs)<=0,-1e9).log_softmax(-1)


def supervised_loss(logits,obs,labels):
    # The 2-D classification kernel supports deterministic CUDA reduction;
    # the spatial nll_loss2d kernel does not in this pinned PyTorch runtime.
    log_probs=masked_log_probs(logits,obs)
    return F.nll_loss(log_probs.reshape(-1,log_probs.shape[-1]),labels.long().reshape(-1),ignore_index=-1)


def load_data(root,direction,*,smoke=False):
    names=('smoke_A',) if smoke else (*TRAIN_FILES,*EVAL_FILES)
    loaded={}; manifest={}; all_games={}
    for name in names:
        path=root/(name+'.pt.gz')
        metadata=json.loads(path.with_suffix('.json').read_text())
        digest=sha256(path)
        if digest!=metadata['sha256']: raise ValueError(f'Bad dataset hash: {path}')
        raw=read_shard(path)
        keys=('spatial',direction,'labels','seat','game_id','step')
        loaded[name]={k:raw[k] for k in keys}
        all_games[name]=set(raw['game_id'].tolist())
        manifest[name]=dict(sha256=digest,rows=len(raw['labels']),meta=raw['meta'])
        print(json.dumps(dict(event='data_loaded',name=name,rows=len(raw['labels']),sha256=digest)),flush=True)
        del raw
    if not smoke:
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if all_games[a] & all_games[b]: raise ValueError(f'Game leakage: {a} / {b}')
        loaded['train']={k:torch.cat([loaded[n][k] for n in TRAIN_FILES]) for k in loaded[TRAIN_FILES[0]]}
        for n in TRAIN_FILES: del loaded[n]
    else:
        loaded['train']=loaded.pop('smoke_A')
        loaded['validation']=loaded['train']
    for name,data in loaded.items():
        loaded[name]={k:v.cuda() for k,v in data.items()}
    return loaded,manifest


@torch.no_grad()
def evaluate(model,data,direction,arm,*,predictor='neural',class_prior=None,feature_control=None):
    model.eval()
    device=data['spatial'].device
    totals=torch.zeros(7,device=device,dtype=torch.float64)
    type_sum=torch.zeros((12,3),device=device,dtype=torch.float64)
    bins=torch.zeros((10,3),device=device,dtype=torch.float64)
    row_stats=[]
    count=len(data['labels'])
    shuffle=torch.roll(torch.arange(count,device=device),max(1,count//3))
    for start in range(0,count,128):
        sl=slice(start,min(start+128,count))
        obs=data['spatial'][sl].float(); labels=data['labels'][sl].long()
        active=labels>=0; targets=labels.clamp_min(0)
        q=rule_prior(obs)
        if predictor=='rule': probs=q
        elif predictor=='frequency':
            probs=(q>0)*class_prior
            probs=probs/probs.sum(-1,keepdim=True).clamp_min(1e-12)
        else:
            extra=data[direction][sl].float()
            if arm=='baseline' or feature_control=='zero': extra=torch.zeros_like(extra)
            elif feature_control=='shuffle': extra=data[direction][shuffle[sl]].float()
            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
            with ctx:
                logits=model(obs,data['seat'][sl].long(),extra)
            probs=masked_log_probs(logits,obs).exp()
        ptrue=probs.gather(-1,targets[...,None]).squeeze(-1)
        nll=-ptrue.clamp_min(1e-12).log()
        brier=probs.square().sum(-1)-2*ptrue+1
        confidence,pred=probs.max(-1)
        correct=pred==targets
        high=confidence>=.9
        entropy=-(probs*probs.clamp_min(1e-12).log()).sum(-1)
        scalars=(active.sum(),(nll*active).sum(),(brier*active).sum(),
                 (correct&active).sum(),(high&~correct&active).sum(),(high&active).sum(),
                 (entropy*active).sum())
        totals+=torch.stack(scalars).double()
        selected=targets[active]
        type_sum[:,0].scatter_add_(0,selected,torch.ones_like(selected,dtype=torch.float64))
        type_sum[:,1].scatter_add_(0,selected,nll[active].double())
        type_sum[:,2].scatter_add_(0,selected,correct[active].double())
        idx=(confidence[active]*10).long().clamp_max(9)
        for col,val in enumerate((torch.ones_like(idx,dtype=torch.float64),
                                  confidence[active].double(),correct[active].double())):
            bins[:,col].scatter_add_(0,idx,val)
        row_stats.append(torch.stack((data['game_id'][sl].double(),data['step'][sl].double(),
            active.sum(-1).double(),(nll*active).sum(-1).double(),(brier*active).sum(-1).double(),
            (correct&active).sum(-1).double()),-1).cpu())
    n,nll,brier,correct,wrong_high,high,entropy=totals.cpu().tolist()
    if n==0: raise ValueError('No evaluation targets')
    bin_np=bins.cpu().numpy()
    ece=float((np.abs(bin_np[:,1]-bin_np[:,2])).sum()/n)
    rows=torch.cat(row_stats).numpy()
    grouped=[]
    for game in np.unique(rows[:,0]):
        sums=rows[rows[:,0]==game,2:].sum(0)
        grouped.append(dict(game_id=int(game),labels=int(sums[0]),nll=float(sums[1]/sums[0]),
                            brier=float(sums[2]/sums[0]),accuracy=float(sums[3]/sums[0])))
    stages={}
    for stage,lo,hi in [('opening',0,32),('middle',33,256),('late',257,100000)]:
        values=rows[(rows[:,1]>=lo)&(rows[:,1]<=hi),2:].sum(0)
        if values[0]: stages[stage]=dict(labels=int(values[0]),nll=float(values[1]/values[0]),
            brier=float(values[2]/values[0]),accuracy=float(values[3]/values[0]))
    return dict(labels=int(n),nll=nll/n,brier=brier/n,accuracy=correct/n,
        wrong_confident_fraction=wrong_high/n,confident_fraction=high/n,
        confident_accuracy=1-wrong_high/high if high else None,entropy=entropy/n,ece=ece,
        games=grouped,stages=stages,type_totals=type_sum.cpu().tolist())


def compact(metrics):
    return {k:v for k,v in metrics.items() if k not in ('games','stages','type_totals')}


def parameter_hash(model):
    digest=hashlib.sha256()
    for k,v in model.state_dict().items():
        digest.update(k.encode()); digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_cell(args,data,manifest,seed,arm,*,cfg):
    cell=Path(args.output)/f'{arm}_s{seed}'
    cell.mkdir(exist_ok=False)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model=FeatureBeliefNet(cfg).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    initial_hash=parameter_hash(model)
    sample_rng=torch.Generator(device='cuda').manual_seed(seed+700000)
    train=data['train']; rows=len(train['labels'])
    config=dict(seed=seed,arm=arm,direction=args.direction,net=asdict(cfg),lr=args.lr,
        batch_size=args.batch_size,max_steps=args.steps,max_seconds=args.cell_seconds,
        eval_every=args.eval_every,gradient_clip=.5,parameter_ema=False,
        loss='deductive-support-masked cross entropy on ambiguous enemy identities',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data={k:v['sha256'] for k,v in manifest.items()},initial_state_sha256=initial_hash,
        parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda)
    write_json(cell/'config.json',config)
    started=time.monotonic(); train_seconds=0.; best=float('inf'); best_step=0
    val=evaluate(model,data['validation'],args.direction,arm)
    initial_val=compact(val)
    best=val['nll']
    curves=[]; step=0; loss_sum=0.; grad_max=0.; last_train_eval=None
    # No early stopping or averaging; validation selects a stored checkpoint.
    # A one-shot signal timer bounds this cell independently of status polling.
    timed_out=False
    def deadline(*_):
        nonlocal timed_out
        timed_out=True
    previous=signal.signal(signal.SIGALRM,deadline)
    signal.setitimer(signal.ITIMER_REAL,args.cell_seconds)
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    try:
        while step<args.steps and not timed_out:
            model.train()
            index=torch.randint(rows,(args.batch_size,),generator=sample_rng,device='cuda')
            obs=train['spatial'][index].float(); label=train['labels'][index].long()
            extra=train[args.direction][index].float()
            if arm=='baseline': extra=torch.zeros_like(extra)
            seats=train['seat'][index].long()
            tick=time.monotonic()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=model(obs,seats,extra)
            loss=supervised_loss(logits,obs,label)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite supervised loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
            if not torch.isfinite(grad): raise RuntimeError('Nonfinite supervised gradient')
            opt.step()
            # CUDA synchronization is also the timing boundary for optimizer cost.
            value=float(loss.detach()); grad_max=max(grad_max,float(grad))
            train_seconds+=time.monotonic()-tick; loss_sum+=value; step+=1
            if step%args.eval_every==0 or step==args.steps or timed_out:
                val=evaluate(model,data['validation'],args.direction,arm)
                if val['nll']<best:
                    best=val['nll']; best_step=step
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    torch.save(dict(net=best_state,config=config,step=step,validation=compact(val)),cell/'best.pt.tmp')
                    (cell/'best.pt.tmp').replace(cell/'best.pt')
                record=dict(step=step,train_loss=loss_sum/(step-(curves[-1]['step'] if curves else 0)),
                    validation=compact(val),elapsed_seconds=time.monotonic()-started,
                    optimizer_seconds=train_seconds,best_step=best_step,gradient_max=grad_max)
                curves.append(record); loss_sum=0.
                with (cell/'metrics.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
                print(json.dumps(dict(event='validation',cell=cell.name,**record)),flush=True)
        # Preserve the actual final trained parameters separately from the selected ones.
        final_hash=parameter_hash(model)
        torch.save(dict(net=model.state_dict(),optimizer=opt.state_dict(),step=step,config=config),cell/'last.pt')
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
    # The final holdouts are first evaluated after all optimization for this cell.
    model.load_state_dict(best_state)
    if not (cell/'best.pt').exists():
        torch.save(dict(net=best_state,config=config,step=best_step,validation=initial_val),cell/'best.pt')
    outputs={name:evaluate(model,data[name],args.direction,arm) for name in data if name!='train'}
    # Training diagnostic uses a fixed prefix, never the held-out game IDs.
    subset={k:v[:min(len(v),4096)] for k,v in train.items()}
    training=evaluate(model,subset,args.direction,arm)
    controls={}
    if arm!='baseline' and not args.smoke:
        for condition in ('zero','shuffle'):
            controls[condition]=evaluate(model,data['test'],args.direction,arm,feature_control=condition)
    summary=dict(status='completed' if step==args.steps else 'partial',seed=seed,arm=arm,direction=args.direction,steps=step,
        best_step=best_step,stop_reason='cell_budget' if timed_out else 'step_limit',
        train_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,
        initial_validation=initial_val,training_subset=compact(training),
        final_parameter_sha256=final_hash,parameters_changed=final_hash!=initial_hash,
        best_adapter_norm=float(model.adapter.weight.detach().norm()),
        evaluations=outputs,feature_controls=controls)
    write_json(cell/'summary.json',summary)
    (cell/('done' if step==args.steps else 'partial')).touch()
    print(json.dumps(dict(event='cell_complete',cell=cell.name,steps=step,best_step=best_step,
        evaluations={k:compact(v) for k,v in outputs.items()})),flush=True)
    del model,opt,best_state
    gc.collect(); torch.cuda.empty_cache()
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--direction',choices=['temporal','relational'],required=True)
    p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--cell-seconds',type=int,default=2100)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--lr',type=float,default=5e-5)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=p.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    data,manifest=load_data(Path(args.data),args.direction,smoke=args.smoke)
    cfg=BeliefNetConfig(n_encoder_layer=4,n_head=8,embed_dim=256,cnn_channels=64,
                        cnn_layers=2,ff_factor=2,dropout=.1)
    write_json(root/'data_manifest.json',manifest)
    # Fixed, label-frequency-only reference fitted exclusively to training labels.
    labels=data['train']['labels'].long()
    frequency=torch.bincount(labels[labels>=0],minlength=12).float()+1
    frequency=frequency/frequency.sum()
    reference_model=FeatureBeliefNet(cfg).cuda()
    references={}
    for name in data:
        if name=='train': continue
        references[name]={pred:evaluate(reference_model,data[name],args.direction,'baseline',
            predictor=pred,class_prior=frequency) for pred in ('rule','frequency')}
    write_json(root/'reference_metrics.json',references)
    del reference_model
    gc.collect(); torch.cuda.empty_cache()
    order=[]
    for i,seed in enumerate((601,) if args.smoke else args.seeds):
        arms=['baseline',args.direction]
        if (i+(args.direction=='relational'))%2: arms.reverse()
        order.extend((seed,arm) for arm in arms)
    write_json(root/'protocol.json',dict(status='running',order=order,network=asdict(cfg),
        args=vars(args),estimated_max_training_seconds=len(order)*args.cell_seconds))
    results=[]
    try:
        for seed,arm in order:
            results.append(fit_cell(args,data,manifest,seed,arm,cfg=cfg))
        complete=all(r['status']=='completed' for r in results)
        write_json(root/'summary.json',dict(status='completed' if complete else 'partial',expected_cells=len(order),
            completed_cells=len(results),cells=[dict(seed=r['seed'],arm=r['arm'],steps=r['steps'],
                best_step=r['best_step'],test=compact(r['evaluations'].get('test',r['evaluations']['validation'])))
                for r in results]))
        (root/('done' if complete else 'partial')).touch()
    except BaseException as e:
        write_json(root/'failed.json',dict(status='failed',error=repr(e),completed_cells=len(results)))
        raise

if __name__=='__main__': main()
