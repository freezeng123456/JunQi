"""Public-subgroup scoring and simple train-only tabular reference models."""
from __future__ import annotations
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import torch
from junqi_core.observation import CHANNEL_LAYOUT


def load(path):
    metadata=json.loads(path.with_suffix('.json').read_text())
    with path.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==metadata['sha256']
    with gzip.open(path,'rb') as f: data=torch.load(f,map_location='cpu',weights_only=False)
    return data


def slot_and_moves(spatial):
    slots=spatial[:,CHANNEL_LAYOUT['piece_slot']].flatten(-2).argmax(1)
    move_slice=CHANNEL_LAYOUT['move_bucket']
    moves=spatial[:,move_slice.start+4:move_slice.stop].flatten(-2).argmax(1)
    return slots.long(),moves.long()


def support(spatial):
    return (spatial[:,CHANNEL_LAYOUT['belief_left_side']]+spatial[:,CHANNEL_LAYOUT['belief_right_side']]).flatten(-2).transpose(-1,-2)>0


def groups(data):
    labels=data['labels']; live=labels>=0
    _,move_bin=slot_and_moves(data['spatial'])
    steps=data['step'].long()[:,None]
    return {'all':live,'moved':live&(move_bin>0),'never_moved':live&(move_bin==0),
        'opening_zero':live&(steps==0),'opening':live&(steps<=32),
        'middle':live&(steps>=33)&(steps<=256),'late':live&(steps>=257),
        'moved_after_128':live&(move_bin>0)&(steps>=128)}


def segment_sum(index,values,length):
    # scatter_add has a deterministic CUDA path in the pinned runtime;
    # floating-weight bincount does not.
    return values.new_zeros(length).scatter_add_(0,index,values)


@torch.no_grad()
def score_predictions(probs,data,*,include_games=True):
    labels=data['labels'].long(); target=labels.clamp_min(0)
    ptrue=probs.gather(-1,target[...,None]).squeeze(-1)
    nll=-ptrue.clamp_min(1e-12).log()
    brier=probs.square().sum(-1)-2*ptrue+1
    confidence,predicted=probs.max(-1)
    correct=predicted==target
    if include_games: game_ids,inverse=torch.unique(data['game_id'],sorted=True,return_inverse=True)
    output={}
    for name,mask in groups(data).items():
        count=int(mask.sum())
        if not count: continue
        games=[]
        if include_games:
            row_count=mask.sum(-1).double()
            grouped=torch.stack([segment_sum(inverse,value,len(game_ids)) for value in
                (row_count,(nll*mask).sum(-1).double(),(brier*mask).sum(-1).double(),(correct&mask).sum(-1).double())],1).cpu()
            grouped_ids=game_ids.cpu().tolist()
            games=[{'game_id':int(game),'labels':int(v[0]),'nll':float(v[1]/v[0]),
                'brier':float(v[2]/v[0]),'accuracy':float(v[3]/v[0])} for game,v in zip(grouped_ids,grouped,strict=True) if v[0]>0]
        bins=(confidence[mask]*10).long().clamp_max(9)
        confidence_sum=segment_sum(bins,confidence[mask].double(),10)
        correct_sum=segment_sum(bins,correct[mask].double(),10)
        types=target[mask]
        type_counts=segment_sum(types,torch.ones_like(types,dtype=torch.float64),12)
        type_correct=segment_sum(types,correct[mask].double(),12)
        type_nll=segment_sum(types,nll[mask].double(),12)
        high=confidence>=.9
        output[name]={'labels':count,'nll':float(nll[mask].double().mean()),
            'brier':float(brier[mask].double().mean()),'accuracy':float(correct[mask].double().mean()),
            'ece':float((confidence_sum-correct_sum).abs().sum()/count),
            'wrong_confident_fraction':float((mask&high&~correct).sum()/count),
            'confident_fraction':float((mask&high).sum()/count),'games':games,
            'type_totals':torch.stack((type_counts,type_nll,type_correct),1).cpu().tolist()}
    return output


def fit_references(root):
    tables={'slot_initial':torch.ones(25,12,dtype=torch.float64),
        'slot_all':torch.ones(25,12,dtype=torch.float64),
        'slot_move':torch.ones(25,4,12,dtype=torch.float64)}
    sources={}
    for name in ('train_a0','train_a1','train_b0','train_b1'):
        path=root/(name+'.pt.gz'); data=load(path)
        labels=data['labels'].long(); valid=labels>=0
        slot,moves=slot_and_moves(data['spatial'])
        assert (data['spatial'][:,CHANNEL_LAYOUT['piece_slot']].sum(1).flatten(-2)[valid]==1).all()
        initial=valid&(data['step'].long()==0)[:,None]
        for key,mask in [('slot_all',valid),('slot_initial',initial)]:
            ids=slot[mask]*12+labels[mask]
            tables[key]+=torch.bincount(ids,minlength=25*12).reshape(25,12)
        ids=(slot[valid]*4+moves[valid])*12+labels[valid]
        tables['slot_move']+=torch.bincount(ids,minlength=25*4*12).reshape(25,4,12)
        sources[name]=json.loads(path.with_suffix('.json').read_text())['sha256']
        print(json.dumps({'event':'reference_fitted_shard','shard':name}),flush=True)
        del data; gc.collect()
    return {k:(v/v.sum(-1,keepdim=True)).float() for k,v in tables.items()},sources


def tabular_probabilities(data,table):
    slot,moves=slot_and_moves(data['spatial'])
    probs=table[slot,moves] if table.ndim==3 else table[slot]
    probs=probs*support(data['spatial'])
    return probs/probs.sum(-1,keepdim=True).clamp_min(1e-12)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    torch.set_num_threads(1)
    tables,sources=fit_references(args.data)
    output={'fit_shards':sources,'add_one_smoothing':1.,'posthoc_interpretive_reference':True,
        'inputs':'Only the existing public starting-slot and exact public movement bucket channels, followed by deductive support masking.',
        'tables':{k:v.tolist() for k,v in tables.items()},'evaluations':{}}
    for split in ('validation','test','ood_attack','ood_random'):
        data=load(args.data/(split+'.pt.gz'))
        output['evaluations'][split]={}
        for name,table in tables.items():
            scores=score_predictions(tabular_probabilities(data,table),data)
            output['evaluations'][split][name]=scores
            print(json.dumps({'event':'reference_scored','split':split,'model':name,
                'nll':scores['all']['nll'],'accuracy':scores['all']['accuracy']}),flush=True)
        del data; gc.collect()
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
