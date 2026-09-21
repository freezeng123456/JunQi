"""Score public subgroups and raw final fit, after each host has finished training."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import torch
from junqi_rl.networks.belief_net import BeliefNetConfig
from experiments.belief_features_20260913.features import FeatureBeliefNet
from experiments.belief_features_20260913.train import load_data,masked_log_probs,TRAIN_FILES
from experiments.belief_features_20260913.dataset import write_json,sha256
from experiments.belief_features_20260913.diagnostics import score_predictions


@torch.no_grad()
def predict(model,data,direction,arm):
    model.eval(); predictions=[]
    for begin in range(0,len(data['labels']),128):
        rows=slice(begin,begin+128)
        obs=data['spatial'][rows].float()
        extra=data[direction][rows].float()
        if arm=='baseline': extra=torch.zeros_like(extra)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(obs,data['seat'][rows].long(),extra)
        predictions.append(masked_log_probs(logits,obs).exp())
    return torch.cat(predictions)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--role',choices=['A','B'],required=True)
    parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    root=args.root; args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    expanded=(root/'data_expanded.ready').exists()
    data_root=root/('data_expanded' if expanded else 'data_main')
    train_files=(*TRAIN_FILES,*[f'train_e{i:02d}' for i in range(16)]) if expanded else TRAIN_FILES
    direction='temporal' if args.role=='A' else 'relational'
    data,manifest=load_data(data_root,direction,train_files=train_files)
    original_rows=sum(manifest[n]['rows'] for n in TRAIN_FILES)
    base_training={k:v[:original_rows] for k,v in data['train'].items()}
    records=[]
    for prefix in ('training','scale'):
        seeds=(601,602,603,604) if prefix=='training' and args.role=='B' else (601,602,603)
        for seed in seeds:
            for arm in ('baseline',direction):
                cell=root/f'{prefix}_{args.role}/{arm}_s{seed}'
                if not (cell/'done').exists(): continue
                if prefix=='scale' and not expanded: raise ValueError('Expanded model without expanded training data')
                tick=time.monotonic()
                checkpoint=torch.load(cell/'best.pt',map_location='cpu',weights_only=False)
                config=checkpoint['config']
                assert config['parameter_ema'] is False
                model=FeatureBeliefNet(BeliefNetConfig(**config['net'])).cuda()
                model.load_state_dict(checkpoint['net'])
                best_step=checkpoint['step']; del checkpoint
                evaluation={}
                for split in ('test','ood_attack','ood_random'):
                    probabilities=predict(model,data[split],direction,arm)
                    evaluation[split]=score_predictions(probabilities,data[split])
                    del probabilities
                training=data['train'] if prefix=='scale' else base_training
                probabilities=predict(model,training,direction,arm)
                selected_training=score_predictions(probabilities,training,include_games=False)
                del probabilities
                last=torch.load(cell/'last.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(last['net']); last_step=last['step']; del last
                probabilities=predict(model,training,direction,arm)
                final_training=score_predictions(probabilities,training,include_games=False)
                del probabilities
                value={'role':args.role,'experiment':prefix,'arm':arm,'seed':seed,
                    'best_step':best_step,'last_step':last_step,'parameter_ema':False,
                    'best_file_sha256':sha256(cell/'best.pt'),'last_file_sha256':sha256(cell/'last.pt'),
                    'data':{k:v['sha256'] for k,v in manifest.items()},'evaluations':evaluation,
                    'selected_training':selected_training,'final_training':final_training,
                    'seconds':time.monotonic()-tick}
                filename=f'{prefix}_{arm}_s{seed}.json'
                write_json(args.output/filename,value)
                records.append(filename)
                print(json.dumps({'event':'diagnostic_model_complete','model':filename,
                    'best_step':best_step,'selected_train_nll':selected_training['all']['nll'],
                    'final_train_nll':final_training['all']['nll'],'test_nll':evaluation['test']['all']['nll'],
                    'seconds':value['seconds']}),flush=True)
                del model,evaluation,selected_training,final_training,value
                gc.collect(); torch.cuda.empty_cache()
    write_json(args.output/'summary.json',{'status':'completed','role':args.role,'models':records,
        'models_evaluated':len(records),'expanded_data_available':expanded})
    (args.output/'done').touch()


if __name__=='__main__': main()
