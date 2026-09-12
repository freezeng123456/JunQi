"""Check the replay anchors and the effect of denser early model selection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.belief_features_20260913.analyze import read, paired_bootstrap, SEEDS, SPLITS, state_hash


def compare(root, draws=5000):
    early = read(root/'analysis_early/analysis.json')
    assert early['status'] == 'complete' and early['steps_per_cell'] == 1000
    checks=[]; comparisons={}; selections={}
    for role,arms in [('A',('baseline','temporal')),('B',('baseline','relational'))]:
        for arm in arms:
            originals=[]; replays=[]
            for seed in SEEDS:
                old=root/f'training_{role}/{arm}_s{seed}'
                new=root/f'early_{role}/{arm}_s{seed}'
                oc,nc=read(old/'config.json'),read(new/'config.json')
                names=('seed','arm','direction','net','lr','batch_size','gradient_clip',
                       'parameter_ema','loss','source_commit','data','initial_state_sha256','parameters')
                assert all(oc[k]==nc[k] for k in names)
                assert oc['eval_every']==500 and nc['eval_every']==50
                before={x['step']:x for x in map(json.loads,(old/'metrics.jsonl').read_text().splitlines())}
                after={x['step']:x for x in map(json.loads,(new/'metrics.jsonl').read_text().splitlines())}
                assert set(after)==set(range(50,1001,50))
                anchor=[]
                for step in (500,1000):
                    identical=before[step]['validation']==after[step]['validation']
                    assert identical,(role,arm,seed,step,'validation replay differs')
                    mean_loss=np.mean([after[t]['train_loss'] for t in range(step-450,step+1,50)])
                    error=abs(float(mean_loss)-before[step]['train_loss'])
                    assert error<1e-10,(role,arm,seed,step,error)
                    anchor.append({'step':step,'all_validation_metrics_identical':identical,
                                   'training_loss_block_mean_difference':error})
                os,ns=read(old/'summary.json'),read(new/'summary.json')
                same_step=os['best_step']==ns['best_step']
                same_weights=state_hash(old/'best.pt')==state_hash(new/'best.pt') if same_step else None
                if same_step: assert same_weights
                checks.append({'role':role,'arm':arm,'seed':seed,'same_initialization_data_optimizer':True,
                    'anchors':anchor,'original_selected_step':os['best_step'],
                    'dense_selected_step':ns['best_step'],'selected_weights_identical_if_step_matches':same_weights})
                originals.append(os); replays.append(ns)
            key=arm if arm!='baseline' or role=='A' else 'baseline_B_check'
            comparisons[key]={split:{metric:paired_bootstrap(
                [x['evaluations'][split] for x in originals],[x['evaluations'][split] for x in replays],
                metric,draws=draws) for metric in ('nll','brier','accuracy')} for split in SPLITS}
            selections[key]={'original_steps':[x['best_step'] for x in originals],
                             'dense_steps':[x['best_step'] for x in replays]}
    return {'status':'early_resolution_verified','early_analysis':early,'replay_checks':checks,
            'dense_selection_minus_primary_selection':comparisons,'selected_steps':selections,
            'scope':'Exploratory validation-resolution replay. Same original 1024 games and optimization; '
                    'the 500/1000-step validation metrics and aggregated training losses are checked against the original run. '
                    'Dense selection considers steps 0,50,...,1000; original selection considers 0,500,...,50000. '
                    'This is not new independent evidence from fresh data.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--draws',type=int,default=5000)
    args=parser.parse_args()
    result=compare(args.root,args.draws)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'replay_checks':len(result['replay_checks']),
                      'selected_steps':result['selected_steps']}),flush=True)
