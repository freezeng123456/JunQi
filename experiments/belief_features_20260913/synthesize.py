"""Join verified primary, expanded-data, public-subgroup and prior diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from experiments.belief_features_20260913.analyze import read,paired_bootstrap,verify_manifest,SEEDS,SPLITS

ARMS=('baseline','temporal','relational')
METRICS=('nll','brier','accuracy','ece','wrong_confident_fraction')


def synthesis(root,draws):
    reports={prefix:read(root/f'analysis_{prefix}/analysis.json') for prefix in ('training','scale')}
    diagnostics={}; coverage={}; agreement=[]
    for role,expected in [('A',12),('B',14)]:
        folder=root/f'diagnostics_{role}'
        files=verify_manifest(folder); summary=read(folder/'summary.json'); launch=read(folder/'launcher_exit.json')
        assert (folder/'done').exists() and summary['status']=='completed' and launch['exit_code']==0 and not launch['deadline_fired']
        assert summary['models_evaluated']==expected
        coverage[role]={'models_evaluated':expected,'verified_files':files,'exit_code':0}
        for name in summary['models']:
            item=read(folder/name)
            key=(item['experiment'],role,item['arm'],item['seed'])
            diagnostics[key]=item
            original=read(root/f"{item['experiment']}_{role}/{item['arm']}_s{item['seed']}/summary.json")
            differences=[]
            for split in SPLITS:
                a=original['evaluations'][split]; b=item['evaluations'][split]['all']
                assert a['labels']==b['labels']
                differences.extend(abs(a[m]-b[m]) for m in METRICS)
            difference=max(differences)
            assert difference<2e-6,(key,difference)
            agreement.append({'model':'/'.join(map(str,key)),'maximum_absolute_metric_difference':difference})
    groups={}; paired={}; fitting={}
    for prefix in ('training','scale'):
        groups[prefix]={}; paired[prefix]={}; fitting[prefix]={}
        for arm in ARMS:
            role='B' if arm=='relational' else 'A'
            models=[diagnostics[prefix,role,arm,seed] for seed in SEEDS]
            groups[prefix][arm]={split:{group:{m:float(np.mean([v['evaluations'][split][group][m] for v in models])) for m in METRICS}
                | {'labels':models[0]['evaluations'][split][group]['labels'],'games':len(models[0]['evaluations'][split][group]['games'])}
                for group in models[0]['evaluations'][split]} for split in SPLITS}
            fitting[prefix][arm]={selection:{m:float(np.mean([v[selection]['all'][m] for v in models])) for m in METRICS}
                for selection in ('selected_training','final_training')}
            if arm=='baseline': continue
            bases=[diagnostics[prefix,role,'baseline',seed] for seed in SEEDS]
            paired[prefix][arm]={split:{group:{m:paired_bootstrap(
                [v['evaluations'][split][group] for v in bases],
                [v['evaluations'][split][group] for v in models],m,draws=draws)
                for m in ('nll','accuracy')} for group in models[0]['evaluations'][split]} for split in SPLITS}
    volume={}; selection_window={}
    for arm in ARMS:
        role='B' if arm=='relational' else 'A'
        before=[diagnostics['training',role,arm,seed] for seed in SEEDS]
        after=[diagnostics['scale',role,arm,seed] for seed in SEEDS]
        selection_window[arm]={'primary_best_steps':[v['best_step'] for v in before],
            'expanded_best_steps':[v['best_step'] for v in after],
            'all_selected_primary_steps_at_most_5000':all(v['best_step']<=5000 for v in before)}
        volume[arm]={split:{m:paired_bootstrap([v['evaluations'][split]['all'] for v in before],
            [v['evaluations'][split]['all'] for v in after],m,draws=draws) for m in ('nll','brier','accuracy')} for split in SPLITS}
    tabular=read(root/'tabular_diagnostics.json')
    opening=read(root/'opening_prior.json')
    reference_comparisons={}
    for prefix in ('training','scale'):
        models=[diagnostics[prefix,'A','baseline',seed] for seed in SEEDS]
        reference_comparisons[prefix]={split:{ref:paired_bootstrap(
            [tabular['evaluations'][split][ref]['all']]*len(SEEDS),
            [v['evaluations'][split]['all'] for v in models],'nll',draws=draws)
            for ref in ('slot_initial','slot_all','slot_move')} for split in SPLITS}
        reference_comparisons[prefix]['opening_analytic']={split:paired_bootstrap(
            [opening['evaluations'][split]]*len(SEEDS),[v['evaluations'][split]['opening_zero'] for v in models],
            'nll',draws=draws) for split in SPLITS}
    raw=[]
    for key,item in diagnostics.items():
        prefix,role,arm,seed=key
        raw.append({'experiment':prefix,'role':role,'arm':arm,'seed':seed,'best_step':item['best_step'],
            'last_step':item['last_step'],'selected_train_nll':item['selected_training']['all']['nll'],
            'selected_train_accuracy':item['selected_training']['all']['accuracy'],
            'last_train_nll':item['final_training']['all']['nll'],
            'last_train_accuracy':item['final_training']['all']['accuracy'],
            'selected_test_nll':item['evaluations']['test']['all']['nll']})
    expanded_data=read(root/'data_expanded/recovery_verified.json')
    assert expanded_data['status']=='all_24_shards_verified' and expanded_data['training_games']==5120
    audit=read(root/'dataset_audit.json')
    assert audit['all_game_partitions_disjoint'] and all(x['temporal_movement_matches_base'] for x in audit['shards'].values())
    return {'status':'all_experiments_and_diagnostics_verified',
        'analysis_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'primary':reports['training'],'expanded':reports['scale'],'diagnostic_coverage':coverage,
        'independent_rescoring_agreement':agreement,'group_means':groups,'paired_group_comparisons':paired,
        'data_volume_comparisons':volume,'selection_window_check':selection_window,
        'full_training_fit':fitting,'all_model_fit_records':raw,'reference_comparisons':reference_comparisons,
        'tabular_reference_fit_uses_original_1024_games':True,
        'opening_distribution_reference':{k:opening[k] for k in ('mean_opening_marginal_entropy_nats','mean_opening_marginal_top1','scope')},
        'bootstrap_draws':draws}


def plot_volume(root,result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'baseline':'#526173','temporal':'#D77830','relational':'#188C8A'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(13,4.4))
    for arm in ARMS:
        role='B' if arm=='relational' else 'A'
        for prefix,style in [('training','--'),('scale','-')]:
            curves=[[json.loads(line) for line in (root/f'{prefix}_{role}/{arm}_s{seed}/metrics.jsonl').read_text().splitlines()
                if json.loads(line)['step']<=5000] for seed in SEEDS]
            steps=[x['step'] for x in curves[0]]
            values=np.array([[x['validation']['nll'] for x in curve] for curve in curves])
            axs[0].plot(steps,values.mean(0),style,color=colors[arm],linewidth=1.7,
                label=f'{arm}: '+('1,024 games' if prefix=='training' else '5,120 games'))
        for j,split in enumerate(SPLITS):
            value=result['data_volume_comparisons'][arm][split]['nll']
            delta=value['delta_candidate_minus_reference']; lo,hi=value['paired_seed_and_game_ci95']
            y=j+(ARMS.index(arm)-1)*.2
            axs[1].errorbar(delta,y,xerr=[[max(0,delta-lo)],[max(0,hi-delta)]],fmt='o',capsize=3,
                color=colors[arm],label=arm if j==0 else None)
    axs[0].set(xlabel='Optimizer updates (first 5,000)',ylabel='Validation NLL',title='More independent games: same models')
    axs[0].legend(frameon=False,fontsize=7.8,ncol=2); axs[0].grid(alpha=.15)
    axs[1].axvline(0,color='#758195',linestyle='--',linewidth=1)
    axs[1].set(yticks=range(3),yticklabels=['Frozen-policy test','Attack-biased random','Uniform random'],
        xlabel='Expanded-data NLL - original-data NLL',title='Paired data-volume effects with 95% intervals')
    axs[1].invert_yaxis(); axs[1].legend(frameon=False); axs[1].grid(axis='x',alpha=.15)
    fig.tight_layout(pad=2)
    fig.savefig(out/'data_volume_results.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/'data_volume_results.svg',bbox_inches='tight'); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--draws',type=int,default=5000)
    args=parser.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    result=synthesis(args.root,args.draws)
    (args.output/'synthesis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    plot_volume(args.root,result,args.output)
    print(json.dumps({'status':result['status'],'diagnostic_coverage':result['diagnostic_coverage']}),flush=True)
