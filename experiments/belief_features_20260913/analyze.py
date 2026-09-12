"""Paired seed/game analysis; never treat repeated chess-piece labels as IID."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS=(601,602,603)
SPLITS=('test','ood_attack','ood_random')
SCALARS=('nll','brier','accuracy','ece','wrong_confident_fraction','confident_fraction')


def read(path): return json.loads(Path(path).read_text())


def verify_manifest(root):
    count=0
    for line in (root/'artifacts.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        path=root/name
        assert '..' not in Path(name).parts and path.is_relative_to(root)
        with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=digest: raise ValueError(f'Artifact mismatch: {path}')
        count+=1
    return count


def paired_bootstrap(reference,candidate,metric,*,draws=5000,rng_seed=620913):
    """Shared game resamples preserve pairing across both arms and all seeds."""
    assert len(reference)==len(candidate)>0
    arrays=[]; weights=None; ids=None
    for a,b in zip(reference,candidate,strict=True):
        left={g['game_id']:g for g in a['games']}
        right={g['game_id']:g for g in b['games']}
        assert left.keys()==right.keys()
        current=sorted(left)
        assert ids is None or ids==current
        ids=current
        wa=np.array([left[g]['labels'] for g in ids],dtype=float)
        wb=np.array([right[g]['labels'] for g in ids],dtype=float)
        np.testing.assert_array_equal(wa,wb)
        if weights is not None: np.testing.assert_array_equal(weights,wa)
        weights=wa
        arrays.append([right[g][metric]-left[g][metric] for g in ids])
    delta=np.array(arrays); nseed,ngame=delta.shape
    per_seed=(delta*weights).sum(1)/weights.sum()
    rng=np.random.default_rng(rng_seed)
    hierarchical=np.empty(draws); conditional=np.empty(draws)
    for i in range(draws):
        games=rng.integers(ngame,size=ngame)
        selected_weights=weights[games]
        per=(delta[:,games]*selected_weights).sum(1)/selected_weights.sum()
        hierarchical[i]=per[rng.integers(nseed,size=nseed)].mean()
        conditional[i]=per.mean()
    return {'delta_candidate_minus_reference':float(per_seed.mean()),
        'paired_seed_deltas':per_seed.tolist(),'training_seed_sd':float(per_seed.std(ddof=1)) if nseed>1 else None,
        'paired_seed_and_game_ci95':np.quantile(hierarchical,[.025,.975]).tolist(),
        'paired_game_only_ci95':np.quantile(conditional,[.025,.975]).tolist(),
        'equal_game_weight_delta':float(delta.mean()),'training_seeds':nseed,
        'test_games':ngame,'labels_per_seed':int(weights.sum()),'bootstrap_draws':draws}


def state_hash(path):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)['net']
    digest=hashlib.sha256()
    for name,value in state.items():
        digest.update(name.encode()); digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def collect(root):
    loaded={}; coverage={}; configs={}
    for role,seeds,direction in [('A',SEEDS,'temporal'),('B',(*SEEDS,604),'relational')]:
        path=root/f'training_{role}'
        verified=verify_manifest(path)
        overview=read(path/'summary.json'); exit_info=read(path/'launcher_exit.json')
        assert overview['status']=='completed' and exit_info['exit_code']==0 and not exit_info['deadline_fired']
        expected=2*len(seeds)
        assert overview['expected_cells']==overview['completed_cells']==expected
        for seed in seeds:
            for arm in ('baseline',direction):
                cell=path/f'{arm}_s{seed}'
                summary=read(cell/'summary.json'); config=read(cell/'config.json')
                assert (cell/'done').exists() and summary['status']=='completed'
                assert summary['steps']==config['max_steps']==50000
                assert not config['parameter_ema'] and config['source_commit']=='57940a09dd3ad0b24534a827e0c7b3ffda252ffd'
                key=(role,arm,seed)
                loaded[key]=summary; configs[key]=config
        coverage[role]={'expected_cells':expected,'completed_cells':expected,'verified_files':verified,
                        'exit_code':exit_info['exit_code'],'deadline_fired':exit_info['deadline_fired']}
    agreement=[]
    for seed in SEEDS:
        a=loaded['A','baseline',seed]; b=loaded['B','baseline',seed]
        ca=configs['A','baseline',seed]; cb=configs['B','baseline',seed]
        facts={'seed':seed,'initial_state_identical':ca['initial_state_sha256']==cb['initial_state_sha256'],
            'data_identical':ca['data']==cb['data'],'steps_identical':a['steps']==b['steps'],
            'last_parameters_identical':a['final_parameter_sha256']==b['final_parameter_sha256'],
            'best_step_identical':a['best_step']==b['best_step'],
            'best_parameters_identical':state_hash(root/f'training_A/baseline_s{seed}/best.pt')==state_hash(root/f'training_B/baseline_s{seed}/best.pt'),
            'max_holdout_scalar_difference':max(abs(a['evaluations'][s][m]-b['evaluations'][s][m]) for s in SPLITS for m in SCALARS)}
        facts['passed']=all(facts[k] for k in facts if k not in ('seed','max_holdout_scalar_difference')) and facts['max_holdout_scalar_difference']==0
        agreement.append(facts)
    return loaded,configs,coverage,agreement


def analyze(root,*,draws=5000):
    loaded,configs,coverage,agreement=collect(root)
    means={}; paired={}; stages={}; controls={}; trajectories={}
    for arm,role in [('baseline','A'),('temporal','A'),('relational','B')]:
        models=[loaded[role,arm,seed] for seed in SEEDS]
        means[arm]={split:{m:float(np.mean([model['evaluations'][split][m] for model in models])) for m in SCALARS} for split in SPLITS}
        stages[arm]={split:{stage:{m:float(np.mean([model['evaluations'][split]['stages'][stage][m] for model in models]))
            for m in ('nll','brier','accuracy','labels')} for stage in models[0]['evaluations'][split]['stages']} for split in SPLITS}
        trajectories[arm]=[{'seed':seed,'best_step':model['best_step'],'selected_model_training_subset':model['training_subset'],
            'last_metrics':json.loads((root/f'training_{role}/{arm}_s{seed}/metrics.jsonl').read_text().splitlines()[-1]),
            'train_seconds':model['train_seconds'],'elapsed_seconds':model['elapsed_seconds']} for seed,model in zip(SEEDS,models,strict=True)]
        if arm=='baseline': continue
        baseline=[loaded[role,'baseline',seed] for seed in SEEDS]
        paired[arm]={split:{m:paired_bootstrap([v['evaluations'][split] for v in baseline],
            [v['evaluations'][split] for v in models],m,draws=draws) for m in ('nll','brier','accuracy')} for split in SPLITS}
        controls[arm]={control:{m:float(np.mean([model['feature_controls'][control][m]-model['evaluations']['test'][m] for model in models]))
            for m in ('nll','brier','accuracy')} for control in ('zero','shuffle')}
    extra={arm:{'best_step':loaded['B',arm,604]['best_step'],'evaluations':{s:{m:loaded['B',arm,604]['evaluations'][s][m]
        for m in SCALARS} for s in SPLITS}} for arm in ('baseline','relational')}
    references=read(root/'training_A/reference_metrics.json')
    assert references==read(root/'training_B/reference_metrics.json')
    parameters={str(k):v['parameters'] for k,v in configs.items()}
    assert len(set(parameters.values()))==1
    return {'status':'complete','primary_seeds':list(SEEDS),'coverage':coverage,
        'parameters_per_arm':next(iter(parameters.values())),'cross_host_baseline_agreement':agreement,
        'primary_means':means,'paired_comparisons':paired,'stages':stages,'feature_control_minus_full':controls,
        'learning_trajectories':trajectories,'supplementary_seed_604':extra,
        'references':{s:{p:{m:references[s][p][m] for m in SCALARS} for p in ('rule','frequency')} for s in SPLITS},
        'ci_scope':'Descriptive paired resampling of the three observed training seeds and complete held-out games; not IID piece labels or a population-wide guarantee.'}


def plot(root,analysis,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'baseline':'#526173','temporal':'#D77830','relational':'#188C8A'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axs=plt.subplots(1,2,figsize=(13,4.5),gridspec_kw={'width_ratios':[1.25,1]})
    for arm,role in [('baseline','A'),('temporal','A'),('relational','B')]:
        curves=[[json.loads(line) for line in (root/f'training_{role}/{arm}_s{s}/metrics.jsonl').read_text().splitlines()] for s in SEEDS]
        steps=[row['step'] for row in curves[0]]
        assert all([row['step'] for row in curve]==steps for curve in curves)
        for kind,style in [('validation','-'),('training','--')]:
            values=np.array([[r['validation']['nll'] if kind=='validation' else r['train_loss'] for r in c] for c in curves])
            axs[0].plot(steps,values.mean(0),style,color=colors[arm],label=f'{arm}: {kind}',linewidth=1.5)
    axs[0].set(xlabel='Optimizer updates',ylabel='NLL / training loss',title='Learning curves: mean of three seeds')
    axs[0].legend(frameon=False,fontsize=8,ncol=2); axs[0].grid(alpha=.15)
    for i,arm in enumerate(('temporal','relational')):
        for j,split in enumerate(SPLITS):
            value=analysis['paired_comparisons'][arm][split]['nll']
            delta=value['delta_candidate_minus_reference']; lo,hi=value['paired_seed_and_game_ci95']
            y=j+(i-.5)*.2
            axs[1].errorbar(delta,y,xerr=[[max(0,delta-lo)],[max(0,hi-delta)]],fmt='o',capsize=3,color=colors[arm],label=arm if j==0 else None)
    axs[1].axvline(0,color='#758195',linewidth=1,linestyle='--')
    axs[1].set(yticks=range(3),yticklabels=['Frozen-policy test','Attack-biased random','Uniform random'],
        xlabel='Feature NLL - baseline NLL (negative is better)',title='Paired differences with 95% intervals')
    axs[1].invert_yaxis(); axs[1].legend(frameon=False); axs[1].grid(axis='x',alpha=.15)
    fig.tight_layout(pad=2)
    fig.savefig(out/'belief_feature_results.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/'belief_feature_results.svg',bbox_inches='tight')
    plt.close(fig)


def markdown(analysis):
    lines=['# BeliefNet 两类特征实验结果','',
        '本表只汇总完成并通过文件哈希校验的固定步数实验。主比较采用配对种子 601–603；补充种子 604 单独保存。','',
        '| 测试分布 | 规则 NLL | 标签频率 NLL | 基线 NLL | 长期轨迹 NLL | 空间关系 NLL |',
        '|---|---:|---:|---:|---:|---:|']
    names={'test':'冻结策略','ood_attack':'偏好攻击的随机行为','ood_random':'均匀随机行为'}
    for split in SPLITS:
        refs=analysis['references'][split]; means=analysis['primary_means']
        lines.append(f"| {names[split]} | {refs['rule']['nll']:.6f} | {refs['frequency']['nll']:.6f} | "+' | '.join(f"{means[a][split]['nll']:.6f}" for a in ('baseline','temporal','relational'))+' |')
    lines+=['','NLL 越低越好。下表为新增特征减去本机配对基线，区间同时按本轮训练种子和完整测试对局重采样。','',
        '| 方向 | 测试分布 | NLL 差值 | 配对 95% 区间 | 三个种子的差值 |','|---|---|---:|---|---|']
    for arm in ('temporal','relational'):
        for split in SPLITS:
            m=analysis['paired_comparisons'][arm][split]['nll']; lo,hi=m['paired_seed_and_game_ci95']
            lines.append(f"| {arm} | {names[split]} | {m['delta_candidate_minus_reference']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | "+', '.join(f'{v:+.6f}' for v in m['paired_seed_deltas'])+' |')
    lines+=['','共有三组主训练种子，区间只描述当前实验的数据和随机初始化，不等于对所有策略、布局分布和长局面的普遍结论。',
        '',f"已完成 A {analysis['coverage']['A']['completed_cells']}/6、B {analysis['coverage']['B']['completed_cells']}/8 个主单元；各单元 50,000 步。两机基线完全一致：{all(a['passed'] for a in analysis['cross_host_baseline_agreement'])}。",'',
        '完整校准指标、阶段指标、特征干预、末期训练曲线、补充种子及完成证据见 analysis.json；解释性结论另见最终研究报告。','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--draws',type=int,default=5000)
    parser.add_argument('--no-plot',action='store_true'); args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    result=analyze(args.root,draws=args.draws)
    (args.output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (args.output/'RESULT_TABLES.md').write_text(markdown(result))
    if not args.no_plot: plot(args.root,result,args.output)
    print(json.dumps({'status':result['status'],'coverage':result['coverage']}),flush=True)
