"""Audit suite seed separation and compute paired inference-knockout differences."""
from pathlib import Path
import collections, hashlib, json
import numpy as np

BASE=Path(__file__).resolve().parent
ROOT=BASE/'recovered/feature_focus_20260921'
SCORES={'win':1.,'loss':0.,'draw':.5}
setups={};values={};total=0
for arm in ('N','M','R'):
    folder=ROOT/arm/'comparison'
    summary=json.loads((folder/'summary.json').read_text())
    assert summary['status']=='complete'
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    for name,stats in summary['matches'].items():
        pairs=collections.defaultdict(list)
        rows=[json.loads(line) for line in (folder/name/'games.jsonl').read_text().splitlines()]
        assert len(rows)==stats['games']
        total+=len(rows)
        for row in rows:
            pairs[(row['setup_seed'],row['random_seed'])].append(row)
        for pair in pairs.values():
            assert len(pair)==2
            assert {row['first_team'] for row in pair}=={0,1}
            assert len({row['setup_sha256'] for row in pair})==1
        setups[name]={key:pair[0]['setup_sha256'] for key,pair in pairs.items()}
        values[name]={key:np.mean([SCORES[row['outcome']] for row in pair]) for key,pair in pairs.items()}
primary=['N_vs_start','M_vs_start','M_vs_N','R_vs_start','R_vs_N','R_vs_M']
for name in primary:
    assert setups[name]==setups[primary[0]]
for name in setups:
    if name not in primary:
        assert not set(setups[name])&set(setups[primary[0]])
full='R_diag_full_vs_N';deltas={}
for group in ('threat','location','flags','information','resources'):
    name='R_without_'+group+'_vs_N'
    assert setups[name]==setups[full]
    differences=np.array([values[name][key]-values[full][key] for key in sorted(values[full])])
    rng=np.random.default_rng(921)
    bootstrap=np.array([rng.choice(differences,len(differences),replace=True).mean() for _ in range(10000)])
    deltas[group]={'games':512,'paired_seeds':256,'score_delta_knockout_minus_full':float(differences.mean()),'pair_bootstrap_delta_ci95':np.quantile(bootstrap,[.025,.975]).tolist()}
assert total==16896
report={'source':'HF revision '+json.loads((BASE/'recovery.json').read_text())['revision'],'total_games':total,'primary_games':12288,'random_games':1536,'diagnostic_games':3072,'main_setup_identity_across_six_matchups':True,'diagnostic_setup_identity':True,'fresh_diagnostic_and_random_seeds':True,'diagnostic_deltas':deltas,'interpretation':'Inference-time knockout of one trained model; not independent retraining, not a feature ranking.'}
(BASE/'independent_suite_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
