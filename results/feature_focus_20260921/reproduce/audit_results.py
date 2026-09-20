from pathlib import Path
import json,hashlib,collections
import numpy as np
base=Path(__file__).resolve().parent;root=base/'recovered/feature_focus_20260921';proof={}
for arm in ('N','M','R'):
 folder=root/arm/'comparison';summary=json.loads((folder/'summary.json').read_text());reports={};verified=[]
 for line in (folder/'SHA256SUMS').read_text().splitlines():
  h,name=line.split('  ',1);assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h;verified.append(name)
 for name,s in summary['matches'].items():
  rows=[json.loads(x) for x in (folder/name/'games.jsonl').read_text().splitlines()];assert len(rows)==s['games']
  c=collections.Counter(x['outcome'] for x in rows);assert dict(c)==s['counts']
  pairs=collections.defaultdict(list)
  for x in rows:pairs[(x['setup_seed'],x['random_seed'])].append(x)
  for p in pairs.values():assert len(p)==2 and {x['first_team'] for x in p}=={0,1} and len({x['setup_sha256'] for x in p})==1
  values=np.array([sum({'win':1,'loss':0,'draw':.5}[x['outcome']] for x in p)/2 for p in pairs.values()]);assert float(values.mean())==s['score'] and c['win']/len(rows)==s['win_fraction']
  rng=np.random.default_rng(919);boot=np.array([rng.choice(values,size=len(values),replace=True).mean() for _ in range(10000)])
  assert np.allclose(np.quantile(boot,[.025,.975]),s['pair_bootstrap_ci95'])
  reports[name]={'games':len(rows),'paired_seeds':len(pairs),'counts_score_ci_verified':True}
 proof[arm]={'comparison_status':summary['status'],'manifest_files':len(verified),'matchups':reports}
(base/'independent_result_audit.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
