"""Publish one arm's complete evidence to the shared results branch via Git."""
import os,sys,json,subprocess,shutil,re,time
from common import *

def main(creds,arm):
 from huggingface_hub import HfApi
 repo='https://github.com/freezeng123456/JunQi.git';branch='results/feature-focus-20260921'
 env=dict(os.environ,JUNQI_GH_PUBLISH_TOKEN=creds['github'],GIT_TERMINAL_PROMPT='0')
 helper='!f() { echo username=x-access-token; echo password=$JUNQI_GH_PUBLISH_TOKEN; }; f'
 def git(*args,cwd=None):
  return subprocess.check_output(['git','-c','credential.helper=','-c','credential.helper='+helper,*args],cwd=cwd,env=env,text=True,stderr=subprocess.PIPE).strip()
 # Check network before creating the local publication checkout.
 refs=git('ls-remote',repo,'refs/heads/'+branch)
 folder=ROOT/('publish_'+arm)
 if not (folder/'.git').exists():
  git('clone','--depth','1','--single-branch','--branch',branch if refs else 'main',repo,str(folder))
 git('config','user.name','JunQi experiment',cwd=folder);git('config','user.email','junqi-experiment@users.noreply.github.com',cwd=folder)
 rel='results/feature_focus_20260921';out=folder/rel/arm
 if out.exists():shutil.rmtree(out)
 shutil.copytree(ROOT/arm/'comparison',out)
 for p in (ROOT/arm).glob('*_upload.json'):shutil.copy2(p,out/p.name)
 for n in ['migration.json','gpu_identity.log','smoke.log']:
  p=ROOT/arm/n
  if p.exists():shutil.copy2(p,out/p.name)
 reproduce=folder/rel/'reproduce';reproduce.mkdir(parents=True,exist_ok=True)
 for p in ROOT.glob('*'):
  if p.is_file() and p.suffix in ('.py','.md','.json') and p.name not in ('controller.pid','lease_guard_status.json'):shutil.copy2(p,reproduce/p.name)
 code=folder/rel/'source';code.mkdir(exist_ok=True)
 for name in ['junqi_rl/networks/relational_features.py','junqi_rl/networks/tactical_features.py','junqi_rl/networks/junqi_net.py','junqi_rl/checkpoint_compat.py','experiments/feature_focus_20260921/migrate.py','experiments/feature_focus_20260921/snapshot.py','experiments/explore_20260921/gpu_identity.py','tests/test_feature_focus_20260921.py']:
  p=code/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'code'/name,p)
 for p in (folder/rel).rglob('*'):
  if p.is_file():assert not re.search(rb'(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]{20,}',p.read_bytes())
 git('add','-f',rel,cwd=folder)
 git('commit','-m','results: preserve JunQi '+arm+' public tactical feature ablation 2026-09-21',cwd=folder)
 # Other arm may have published first. Merge preserves both complete directories.
 for _ in range(3):
  try:
   if git('ls-remote',repo,'refs/heads/'+branch):
    git('fetch','--depth','50','origin',branch,cwd=folder)
    git('merge','--no-edit','-X','ours','FETCH_HEAD',cwd=folder)
   git('push','origin','HEAD:refs/heads/'+branch,cwd=folder)
   commit=git('rev-parse','HEAD',cwd=folder)
   assert git('ls-remote',repo,'refs/heads/'+branch).split()[0]==commit
   result={'status':'github_remote_ref_verified','branch':branch,'commit':commit,'arm':arm}
   save(ROOT/arm/'github_upload.json',result)
   delivery=ROOT/arm/'delivery';delivery.mkdir(exist_ok=True);save(delivery/'github_upload.json',result)
   save(ROOT/arm/'delivery_upload.json',publish(HfApi(token=creds['hf']),delivery,PREFIX+'/'+arm+'/delivery'))
   return result
  except subprocess.CalledProcessError:time.sleep(10)
 raise RuntimeError('Git publication failed; recover from HF')
