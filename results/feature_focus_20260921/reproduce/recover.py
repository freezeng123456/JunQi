"""One-shot private artifact recovery with manifest verification; no training control."""
import os,json,pathlib,re,hashlib,sys
for k in ['HF_HUB_DISABLE_XET','HF_HUB_DISABLE_PROGRESS_BARS','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='60'
os.environ['HF_HUB_ETAG_TIMEOUT']='60'
from huggingface_hub import HfApi,snapshot_download
base=pathlib.Path(__file__).resolve().parent
token=os.environ.get('HF_TOKEN');assert token, 'Provide HF_TOKEN in the process environment'
api=HfApi(token=token);repo='a1390892757/junqi-guarded-continue-20260914';prefix='feature_focus_20260921'
info=api.model_info(repo,files_metadata=True);assert info.private
patterns=[prefix+'/*'] if '--all' in sys.argv else [prefix+'/*/comparison/*',prefix+'/*/delivery/*',prefix+'/*/primary/*',prefix+'/*/terminal/*']
local=base/'recovered'
snapshot_download(repo,revision=info.sha,allow_patterns=patterns,local_dir=str(local),token=token,max_workers=2)
checks=[]
for manifest in (local/prefix).rglob('SHA256SUMS'):
 for line in manifest.read_text().splitlines():
  expected,name=line.split('  ',1);p=manifest.parent/name
  assert p.is_file(),f"missing manifest entry: {p}"
  if p.exists():
   with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
   assert actual==expected,(p,actual,expected);checks.append(str(p.relative_to(local)))
report={'revision':info.sha,'verified_files':len(checks),'local':str(local),'selection':patterns}
(base/'recovery.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
