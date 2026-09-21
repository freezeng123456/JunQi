import hashlib,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO='a1390892757/junqi-guarded-continue-20260914'
PREFIX='feature_focus_20260921'
PYTHON='/jizhicfs/yuyechen/miniconda3/envs/cl/bin/python'
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def publish(api,folder,prefix):
 files={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '.cache' not in p.parts}
 (folder/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(files.items())))
 files['SHA256SUMS']=sha(folder/'SHA256SUMS')
 last=None
 for attempt in range(3):
  try:
   c=api.upload_folder(repo_id=REPO,folder_path=str(folder),path_in_repo=prefix,ignore_patterns=['.cache/**'])
   info=api.model_info(REPO,revision=c.oid,files_metadata=True);assert info.private
   lookup={x.rfilename:x for x in info.siblings}
   for n,h in files.items():
    x=lookup[prefix+'/'+n];p=folder/n;assert x.size==p.stat().st_size
    if x.lfs:assert x.lfs.sha256==h
    else:
     data=p.read_bytes();assert x.blob_id==hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
   return {'commit':c.oid,'prefix':prefix,'files':len(files),'hashes_verified':True}
  except Exception as e:
   last=type(e).__name__;time.sleep(20)
 raise RuntimeError('publication failed: '+str(last))
