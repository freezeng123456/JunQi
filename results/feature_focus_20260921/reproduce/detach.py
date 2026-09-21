import subprocess,sys,json,os
from common import *
credentials=json.loads(sys.stdin.readline())
assert not (ROOT/'controller.pid').exists(),'already launched; inspect status before resubmitting'
for name in ('lease_guard','controller'):
 with (ROOT/(name+'.log')).open('a') as log:
  p=subprocess.Popen([PYTHON,str(ROOT/(name+'.py'))]+([sys.argv[1]] if name=='controller' else []),stdin=subprocess.PIPE if name=='controller' else subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT,env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_DISABLE_XET='1'))
 if name=='controller':p.stdin.write((json.dumps(credentials)+'\n').encode());p.stdin.close()
 (ROOT/(name+'.pid')).write_text(str(p.pid)+'\n')
 print(name,p.pid)
