"""Independent deadline enforcement; only this experiment's compute processes."""
import os,time,signal
from datetime import datetime
from pathlib import Path
from common import ROOT,save
train=datetime.fromisoformat('2026-09-21T19:30:00+08:00').timestamp()
eval_end=datetime.fromisoformat('2026-09-21T22:30:00+08:00').timestamp()
lease=datetime.fromisoformat('2026-09-21T23:59:00+08:00').timestamp()
seen={}
while time.time()<lease:
 now=time.time()
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:
   args=p.read_bytes().split(b'\0');joined=b' '.join(args)
   if str(ROOT).encode() not in joined:continue
   istrain=b'scripts/train.py' in args
   iseval=str(ROOT/'evaluate.py').encode() in args
   cutoff=train if istrain else eval_end
   if not (istrain or iseval) or now<cutoff:continue
   pid=int(p.parent.name)
   if pid not in seen:os.killpg(pid,signal.SIGTERM);seen[pid]=now
   elif now-seen[pid]>180:os.killpg(pid,signal.SIGKILL)
  except (OSError,ValueError):pass
 save(ROOT/'lease_guard_status.json',{'time':now,'signaled_pids':seen,'train_cutoff':train,'eval_cutoff':eval_end})
 time.sleep(30)
