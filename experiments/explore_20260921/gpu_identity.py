import sys,json,torch
from junqi_rl.training.checkpoint import load_evaluation_policy
from tests.test_explore_20260921 import sample
old=load_evaluation_policy(sys.argv[1],device='cuda');new=load_evaluation_policy(sys.argv[2],device='cuda')
obs,g,mask,_=sample();obs=obs.cuda();g=g.cuda();mask=torch.ones((1,16641),dtype=torch.bool,device='cuda')
proof={}
for bf in (False,True):
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf):
  a=old._encode(obs,g);b=new._encode(obs,g)
  x=old._policy_logits(a[1],mask,obs);y=new._policy_logits(b[1],mask,obs)
  torch.testing.assert_close(x,y,rtol=0,atol=0);torch.testing.assert_close(old._value(a[0]),new._value(b[0]),rtol=0,atol=0)
  proof['bf16' if bf else 'fp32']='exact_initial_policy_and_value'
print(json.dumps(proof))
