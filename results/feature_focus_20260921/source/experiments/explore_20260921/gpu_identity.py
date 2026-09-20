import sys,json,torch
from junqi_rl.training.checkpoint import load_evaluation_policy
import random
from junqi_core.state import GameState
from junqi_core.setup import generate_random_setup
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import build_observation
old=load_evaluation_policy(sys.argv[1],device='cuda');new=load_evaluation_policy(sys.argv[2],device='cuda')
state=GameState.new_game(generate_random_setup(random.Random(921)))
belief=BeliefTensor.initial(state,state.turn);observation=build_observation(state,belief,state.turn).snapshot()
obs=torch.from_numpy(observation.spatial)[None].cuda();g=torch.from_numpy(observation.global_)[None].cuda();mask=torch.ones((1,16641),dtype=torch.bool,device='cuda')
proof={}
for bf in (False,True):
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf):
  a=old._encode(obs,g);b=new._encode(obs,g)
  x=old._policy_logits(a[1],mask,obs);y=new._policy_logits(b[1],mask,obs)
  torch.testing.assert_close(x,y,rtol=0,atol=0);torch.testing.assert_close(old._value(a[0]),new._value(b[0]),rtol=0,atol=0)
  proof['bf16' if bf else 'fp32']='exact_initial_policy_and_value'
print(json.dumps(proof))
