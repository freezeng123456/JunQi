"""Information boundary, neutral migration, and optimizer continuity contracts."""
import copy
import os
from pathlib import Path
import pytest
import torch
from junqi_core.rules import PieceType,Seat,ShowMode,Event,resolve_combat
from junqi_core.info_model import BeliefTensor,TRACKED_TYPES
from junqi_rl.networks.tactical_features import TacticalFeatureHead
from tests.test_feature_information_boundaries import position
from tests.test_combat_outcome_features import inputs
from junqi_core.state import Action
from experiments.explore_20260921.migrate import migrate

START=Path(os.environ.get('JUNQI_START_CHECKPOINT',str(Path(__file__).resolve().parents[3]/'outputs/JunQi-lr-20260920/payload/start.pt')))

def sample(defender=PieceType.LIANZH,bright=False):
    s=position(PieceType.PAIZH,defender)
    if bright:s.show_mode=ShowMode.BRIGHT
    return inputs(s,BeliefTensor.initial(s,Seat.SOUTH),[Action(Seat.SOUTH,(2,7),(1,7))])

def test_tactical_known_combat_and_information_boundary():
    head=TacticalFeatureHead()
    torch.nn.init.normal_(head.mlp[-1].weight)
    seen=[]
    for defender in (PieceType.DILEI,PieceType.LIANZH):
        obs,_,_,_=sample(defender)
        seen.append((obs,head(obs)))
    torch.testing.assert_close(seen[0][0],seen[1][0],rtol=0,atol=0)
    torch.testing.assert_close(seen[0][1],seen[1][1],rtol=0,atol=0)
    for defender in TRACKED_TYPES:
        obs,_,_,actions=sample(defender,True);own,feat,valid=head.features(obs)
        a=TRACKED_TYPES.index(PieceType.PAIZH);d=TRACKED_TYPES.index(defender);j=int(actions[0]%129)
        event=resolve_combat(PieceType.PAIZH,defender)
        torch.testing.assert_close(feat[0,a,j,0],head.values[d] if event in (Event.EAT,Event.BOMB) else torch.tensor(0.))
        torch.testing.assert_close(feat[0,a,j,1],head.values[a] if event in (Event.KILLED,Event.BOMB) else torch.tensor(0.))
        assert feat[0,a,j,2]==0 and valid[0,a,j]
    # Even a learned bias cannot give an attack feature to an empty destination.
    obs,_,_,_=sample();empty=torch.zeros_like(obs)
    torch.nn.init.constant_(head.mlp[-1].bias,1)
    assert torch.count_nonzero(head(empty))==0

@pytest.mark.parametrize('kind',['F','D'])
def test_migration_preserves_function_and_moments_and_can_update(kind,tmp_path):
    torch.set_num_threads(1)
    old,new,sd=migrate(START,tmp_path/'migrated.pt',kind)
    obs,g,mask,_=sample();obs=obs.expand(2,-1,-1,-1).clone();g=g.expand(2,-1).clone()
    # Use multiple legal choices so added modules get a nonzero policy gradient.
    mask=torch.ones(2,16641,dtype=torch.bool)
    for mode in ('float32','bfloat16'):
        with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=mode=='bfloat16'):
            a=old._encode(obs,g);b=new._encode(obs,g)
            torch.testing.assert_close(old._policy_logits(a[1],mask,obs),new._policy_logits(b[1],mask,obs),rtol=0,atol=0)
            torch.testing.assert_close(old._value(a[0]),new._value(b[0]),rtol=0,atol=0)
    orig=torch.load(START,map_location='cpu',weights_only=False)
    old_names=[n for n,_ in old.named_parameters()];new_names=[n for n,_ in new.named_parameters()]
    for n in old_names:
        for k,v in orig['optimizer']['state'][old_names.index(n)].items():
            if isinstance(v,torch.Tensor):assert torch.equal(v,sd['optimizer']['state'][new_names.index(n)][k])
    new.train();cls,cells=new._encode(obs,g);logits=new._policy_logits(cells,mask,obs)
    (logits.square().mean()+new._value(cls).square().mean()).backward()
    added=[(n,p) for n,p in new.named_parameters() if n not in old_names]
    assert any(p.grad is not None and p.grad.abs().sum()>0 for _,p in added)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for _,p in new.named_parameters())
