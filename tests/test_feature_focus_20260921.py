import os,random
from pathlib import Path
import numpy as np
import pytest
import torch
from junqi_core.board import COMPACT_TO_FLAT,FLAT_TO_COMPACT,is_stronghold
from junqi_core.rules import PieceType,Seat,ShowMode
from junqi_core.move_gen import PieceRef,is_legal_move
from junqi_core.state import GameState,Action
from junqi_core.setup import generate_random_setup
from junqi_core.info_model import BeliefTensor,TRACKED_TYPES
from junqi_core.observation import build_observation,CHANNEL_LAYOUT
from junqi_rl.networks.relational_features import RelationalFeatureHead,DEST_NAMES
from junqi_rl.networks.tactical_features import TacticalFeatureHead
from tests.test_feature_information_boundaries import position
from tests.test_combat_outcome_features import inputs
from experiments.feature_focus_20260921.migrate import migrate
START=Path(os.environ.get('JUNQI_START_CHECKPOINT',str(Path(__file__).resolve().parents[3]/'outputs/JunQi-feature-focus-20260921/payload/start.pt')))

def idx(x,y):return int(FLAT_TO_COMPACT[y*17+x])
def observe(st,observer=Seat.SOUTH):
 ob=build_observation(st,BeliefTensor.initial(st,observer),observer).snapshot()
 return torch.from_numpy(ob.spatial)[None],torch.from_numpy(ob.global_)[None]

def test_material_values_follow_rank_strength_not_enum_direction():
 h=TacticalFeatureHead();v=dict(zip(TRACKED_TYPES,h.values.tolist()))
 assert v[PieceType.SILING]==pytest.approx(1.)
 assert v[PieceType.GONGB]==pytest.approx(1/9)
 assert v[PieceType.PAIZH]==pytest.approx(2/9)
 ordered=[PieceType.GONGB,PieceType.PAIZH,PieceType.LIANZH,PieceType.YINGZH,PieceType.TUANZH,PieceType.LVZH,PieceType.SHIZH,PieceType.JUNZH,PieceType.SILING]
 assert all(v[a]<v[b] for a,b in zip(ordered,ordered[1:]))
 assert all(0<float(x)<=1 for x in h.values)


def test_public_reach_matches_real_nonengineer_rules_for_random_occupancy():
 torch.set_num_threads(1);head=RelationalFeatureHead();rng=random.Random(9215)
 positions=[(int(f)%17,int(f)//17) for f in COMPACT_TO_FLAT]
 for _ in range(4):
  st=GameState.new_game(generate_random_setup(rng));obs,_=observe(st)
  _,_,reach,_,_=head.context(obs);reach=reach[0].numpy()>0
  pieces=dict(st.pieces)
  # Same SOUTH/canonical frame; replace source's identity only in the independent
  # move-generator oracle. Occupancy, intervening pieces and destination stay.
  for s in rng.sample(positions,20):
   for d in rng.sample(positions,20):
    if s==d or (d in pieces and pieces[d].seat in (Seat.SOUTH,Seat.NORTH)):continue
    counterfactual=pieces.copy();counterfactual[s]=PieceRef(Seat.SOUTH,PieceType.LIANZH)
    # Potential reach deliberately includes occupied camps; attackability is
    # represented separately as zero danger at camp destinations.
    from junqi_core.board import is_camp
    if d in pieces and is_camp(*d):continue
    assert reach[idx(*s),idx(*d)]==is_legal_move(counterfactual,s,d,Seat.SOUTH),(s,d)


def test_pressure_camp_and_hidden_identity_boundary():
 head=RelationalFeatureHead();out=[]
 for defender in (PieceType.DILEI,PieceType.LIANZH):
  st=position(PieceType.PAIZH,defender);obs,_=observe(st)
  own,features,reach,occ,known=head.context(obs);out.append(features)
 torch.testing.assert_close(out[0],out[1],rtol=0,atol=0)
 # All camps are protected from direct attack pressure, including empty camps
 # that a candidate could enter. Headquarters cannot emit threat/support moves.
 camps=head.camp;features=out[0]
 assert torch.count_nonzero(features[:,:,camps,DEST_NAMES.index('removal_pressure')])==0
 assert torch.count_nonzero(reach[:,head.stronghold])==0
 st=position(PieceType.PAIZH,PieceType.SILING);st.show_mode=ShowMode.BRIGHT;obs,_=observe(st)
 _,f,_,_,_=head.context(obs)
 # (2,7) is a protected camp; (1,6) is an adjacent exposed railway square.
 assert f[0,TRACKED_TYPES.index(PieceType.PAIZH),idx(1,6),0]>0
 # Engineer-only rail travel discloses the identity, but an ordinary adjacent
 # action does not. Use the independent move-generator to find such a route.
 empty=torch.zeros_like(obs);src=idx(6,6)
 empty[0,CHANNEL_LAYOUT['piece_own'].start+TRACKED_TYPES.index(PieceType.GONGB),6,6]=1
 _,_,normal,_,_=head.context(empty)
 destinations=[p for p in [(int(f)%17,int(f)//17) for f in COMPACT_TO_FLAT]
  if is_legal_move({(6,6):PieceRef(Seat.SOUTH,PieceType.GONGB)},(6,6),p,Seat.SOUTH) and not normal[0,src,idx(*p)]]
 assert destinations
 mask=torch.zeros(1,16641,dtype=torch.bool);target=idx(*destinations[0]);mask[0,src*129+target]=True
 with torch.no_grad():head.special_weights[2]=1
 assert head(empty,mask)[0,src*129+target]==1
 empty[0,CHANNEL_LAYOUT['cm_my_is_gongb'].start,6,6]=1
 assert head(empty,mask)[0,src*129+target]==0

@pytest.mark.parametrize('kind',['M','R'])
def test_feature_migration_is_neutral_and_new_parameters_learn(kind,tmp_path):
 torch.set_num_threads(1);old,new,state=migrate(START,tmp_path/'initial.pt',kind)
 original=torch.load(START,map_location='cpu',weights_only=False)
 old_names=[n for n,_ in old.named_parameters()];new_names=[n for n,_ in new.named_parameters()]
 for n in old_names:
  for key,value in original['optimizer']['state'][old_names.index(n)].items():
   if isinstance(value,torch.Tensor):
    assert torch.equal(value,state['optimizer']['state'][new_names.index(n)][key]),(n,key)
 obs,g=observe(GameState.new_game(generate_random_setup(random.Random(9216))))
 mask=torch.ones(1,16641,dtype=torch.bool)
 for bf in (False,True):
  with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,enabled=bf):
   a=old._encode(obs,g);b=new._encode(obs,g)
   torch.testing.assert_close(old._policy_logits(a[1],mask,obs),new._policy_logits(b[1],mask,obs),atol=0,rtol=0)
   torch.testing.assert_close(old._value(a[0]),new._value(b[0]),atol=0,rtol=0)
 new.train();cls,cells=new._encode(obs,g);logits=new._policy_logits(cells,mask,obs)
 (logits.square().mean()+new._value(cls).square().mean()).backward()
 extra=[(n,p) for n,p in new.named_parameters() if n not in old.state_dict()]
 assert any(p.grad is not None and p.grad.abs().sum()>0 for _,p in extra)
 assert all(p.grad is None or torch.isfinite(p.grad).all() for _,p in new.named_parameters())
 if kind=='R':assert new.relational_head.weights.grad.abs().sum()>0

def test_old_material_semantics_are_rejected():
 from junqi_rl.networks.junqi_net import JunqiNet,JunqiNetConfig
 from junqi_rl.checkpoint_compat import current_checkpoint_metadata,validate_policy_checkpoint
 net=JunqiNet(JunqiNetConfig(tactical_features=True));meta=current_checkpoint_metadata(net)
 meta['tactical_feature_version']=1
 with pytest.raises(ValueError,match='tactical_feature_version'):
  validate_policy_checkpoint(net,{'policy':net.state_dict(),'checkpoint_meta':meta})
