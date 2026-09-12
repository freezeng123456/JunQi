import inspect
import torch
import pytest
from junqi_core.observation import CHANNEL_LAYOUT, OBS_CHANNELS
from junqi_rl.networks.belief_net import BeliefNetConfig
from experiments.belief_features_20260913.features import (
    PublicTrajectory, PublicRelations, FeatureBeliefNet, TEMPORAL_NAMES,
)


def test_public_trajectory_surviving_defender_stays_in_place():
    x=torch.tensor([[2,3]]); y=torch.tensor([[2,2]]); alive=torch.tensor([[True,True]])
    tracker=PublicTrajectory(x,y,alive)
    # Attacker dies at the target; the defender must not inherit its movement.
    tracker.update(torch.tensor([[-1,3]]),torch.tensor([[-1,2]]),torch.tensor([[False,True]]))
    assert tracker.moves[0,1]==0
    assert tracker.unique[0,1]==1
    assert torch.all(tracker.world_features()[0,:,2,2]==0)
    assert torch.isfinite(tracker.world_features()).all()


def test_public_trajectory_long_history_reversal_and_window_expiration():
    x=torch.tensor([[2]]); y=torch.tensor([[2]]); live=torch.tensor([[True]])
    tracker=PublicTrajectory(x,y,live)
    tracker.update(torch.tensor([[5]]),y,live)
    tracker.update(x,y,live)
    assert tracker.moves.item()==2
    assert tracker.long_moves.item()==2
    assert tracker.reversals.item()==1
    assert tracker.unique.item()==2
    assert tracker.revisits.item()==1
    for _ in range(260): tracker.update(x,y,live)
    assert tracker.recent.count_nonzero()==0
    assert tracker.recent_long.count_nonzero()==0
    assert tracker.moves.item()==2
    assert tracker.piece_features().shape==(1,len(TEMPORAL_NAMES),1)
    tracker.reset(x,y,live)
    assert tracker.moves.item()==0
    assert tracker.unique.item()==1


def test_feature_interfaces_exclude_hidden_types_and_labels():
    assert list(inspect.signature(PublicTrajectory.update).parameters)==['self','x','y','alive']
    assert list(inspect.signature(PublicRelations.forward).parameters)==['self','obs']


def test_attack_behavior_scores_only_public_enemy_destinations():
    from junqi_core.board import COMPACT_TO_FLAT
    from experiments.belief_features_20260913.dataset import public_enemy_destinations
    obs=torch.zeros(2,OBS_CHANNELS,17,17)
    flat=int(COMPACT_TO_FLAT[10])
    obs[0,CHANNEL_LAYOUT['piece_left_side_enemy'],flat//17,flat%17]=1
    enemy=public_enemy_destinations(obs)
    assert enemy.shape==(2,len(COMPACT_TO_FLAT))
    assert enemy[0,10] and enemy.sum()==1


def test_relations_proximity_and_own_type_are_observer_only():
    obs=torch.zeros(1,OBS_CHANNELS,17,17)
    own=CHANNEL_LAYOUT['piece_own'].start
    obs[0,own,8,8]=1
    rel=PublicRelations()
    feature=rel(obs)
    assert feature.shape==(1,32,17,17)
    assert feature[0,0,8,9]>feature[0,0,8,10]>feature[0,0,8,11]
    assert feature[0,24,8,8]==1
    assert feature[0,25:28].count_nonzero()==0
    # Changes to an unrelated, public history channel cannot change relations.
    changed=obs.clone(); changed[:,CHANNEL_LAYOUT['move_history']]=.4
    torch.testing.assert_close(rel(changed),feature,atol=0,rtol=0)
    assert torch.isfinite(feature).all()


def test_adapter_starts_with_identical_predictions_then_learns_features():
    cfg=BeliefNetConfig(n_encoder_layer=1,n_head=2,embed_dim=16,cnn_channels=8,cnn_layers=1,ff_factor=2)
    torch.manual_seed(3)
    net=FeatureBeliefNet(cfg).eval()
    obs=torch.randn(2,OBS_CHANNELS,17,17); seats=torch.tensor([0,1])
    extra=torch.randn(2,32,17,17)
    baseline=net(obs,seats,torch.zeros_like(extra))
    candidate=net(obs,seats,extra)
    torch.testing.assert_close(candidate,baseline,rtol=0,atol=0)
    candidate[0,100,3].backward()
    assert net.adapter.weight.grad.abs().sum()>0


def test_loss_respects_hard_support_and_excludes_unlabelled_cells():
    from experiments.belief_features_20260913.train import supervised_loss
    obs=torch.zeros(1,OBS_CHANNELS,17,17)
    obs[0,CHANNEL_LAYOUT['belief_left_side'].start:CHANNEL_LAYOUT['belief_left_side'].start+2,0,0]=.5
    logits=torch.zeros(1,289,12,requires_grad=True)
    labels=torch.full((1,289),-1,dtype=torch.long); labels[0,0]=1
    loss=supervised_loss(logits,obs,labels)
    assert float(loss.detach())==pytest.approx(.69314718056)
    loss.backward()
    assert logits.grad[0,0,1]<0
    assert logits.grad[0,0,2:].count_nonzero()==0
    assert logits.grad[0,1:].count_nonzero()==0


def test_reported_scores_match_hand_computation_and_group_games():
    import math
    from experiments.belief_features_20260913.train import evaluate
    obs=torch.zeros(2,OBS_CHANNELS,17,17)
    start=CHANNEL_LAYOUT['belief_left_side'].start
    obs[:,start,0,0]=.7; obs[:,start+1,0,0]=.3
    labels=torch.full((2,289),-1,dtype=torch.long)
    labels[0,0]=0; labels[1,0]=1
    data=dict(spatial=obs,labels=labels,game_id=torch.tensor([101,102]),step=torch.tensor([0,128]))
    metrics=evaluate(torch.nn.Identity(),data,'temporal','baseline',predictor='rule')
    assert metrics['labels']==2
    assert metrics['nll']==pytest.approx((-math.log(.7)-math.log(.3))/2)
    assert metrics['brier']==pytest.approx(.58)
    assert metrics['accuracy']==.5
    assert metrics['games'][0]['nll']==pytest.approx(-math.log(.7))
    assert metrics['stages']['middle']['accuracy']==0


def test_legal_hidden_swap_is_indistinguishable_to_both_feature_arms():
    from experiments.belief_features_20260913.ambiguity_probe import construct
    proof=construct()
    assert proof['status']=='verified_constructive_ambiguity'
    assert proof['configuration_1']!=proof['configuration_2']


def test_paired_intervals_resample_games_not_their_thousand_repeated_labels():
    from experiments.belief_features_20260913.analyze import paired_bootstrap
    reference={'games':[{'game_id':10,'labels':1000,'nll':2.},
                        {'game_id':11,'labels':1,'nll':2.}]}
    candidate={'games':[{'game_id':10,'labels':1000,'nll':1.},
                        {'game_id':11,'labels':1,'nll':3.}]}
    answer=paired_bootstrap([reference]*3,[candidate]*3,'nll',draws=400)
    assert answer['delta_candidate_minus_reference']==pytest.approx(-999/1001)
    assert answer['equal_game_weight_delta']==0
    assert answer['paired_seed_and_game_ci95']==[-1.,1.]
    assert answer['test_games']==2 and answer['labels_per_seed']==1001


def test_diagnostics_separate_public_movement_and_late_int16_steps():
    import math
    from experiments.belief_features_20260913.diagnostics import score_predictions
    obs=torch.zeros(2,OBS_CHANNELS,17,17)
    move=CHANNEL_LAYOUT['move_bucket'].start+4
    obs[0,move,0,0]=1; obs[1,move+1,0,0]=1
    probs=torch.zeros(2,289,12); probs[:,:,0]=.7; probs[:,:,1]=.3
    labels=torch.full((2,289),-1,dtype=torch.int8); labels[0,0]=0; labels[1,0]=1
    data={'spatial':obs,'labels':labels,'step':torch.tensor([0,512],dtype=torch.int16),
          'game_id':torch.tensor([3,3])}
    result=score_predictions(probs,data)
    assert result['all']['nll']==pytest.approx(-math.log(.21)/2)
    assert result['all']['brier']==pytest.approx(.58)
    assert result['moved']['labels']==result['never_moved']['labels']==result['late']['labels']==1
    assert result['moved']['accuracy']==0 and result['never_moved']['accuracy']==1
    assert result['all']['games'][0]['labels']==2


def test_successful_capture_counts_as_movement_without_a_quiet_move():
    from experiments.belief_features_20260913.diagnostics import slot_and_moves,groups
    obs=torch.zeros(1,OBS_CHANNELS,17,17)
    obs[0,CHANNEL_LAYOUT['move_bucket'].start+4,0,0]=1
    eat=CHANNEL_LAYOUT['active_eat_bucket'].start+4
    obs[0,eat:eat+2,0,0]=1  # cumulative capture counter: exactly one capture
    _,moves=slot_and_moves(obs)
    assert moves[0,0]==1
    labels=torch.full((1,289),-1); labels[0,0]=0
    result=groups({'spatial':obs,'labels':labels,'step':torch.tensor([8])})
    assert result['moved'][0,0] and not result['never_moved'][0,0]


def test_analytic_opening_prior_respects_inventory_and_constrained_slots():
    from junqi_core.info_model import TRACKED_TYPES
    from junqi_core.rules import PieceType,PIECE_COUNTS
    from experiments.belief_features_20260913.opening_prior import exact_marginal_table
    slots,table=exact_marginal_table()
    assert table[slots.index(26),TRACKED_TYPES.index(PieceType.JUNQI)]==.5
    assert table[slots.index(20),TRACKED_TYPES.index(PieceType.DILEI)]==pytest.approx(1/3)
    assert table[slots.index(0),TRACKED_TYPES.index(PieceType.ZHADAN)]==0
    assert table.sum(0).tolist()==pytest.approx([PIECE_COUNTS[t] for t in TRACKED_TYPES])


def test_expanded_loader_uses_requested_training_shards_and_rejects_game_leakage(tmp_path,monkeypatch):
    import gzip
    import json
    from experiments.belief_features_20260913.dataset import sha256
    from experiments.belief_features_20260913.train import load_data,EVAL_FILES
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*args,**kwargs:self)
    def shard(name,game):
        path=tmp_path/(name+'.pt.gz')
        data={'spatial':torch.zeros(1,OBS_CHANNELS,17,17,dtype=torch.float16),
            'temporal':torch.zeros(1,32,17,17,dtype=torch.float16),
            'labels':torch.full((1,289),-1,dtype=torch.int8),'seat':torch.zeros(1,dtype=torch.int8),
            'game_id':torch.tensor([game]),'step':torch.zeros(1,dtype=torch.int16),'meta':{'name':name}}
        with gzip.open(path,'wb') as f: torch.save(data,f)
        path.with_suffix('.json').write_text(json.dumps({'sha256':sha256(path)}))
    names=('custom0','custom1',*EVAL_FILES)
    for i,name in enumerate(names): shard(name,100+i)
    data,manifest=load_data(tmp_path,'temporal',train_files=('custom0','custom1'))
    assert data['train']['game_id'].tolist()==[100,101]
    assert set(manifest)==set(names)
    shard('test',100)
    with pytest.raises(ValueError,match='Game leakage'):
        load_data(tmp_path,'temporal',train_files=('custom0','custom1'))
