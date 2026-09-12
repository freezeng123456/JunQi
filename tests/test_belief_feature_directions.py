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
