"""Search privacy, public replay, value orientation, and a bounded CPU probe."""
import random

import numpy as np
import pytest
import torch

from junqi_core.rules import ALL_SEATS, ShowMode, Seat
from junqi_core.setup import generate_random_setup
from junqi_rl.analysis.legacy_joint_search import (BatchedBlueprint, JointSearch, SearchConfig,
    advance_public, magnetic_update, new_world, public_result, sampled_setups, scalar_value, step_world)
from junqi_rl.belief.legacy_joint import build_joint_input, joint_training_targets
from junqi_rl.env import JunqiEnv
from junqi_rl.networks.joint_belief_net import JointBeliefNet, JointBeliefNetConfig
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def fixture():
    env=JunqiEnv(show_mode=ShowMode.DARK,include_global=False); env.reset(20261001)
    soa={k:getattr(env.state,k)[None] for k in
         ('alive','pos_x','pos_y','move_count_arr','active_eat_arr','passive_surv_arr','piece_type_arr')}
    rules=np.stack([env.beliefs[s].to_world_tensor().transpose(1,0,2).reshape(289,12).T for s in ALL_SEATS])[None]
    inp=build_joint_input(soa,rules,np.array([0]),np.array([0]))
    labels=joint_training_targets(soa,np.array([0]),inp)[0].tolist()
    return env,inp.piece_ids[0].tolist(),labels


def test_replay_uses_only_public_events_and_rejects_wrong_broadcast():
    env,ids,labels=fixture(); setup=sampled_setups(env,Seat.SOUTH,ids,labels)
    history=[]
    for _ in range(12):
        result=step_world(env,int(env.legal_action_ids()[0])); history.append(public_result(result))
    shadow=new_world(setup)
    assert advance_public(shadow,history)
    np.testing.assert_array_equal(env.state.cell_piece_id,shadow.state.cell_piece_id)
    bad=list(history); last=list(bad[-1]); last[6]=not last[6]; bad[-1]=tuple(last)
    assert not advance_public(new_world(setup),bad)


def test_counterfactual_setup_does_not_depend_on_other_real_identities():
    env,ids,labels=fixture(); first=sampled_setups(env,Seat.SOUTH,ids,labels)
    # Change every hidden rank in the engine: the supplied joint assignment
    # must remain the sole source of hidden initial identities.
    env.state.piece_type_arr[30:]=-1
    for pos,ref in list(env.state.zero_board.items()):
        if ref.seat!=Seat.SOUTH: del env.state.zero_board[pos]
    assert sampled_setups(env,Seat.SOUTH,ids,labels)==first


def test_magnetic_update_and_value_category_order():
    legal=torch.zeros(1,129*129,dtype=torch.bool); legal[0,[0,1,129]]=True
    prior=torch.full_like(legal,-torch.inf,dtype=torch.float32)
    prior[0,[0,1,129]]=torch.tensor([.2,.3,.5]).log()
    result=magnetic_update(prior,torch.zeros_like(prior),legal,alpha=0,beta=.02)
    torch.testing.assert_close(result[legal],prior[legal])
    magnet=magnetic_update(prior,torch.zeros_like(prior),legal,alpha=1,beta=1e-8).exp()
    torch.testing.assert_close(magnet[0,[0,1,129]],torch.tensor([.25,.25,.5]))
    assert scalar_value(torch.tensor([[.1,.2,.7]]).log()).item()==pytest.approx(.6)


def test_small_search_returns_legal_action_and_reuses_particles():
    env,_,_=fixture()
    policy=JunqiNet(JunqiNetConfig(use_global_summary=False,cnn_channels=8,cnn_layers=1,
         embed_dim=16,n_head=2,depth=1,action_key_dim=8)).eval()
    joint=JointBeliefNet(JointBeliefNetConfig(embed_dim=16,n_head=2,cnn_channels=8,
         cnn_layers=1,n_encoder_layer=1,n_decoder_layer=1)).eval()
    search=JointSearch(BatchedBlueprint(policy,'cpu'),joint,
                      SearchConfig(particles=1,proposal_samples=1,proposal_batches=1,candidates=2,depth=2))
    action=search.action(env,[])
    assert action in env.legal_action_ids()
    assert search.report()['accepted_proposals']==1
    # Reuse at the same information set: no second AR decode.
    search.action(env,[])
    assert search.report()['joint_refreshes']==1


def test_search_advances_public_cache_across_other_players_moves():
    env,_,_=fixture()
    policy=JunqiNet(JunqiNetConfig(use_global_summary=False,cnn_channels=8,cnn_layers=1,
         embed_dim=16,n_head=2,depth=1,action_key_dim=8)).eval()
    joint=JointBeliefNet(JointBeliefNetConfig(embed_dim=16,n_head=2,cnn_channels=8,
         cnn_layers=1,n_encoder_layer=1,n_decoder_layer=1)).eval()
    search=JointSearch(BatchedBlueprint(policy,'cpu'),joint,
                      SearchConfig(particles=1,proposal_samples=1,proposal_batches=1,candidates=2,depth=2))
    history=[]
    for ply in range(64):
        action=search.action(env,history)
        assert action in env.legal_action_ids()
        history.append(public_result(step_world(env,action)))
        if env.state.terminated: break
    assert search.metrics['rollout_world_steps']>0
