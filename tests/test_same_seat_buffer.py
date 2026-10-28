"""CPU integration of the production buffer's prefix, targets and minibatches."""
import copy
from types import SimpleNamespace
import pytest
import torch
from junqi_rl.training.rollout_gpu import RolloutBufferGPU, OBS_CHANNELS, BOARD_SIZE, FLAT_ACTION_DIM
from junqi_rl.training.seat_pending import PendingSeatTransitions
from junqi_rl.training.config import TrainConfig, validate_config
from junqi_rl.training.game_progress import runtime_contract


def cpu_buffer(compact=False):
    b = RolloutBufferGPU.__new__(RolloutBufferGPU)
    b.num_envs, b.steps_per_env, b.storage_steps, b.prefix_steps = 1, 5, 9, 4
    b.same_seat=True; b.return_perspective="same_seat"; b.advantage_normalization="raw"
    b.device=torch.device("cpu"); b.gamma=1.; b.gae_lambda=.5; b.td_lambda=.8
    b.random_opponent=False; b.train_value_on_random_seats=False
    b.obs_global_dims=0; b.obs_storage_dtype=torch.float16
    b.uses_compact_history=compact; b.csr_legal_mask=False
    b.adv_filter_scope="rollout"; b.value_sample_scope="policy"; b.minibatch_group="timestep"
    b.adv_filt_rate=1.; b.adv_filt_thresh=0.
    b.actions=torch.zeros(9,1,dtype=torch.int32)
    b.seats=torch.zeros(9,1,dtype=torch.int8)
    for name in ("log_probs","values","rewards","returns_","advantages_"):
        setattr(b,name,torch.zeros(9,1))
    b.dones=torch.zeros(9,1,dtype=torch.bool)
    b.input_valid=torch.ones_like(b.dones); b.ready_mask=torch.ones_like(b.dones)
    b.obs_spatial=torch.zeros(9,1,OBS_CHANNELS,BOARD_SIZE,BOARD_SIZE,dtype=torch.float16)
    b.obs_global=torch.empty(9,1,0,dtype=torch.float16)
    b.legal_mask=torch.zeros(9,1,FLAT_ACTION_DIM,dtype=torch.bool)
    if compact:
        # Independent history fixture: only the five newly collected states.
        history_sp=b.obs_spatial[4:].clone();history_gl=b.obs_global[4:].clone();history_lm=b.legal_mask[4:].clone()
        def reconstruct(idx,seats,dtype):
            assert (idx>=0).all() and (idx<5).all()
            return history_sp[:,0][idx].to(dtype),history_gl[:,0][idx].to(dtype),history_lm[:,0][idx]
        b.history=SimpleNamespace(reconstruct=reconstruct,history_bytes=0,sp=history_sp,lm=history_lm)
    b.pending=PendingSeatTransitions(1,"cpu",obs_global_dims=0); b._carried=None
    return b


def add(b, seat, action, value, terminal=False, reward=0.):
    lm=torch.zeros(1,FLAT_ACTION_DIM,dtype=torch.bool); lm[0,action]=True
    behavior=torch.full((1,FLAT_ACTION_DIM),-torch.inf);behavior[0,action]=0.
    sp=torch.full((1,OBS_CHANNELS,BOARD_SIZE,BOARD_SIZE),float(action))
    data=dict(obs_spatial=sp,obs_global=torch.empty(1,0),legal_mask=lm,
        actions=torch.tensor([action],dtype=torch.int32),log_probs=torch.zeros(1),
        values=torch.tensor([value]),seats=torch.tensor([seat],dtype=torch.int8))
    if b.uses_compact_history:
        t=b._ptr-4;b.history.sp[t]=sp;b.history.lm[t]=lm
    b.pending.record(**data,behavior_log_probs=behavior)
    b.add(**data,rewards=torch.tensor([reward]),dones=torch.tensor([terminal]))
    b.pending.terminal(torch.tensor([terminal]))


@pytest.mark.parametrize("compact",[False,True])
def test_pending_action_closes_once_and_keeps_old_observation_and_policy(compact):
    b=cpu_buffer(compact);b.reset()
    for seat,action,value in [(0,10,.1),(1,11,.2),(2,12,.3),(3,13,.4),(0,14,.5)]:
        add(b,seat,action,value)
    b.compute_returns()
    first=list(b.minibatches(8,shuffle=False))
    assert [int(x.actions[0]) for x in first]==[10]
    assert first[0].advantages.item()==pytest.approx(.4)
    assert first[0].returns.item()==pytest.approx(.5)
    b.reset()
    # Seat 1 never acts again; seat 0's teammate wins at the third action.
    for seat,action,value,done,reward in [(2,20,.6,False,0.),(3,21,.7,False,0.),
            (0,22,.8,True,1.),(0,30,-.1,False,0.),(1,31,-.2,False,0.)]:
        add(b,seat,action,value,done,reward)
    b.compute_returns();batches=list(b.minibatches(8,shuffle=False))
    actions=[int(x.actions[0]) for x in batches]
    assert set(actions)=={11,12,13,14,20,21,22} and len(actions)==7
    assert 10 not in actions and 30 not in actions and 31 not in actions
    for x in batches:
        action=int(x.actions[0]);assert x.obs_spatial[0,0,0,0].item()==action
        if action<20:
            assert x.behavior_log_probs is not None
            assert x.behavior_log_probs[0,action].item()==0.
        if action==11:
            assert x.returns.item()==-1.  # eliminated seat gets final team loss
        if action==22:assert x.returns.item()==1.
    assert b.pending.valid[:,0].tolist()==[True,True,False,False]


def test_runtime_contract_records_learning_target_semantics():
    cfg=TrainConfig();legacy=runtime_contract(cfg)
    cfg.ppo.return_perspective="same_seat";cfg.ppo.advantage_normalization="raw"
    contract=runtime_contract(cfg)
    assert contract!=legacy and contract["return_learning"]["gae_lambda"]==.5


def test_same_seat_config_rejects_silent_cpu_or_shaped_fallback():
    cfg=TrainConfig();cfg.ppo.return_perspective="same_seat"
    with pytest.raises(ValueError,match="same_seat"):
        validate_config(cfg)
