import copy
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.networks.ataraxos_block import AtaraxosBlock
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout import RolloutBatch
from junqi_rl.training.ema import PolicyEMA
from junqi_rl.training.advantage_history import AdvantageHistory, history_advantages
from junqi_rl.training.seat_returns import same_seat_returns
from junqi_rl.training.config import TrainConfig, load_yaml_config, _dict_to_dataclass, validate_config


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_reference_contract_rejects_drift():
    cfg=_dict_to_dataclass(TrainConfig,load_yaml_config('configs/v100_ataraxos_reference.yaml'))
    validate_config(cfg)
    assert cfg.ppo.ema_decay==cfg.arr.ppo.ema_decay==.999
    for obj,key,bad in [(cfg.net,'embed_dim',320),(cfg.ppo,'timestep_order','chronological'),
                         (cfg.arr,'reset_reward_window',False),(cfg.env,'num_envs',128)]:
        old=getattr(obj,key);setattr(obj,key,bad)
        with pytest.raises(ValueError,match='reference contract'):validate_config(cfg)
        setattr(obj,key,old)


def test_block_matches_explicit_upstream_attention_and_relu():
    torch.manual_seed(43);block=AtaraxosBlock(16,2).double()
    x=torch.randn(3,7,16,dtype=torch.float64,requires_grad=True)
    z=block.norm1(x)
    q,k,v=[p(z).reshape(3,7,2,8).transpose(1,2) for p in (block.q_proj,block.k_proj,block.v_proj)]
    attn=(q@k.transpose(-1,-2)/8**.5).softmax(-1)@v
    mid=x+block.out_proj(attn.transpose(1,2).reshape_as(x))
    ref=mid+block.linear2(F.relu(block.linear1(block.norm2(mid))))
    actual=block(x)
    torch.testing.assert_close(actual,ref,atol=1e-12,rtol=1e-12)
    ga=torch.autograd.grad(actual.square().sum(),x,retain_graph=True)[0]
    gr=torch.autograd.grad(ref.square().sum(),x)[0]
    torch.testing.assert_close(ga,gr,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('lam',[0.,.5,1.])
def test_parallel_history_trace_matches_independent_seat_recursion(lam):
    gen=torch.Generator().manual_seed(61)
    p=torch.rand(97,7,3,generator=gen);p/=p.sum(-1,keepdim=True)
    seats=torch.randint(4,(97,7),generator=gen)
    valid=torch.rand(97,7,generator=gen)>.1
    dones=(torch.rand(97,7,generator=gen)<.06)&valid
    reward=(torch.randint(3,(97,7),generator=gen)-1).float()*dones
    expected=same_seat_returns(p,reward,dones,seats,valid,gamma=1.,gae_lambda=lam,td_lambda=.8)[0]
    actual=history_advantages(p,reward,dones,seats,valid,lam)
    torch.testing.assert_close(actual,expected,atol=4e-6,rtol=4e-6)
    hist=AdvantageHistory(97);hist.append(values=p,rewards=reward,dones=dones,seats=seats,valid=valid)
    assert hist.threshold(lam,.25,.01)==pytest.approx(max(.01,expected[valid].abs().quantile(.75).item()),abs=1e-6)
    restored=AdvantageHistory(97);restored.load_state_dict(hist.state_dict(),'cpu')
    assert restored.threshold(lam,.25,.01)==hist.threshold(lam,.25,.01)


def tiny_move():
    cfg=JunqiNetConfig(backbone='ataraxos',embed_dim=16,n_head=2,depth=1,action_key_dim=16,
        pos_emb_std=.1,use_global_summary=False,identity_encoding='known_only',use_cat_vf=True)
    torch.manual_seed(71);model=JunqiNet(cfg)
    ppo=PPOConfig(net=cfg,dtype='float32',return_perspective='same_seat',advantage_normalization='raw',
        lr_schedule_unit='rollout',temperature_schedule_unit='rollout',magnet_shape='piece_then_dest',
        temperature_decay=.3,temperature_floor=.001,ema_decay=.999)
    return model,ppo


@pytest.mark.parametrize('carried',[False,True])
def test_chunked_move_matches_one_full_adam_step(carried):
    model,cfg=tiny_move();whole=PPOTrainer(copy.deepcopy(model),cfg)
    chunk=PPOTrainer(copy.deepcopy(model),replace(cfg,train_chunk_size=3))
    n=7;obs=torch.randn(n,317,17,17);glob=torch.empty(n,0)
    legal=torch.zeros(n,16641,dtype=torch.bool);legal[:,[0,1,129,130]]=True
    actions=torch.tensor([0,1,129,130,0,1,129])
    with torch.no_grad():old=model(obs,glob,legal,actions=actions)['log_probs']
    if carried:
        old=old.clone();old[:,[0,1,129,130]]=torch.tensor([.1,.2,.3,.4]).log()
    batch=RolloutBatch(obs,glob,legal,actions,old.gather(1,actions[:,None]).squeeze(1),
        torch.linspace(-.9,1.,n),torch.full((n,3),1/3),torch.full((n,3),1/3),
        torch.ones(n,dtype=torch.bool),behavior_log_probs=old if carried else None)
    a=whole._update_step(batch);b=chunk._update_step(batch)
    for key in ['policy_loss','value_loss','entropy_loss','kl_loss','total_loss']:
        torch.testing.assert_close(a['train/'+key],b['train/'+key],rtol=1e-5,atol=1e-6)
    for pa,pb in zip(whole.policy.parameters(),chunk.policy.parameters()):
        torch.testing.assert_close(pa,pb,atol=3e-6,rtol=1e-5)
    assert whole.num_train_step==chunk.num_train_step==1
    assert all(s['step'].item()==1 for s in chunk.optimizer.state.values())


def test_ema_is_separate_and_restart_exact():
    model=torch.nn.Linear(3,2);ema=PolicyEMA(model,.999)
    before=copy.deepcopy(ema.model.state_dict())
    with torch.no_grad():model.weight.add_(2.)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight,before['weight']*.999+model.weight*.001)
    restored=PolicyEMA(model,.999);restored.load_state_dict(copy.deepcopy(ema.state_dict()))
    with torch.no_grad():model.bias.sub_(1.)
    ema.update(model);restored.update(model)
    for a,b in zip(ema.model.parameters(),restored.model.parameters()):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert ema.updates==restored.updates==2


def test_backbone_has_only_official_input_and_value_token():
    model,_=tiny_move()
    assert isinstance(model.stem,torch.nn.Linear)
    assert 'cls_token' not in dict(model.named_parameters())
    assert model.q_proj.bias is not None and model.k_proj.bias is not None


def test_chunked_setup_matches_full_batch_and_iteration_ema():
    from junqi_rl.networks.arrangement_net import ArrangementNet, ArrangementNetConfig
    from junqi_rl.training.arr_ppo import ArrangementPPOTrainer, ArrangementPPOConfig
    from junqi_rl.arrangement.sampling import generate_arrangements
    from junqi_rl.arrangement.buffer import Batch
    from collections import defaultdict
    torch.manual_seed(12)
    net=ArrangementNet(ArrangementNetConfig(depth=1,embed_dim=16,n_head=2,seat_conditioning=False))
    cfg=ArrangementPPOConfig(batch_size=7,ema_decay=.999)
    a=ArrangementPPOTrainer(copy.deepcopy(net),cfg)
    b=ArrangementPPOTrainer(copy.deepcopy(net),replace(cfg,train_chunk_size=3))
    gen=generate_arrangements(7,net,max_resample=0)
    batch=Batch(gen.samples,gen.seat_idx,gen.log_probs,
        torch.full((7,30,3),1/3),torch.randn(7,30),torch.randn(7,30))
    # Float64 separates algorithmic equivalence from FP32 summation-order
    # noise amplified by Adam on mathematically zero attention-bias gradients.
    a.net.double();b.net.double()
    batch=Batch(*(x.double() if x.is_floating_point() else x for x in
        (batch.arrangements,batch.seat_idx,batch.log_probs,batch.returns,batch.reg_returns,batch.advantages)))
    sa,sb=defaultdict(list),defaultdict(list)
    a._step(batch,sa);b._step(batch,sb)
    for key in ('policy_loss','value_loss','entropy_loss','kl_loss','total_loss'):
        assert sa['arr_train/'+key][0]==pytest.approx(sb['arr_train/'+key][0],rel=1e-5,abs=1e-6)
    for pa,pb in zip(a.net.parameters(),b.net.parameters()):
        torch.testing.assert_close(pa,pb,atol=1e-10,rtol=1e-10)
    assert a._total_batches==b._total_batches==1
    assert a.ema.updates==b.ema.updates==0  # minibatches never update EMA


def test_eval_loads_ema_and_explicit_raw_and_resume_preserves_ema(tmp_path):
    from dataclasses import asdict
    from junqi_rl.training.checkpoint import load_evaluation_policy
    net,cfg=tiny_move();a=PPOTrainer(net,cfg)
    with torch.no_grad():a.policy.value_head.bias.add_(2.)
    a.ema.update(a.policy)
    state=a.state_dict();state['train_cfg']={'net':asdict(net.cfg)}
    path=tmp_path/'weights.pt';torch.save(state,path)
    em=load_evaluation_policy(path);raw=load_evaluation_policy(path,weights='raw')
    torch.testing.assert_close(em.value_head.bias,a.ema.model.value_head.bias)
    torch.testing.assert_close(raw.value_head.bias,a.policy.value_head.bias)
    restored=PPOTrainer(JunqiNet(net.cfg),cfg);restored.load_state_dict(state)
    assert restored.ema.updates==1
    torch.testing.assert_close(restored.ema.model.value_head.bias,a.ema.model.value_head.bias)
