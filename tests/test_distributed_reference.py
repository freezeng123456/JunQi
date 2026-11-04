"""Real two-process reference updates vs the unpartitioned logical batch."""
import copy
from collections import defaultdict
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from junqi_rl.training import distributed_reference as sync
from junqi_rl.training.config import TrainConfig, load_yaml_config, _dict_to_dataclass, validate_config
from junqi_rl.training.ppo import PPOConfig, PPOTrainer
from junqi_rl.training.rollout import RolloutBatch
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.arr_ppo import ArrangementPPOConfig, ArrangementPPOTrainer
from junqi_rl.networks.arrangement_net import ArrangementNet, ArrangementNetConfig
from junqi_rl.arrangement.buffer import Batch
from junqi_rl.arrangement.sampling import generate_arrangements


def _select(batch, indices):
    return type(batch)(**{f.name: (getattr(batch,f.name)[indices] if isinstance(getattr(batch,f.name),torch.Tensor)
                                else getattr(batch,f.name)) for f in fields(batch)})


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.manual_seed(77)
    netcfg=JunqiNetConfig(backbone='ataraxos',embed_dim=16,n_head=2,depth=1,action_key_dim=16,
        use_global_summary=False,identity_encoding='known_only',use_cat_vf=True)
    cfg=PPOConfig(net=netcfg,dtype='float32',train_chunk_size=2,return_perspective='same_seat',
        advantage_normalization='raw',magnet_shape='piece_then_dest',ema_decay=.999)
    model=JunqiNet(netcfg)
    oracle=PPOTrainer(copy.deepcopy(model),cfg)
    # SGD makes the exact aggregated gradient observable, including tiny attention-bias gradients.
    oracle.optimizer=torch.optim.SGD(oracle.policy.parameters(),lr=.01,momentum=.9)
    n=7;obs=torch.randn(n,317,17,17);glob=torch.empty(n,0)
    legal=torch.zeros(n,16641,dtype=torch.bool);legal[:,[0,1,129,130]]=True
    actions=torch.tensor([0,1,129,130,0,1,129])
    with torch.no_grad():old=model(obs,glob,legal,actions=actions)['log_probs']
    batch=RolloutBatch(obs,glob,legal,actions,old.gather(1,actions[:,None]).squeeze(1),
        torch.linspace(-.9,1,n),torch.full((n,3),1/3),torch.full((n,3),1/3),
        torch.ones(n,dtype=torch.bool),behavior_log_probs=old)
    expected=[]
    for _ in range(3):
        metrics=oracle._update_step(batch)
        expected.append((copy.deepcopy(oracle.policy.state_dict()),metrics))
    setup=ArrangementNet(ArrangementNetConfig(depth=1,embed_dim=16,n_head=2,seat_conditioning=False))
    scfg=ArrangementPPOConfig(batch_size=2,train_chunk_size=2,ema_decay=.999)
    arr_oracle=ArrangementPPOTrainer(copy.deepcopy(setup),scfg)
    generated=generate_arrangements(8,setup,max_resample=0)
    sbatch=Batch(generated.samples,generated.seat_idx,generated.log_probs,
        torch.full((8,30,3),1/3),torch.randn(8,30),torch.randn(8,30))
    arr_oracle.optim=torch.optim.SGD(arr_oracle.net.parameters(),lr=.01,momentum=.9)
    astats=defaultdict(list);arr_oracle._step(sbatch,astats)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2,timeout=timedelta(seconds=90))
    try:
        # Deliberately different initial weights test initial synchronization, including EMA.
        torch.manual_seed(77+rank)
        learner=PPOTrainer(copy.deepcopy(model) if rank==0 else JunqiNet(netcfg),cfg)
        learner.optimizer=torch.optim.SGD(learner.policy.parameters(),lr=.01,momentum=.9)
        for step,split in enumerate((1,0,5)):
            indices=slice(0,split) if rank==0 else slice(split,n)
            actual=learner._update_step(_select(batch,indices))
            weights,metric=expected[step]
            for k,v in learner.policy.state_dict().items():
                torch.testing.assert_close(v,weights[k],atol=3e-7,rtol=2e-5)
            for k in ('policy_loss','value_loss','total_loss','g_norm'):
                torch.testing.assert_close(actual['train/'+k],metric['train/'+k],atol=2e-6,rtol=2e-5)
            assert actual['train/batch_size']==7 and learner.num_train_step==step+1
        arr=ArrangementPPOTrainer(copy.deepcopy(setup),scfg)
        arr.optim=torch.optim.SGD(arr.net.parameters(),lr=.01,momentum=.9)
        stats=defaultdict(list)
        arr._step(_select(sbatch,slice(0,1) if rank==0 else slice(1,None)),stats)
        for a,b in zip(arr.net.parameters(),arr_oracle.net.parameters()):
            torch.testing.assert_close(a,b,atol=3e-7,rtol=2e-5)
        assert stats['arr_train/total_loss'][0]==pytest.approx(astats['arr_train/total_loss'][0],rel=2e-5)
        # Exact global quantile, ragged gathering, and common row ordering.
        values=torch.tensor([1.,9.,3.]) if rank==0 else torch.tensor([2.,8.])
        assert sync.global_quantile(values,.75,.01)==8.
        order=sync.common_order(np.array([2,0,1]) if rank==0 else np.array([1,2,0]),'cpu')
        assert order.tolist()==[2,0,1]
        # Every global ready setup appears once per epoch, even with an empty rank.
        count=0 if rank==0 else 7
        data=torch.arange(count,dtype=torch.float32)[:,None]
        buffer=SimpleNamespace(device=torch.device('cpu'),ready_flags=torch.ones(count,dtype=torch.bool),
            arrangements=data,seat_idx=data[:,0].long(),log_probs=data,val_est=data,reg_val_est=data,adv_est=data)
        seen=[]
        def step(b, stats):
            seen.extend(b.arrangements[:,0].tolist());stats['arr_train/n_batches'].append(1.)
            return True
        fake=SimpleNamespace(cfg=SimpleNamespace(batch_size=2,num_epoch_per_train=2),
            _net_for_train=torch.nn.Linear(1,1),_step=step,ema=None,_grad_nan_skip_count=0)
        result=sync.setup_epoch(fake,buffer)
        assert result['arr_train/n_batches']==4
        assert sorted(x for rows in sync.objects(seen) for x in rows)==sorted(list(range(7))*2)
        # Durable rank-specific runtime and exact RNG round trip.
        from junqi_rl.training.checkpoint import save_checkpoint
        from junqi_rl.training.distributed_checkpoint import load_runtime
        from junqi_rl.training.game_progress import capture_rng, restore_rng
        rng=np.random.default_rng(rank+3)
        learner.runtime_state_provider=lambda:dict(games={'completed_games':rank+1},rng=capture_rng(rng))
        config=TrainConfig();config.ppo=cfg;config.net=netcfg;config.checkpoint_keep=1
        path=save_checkpoint(learner,config,1,directory,arr_trainer=arr)
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['runtime_state']['games']['completed_games']==3
        runtime=load_runtime(path,state['runtime_state'])
        assert runtime['games']['completed_games']==rank+1
        expected_rng=rng.random(4);restore_rng(runtime['rng'],rng)
        np.testing.assert_array_equal(rng.random(4),expected_rng)
        with pytest.raises(ValueError,match='same world size'):
            load_runtime(path,dict(state['runtime_state'],world_size=4))
        save_checkpoint(learner,config,2,directory,arr_trainer=arr)
        assert not Path(path).exists()
        assert not list(Path(directory).glob('ckpt_000001.rank*.runtime.pt'))
        # One rank's nonfinite loss must fail on every rank before any optimizer step.
        with pytest.raises(FloatingPointError,match='intentional'):
            sync.fail_together(rank==1,'cpu','intentional nonfinite')
    finally:
        dist.destroy_process_group()


def test_reference_two_ranks_match_global_batch_and_resume(tmp_path):
    mp.spawn(_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'checkpoints')),nprocs=2,join=True)


def test_reference_distributed_preserves_global_configuration():
    cfg=_dict_to_dataclass(TrainConfig,load_yaml_config('configs/v100_ataraxos_reference_8gpu.yaml'))
    validate_config(cfg)
    assert cfg.env.num_envs*cfg.distributed_world_size==1536
    assert cfg.arr.n_arr*cfg.distributed_world_size==4000
    assert cfg.arr.ppo.batch_size*cfg.distributed_world_size==1024
    assert cfg.target_completed_games==1_000_000
    cfg.env.num_envs=1536
    with pytest.raises(ValueError,match='reference contract'):
        validate_config(cfg)
