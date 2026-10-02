"""Causality, incremental-decoder parity, inventory feasibility, and privacy."""
import itertools

import numpy as np
import pytest
import torch

from junqi_core.rules import ALL_SEATS, ShowMode
from junqi_rl.belief.legacy_joint import build_joint_input, joint_training_targets
from junqi_rl.env import JunqiEnv
from junqi_rl.networks.joint_belief_net import JointBeliefNet, JointBeliefNetConfig


def model():
    torch.manual_seed(4)
    return JointBeliefNet(JointBeliefNetConfig(n_encoder_layer=1,n_decoder_layer=1,
        embed_dim=16,n_head=2,ff_factor=2,cnn_channels=8,cnn_layers=1)).eval()


def test_causal_prefix_and_cached_decoder_agree():
    net=model()
    memory=torch.randn(2,7,16)
    context=torch.randn(2,5,16)
    targets=torch.randint(12,(2,5))
    with torch.no_grad():
        parallel=net.decode(memory,context,targets)
        changed=targets.clone(); changed[:,2:]=(changed[:,2:]+1)%12
        assert torch.allclose(parallel[:,:3],net.decode(memory,context,changed)[:,:3])
        caches=[(torch.empty(2,2,5,8),torch.empty(2,2,5,8))]
        mem=[b.memory(memory) for b in net.decoder]
        actual=[]
        for i in range(5):
            prev=torch.full((2,),12) if i==0 else targets[:,i-1]
            x=context[:,i:i+1]+net.type_embedding(prev)[:,None]
            for block,kv,cache in zip(net.decoder,mem,caches):
                x=block.step(x,kv,cache,i)
            actual.append(net.head(net.decoder_norm(x)))
        torch.testing.assert_close(parallel,torch.cat(actual,1),atol=2e-6,rtol=2e-6)


def test_feasibility_matches_enumerated_small_problem():
    net=model()
    # Choosing type 0 first strands the two later pieces; only type 2 works.
    support=torch.zeros(1,3,12,dtype=torch.bool)
    support[0,0,:3]=True; support[0,1,0]=True; support[0,2,1]=True
    owners=torch.ones(1,3,dtype=torch.long)
    counts=torch.zeros(1,12,dtype=torch.long); counts[0,:3]=1
    allowed=support[:,0]&net.feasible_types(counts,net.future_demand(support,owners)[:,0])
    completions=[p for p in itertools.permutations(range(3))
                 if all(support[0,i,t] for i,t in enumerate(p))]
    assert set(allowed[0].nonzero().flatten().tolist())=={p[0] for p in completions}=={2}


def test_joint_sampling_preserves_counts_and_support():
    net=model()
    obs=torch.randn(1,317,17,17); seats=torch.zeros(1,dtype=torch.long)
    cells=torch.tensor([[193,194]]); owners=torch.ones(1,2,dtype=torch.long)
    features=torch.zeros(1,2,4)
    support=torch.zeros(1,2,12,dtype=torch.bool); support[:,:,:2]=True
    inventory=torch.zeros(1,4,12,dtype=torch.long); inventory[:,1,:2]=1
    samples=net.sample(obs,seats,cells,owners,features,support,inventory,n_samples=32)
    assert samples.shape==(1,32,2)
    assert (samples.sort(-1).values==torch.tensor([0,1])).all()


def public_fixture():
    env=JunqiEnv(show_mode=ShowMode.DARK,include_global=False)
    env.reset(20261001); state=env.state
    names=('alive','pos_x','pos_y','move_count_arr','active_eat_arr','passive_surv_arr','piece_type_arr')
    soa={name:getattr(state,name)[None].copy() for name in names}
    rules=np.stack([env.beliefs[s].to_world_tensor().transpose(1,0,2).reshape(289,12).T
                    for s in ALL_SEATS])[None]
    return soa,rules


def test_public_metadata_does_not_read_hidden_types():
    soa,rules=public_fixture()
    first=build_joint_input(soa,rules,np.array([0]),np.array([0]))
    targets=joint_training_targets(soa,np.array([0]),first)
    assert first.support.gather(-1,targets[...,None]).all()
    soa.pop('piece_type_arr')
    other=build_joint_input(soa,rules,np.array([0]),np.array([0]))
    for key,value in first.kwargs().items():
        assert torch.equal(value,other.kwargs()[key])
    assert first.piece_ids.shape==(1,75)  # includes unknown teammate


def test_full_army_teacher_loss_is_finite_and_trainable():
    soa,rules=public_fixture()
    inputs=build_joint_input(soa,rules,np.array([0]),np.array([0]))
    targets=joint_training_targets(soa,np.array([0]),inputs)
    net=model().train()
    logits=net.teacher_logits(torch.randn(1,317,17,17),torch.tensor([0]),targets=targets,
                             **inputs.kwargs())
    loss=-logits.log_softmax(-1).gather(-1,targets[...,None]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert net.type_embedding.weight.grad.norm()>0
    assert any(p.grad is not None and p.grad.norm()>0 for p in net.cnn.parameters())


def test_inconsistent_constraints_fail_explicitly():
    net=model(); support=torch.zeros(1,2,12,dtype=torch.bool); support[:,:,0]=True
    inv=torch.zeros(1,4,12,dtype=torch.long); inv[:,1,:2]=1
    with pytest.raises(ValueError,match='complete assignment'):
        net.sample(torch.randn(1,317,17,17),torch.tensor([0]),torch.tensor([[193,194]]),
                   torch.ones(1,2,dtype=torch.long),torch.zeros(1,2,4),support,inv,n_samples=1)


def test_joint_checkpoint_contract_and_parent_policy(tmp_path):
    from dataclasses import asdict
    import hashlib
    from junqi_rl.belief.joint_checkpoint import INPUT_SPEC, load_joint_belief
    net=model(); policy=tmp_path/'policy.pt'; policy.write_bytes(b'parent-policy')
    state={'architecture_id':net.architecture_id,'config':asdict(net.cfg),
           'input_spec':INPUT_SPEC,'net':net.state_dict(),
           'parent_policy_sha256':hashlib.sha256(policy.read_bytes()).hexdigest()}
    checkpoint=tmp_path/'joint.pt'; torch.save(state,checkpoint)
    loaded=load_joint_belief(checkpoint,policy_path=policy)
    for key,value in net.state_dict().items(): torch.testing.assert_close(loaded.state_dict()[key],value)
    policy.write_bytes(b'other-policy')
    with pytest.raises(ValueError,match='parent policy mismatch'):
        load_joint_belief(checkpoint,policy_path=policy)
