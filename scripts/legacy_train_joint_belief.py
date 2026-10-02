"""LEGACY probability-input autoregressive belief training for DARK JunQi."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import time

import numpy as np
import torch

from junqi_rl.belief.legacy_joint import JointBeliefInput, build_joint_input, joint_training_targets
from junqi_rl.belief.joint_checkpoint import INPUT_SPEC
from junqi_rl.networks.joint_belief_net import JointBeliefNet, JointBeliefNetConfig
from junqi_rl.training.checkpoint import load_evaluation_policy
from junqi_rl.training._checkpoint_io import atomic_write


def gather(inputs, indices, device):
    return JointBeliefInput(**{f.name: getattr(inputs, f.name)[indices].to(device)
                              for f in fields(inputs)})


@torch.no_grad()
def collect(policy, *, rows, seed, device, num_envs=8):
    if policy.identity_encoding != "legacy_probabilities":
        raise ValueError("Legacy joint v1 collector cannot consume a facts-only policy")
    from junqi_rl.gpu_rollout import GpuRollout
    world=GpuRollout(num_envs,include_global=False)
    world.reset(seed_base=seed)
    observations, seats, targets, metadata, folds = [],[],[],[],[]
    start=time.perf_counter()
    for step in range(rows*16):
        if step % 16 == 0:
            env=(step//16)%num_envs
            all_obs,_=world.build_all_seat_observations_torch()
            obs=all_obs[env].detach().to('cpu',dtype=torch.float16).clone()
            rules=world.rule_beliefs_torch().detach().cpu().numpy()
            soa=world.state.copy_to_host()
            observers=np.arange(4,dtype=np.int64)
            env_ids=np.full(4,env,dtype=np.int64)
            inp=build_joint_input(soa,rules,env_ids,observers)
            label=joint_training_targets(soa,env_ids,inp)
            observations.append(obs); seats.append(torch.from_numpy(observers))
            targets.append(label); metadata.append(inp); folds.extend([env%4==0]*4)
            if len(folds)>=rows:
                break
        acting=world.turn_torch().clone()
        obs,global_=world.build_acting_seat_observation_torch(acting)
        legal=world.legal_mask_canonical_torch_device(acting)
        actions,_,_=policy.act(obs,global_,legal)
        result=world.step_device_torch(actions.to(torch.int32),acting)
        world.update_beliefs_device(result,acting)
        world.reset_terminated_device(seed=seed+step+1)
    inputs=JointBeliefInput(**{f.name:torch.cat([getattr(m,f.name) for m in metadata])[:rows]
                              for f in fields(JointBeliefInput)})
    return (torch.cat(observations)[:rows],torch.cat(seats)[:rows],torch.cat(targets)[:rows],
            inputs,torch.tensor(folds[:rows]),{'rows':rows,'collection_steps':step,
            'environment_steps':step*num_envs,'collection_seconds':time.perf_counter()-start,
            'split':'environment-disjoint; all four observers remain in same fold'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--rows',type=int,default=1024)
    parser.add_argument('--updates',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=20261002)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--resume',default='')
    args=parser.parse_args()
    torch.set_num_threads(2); torch.manual_seed(args.seed); np.random.seed(args.seed)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    policy=load_evaluation_policy(args.policy,device=device)
    for p in policy.parameters(): p.requires_grad_(False)
    parent_sha=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    obs,seats,labels,inputs,validation,report=collect(policy,rows=args.rows,seed=args.seed,device=device)
    del policy
    cfg=JointBeliefNetConfig(n_encoder_layer=2,n_decoder_layer=2,embed_dim=64,n_head=4,
                           ff_factor=2,cnn_channels=32,cnn_layers=2,dropout=0.0)
    net=JointBeliefNet(cfg).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=5e-5)
    start_update=0
    torch_rng=None
    cuda_rng=None
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False)
        if (saved['architecture_id']!=net.architecture_id or saved['parent_policy_sha256']!=parent_sha
                or saved.get('input_spec')!=INPUT_SPEC or saved['config']!=asdict(cfg)
                or saved['seed']!=args.seed or saved['report']['rows']!=args.rows):
            raise ValueError('Joint checkpoint architecture/parent policy mismatch')
        net.load_state_dict(saved['net'],strict=True); optimizer.load_state_dict(saved['optimizer'])
        start_update=saved['updates']
        torch_rng=saved['torch_rng'].cpu()
        cuda_rng=saved.get('cuda_rng')
    training=(~validation).nonzero().flatten()
    heldout=validation.nonzero().flatten()
    if args.updates<=start_update: raise ValueError('Resume must advance optimizer updates')
    if not len(training) or not len(heldout): raise ValueError('Empty train/validation split')

    def logits(indices):
        batch=gather(inputs,indices,device)
        return net.teacher_logits(obs[indices].to(device,dtype=torch.float32),seats[indices].to(device),
                                 targets=labels[indices].to(device),**batch.kwargs())

    @torch.no_grad()
    def evaluate():
        net.eval(); total=correct=count=0.0
        for indices in heldout.split(args.batch_size):
            output=logits(indices)
            truth=labels[indices].to(device)
            unknown=inputs.support[indices].sum(-1).to(device)>1
            nll=-output.log_softmax(-1).gather(-1,truth[...,None]).squeeze(-1)
            total+=float(nll[unknown].sum()); count+=int(unknown.sum())
            correct+=int(((output.argmax(-1)==truth)&unknown).sum())
        return {'conditional_nll_per_unknown_piece':total/max(count,1),
                'teacher_forced_accuracy':correct/max(count,1),'unknown_targets':int(count)}

    report.update(parent_policy_sha256=parent_sha,architecture_id=net.architecture_id,
                  config=asdict(cfg),seed=args.seed,parameters=net.num_parameters(),
                  initial_validation=evaluate(),train_rows=len(training),validation_rows=len(heldout))
    if torch_rng is not None: torch.set_rng_state(torch_rng)
    if cuda_rng is not None and device.type=='cuda': torch.cuda.set_rng_state_all([v.cpu() for v in cuda_rng])
    train_start=time.perf_counter()
    for update in range(start_update,args.updates):
        net.train()
        indices=training[torch.randint(len(training),(args.batch_size,))]
        output=logits(indices)
        truth=labels[indices].to(device)
        loss=-output.log_softmax(-1).gather(-1,truth[...,None]).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite joint NLL')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(net.parameters(),0.5)
        if not torch.isfinite(grad): raise FloatingPointError('Nonfinite joint gradients')
        optimizer.step()
        if (update+1)%64==0:
            print(json.dumps({'update':update+1,'train_nll':float(loss.detach()),
                              'validation':evaluate()}),flush=True)
    report.update(updates=args.updates,final_validation=evaluate(),
                  training_seconds=time.perf_counter()-train_start)
    net.eval()
    idx=heldout[:1]; sample_input=gather(inputs,idx,device)
    if device.type=='cuda': torch.cuda.synchronize()
    begin=time.perf_counter()
    samples=net.sample(obs[idx].to(device,dtype=torch.float32),seats[idx].to(device),
                       n_samples=16,**sample_input.kwargs())
    if device.type=='cuda': torch.cuda.synchronize()
    report.update(sample_seconds=time.perf_counter()-begin,sample_count=16,
                  sampled_assignment_shape=list(samples.shape),
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
    state={'architecture_id':net.architecture_id,'config':asdict(cfg),'net':net.state_dict(),
           'optimizer':optimizer.state_dict(),'updates':args.updates,
           'parent_policy_sha256':parent_sha,'seed':args.seed,'report':report,
           'input_spec':INPUT_SPEC,'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
    assert all(torch.isfinite(v).all() for v in state['net'].values())
    atomic_write(str(out/'joint-belief.pt'),lambda handle:torch.save(state,handle))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
