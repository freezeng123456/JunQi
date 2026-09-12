"""Bounded, paired self-play pilot for belief/policy co-training.

Both arms train BeliefNet. Only the guarded arm publishes it to the policy.
Version-mismatched policy weights require an explicit warm-start flag; no
optimizer, schedule counter, belief state, or old experiment is resumed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from junqi_core.batched_state import BatchedGameState
from junqi_core.board import FLAT_TO_COMPACT
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import CHANNEL_LAYOUT, ObservationBuilder
from junqi_core.rotation import world_to_canonical
from junqi_core.rules import Seat, ShowMode
from junqi_core.setup import generate_random_setup
from junqi_core.state import GameState
from junqi_rl.belief.buffer import BeliefBuffer
from junqi_rl.belief.inference import (
    constrain_neural_beliefs, guarded_refresh_beliefs_neural, _live_enemy_world_mask,
)
from junqi_rl.belief.sampling import MidgameBeliefSampler, current_hidden_labels
from junqi_rl.checkpoint_compat import OBSERVATION_SEMANTICS_VERSION
from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig
from junqi_rl.networks.junqi_net import JunqiNet
from junqi_rl.training.belief_ppo import BeliefPPOConfig, BeliefPPOTrainer
from junqi_rl.training.ppo import PPOTrainer


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def make_probe(path):
    fields = {key: [] for key in ('spatial', 'global_', 'rules', 'labels', 'mask', 'legal')}
    for seed in range(910000, 910008):
        rng = random.Random(seed)
        state = GameState.new_game(generate_random_setup(rng), show_mode=ShowMode.DARK)
        beliefs = {seat: BeliefTensor.initial(state, seat) for seat in Seat}
        for step in range(129):
            if step in (0, 16, 64, 128):
                spatial, global_, legal = [], [], []
                rules = np.zeros((1, 4, 12, 289), dtype=np.float32)
                for seat in Seat:
                    obs = ObservationBuilder().build(state, beliefs[seat], seat)
                    spatial.append(obs.spatial.copy()); global_.append(obs.global_.copy())
                    legal_mask = np.zeros(16641, dtype=bool)
                    for action in state.legal_actions(seat):
                        sx, sy = world_to_canonical(*action.src, seat)
                        dx, dy = world_to_canonical(*action.dst, seat)
                        legal_mask[int(FLAT_TO_COMPACT[sy*17+sx])*129 + int(FLAT_TO_COMPACT[dy*17+dx])] = True
                    legal.append(legal_mask)
                    for (x,y), vector in beliefs[seat].probs.items():
                        rules[0,seat.value,:,y*17+x] = vector
                batch = BatchedGameState.from_game_states([state])
                soa = {name: getattr(batch,name) for name in
                       ('piece_seat_arr','piece_type_arr','alive','pos_x','pos_y')}
                labels, mask = current_hidden_labels(soa, rules, np.array([0]))
                for key, value in dict(spatial=np.array(spatial,dtype=np.float16),
                        global_=np.array(global_), rules=rules[0], labels=labels.astype(np.int8),
                        mask=mask, legal=np.array(legal)).items():
                    fields[key].append(value)
            if state.terminated:
                break
            actions = state.legal_actions()
            captures = [a for a in actions if a.dst in state.pieces]
            action = rng.choice(captures if captures else actions)
            after, event = state.step(action)
            for belief in beliefs.values():
                belief.update(state, after, event)
            state = after
    np.savez_compressed(path, **{key: np.stack(value) for key,value in fields.items()})
    print(json.dumps({'probe_positions':len(fields['spatial']),
                      'observer_rows':4*len(fields['spatial']),
                      'labels':int(np.stack(fields['mask']).sum())}), flush=True)


@torch.no_grad()
def probe_metrics(belief, policy, data, weight):
    spatial, global_, rules = data['spatial'], data['global_'], data['rules']
    n = len(spatial)
    inputs = spatial.flatten(0,1)
    seats = torch.arange(4,device=inputs.device).repeat(n)
    belief.eval(); policy.eval()
    logits = torch.cat([belief(inputs[i:i+32],seat_idx=seats[i:i+32])['logits']
                        for i in range(0,len(inputs),32)])
    probs = logits.float().softmax(-1).reshape(n,4,289,12)
    live = _live_enemy_world_mask(spatial)
    predicted = constrain_neural_beliefs(probs,rules,live,neural_weight=1)
    mixed = constrain_neural_beliefs(probs,rules,live,neural_weight=weight,max_kl=.05)
    mask = data['mask'].bool()
    labels = data['labels'].long()[mask]
    result = {}
    for name, world in [('rule',rules),('neural',predicted),('mixed',mixed)]:
        rows = world.transpose(-1,-2)[mask]
        ptrue = rows.gather(-1,labels[:,None]).squeeze(-1)
        target = torch.nn.functional.one_hot(labels,12).float()
        confidence, argmax = rows.max(-1)
        result[name+'_nll'] = float(-ptrue.clamp_min(1e-12).log().mean())
        result[name+'_brier'] = float(((rows-target)**2).sum(-1).mean())
        result[name+'_accuracy'] = float((argmax==labels).float().mean())
        result[name+'_wrong_confident_fraction'] = float(((argmax!=labels)&(confidence>.9)).float().mean())
    result['valid_labels'] = int(mask.sum())
    policy_input = spatial.clone()
    for seat,k in enumerate((0,1,2,-1)):
        canonical = torch.rot90(mixed[:,seat].reshape(n,12,17,17),k,(-2,-1))
        for name in ('belief_left_side','belief_right_side'):
            group = CHANNEL_LAYOUT[name]
            occupied = spatial[:,seat,group].sum(1,keepdim=True)>0
            policy_input[:,seat,group] = canonical*occupied
    updated_global = global_.clone()
    for offset,name in ((0,'belief_left_side'),(12,'belief_right_side')):
        updated_global[:,:,offset:offset+12] = policy_input[:,:,CHANNEL_LAYOUT[name]].sum((-2,-1))
    legal = data['legal'].flatten(0,1).bool()
    sp = policy_input.flatten(0,1); gl = updated_global.flatten(0,1)
    logp = torch.cat([policy(sp[i:i+32],gl[i:i+32],legal[i:i+32],
                    actions=legal[i:i+32].to(torch.int8).argmax(-1))['log_probs']
                    for i in range(0,len(sp),32)])
    active = legal.sum(-1)>1
    entropy = -(logp.exp()*logp).sum(-1)
    result['policy_entropy'] = float(entropy[active].mean())
    result['policy_normalized_entropy'] = float((entropy[active]/legal.sum(-1)[active].float().log()).mean())
    result['policy_max_prob'] = float(logp.exp().max(-1).values[active].mean())
    return result


def run(args):
    from junqi_rl.gpu_rollout import GpuRollout
    from junqi_rl.training.gpu_collector import collect_rollout_gpu_v2
    from junqi_rl.training.rollout_gpu import RolloutBufferGPU
    root = Path(args.output); root.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); deadline = started+args.max_seconds
    checkpoint = torch.load(args.init_policy,map_location='cpu',weights_only=False)
    old_version = checkpoint.get('checkpoint_meta',{}).get('observation_semantics_version')
    if old_version != OBSERVATION_SEMANTICS_VERSION and not args.allow_observation_migration:
        raise ValueError('An explicit observation-semantics warm-start migration is required')
    policy_cfg = replace(checkpoint['cfg'],num_epochs_per_rollout=1,minibatch_size=128,
        lr_coef=1e-5,lr_ceil=1e-5,lr_floor=1e-5,lr_decay=0,lr_schedule_unit='rollout',
        temperature_coef=.03,temperature_floor=.03,temperature_ceil=.03,temperature_decay=0,
        temperature_schedule_unit='rollout',torch_compile=False)
    belief_cfg = BeliefNetConfig(n_encoder_layer=2,n_head=4,embed_dim=64,cnn_channels=32,
                                 cnn_layers=2,ff_factor=2,dropout=0)
    belief_train_cfg = BeliefPPOConfig(lr=5e-5,batch_size=32,epochs_per_rollout=4,
                                      ema_decay=.99,autocast_dtype='bfloat16')
    order = [(501,'shadow'),(501,'guarded'),(502,'guarded'),(502,'shadow')]
    provenance = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        input_checkpoint=str(args.init_policy),input_sha256=hashlib.sha256(Path(args.init_policy).read_bytes()).hexdigest(),
        from_observation_semantics=old_version,to_observation_semantics=OBSERVATION_SEMANTICS_VERSION,
        migration='policy weights only; fresh optimizer, counters, and BeliefNet',
        probe_sha256=hashlib.sha256(Path(args.probe).read_bytes()).hexdigest(),
        cells=order,rollouts=args.rollouts,num_envs=32,steps_per_env=128,self_play=True,
        policy_config=asdict(policy_cfg),belief_config=asdict(belief_cfg),belief_train_config=asdict(belief_train_cfg),
        warmup=8,ramp=16,max_neural_weight=.25,max_belief_kl=.05,max_publication_policy_kl=.02,
        min_publication_entropy_ratio=.95,max_seconds=args.max_seconds,
        stop_rules={'nonfinite':'stop immediately','probe_normalized_entropy':'stop below 50% of initial',
                    'probe_neural_nll':'stop if > initial + 0.25 for 3 consecutive probes'},
        gpu=torch.cuda.get_device_name(),torch=torch.__version__)
    save_json(root/'provenance.json',provenance)
    with np.load(args.probe) as d:
        probe={key:torch.as_tensor(d[key],device='cuda',dtype=(torch.bool if key in ('legal','mask') else
                   torch.int64 if key=='labels' else torch.float32)) for key in d.files}
    summaries=[]
    for seed,arm in order:
        if time.monotonic()>deadline: raise TimeoutError('Pilot budget exhausted before next cell')
        cell=root/f'{arm}_s{seed}'; cell.mkdir()
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        policy=JunqiNet(policy_cfg.net).cuda(); policy.load_state_dict(checkpoint['policy'],strict=True)
        trainer=PPOTrainer(policy,policy_cfg,device='cuda')
        belief=BeliefNet(belief_cfg).cuda()
        belief_trainer=BeliefPPOTrainer(belief,belief_train_cfg,device='cuda')
        buffer=BeliefBuffer(capacity=2048,seed=seed)
        sampler=MidgameBeliefSampler(buffer,every_steps=16,envs_per_sample=2,seed=seed)
        world=GpuRollout(num_envs=32)
        world.reset(seed_base=130000+seed)
        history=world.create_rollout_history(128)
        rollout=RolloutBufferGPU(num_envs=32,steps_per_env=128,history=history,storage_mode='compact_history',
            device='cuda',random_opponent=False,adv_filt_rate=policy_cfg.adv_filt_rate,
            adv_filt_thresh=policy_cfg.adv_filt_thresh,adv_filter_scope=policy_cfg.adv_filter_scope,
            minibatch_group=policy_cfg.minibatch_group,gamma=policy_cfg.gamma,
            gae_lambda=policy_cfg.gae_lambda,td_lambda=policy_cfg.td_lambda)
        initial=probe_metrics(belief_trainer.ema.model,policy,probe,0)
        save_json(cell/'initial-probe.json',initial)
        bad_probe=0; publications=0; cell_start=time.monotonic(); probes=[]; effective_beta=0.0
        initial_belief={k:v.detach().cpu().clone() for k,v in belief.state_dict().items()}
        for r in range(args.rollouts):
            if time.monotonic()>deadline: raise TimeoutError('Pilot wall-time limit reached')
            collect_rollout_gpu_v2(world,policy,rollout,seed_base=seed*10000+r,reset_at_start=False,
                                  random_opponent=False,use_compile=False,on_observation=sampler)
            metrics=trainer.train_epoch(rollout,rng=np.random.default_rng(seed*1000+r))
            metrics.update(belief_trainer.train_epoch(buffer)); metrics.update(sampler.stats())
            beta=.25*min(1,(r-7)/16) if arm=='guarded' and r>=8 else 0.0
            if beta:
                metrics.update(guarded_refresh_beliefs_neural(world,belief_trainer.ema.model,policy,
                    neural_weight=beta,max_kl=.05,rule_only_input=True,chunk_size=32))
                publications+=int(metrics.get('belief_infer/neural_weight',0)>0)
                if not metrics.get('belief_guard/rejected',0):
                    effective_beta=metrics['belief_infer/neural_weight']
            nonfinite = {key: str(value) for key, value in metrics.items() if not np.isfinite(value)}
            if nonfinite:
                save_json(cell/'nonfinite-metrics.json', {'rollout':r+1,'metrics':nonfinite})
                raise FloatingPointError(f'Non-finite pilot metrics: {sorted(nonfinite)}')
            if trainer._nan_skip_count or trainer._grad_nan_skip_count or belief_trainer._nan_skip_count or belief_trainer._grad_skip_count:
                raise FloatingPointError('Pilot encountered a numerical update skip')
            record={'rollout':r+1,'elapsed':time.monotonic()-cell_start,**metrics}
            with (cell/'metrics.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
            if (r+1)%8==0 or r+1==args.rollouts:
                p=probe_metrics(belief_trainer.ema.model,policy,probe,effective_beta)
                p['requested_neural_weight']=beta
                p['last_accepted_neural_weight']=effective_beta
                p['rollout']=r+1; probes.append(p)
                save_json(cell/'probes.json',probes)
                if p['policy_normalized_entropy'] < .5*initial['policy_normalized_entropy']:
                    raise RuntimeError('Predeclared policy entropy stop condition')
                bad_probe=bad_probe+1 if p['neural_nll']>initial['neural_nll']+.25 else 0
                if bad_probe>=3: raise RuntimeError('Predeclared belief deterioration stop condition')
                print(json.dumps({'cell':cell.name,**p}),flush=True)
        trainer_state=trainer.state_dict(); trainer_state['belief']=belief_trainer.state_dict()
        torch.save(trainer_state,cell/'checkpoint.pt')
        policy_delta=max(float((v.detach().cpu()-checkpoint['policy'][k]).abs().max()) for k,v in policy.state_dict().items())
        belief_delta=max(float((v.detach().cpu()-initial_belief[k]).abs().max()) for k,v in belief.state_dict().items())
        summary=dict(status='completed',seed=seed,arm=arm,rollouts=args.rollouts,
            environment_steps=args.rollouts*32*128,policy_updates=trainer.num_train_step,
            belief_updates=belief_trainer.num_train_step,publications=publications,
            policy_max_parameter_change=policy_delta,belief_max_parameter_change=belief_delta,
            elapsed_seconds=time.monotonic()-cell_start,initial=initial,final=probes[-1],**sampler.stats())
        save_json(cell/'summary.json',summary); (cell/'done').touch(); summaries.append(summary)
        print(json.dumps({'cell_done':cell.name,'seconds':summary['elapsed_seconds']}),flush=True)
        del trainer_state,trainer,policy,belief_trainer,belief,rollout,history,world,buffer,sampler
        gc.collect(); torch.cuda.empty_cache()
    save_json(root/'summary.json',{'status':'completed','expected_cells':4,'completed_cells':len(summaries),
                                 'elapsed_seconds':time.monotonic()-started,'cells':summaries})
    (root/'done').touch()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--make-probe')
    parser.add_argument('--probe')
    parser.add_argument('--init-policy')
    parser.add_argument('--allow-observation-migration',action='store_true')
    parser.add_argument('--output')
    parser.add_argument('--rollouts',type=int,default=48)
    parser.add_argument('--max-seconds',type=int,default=900)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.make_probe:
        make_probe(args.make_probe)
    else:
        try:
            run(args)
        except Exception as exc:
            if args.output and Path(args.output).exists():
                save_json(Path(args.output)/'failed.json',{'error':type(exc).__name__,'message':str(exc)})
            raise
