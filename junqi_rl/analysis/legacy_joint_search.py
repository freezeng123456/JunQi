"""Historical rule-probability joint-search prototype.

A single frozen blueprint acts for every player in each sampled world.
Counterfactual worlds are built from the observer's own setup and public
history, never by cloning the real hidden game or its private combat memory.
CPU rule simulation batches policy inference on the selected torch device.
This legacy adapter rejects facts-only policies and is not the next belief path.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from junqi_core.board import index_to_pos
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import ObservationBuilder
from junqi_core.rules import ALL_SEATS, CAMP_INDICES, PieceType, Seat, ShowMode
from junqi_core.state import Action, GameState, MoveResult
from junqi_rl.belief.legacy_joint import build_joint_input
from junqi_rl.env import (JunqiEnv, action_id_to_src_dst, rotate_to_compact_action_id,
                          src_dst_to_action_id, unrotate_compact_action_id)


@dataclass(frozen=True)
class SearchConfig:
    particles: int = 2
    proposal_samples: int = 8
    proposal_batches: int = 2
    candidates: int = 8
    depth: int = 4
    failed_refresh_plies: int = 32
    alpha: float = 0.002
    beta: float = 0.02

    def __post_init__(self):
        if min(self.particles,self.proposal_samples,self.proposal_batches,self.candidates,self.depth)<1:
            raise ValueError('Search budgets must be positive')
        if self.alpha<0 or self.beta<=0:
            raise ValueError('Require alpha >= 0 and beta > 0')


_PUBLIC_FIELDS=('seat','src','dst','event','flag_reveal_src','flag_reveal_dst',
                'flag_captured','terminated_after','winner_team_after','draw_after',
                'seats_died_this_step')


def public_result(result: MoveResult) -> tuple:
    return tuple(getattr(result,k) for k in _PUBLIC_FIELDS)


def step_world(env: JunqiEnv, action_id: int) -> MoveResult:
    """Update rules/beliefs without building four unused observations."""
    prev=env.state
    src,dst=action_id_to_src_dst(action_id)
    new,result=prev.step(Action(prev.turn,src,dst))
    for belief in env.beliefs.values(): belief.update(prev,new,result)
    env._state=new
    return result


def new_world(setups) -> JunqiEnv:
    world=JunqiEnv(show_mode=ShowMode.DARK,include_global=False)
    world._state=GameState.new_game(setups,show_mode=ShowMode.DARK)
    world._beliefs={s:BeliefTensor.initial(world.state,s) for s in ALL_SEATS}
    return world


def sampled_setups(env: JunqiEnv, observer: Seat, piece_ids, assignment):
    """Only the observer's known original identities are read from real state."""
    setup=[[PieceType.NONE]*30 for _ in ALL_SEATS]
    for slot in range(30):
        if slot not in CAMP_INDICES:
            setup[int(observer)][slot]=env.state.zero_board[index_to_pos(observer,slot)].piece_type
    for pid,kind in zip(piece_ids,assignment):
        owner,slot=divmod(int(pid),30)
        if owner==int(observer): raise ValueError('Joint input contains observer pieces')
        setup[owner][slot]=PieceType(int(kind)+2)
    return tuple(tuple(row) for row in setup)


def advance_public(world: JunqiEnv, history, start=0) -> bool:
    for expected in history[start:]:
        if world.state.terminated or int(world.state.turn)!=int(expected[0]): return False
        try:
            result=step_world(world,src_dst_to_action_id(expected[1],expected[2]))
        except (ValueError,RuntimeError):
            return False
        if public_result(result)!=expected: return False
    return True


def clone_world(world: JunqiEnv) -> JunqiEnv:
    # This clones an already reconstructed counterfactual world, not truth.
    for belief in world.beliefs.values(): belief.ensure_synced()
    clone=JunqiEnv(show_mode=ShowMode.DARK,include_global=False)
    clone._state=world.state.clone()
    clone._beliefs=copy.deepcopy(world.beliefs)
    return clone


def magnetic_update(log_policy, q, legal, *, alpha, beta):
    """One magnetic mirror-descent update, with the piece-uniform magnet."""
    moves=legal.reshape(-1,129,129).sum(-1)
    pieces=(moves>0).sum(-1,keepdim=True)
    log_magnet=-moves.clamp_min(1).float().log()-pieces.clamp_min(1).float().log()
    log_magnet=log_magnet.repeat_interleave(129,-1)
    logits=(q+alpha*log_magnet+beta*log_policy)/(alpha+beta)
    return logits.masked_fill(~legal,-torch.inf).log_softmax(-1)


def scalar_value(value):
    return value.exp()[...,2]-value.exp()[...,0] if value.ndim==2 else value.reshape(-1)


class BatchedBlueprint:
    """Reuse observation staging buffers; one policy call per search depth."""
    def __init__(self,policy,device):
        if getattr(policy, "identity_encoding", "legacy_probabilities") != "legacy_probabilities":
            raise ValueError("Legacy joint search cannot consume known_only; use the facts interface")
        self.policy=policy; self.device=torch.device(device)
        self.builder=ObservationBuilder(include_global=False)
        self.capacity=0

    def inputs(self,worlds,*,include_legal=True):
        n=len(worlds)
        if n>self.capacity:
            self.capacity=n
            self.spatial=np.empty((n,317,17,17),dtype=np.float32)
            self.global_=np.empty((n,0),dtype=np.float32)
            self.legal=np.empty((n,129*129),dtype=bool)
        seats=[w.state.turn for w in worlds]
        self.builder.build_observations_batch([w.state for w in worlds],
            [w.beliefs[s] for w,s in zip(worlds,seats)],seats,
            self.spatial[:n],self.global_[:n])
        for i,(world,seat) in enumerate(zip(worlds,seats)) if include_legal else ():
            self.legal[i].fill(False)
            for aid in world.legal_action_ids():
                self.legal[i,rotate_to_compact_action_id(aid,seat)]=True
        return (torch.from_numpy(self.spatial[:n]).to(self.device),
                torch.from_numpy(self.global_[:n]).to(self.device),
                torch.from_numpy(self.legal[:n]).to(self.device))

    @torch.no_grad()
    def distribution(self,worlds):
        obs,global_,legal=self.inputs(worlds)
        result=self.policy(obs,global_,legal,actions=legal.long().argmax(-1))
        return result['log_probs'],scalar_value(result['value']),legal,obs

    @torch.no_grad()
    def values(self,worlds):
        obs,global_,_=self.inputs(worlds,include_legal=False)
        return scalar_value(self.policy.forward_value(obs,global_))


class JointSearch:
    def __init__(self,blueprint: BatchedBlueprint,joint,config=SearchConfig(),*,seed=0):
        self.blueprint=blueprint; self.joint=joint.eval(); self.config=config
        self.generator=torch.Generator(device=blueprint.device).manual_seed(seed)
        self.caches={s:(0,[]) for s in ALL_SEATS}
        self.last_failed={}
        self.metrics={'decisions':0,'fallbacks':0,'proposals':0,'accepted_proposals':0,
                      'rollout_world_steps':0,'joint_refreshes':0,'candidate_actions':0,
                      'legal_actions':0,'cooldown_decisions':0,'public_replay_steps':0}
        self.latencies=[]

    @torch.no_grad()
    def particles(self,env,observer,history,obs):
        consumed,cached=self.caches[observer]
        worlds=[]
        for world in cached:
            before=world.state.move_counter
            accepted=advance_public(world,history,consumed)
            self.metrics['public_replay_steps']+=world.state.move_counter-before
            if accepted: worlds.append(world)
        # Surviving particles remain an approximate posterior conditioned on
        # new public events. Decode again only if this cache is empty.
        if not worlds:
            failed_at=self.last_failed.get(observer)
            if failed_at is not None and len(history)-failed_at<self.config.failed_refresh_plies:
                self.metrics['cooldown_decisions']+=1
                self.caches[observer]=(len(history),[])
                return []
            soa={k:getattr(env.state,k)[None] for k in
                 ('alive','pos_x','pos_y','move_count_arr','active_eat_arr','passive_surv_arr')}
            rules=np.zeros((1,4,12,289),dtype=np.float32)
            rules[0,int(observer)]=env.beliefs[observer].to_world_tensor().transpose(1,0,2).reshape(289,12).T
            inputs=build_joint_input(soa,rules,np.array([0]),np.array([int(observer)]))
            ids=inputs.piece_ids[0].tolist(); inputs=inputs.to(self.blueprint.device)
            seats=torch.tensor([int(observer)],device=self.blueprint.device)
            self.metrics['joint_refreshes']+=1
            for _ in range(self.config.proposal_batches):
                samples=self.joint.sample(obs,seats,n_samples=self.config.proposal_samples,
                                          generator=self.generator,**inputs.kwargs())[0].cpu().tolist()
                self.metrics['proposals']+=len(samples)
                for assignment in samples:
                    world=new_world(sampled_setups(env,observer,ids,assignment))
                    accepted=advance_public(world,history)
                    self.metrics['public_replay_steps']+=world.state.move_counter
                    if accepted:
                        worlds.append(world); self.metrics['accepted_proposals']+=1
                        if len(worlds)>=self.config.particles: break
                if worlds: break
            if worlds: self.last_failed.pop(observer,None)
            else: self.last_failed[observer]=len(history)
        self.caches[observer]=(len(history),worlds)
        return worlds

    @torch.no_grad()
    def action(self,env,history):
        started=time.perf_counter(); self.metrics['decisions']+=1
        seat=env.state.turn; root_team=int(seat)&1
        log_pi,value,legal,obs=self.blueprint.distribution([env])
        # CPU staging tensors share numpy storage. Future batch assembly
        # must not replace the root legal set used by the final update.
        legal=legal.clone()
        worlds=self.particles(env,seat,history,obs)
        if not worlds:
            self.metrics['fallbacks']+=1
            distribution=log_pi
        else:
            valid=legal[0].nonzero().flatten()
            candidates=valid[log_pi[0,valid].topk(min(self.config.candidates,len(valid))).indices]
            self.metrics['candidate_actions']+=len(candidates)
            self.metrics['legal_actions']+=len(valid)
            simulations=[]; origins=[]
            for i,aid in enumerate(candidates.cpu().tolist()):
                action=unrotate_compact_action_id(aid,seat)
                for world in worlds:
                    copy_world=clone_world(world)
                    step_world(copy_world,action)
                    simulations.append(copy_world); origins.append(i)
            self.metrics['rollout_world_steps']+=len(simulations)
            for _ in range(self.config.depth-1):
                active=[w for w in simulations if not w.state.terminated]
                if not active: break
                log_future,_,_,_=self.blueprint.distribution(active)
                aids=torch.multinomial(log_future.exp(),1,generator=self.generator).flatten().cpu().tolist()
                for world,aid in zip(active,aids):
                    step_world(world,unrotate_compact_action_id(aid,world.state.turn))
                self.metrics['rollout_world_steps']+=len(active)
            returns=np.zeros(len(simulations),dtype=np.float32)
            unfinished=[]; positions=[]
            for i,world in enumerate(simulations):
                if world.state.terminated:
                    returns[i]=float((world.state.team_rewards() or (0,0,0,0))[int(seat)])
                else: unfinished.append(world); positions.append(i)
            if unfinished:
                leaf=self.blueprint.values(unfinished).cpu().numpy()
                for pos,world,v in zip(positions,unfinished,leaf):
                    returns[pos]=v if (int(world.state.turn)&1)==root_team else -v
            q=value[:,None].expand_as(log_pi).clone()
            for i,aid in enumerate(candidates):
                q[0,aid]=float(returns[np.array(origins)==i].mean())
            distribution=magnetic_update(log_pi,q,legal,alpha=self.config.alpha,beta=self.config.beta)
        aid=int(torch.multinomial(distribution.exp(),1,generator=self.generator)[0,0])
        self.latencies.append(time.perf_counter()-started)
        return unrotate_compact_action_id(aid,seat)

    def report(self):
        return {**self.metrics,'config':asdict(self.config),
                'decision_seconds_total':sum(self.latencies),
                'decision_seconds_p50':float(np.median(self.latencies)) if self.latencies else 0,
                'decision_seconds_p95':float(np.quantile(self.latencies,.95)) if self.latencies else 0,
                'backend':'public-history CPU rules; batched torch blueprint; cached joint decoder'}
