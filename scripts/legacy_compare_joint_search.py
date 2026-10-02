"""LEGACY probability-input DARK matches: frozen blueprint vs joint search."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import asdict
import hashlib
import json
import time

import torch

from junqi_core.rules import ShowMode
from junqi_core.state import classify_termination
from junqi_rl.analysis.legacy_joint_search import (BatchedBlueprint, JointSearch, SearchConfig,
    public_result, step_world)
from junqi_rl.analysis.protocol import EvaluationCounts
from junqi_rl.belief.joint_checkpoint import load_joint_belief
from junqi_rl.env import JunqiEnv, unrotate_compact_action_id
from junqi_rl.training.checkpoint import load_evaluation_policy
from junqi_rl.training._checkpoint_io import atomic_write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',required=True); p.add_argument('--joint',required=True)
    p.add_argument('--output',required=True); p.add_argument('--pairs',type=int,default=16)
    p.add_argument('--seed',type=int,default=20261003); p.add_argument('--max-moves',type=int,default=4000)
    p.add_argument('--particles',type=int,default=2); p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--depth',type=int,default=4); p.add_argument('--device',default='cuda')
    p.add_argument('--wall-seconds',type=float,default=5400)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.pairs<1 or a.max_moves<1: raise ValueError('Positive evaluation budgets required')
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    policy=load_evaluation_policy(a.policy,device=a.device)
    if policy.identity_encoding != "legacy_probabilities":
        raise ValueError("Legacy joint v1 search requires its matching input schema")
    joint=load_joint_belief(a.joint,policy_path=a.policy,device=a.device)
    blueprint=BatchedBlueprint(policy,a.device)
    cfg=SearchConfig(particles=a.particles,candidates=a.candidates,depth=a.depth)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    games=[]; reports=[]; began=time.perf_counter(); deadline=began+a.wall_seconds
    requested=a.pairs*2
    previous_seconds=0
    if a.resume and (out/'comparison.json').exists():
        previous=json.loads((out/'comparison.json').read_text())
        expected={'seed':a.seed,'pairs':a.pairs,'max_moves':a.max_moves,'search_config':asdict(cfg),
                  'policy_sha256':hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),
                  'joint_sha256':hashlib.sha256(Path(a.joint).read_bytes()).hexdigest()}
        if any(previous[k]!=v for k,v in expected.items()):
            raise ValueError('Evaluation resume protocol/checkpoints mismatch')
        completed=[(g,r) for g,r in zip(previous['games'],previous['search_reports']) if g['outcome']!='ongoing']
        games=[g for g,_ in completed]; reports=[r for _,r in completed]
        previous_seconds=previous['seconds']
    finished_keys={(g['pair'],g['candidate_team']) for g in games}

    def save():
        counts=EvaluationCounts(sum(g['outcome']=='win' for g in games),
              sum(g['outcome']=='loss' for g in games),sum(g['outcome']=='draw' for g in games),
              requested-sum(g['outcome']!='ongoing' for g in games))
        report={'protocol':'same setups, both team assignments; frozen identical policy weights',
                'policy_sha256':hashlib.sha256(Path(a.policy).read_bytes()).hexdigest(),
                'joint_sha256':hashlib.sha256(Path(a.joint).read_bytes()).hexdigest(),
                'seed':a.seed,'pairs':a.pairs,'max_moves':a.max_moves,'search_config':asdict(cfg),
                'candidate':counts.as_metrics(prefix='candidate'),
                'baseline':EvaluationCounts(counts.losses,counts.wins,counts.draws,counts.ongoing).as_metrics(prefix='baseline'),
                'seconds':previous_seconds+time.perf_counter()-began,'games':games,'search_reports':reports,
                'finished':len(games)==requested and counts.ongoing==0,
                'pair_dependence':'Wilson intervals are descriptive per-match intervals; paired outcomes are not independent'}
        atomic_write(str(out/'comparison.json'),lambda f:f.write((json.dumps(report,indent=2)+'\n').encode()))
        return report

    for pair in range(a.pairs):
        for team in (0,1):
            if (pair,team) in finished_keys: continue
            if time.perf_counter()>=deadline: print(json.dumps(save()),flush=True); return
            env=JunqiEnv(show_mode=ShowMode.DARK,include_global=False); env.reset(a.seed+pair)
            setup_hash=hashlib.sha256(env.state.piece_type_arr.tobytes()).hexdigest()
            search=JointSearch(blueprint,joint,cfg,seed=a.seed+pair*2+team)
            base_rng=torch.Generator(device=a.device).manual_seed(a.seed+pair*2+team+100000)
            history=[]; last=None
            for ply in range(a.max_moves):
                if env.state.terminated or time.perf_counter()>=deadline: break
                seat=env.state.turn
                if int(seat)&1==team:
                    action=search.action(env,history)
                else:
                    log_pi,_,_,_=blueprint.distribution([env])
                    aid=int(torch.multinomial(log_pi.exp(),1,generator=base_rng)[0,0])
                    action=unrotate_compact_action_id(aid,seat)
                last=step_world(env,action); history.append(public_result(last))
            if env.state.terminated:
                outcome='draw' if env.state.draw else ('win' if env.state.winner_team==team else 'loss')
                reason=classify_termination(env.state,last)
            else: outcome='ongoing'; reason='wall_budget' if time.perf_counter()>=deadline else 'evaluation_move_cap'
            game={'pair':pair,'setup_seed':a.seed+pair,'setup_sha256':setup_hash,
                  'candidate_team':team,'outcome':outcome,'plies':len(history),'reason':reason}
            games.append(game); reports.append(search.report()); save()
            print(json.dumps(game),flush=True)
    print(json.dumps(save(),indent=2),flush=True)


if __name__=='__main__': main()
