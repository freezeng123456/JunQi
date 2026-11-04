#!/usr/bin/env python3
"""Bounded real-engine acceptance; never starts the production campaign.

Run under torchrun with --mode continuous/initial/resume/benchmark. The first
three compare a short full pipeline across a restart; benchmark keeps the
production architecture, global batch, and ordinary game rules.
"""
import argparse
import os
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'junqi_rl')]
import json
import torch
import torch.distributed as dist
from junqi_rl.training.config import TrainConfig, _dict_to_dataclass, load_yaml_config, validate_config
from scripts.train import train

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=('continuous','initial','resume','benchmark'),required=True)
p.add_argument('--output',required=True)
args=p.parse_args()
world=int(os.environ['WORLD_SIZE'])
cfg=_dict_to_dataclass(TrainConfig,load_yaml_config('configs/v100_ataraxos_reference_8gpu.yaml'))
cfg.distributed_world_size=world
cfg.env.num_envs=1536//world;cfg.arr.n_arr=4000//world;cfg.arr.ppo.batch_size=1024//world
cfg.export_milestone_weights=False;cfg.checkpoint_keep=2
cfg.target_completed_games=1_000_000;cfg.save_every_games=1_000_000
cfg.total_rollouts=3
if args.mode!='benchmark':
    cfg.reference_profile=''
    cfg.seed=314;cfg.env.seed=314
    cfg.net.embed_dim=32;cfg.net.n_head=4;cfg.net.depth=1;cfg.net.action_key_dim=32
    cfg.arr.net.embed_dim=32;cfg.arr.net.n_head=4;cfg.arr.net.depth=1
    cfg.env.num_envs=4;cfg.env.steps_per_env=9;cfg.env.max_num_moves=17
    cfg.arr.n_arr=16;cfg.arr.ppo.batch_size=2;cfg.arr.ppo.train_chunk_size=2
    cfg.ppo.advantage_history_steps=40;cfg.ppo.train_chunk_size=2;cfg.ppo.act_chunk_size=2
    cfg.ppo.adv_filt_rate=.25;cfg.ppo.adv_filt_thresh=.000001
    cfg.torch_deterministic=True
    cfg.save_dir=str(Path(args.output)/('continuous' if args.mode=='continuous' else 'restarted'))
    if args.mode=='initial':cfg.total_rollouts=1
    if args.mode=='resume':cfg.resume=str(Path(cfg.save_dir)/'ckpt_000001.pt')
else:
    cfg.save_dir=str(Path(args.output)/'benchmark')
cfg.ppo.net=cfg.net
validate_config(cfg)
try:
    train(cfg)
except RuntimeError as exc:
    if str(exc) != 'rollout safety ceiling reached before completed-game target':
        raise
    # This harness intentionally stops at a bounded rollout count, not a game target.
    dist.barrier()
    if dist.get_rank()==0:
        result=json.loads((Path(cfg.save_dir)/'training_result.json').read_text())
        assert result['status']=='rollout_ceiling' and result['rollout']==cfg.total_rollouts
        result['acceptance_mode']=args.mode
        (Path(cfg.save_dir)/'bounded_result.json').write_text(json.dumps(result,indent=2)+'\n')
    dist.destroy_process_group()
