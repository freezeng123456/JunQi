"""Public-information feature directions for a paired supervised belief study.

Temporal state accepts only public piece positions, existence and persistent
identities. Relational features accept only the observer's existing observation.
Neither interface accepts hidden piece types, truth labels, or enemy action lists.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from junqi_core.info_model import TRACKED_TYPES
from junqi_core.observation import CHANNEL_LAYOUT, OBS_CHANNELS
from junqi_core.rules import Event, resolve_combat
from junqi_rl.networks.belief_net import BeliefNet, BeliefNetConfig

EXTRA_CHANNELS = 32
TEMPORAL_NAMES = (
    'log_moves', 'idle_age', 'ever_moved', 'start_displacement',
    'distance_total', 'distance_mean', 'distance_max', 'long_moves',
    'direction_changes', 'reversals', 'unique_cells', 'first_move_age',
    'has_first_move', 'time_since_first_move', 'net_dx', 'net_dy',
    'last_dx', 'last_dy', 'horizontal_moves', 'vertical_moves',
    'right_angle_turns', 'revisits', 'move_rate', 'travel_rate',
    'moves_last_4', 'moves_last_16', 'moves_last_64', 'moves_last_256',
    'long_last_4', 'long_last_16', 'long_last_64', 'long_last_256',
)


class PublicTrajectory:
    """Track an entire cohort of games without observing piece types.

    Dead pieces never migrate back to an attacker's source. Call reset for a
    new cohort; the dataset generator never replaces games within a cohort.
    """
    windows = (4, 16, 64, 256)

    def __init__(self, x: torch.Tensor, y: torch.Tensor, alive: torch.Tensor):
        self.reset(x, y, alive)

    @torch.no_grad()
    def reset(self, x, y, alive):
        self.x = x.long().clone(); self.y = y.long().clone()
        self.alive = alive.bool().clone()
        self.start_x = self.x.clone(); self.start_y = self.y.clone()
        self.age = 0
        shape = self.x.shape
        zeros = lambda: torch.zeros(shape, device=x.device)
        for name in ('moves', 'distance', 'max_distance', 'long_moves', 'turns',
                     'reversals', 'horizontal', 'vertical', 'right_angles',
                     'revisits', 'last_step', 'last_dx', 'last_dy'):
            setattr(self, name, zeros())
        self.first_step = zeros() - 1
        self.unique = self.alive.float()
        self.visited = torch.zeros((*shape, 289), device=x.device, dtype=torch.bool)
        flat = (self.y * 17 + self.x).clamp(0, 288)
        self.visited.scatter_(2, flat[..., None], self.alive[..., None])
        self.move_ring = torch.zeros((256, *shape), device=x.device, dtype=torch.bool)
        self.long_ring = torch.zeros_like(self.move_ring)
        self.recent = torch.zeros((4, *shape), device=x.device)
        self.recent_long = torch.zeros_like(self.recent)

    @torch.no_grad()
    def update(self, x, y, alive):
        x=x.long(); y=y.long(); alive=alive.bool()
        dx=(x-self.x).float(); dy=(y-self.y).float()
        moved=alive & self.alive & ((dx != 0) | (dy != 0))
        dx=torch.where(moved, dx, 0); dy=torch.where(moved, dy, 0)
        distance=dx.abs()+dy.abs()
        long=moved & (distance > 1)
        previous=self.moves > 0
        cross=self.last_dx*dy-self.last_dy*dx
        dot=self.last_dx*dx+self.last_dy*dy
        self.turns += (moved & previous & (cross != 0)).float()
        self.right_angles += (moved & previous & (dot == 0)).float()
        self.reversals += (moved & previous & (dot < 0)).float()
        self.age += 1
        slot=(self.age-1) % 256
        # Subtract before writing: the 256-step window reads the current slot.
        for i,w in enumerate(self.windows):
            old=(self.age-1-w) % 256
            self.recent[i] += moved.float()-self.move_ring[old].float()
            self.recent_long[i] += long.float()-self.long_ring[old].float()
        self.move_ring[slot].copy_(moved)
        self.long_ring[slot].copy_(long)
        self.first_step=torch.where(moved & ~previous, self.age, self.first_step)
        self.last_step=torch.where(moved, self.age, self.last_step)
        self.moves += moved.float(); self.distance += distance
        self.max_distance=torch.maximum(self.max_distance, distance)
        self.long_moves += long.float()
        self.horizontal += (moved & (dx != 0)).float()
        self.vertical += (moved & (dy != 0)).float()
        flat=(y*17+x).clamp(0,288)
        visited=self.visited.gather(2, flat[...,None]).squeeze(-1)
        self.revisits += (moved & visited).float()
        self.unique += (moved & ~visited).float()
        self.visited.scatter_(2, flat[...,None], (visited | alive)[...,None])
        self.last_dx=torch.where(moved,dx,self.last_dx)
        self.last_dy=torch.where(moved,dy,self.last_dy)
        self.x=x.clone(); self.y=y.clone(); self.alive=alive.clone()

    @torch.no_grad()
    def piece_features(self):
        moved=self.moves > 0
        age=max(self.age,1)
        net_dx=(self.x-self.start_x).float(); net_dy=(self.y-self.start_y).float()
        f=[self.moves.log1p()/math.log(1025),
           ((self.age-self.last_step)/256).clamp(0,1), moved.float(),
           (net_dx.abs()+net_dy.abs())/32,
           (self.distance/256).clamp(0,1),
           self.distance/self.moves.clamp_min(1)/16, self.max_distance/16,
           (self.long_moves/128).clamp(0,1), (self.turns/128).clamp(0,1),
           (self.reversals/128).clamp(0,1), self.unique/129,
           self.first_step.clamp_min(0)/1024, moved.float(),
           torch.where(moved,(self.age-self.first_step)/1024,0),
           net_dx/16, net_dy/16, self.last_dx/16, self.last_dy/16,
           (self.horizontal/128).clamp(0,1), (self.vertical/128).clamp(0,1),
           (self.right_angles/128).clamp(0,1), (self.revisits/128).clamp(0,1),
           self.moves/age, self.distance/age/16]
        f += [(self.recent[i]/max(1,w/4)).clamp(0,1) for i,w in enumerate(self.windows)]
        f += [(self.recent_long[i]/max(1,w/4)).clamp(0,1) for i,w in enumerate(self.windows)]
        return torch.stack(f,1)*self.alive[:,None]

    @torch.no_grad()
    def world_features(self):
        pieces=self.piece_features()
        flat=(self.y*17+self.x).clamp(0,288)
        out=pieces.new_zeros((len(self.x),EXTRA_CHANNELS,289))
        out.scatter_add_(2,flat[:,None].expand(-1,EXTRA_CHANNELS,-1),pieces)
        return out.reshape(-1,EXTRA_CHANNELS,17,17)

    @torch.no_grad()
    def all_observers(self):
        world=self.world_features()
        return torch.stack([torch.rot90(world,k,(-2,-1)) for k in (0,1,2,-1)],1)


class PublicRelations(nn.Module):
    """Geometric proximity and hypothetical matchups, not legal-move claims."""
    def __init__(self):
        super().__init__()
        axis=torch.arange(-2,3)
        dy,dx=torch.meshgrid(axis,axis,indexing='ij')
        distance=dx.abs()+dy.abs()
        kernel=torch.where(distance==1,1.,torch.where(distance==2,.5,0.))
        self.register_buffer('kernel',kernel[None,None],persistent=False)
        axis=torch.arange(17)
        y,x=torch.meshgrid(axis,axis,indexing='ij')
        points=torch.stack((x.flatten(),y.flatten()),-1).float()
        self.register_buffer('distance',(points[:,None]-points[None]).abs().sum(-1),persistent=False)
        table=torch.zeros(12,12)
        for a,attacker in enumerate(TRACKED_TYPES):
            if attacker.is_immobile: continue
            for d,defender in enumerate(TRACKED_TYPES):
                event=resolve_combat(attacker,defender)
                table[a,d]=1 if event is Event.EAT else -1 if event is Event.KILLED else 0
        self.register_buffer('matchups',table,persistent=False)

    @torch.no_grad()
    def forward(self,obs):
        own=obs[:,CHANNEL_LAYOUT['piece_own']].float()
        pressure=F.conv2d(own,self.kernel.expand(12,-1,-1,-1),padding=2,groups=12)
        candidate=torch.einsum('bdhw,ad->bahw',pressure,self.matchups)
        candidate=candidate/pressure.sum(1,keepdim=True).clamp_min(1)
        groups=torch.cat((own.sum(1,keepdim=True)>0,
            (obs[:,CHANNEL_LAYOUT['prob_teammate']].sum(1,keepdim=True)>0)
                | (obs[:,CHANNEL_LAYOUT['dark_teammate']]>0),
            obs[:,CHANNEL_LAYOUT['piece_left_side_enemy']]>0,
            obs[:,CHANNEL_LAYOUT['piece_right_side_enemy']]>0),1)
        density=F.conv2d(groups.float(),self.kernel.expand(4,-1,-1,-1),padding=2,groups=4)/5
        near=[]
        for i in range(4):
            occupied=groups[:,i].flatten(1)
            d=self.distance[None].masked_fill(~occupied[:,None],1000).amin(-1)
            near.append(torch.exp(-d/4).reshape(-1,1,17,17))
        return torch.cat(((pressure/5).clamp(0,1),candidate,*near,density.clamp(0,1)),1)


class FeatureBeliefNet(nn.Module):
    """Same parameters in every arm; zero initial adapter preserves baseline."""
    def __init__(self,cfg:BeliefNetConfig):
        super().__init__()
        self.core=BeliefNet(cfg)
        self.adapter=nn.Conv2d(EXTRA_CHANNELS,OBS_CHANNELS,1,bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self,obs,seat,extra):
        return self.core(obs+self.adapter(extra),seat_idx=seat)['logits']
