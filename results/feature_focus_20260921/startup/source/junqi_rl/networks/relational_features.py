"""Public-position relationships for JunQi action selection.

Current-position, non-engineer reach potential: roads, clear straight rails and
legal corner curves. It is not a post-move search or an exact next-turn threat
probability. Enemy engineer turns are intentionally not included in this map;
our own engineer-only moves use the exact actor legal mask to expose reveal cost.
"""
from collections import deque
import math
import numpy as np
import torch
from torch import nn
from junqi_core.board import (COMPACT_TO_FLAT, FLAT_TO_COMPACT, COMPACT_ROAD_NEIGHBORS,
    COMPACT_RAIL_NEIGHBORS, cell_info, is_camp, is_stronghold, is_railway)
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_core.info_model import TRACKED_TYPES
from junqi_core.rules import PieceType, Event, resolve_combat

RELATIONAL_FEATURE_VERSION=1
DEST_NAMES=('removal_pressure','survival_pressure','friendly_support','enemy_mobile_pressure',
            'camp','stronghold_lock','rail','own_flag_proximity','ally_flag_proximity',
            'enemy_flag_proximity','own_flag_alarm_proximity','enemy_flag_probability',
            'attack_outcome_information')

def topology():
    n=len(COMPACT_TO_FLAT);coords=[(int(f)%17,int(f)//17) for f in COMPACT_TO_FLAT]
    road=np.zeros((n,n),dtype=np.float32)
    for i,neighbors in enumerate(COMPACT_ROAD_NEIGHBORS):
        for j in neighbors:
            if j<n:road[i,j]=1
    rail=[list(map(int,row[row<n])) for row in COMPACT_RAIL_NEIGHBORS]
    path_indices=[];interiors=[]
    for i,(sx,sy) in enumerate(coords):
        for j,(dx,dy) in enumerate(coords):
            if i==j or road[i,j] or not (is_railway(sx,sy) and is_railway(dx,dy)):continue
            if sx==dx:allowed={k for k,(x,y) in enumerate(coords) if x==sx}
            elif sy==dy:allowed={k for k,(x,y) in enumerate(coords) if y==sy}
            else:
                curve=cell_info(sx,sy).curve_rail
                if not curve or curve!=cell_info(dx,dy).curve_rail:continue
                allowed={k for k,(x,y) in enumerate(coords) if cell_info(x,y).curve_rail==curve}
            q=deque([i]);parents={i:None}
            while q and j not in parents:
                cur=q.popleft()
                for nb in rail[cur]:
                    if nb in allowed and nb not in parents:parents[nb]=cur;q.append(nb)
            if j not in parents:continue
            row=np.zeros(n,dtype=np.float32);cur=parents[j]
            while cur!=i:row[cur]=1;cur=parents[cur]
            path_indices.append(i*n+j);interiors.append(row)
    distances=np.full((n,n),99,dtype=np.float32)
    for i in range(n):
        q=deque([i]);distances[i,i]=0
        while q:
            cur=q.popleft()
            for nb in np.flatnonzero(road[cur]):
                if distances[i,nb]==99:distances[i,nb]=distances[i,cur]+1;q.append(int(nb))
    return road,np.asarray(path_indices),np.stack(interiors),np.exp(-distances/4)

class RelationalFeatureHead(nn.Module):
    def __init__(self):
        super().__init__();road,indices,paths,proximity=topology()
        coords=[(int(f)%17,int(f)//17) for f in COMPACT_TO_FLAT]
        table=torch.zeros(12,12,3)
        for a,attacker in enumerate(TRACKED_TYPES):
            if attacker.is_immobile:continue
            for d,defender in enumerate(TRACKED_TYPES):table[a,d,(Event.EAT,Event.KILLED,Event.BOMB).index(resolve_combat(attacker,defender))]=1
        for name,value in [('road',torch.from_numpy(road)),('path_indices',torch.tensor(indices,dtype=torch.long)),
          ('interiors',torch.from_numpy(paths)),('proximity',torch.from_numpy(proximity)),
          ('camp',torch.tensor([is_camp(*p) for p in coords])),('stronghold',torch.tensor([is_stronghold(*p) for p in coords])),
          ('rail',torch.tensor([is_railway(*p) for p in coords])),('on_board',torch.tensor(COMPACT_TO_FLAT,dtype=torch.long)),
          ('mobile',torch.tensor([not t.is_immobile for t in TRACKED_TYPES],dtype=torch.float32)),('outcomes',table)]:
            self.register_buffer(name,value,persistent=False)
        self.weights=nn.Parameter(torch.zeros(12,len(DEST_NAMES)))
        self.special_weights=nn.Parameter(torch.zeros(3))

    def context(self,obs):
        flat=obs.flatten(-2).index_select(-1,self.on_board).float()
        own=flat[:,CHANNEL_LAYOUT['piece_own']]
        ally=flat[:,CHANNEL_LAYOUT['prob_teammate']]
        enemy=flat[:,CHANNEL_LAYOUT['belief_left_side']]+flat[:,CHANNEL_LAYOUT['belief_right_side']]
        enemy_occ=(flat[:,CHANNEL_LAYOUT['piece_left_side_enemy']]+flat[:,CHANNEL_LAYOUT['piece_right_side_enemy']]).squeeze(1)>0
        enemy=enemy*enemy_occ[:,None,:]
        occupied=(own.sum(1)+ally.sum(1)+flat[:,CHANNEL_LAYOUT['dark_teammate']].squeeze(1)+enemy_occ)>0
        # Matrix multiplication uses only binary occupancy/path entries, so BF16
        # exactly represents the small integer intermediate counts.
        clear=(occupied.float()@self.interiors.T)==0
        reach=self.road[None].expand(obs.shape[0],-1,-1).clone().flatten(1)
        reach[:,self.path_indices]=clear.to(reach.dtype)
        reach=reach.reshape(-1,129,129)*(~self.stronghold)[None,:,None]
        # Potential loss if an enemy could attack a defender of each own type.
        removal=self.outcomes[:,:,0]+self.outcomes[:,:,2]
        enemy_loss=torch.einsum('bai,ad->bdi',enemy,removal)
        enemy_survive=torch.einsum('bai,ad->bdi',enemy,self.outcomes[:,:,1])
        loss=torch.bmm(enemy_loss,reach)*(~self.camp)[None,None,:]
        survive=torch.bmm(enemy_survive,reach)*(~self.camp)[None,None,:]
        ally_mobile=((own+ally)*self.mobile[None,:,None]).sum(1)
        enemy_mobile=(enemy*self.mobile[None,:,None]).sum(1)
        support=torch.bmm(ally_mobile[:,None,:],reach).squeeze(1)
        pressure=torch.bmm(enemy_mobile[:,None,:],reach).squeeze(1)*(~self.camp)[None,:]
        flag=TRACKED_TYPES.index(PieceType.JUNQI)
        def flag_map(mass):return (mass@self.proximity)/mass.sum(-1,keepdim=True).clamp_min(1)
        own_flag=flag_map(own[:,flag]);ally_flag=flag_map(ally[:,flag]);enemy_flag=flag_map(enemy[:,flag])
        alarm=(loss[:,flag]*own[:,flag]).sum(1,keepdim=True)
        outcomes=torch.einsum('bdj,ado->bajo',enemy,self.outcomes)
        information=-(outcomes*outcomes.clamp_min(1e-8).log()).sum(-1)/math.log(3)
        b=obs.shape[0]
        def repeat(x):return x[:,None,:].expand(b,12,129)
        def static(x):return x.float()[None,None,:].expand(b,12,129)
        features=torch.stack((loss/(1+loss),survive/(1+survive),repeat(support/(1+support)),repeat(pressure/(1+pressure)),
            static(self.camp),static(self.stronghold),static(self.rail),repeat(own_flag),repeat(ally_flag),repeat(enemy_flag),
            repeat(own_flag*alarm/(1+alarm)),repeat(enemy[:,flag]),information),-1)
        known_engineer=flat[:,CHANNEL_LAYOUT['cm_my_is_gongb']].squeeze(1)
        return own,features,reach,enemy_occ,known_engineer

    def forward(self,obs,legal_mask):
        own,features,reach,enemy_occ,known_engineer=self.context(obs)
        type_scores=(features*self.weights[None,:,None,:]).sum(-1)
        logits=torch.bmm(own.transpose(1,2),type_scores)
        # Destination-minus-source features encode the change made by the move.
        logits=logits-(own*type_scores).sum(1)[:,:,None]
        engineer=own[:,TRACKED_TYPES.index(PieceType.GONGB)]
        bomb=own[:,TRACKED_TYPES.index(PieceType.ZHADAN)]
        eng_scarcity=engineer/engineer.sum(-1,keepdim=True).clamp_min(1)
        bomb_scarcity=bomb/bomb.sum(-1,keepdim=True).clamp_min(1)
        logits=logits+self.special_weights[0]*eng_scarcity[:,:,None]*enemy_occ[:,None,:]
        logits=logits+self.special_weights[1]*bomb_scarcity[:,:,None]*enemy_occ[:,None,:]
        reveal=engineer[:,:,None]*(1-known_engineer[:,:,None])*(reach==0)*legal_mask.reshape(-1,129,129)
        return (logits+self.special_weights[2]*reveal).flatten(1)
