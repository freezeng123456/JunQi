"""Actor-observation-only material and uncertainty residual for attack logits.

No reward change or hidden piece identity input. Material values are declared
heuristic features, not labels: flag 1, mine .3, bomb .6, ranks 1/9 through 1.
The policy learns their relevance; the zero output layer preserves the start.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
from junqi_core.board import COMPACT_TO_FLAT
from junqi_core.info_model import TRACKED_TYPES
from junqi_core.observation import CHANNEL_LAYOUT
from junqi_core.rules import Event, PieceType, resolve_combat

TACTICAL_FEATURE_VERSION = 2
FEATURE_NAMES = ('expected_capture_material', 'expected_own_material_loss',
                 'enemy_belief_entropy', 'flag_probability', 'mine_probability',
                 'bomb_probability', 'own_material', 'own_bomb', 'own_engineer')

class TacticalFeatureHead(nn.Module):
    def __init__(self):
        super().__init__()
        values = torch.tensor([1. if t == PieceType.JUNQI else .3 if t == PieceType.DILEI
                               else .6 if t == PieceType.ZHADAN else (int(PieceType.GONGB)-int(t)+1)/9
                               for t in TRACKED_TYPES])
        captures, losses = torch.zeros(12,12), torch.zeros(12,12)
        for a,attacker in enumerate(TRACKED_TYPES):
            if attacker.is_immobile:continue
            for d,defender in enumerate(TRACKED_TYPES):
                event=resolve_combat(attacker,defender)
                captures[a,d]=values[d] if event in (Event.EAT,Event.BOMB) else 0
                losses[a,d]=values[a] if event in (Event.KILLED,Event.BOMB) else 0
        for name,tensor in [('values',values),('captures',captures),('losses',losses),
                            ('on_board',torch.tensor(COMPACT_TO_FLAT,dtype=torch.long)),
                            ('mobile',torch.tensor([not t.is_immobile for t in TRACKED_TYPES]))]:
            self.register_buffer(name,tensor,persistent=False)
        self.mlp=nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)

    def features(self,obs:Tensor):
        flat=obs.flatten(-2).index_select(-1,self.on_board).float()
        own=flat[:,CHANNEL_LAYOUT['piece_own']]
        occupied=(flat[:,CHANNEL_LAYOUT['piece_left_side_enemy']]+flat[:,CHANNEL_LAYOUT['piece_right_side_enemy']])>0
        enemy=(flat[:,CHANNEL_LAYOUT['belief_left_side']]+flat[:,CHANNEL_LAYOUT['belief_right_side']])*occupied
        b,_,n=enemy.shape
        cap=torch.einsum('bdj,ad->baj',enemy,self.captures)
        loss=torch.einsum('bdj,ad->baj',enemy,self.losses)
        entropy=-(enemy*enemy.clamp_min(1e-8).log()).sum(1)/math.log(12)
        def shared(x):return x[:,None,:].expand(b,12,n)
        def flag(t):return shared(enemy[:,TRACKED_TYPES.index(t)])
        material=self.values[None,:,None].expand(b,12,n)
        bomb=(torch.arange(12,device=obs.device)==TRACKED_TYPES.index(PieceType.ZHADAN))[None,:,None].expand(b,12,n)
        engineer=(torch.arange(12,device=obs.device)==TRACKED_TYPES.index(PieceType.GONGB))[None,:,None].expand(b,12,n)
        features=torch.stack((cap,loss,shared(entropy),flag(PieceType.JUNQI),flag(PieceType.DILEI),flag(PieceType.ZHADAN),material,bomb,engineer),dim=-1)
        valid=occupied.expand(b,12,n)&self.mobile[None,:,None]
        return own,features,valid

    def forward(self,obs:Tensor)->Tensor:
        own,features,valid=self.features(obs)
        scores=self.mlp(features).squeeze(-1).float()*valid
        return torch.bmm(own.transpose(1,2),scores).flatten(1)
