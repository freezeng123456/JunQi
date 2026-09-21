"""Construct legal hidden configurations indistinguishable to all tested inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from junqi_core.info_model import BeliefTensor
from junqi_core.observation import build_observation
from junqi_core.rules import PieceType,Seat,ShowMode
from junqi_core.setup import generate_random_setup,validate_setup
from junqi_core.state import GameState
from experiments.belief_features_20260913.features import PublicRelations,PublicTrajectory


def construct():
    original=generate_random_setup(random.Random(2026091301))
    changed=[list(lineup) for lineup in original]
    enemy=Seat.SOUTH.left_side_enemy
    a=changed[enemy.value].index(PieceType.PAIZH)
    b=changed[enemy.value].index(PieceType.LIANZH)
    changed[enemy.value][a],changed[enemy.value][b]=changed[enemy.value][b],changed[enemy.value][a]
    changed=tuple(tuple(lineup) for lineup in changed)
    assert validate_setup(original) and validate_setup(changed)
    observations=[]; extras=[]
    for setup in [original,changed]:
        state=GameState.new_game(setup,show_mode=ShowMode.DARK)
        obs=build_observation(state,BeliefTensor.initial(state,Seat.SOUTH),Seat.SOUTH).snapshot()
        observations.append(obs)
        # Exactly the same public-only inputs accepted by the feature tracker.
        pieces=sorted(state.pieces.items(),key=lambda item:item[1].piece_id)
        x=torch.tensor([[pos[0] for pos,piece in pieces]])
        y=torch.tensor([[pos[1] for pos,piece in pieces]])
        alive=torch.ones_like(x,dtype=torch.bool)
        temporal=PublicTrajectory(x,y,alive).all_observers()[:,Seat.SOUTH.value]
        spatial=torch.from_numpy(obs.spatial)[None]
        relational=PublicRelations()(spatial)
        extras.append((temporal,relational))
    np.testing.assert_array_equal(observations[0].spatial,observations[1].spatial)
    np.testing.assert_array_equal(observations[0].global_,observations[1].global_)
    for x,y in zip(extras[0],extras[1]): torch.testing.assert_close(x,y,rtol=0,atol=0)
    return {'status':'verified_constructive_ambiguity','both_configurations_legal':True,
        'observer':Seat.SOUTH.name,'hidden_enemy':enemy.name,'enemy_setup_slots':[a,b],
        'configuration_1':['PAIZH','LIANZH'],'configuration_2':['LIANZH','PAIZH'],
        'identical_inputs':['317 spatial channels','28 public global values','32 temporal channels','32 relational channels'],
        'spatial_sha256':hashlib.sha256(observations[0].spatial.tobytes()).hexdigest(),
        'interpretation':'Identical public inputs can have different correct hidden labels. This proves perfect identification is not universally attainable; it does not estimate the dataset-wide Bayes error.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); torch.set_num_threads(1)
    result=construct(); args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
