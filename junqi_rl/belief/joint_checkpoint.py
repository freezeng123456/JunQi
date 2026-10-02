"""Historical joint v1 checkpoint contract, using ``legacy_joint`` inputs.

These weights do not consume the new facts/history schema.
"""
from pathlib import Path
import hashlib
import torch
from junqi_rl.networks.joint_belief_net import JointBeliefNet, JointBeliefNetConfig

INPUT_SPEC={'spatial_channels':317,'global_dims':0,'board_tokens':129,
            'hidden_tokens':75,'types':12,'sequence':'relative-owner-then-initial-slot',
            'mode':'DARK','includes_dead_pieces':True}


def load_joint_belief(path,*,policy_path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('architecture_id')!=JointBeliefNet.architecture_id:
        raise ValueError('Joint checkpoint architecture mismatch')
    if saved.get('input_spec')!=INPUT_SPEC:
        raise ValueError('Joint checkpoint input contract mismatch')
    parent=hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    if saved.get('parent_policy_sha256')!=parent:
        raise ValueError('Joint checkpoint parent policy mismatch')
    net=JointBeliefNet(JointBeliefNetConfig(**saved['config']))
    net.load_state_dict(saved['net'],strict=True)
    return net.to(device).eval()
