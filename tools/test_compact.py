"""Test compact sequence end-to-end."""
import sys
sys.path.insert(0, '/data/home/freezeng/data/workspace/JunQi')
import torch
from junqi_rl.gpu_rollout import GpuRollout, FLAT_ACTION_DIM
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.networks.junqi_net import FLAT_ACTION_DIM as NET_FAD

print(f"FLAT_ACTION_DIM: net={NET_FAD}, gpu_rollout={FLAT_ACTION_DIM}")

cfg = JunqiNetConfig(cnn_channels=32, cnn_layers=1, depth=2, embed_dim=64,
                     n_head=2, ff_factor=2, action_key_dim=16)
policy = JunqiNet(cfg).to('cuda').eval()
print(f"pos_emb: {policy.pos_emb.shape}")

r = GpuRollout(num_envs=4)
r.reset(seed_base=0)
turn = r.turn_torch()
lm = r.legal_mask_canonical_torch_device(turn)
print(f"legal_mask: {lm.shape}, per_env: {lm.sum(dim=-1).tolist()}")

obs_sp, obs_gl = r.build_acting_seat_observation_torch(turn)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
    actions, log_probs, values = policy.act(obs_sp, obs_gl, lm)
legal = lm[torch.arange(4, device='cuda'), actions.long()]
print(f"actions: {actions.tolist()}, all_legal: {legal.all().item()}")
print("COMPACT TEST PASSED" if legal.all() else "FAILED")
