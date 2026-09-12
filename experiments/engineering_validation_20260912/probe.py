"""Bounded device evidence: exact initialization and paired policy timing."""
from __future__ import annotations

import dataclasses
import json
import statistics
import sys
from pathlib import Path

import torch
import yaml

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.rollout_storage import COMPACT_HISTORY_BYTES_PER_TRANSITION


def main():
    output = Path(sys.argv[1])
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    world = GpuRollout(num_envs=16)
    world.reset(seed_base=800)
    for _ in range(16):
        acting = world.turn_torch().clone()
        legal = world.legal_mask_canonical_torch_device(acting)
        action = legal.long().argmax(-1).int()
        result = world.step_device_torch(action, acting)
        world.update_beliefs_device(result, acting)
    acting = world.turn_torch().clone()
    sp, gl = world.build_acting_seat_observation_torch(acting)
    legal = world.legal_mask_canonical_torch_device(acting)
    action = legal.long().argmax(-1)
    net_config = yaml.safe_load(Path('configs/iteration_baseline.yaml').read_text())['net']
    models = []
    for enabled in (False, True):
        torch.manual_seed(241)
        cfg = JunqiNetConfig(**net_config, combat_outcome_features=enabled)
        models.append(JunqiNet(cfg).cuda().eval())
    with torch.inference_mode():
        reference = models[0](sp, gl, legal, actions=action)
        candidate = models[1](sp, gl, legal, actions=action)
        for key in reference:
            torch.testing.assert_close(reference[key], candidate[key], rtol=0, atol=0)
        for model in models:
            for _ in range(20):
                model(sp, gl, legal, actions=action)
        durations = [[], []]
        for repetition in range(3):
            for index in ((0, 1) if repetition % 2 == 0 else (1, 0)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(50):
                    models[index](sp, gl, legal, actions=action)
                end.record()
                torch.cuda.synchronize()
                durations[index].append(start.elapsed_time(end) / 50)
    history = world.create_rollout_history(8)
    assert history.history_bytes == 8 * 16 * COMPACT_HISTORY_BYTES_PER_TRANSITION
    payload = {
        'status': 'PASS', 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'policy_batch_size': 16, 'policy_dtype': 'float32',
        'policy_config': dataclasses.asdict(models[1].cfg),
        'zero_residual_outputs_bit_exact': True,
        'parameters': [m.num_parameters() for m in models],
        'forward_ms_3_repetitions': durations,
        'forward_ms_medians': [statistics.median(x) for x in durations],
        'history_bytes_per_transition_before': 47243,
        'history_bytes_per_transition_after': COMPACT_HISTORY_BYTES_PER_TRANSITION,
        'history_actual_bytes_N16_T8': history.history_bytes,
        'history_bytes_saved_N128_T512': (47243 - COMPACT_HISTORY_BYTES_PER_TRANSITION) * 128 * 512,
        'scope': 'engineering validation; timing excludes collection/PPO/backward; no playing-strength training',
    }
    (output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
