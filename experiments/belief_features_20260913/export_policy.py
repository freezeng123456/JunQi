"""Export the exact verified winning raw policy for inference, without auxiliary state."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from junqi_rl.training.checkpoint import load_evaluation_policy


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def export(root):
    audit = json.loads((root / 'analysis_policy/current_gpu.json').read_text())
    assert audit['status'] == 'all_paired_policy_results_verified'
    role = next(key for key, value in audit['labels'].items() if value == 'guarded_s501_v4')
    verified = audit['runs'][role]
    metrics = verified['summary']['metrics']
    assert metrics['eval/wins'] == metrics['eval/num_games'] == 2048
    assert metrics['eval/losses'] == metrics['eval/draws'] == metrics['eval/ongoing'] == 0
    source = root / 'source_inputs/frozen_policy_v4.pt'
    source_hash = digest(source)
    assert source_hash == verified['provenance']['checkpoint_sha256']
    cpu = json.loads((root / 'analysis_policy/current_cpu.json').read_text())
    assert cpu['status'] == 'all_paired_policy_results_verified'
    cpu_role = next(key for key, value in cpu['labels'].items() if value == 'guarded_s501_v4')
    cpu_run = cpu['runs'][cpu_role]
    assert cpu_run['provenance']['checkpoint_sha256'] == source_hash
    cpu_metrics = cpu_run['summary']['metrics']
    state = torch.load(source, map_location='cpu', weights_only=False)
    assert state['checkpoint_meta']['observation_semantics_version'] == 4
    output = root / 'analysis_policy/guarded_s501_raw_policy_v4_2048wins.pt'
    assert not output.exists()
    exported = {'policy': state['policy'], 'checkpoint_meta': state['checkpoint_meta'],
                'train_cfg': {'net': state['train_cfg']['net']},
                'inference_export': {'source_sha256': source_hash, 'learned_belief': False,
                                     'ema': False, 'optimizer_included': False}}
    torch.save(exported, output)
    torch.set_num_threads(1)
    loaded = load_evaluation_policy(str(output), device='cpu')
    assert loaded.state_dict().keys() == state['policy'].keys()
    assert all(torch.equal(value, state['policy'][key]) for key, value in loaded.state_dict().items())
    assert digest(source) == source_hash
    report = {'status': 'inference_export_reloaded_and_all_policy_tensors_identical',
              'checkpoint': str(output.resolve()), 'sha256': digest(output),
              'source_sha256': source_hash, 'policy_tensors': len(state['policy']),
              'gpu_acceptance': {'wins': 2048, 'losses': 0, 'draws': 0, 'games': 2048},
              'cpu_separate_acceptance': {key: int(cpu_metrics['eval/' + metric]) for key, metric in
                                         [('wins', 'wins'), ('losses', 'losses'), ('draws', 'draws'), ('games', 'num_games')]},
              'learned_belief': False, 'ema': False, 'optimizer_included': False,
              'scope': 'Inference-only export of exactly the tested raw weights. No new training '
                       'or additional game evaluation was performed during export. Not a full training-resume checkpoint.'}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    export(parser.parse_args().root)
