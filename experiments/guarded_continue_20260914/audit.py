"""Strictly audit and export the actual terminal checkpoint; no best selection."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from junqi_rl.analysis.random_eval import evaluate_paired_head_to_head
from junqi_rl.training.checkpoint import load_evaluation_policy


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--h2h', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(914)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = args.root
    baseline = root / 'guarded_s501_raw_policy_v4_2048wins.pt'
    result = root / 'results'
    source = (result / 'train/ckpt_latest.pt').resolve(strict=True)
    state = torch.load(source, map_location='cpu', weights_only=False)
    prior = torch.load(baseline, map_location='cpu', weights_only=False)
    assert state['num_rollout'] > 0 and state['num_train_step'] > 0
    assert not state['train_cfg']['belief']['enabled'] and not state['train_cfg']['arr']['enabled']
    assert not any('ema' in str(k).lower() or k == 'belief' for k in state)
    finite_count = 0

    def finite(value):
        nonlocal finite_count
        if torch.is_tensor(value):
            assert torch.isfinite(value).all().item()
            finite_count += 1
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)

    finite(state)
    policy = load_evaluation_policy(source, device='cpu')
    assert state['policy'].keys() == prior['policy'].keys()
    changed = [k for k, v in state['policy'].items() if not torch.equal(v, prior['policy'][k])]
    assert changed and state['optimizer']['state']
    if args.h2h:
        records = []
        metrics = evaluate_paired_head_to_head(
            policy.to('cuda').eval(), load_evaluation_policy(baseline, device='cuda'),
            num_games=512, num_envs=64, device='cuda', seed=4391400,
            setup_seed=3391400, max_moves=4000, autocast_dtype=torch.bfloat16,
            greedy=True, game_records=records)
        assert len(records) == 512 and metrics['h2h/num_games'] == 512
        assert metrics['h2h/ongoing'] == 0
        (result / 'h2h_games.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        (result / 'h2h.json').write_text(json.dumps({'metrics': metrics, 'candidate_sha256': digest(source),
            'baseline_sha256': digest(baseline), 'setup_seed': 3391400, 'game_seed': 4391400}, indent=2))
        print(json.dumps(metrics), flush=True)
        return
    output = result / 'candidate_raw_policy_v4.pt'
    assert not output.exists()
    torch.save({'policy': state['policy'], 'checkpoint_meta': state['checkpoint_meta'],
        'train_cfg': {'net': asdict(policy.cfg)},
        'inference_export': {'source_sha256': digest(source), 'baseline_sha256': digest(baseline),
            'ema': False, 'learned_belief': False, 'optimizer_included': False}}, output)
    reloaded = load_evaluation_policy(output)
    assert all(torch.equal(v, reloaded.state_dict()[k]) for k,v in state['policy'].items())
    report = {'status': 'finite_strict_load_and_export_verified', 'terminal_checkpoint': str(source),
        'terminal_sha256': digest(source), 'export_sha256': digest(output),
        'baseline_sha256': digest(baseline), 'rollouts': state['num_rollout'],
        'optimizer_updates': state['num_train_step'], 'finite_tensors': finite_count,
        'changed_policy_tensors': len(changed), 'all_policy_tensors': len(state['policy']),
        'ema': False, 'learned_belief': False, 'optimizer_state_entries': len(state['optimizer']['state'])}
    (result / 'checkpoint_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
