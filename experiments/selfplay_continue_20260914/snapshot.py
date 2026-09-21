"""Audit a stable numbered checkpoint on CPU and prepare an inference export."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import torch
from junqi_rl.training.checkpoint import load_evaluation_policy


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    a.output.mkdir(parents=True, exist_ok=False)
    before = digest(a.checkpoint)
    target = a.output / a.checkpoint.name
    shutil.copy2(a.checkpoint, target)
    assert digest(target) == before == digest(a.checkpoint)
    state = torch.load(target, map_location='cpu', weights_only=False)
    baseline = torch.load(a.baseline, map_location='cpu', weights_only=False)
    count = [0]
    def check(v):
        if isinstance(v, torch.Tensor):
            count[0] += 1
            assert torch.isfinite(v).all(), 'nonfinite checkpoint tensor'
        elif isinstance(v, dict):
            for x in v.values(): check(x)
        elif isinstance(v, (tuple, list)):
            for x in v: check(x)
    check(state)
    cfg = state['train_cfg']
    assert not cfg['random_opponent'] and not cfg['reward_shaping']
    assert not cfg['belief']['enabled'] and not cfg['arr']['enabled']
    assert state['num_rollout'] > 0 and state['num_train_step'] > 0
    policy = load_evaluation_policy(target, device='cpu')
    changed = sum(not torch.equal(v, baseline['policy'][k]) for k, v in state['policy'].items())
    assert changed > 0
    export = dict(policy=state['policy'], checkpoint_meta=state['checkpoint_meta'],
                  train_cfg={'net': asdict(policy.cfg)},
                  inference_export={'source_sha256': before, 'selection': 'unpromoted self-play candidate'})
    torch.save(export, a.output / 'candidate_raw_policy_v4.pt')
    restored = load_evaluation_policy(a.output / 'candidate_raw_policy_v4.pt', device='cpu')
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state['policy'].items())
    report = dict(status='finite_strict_reload_verified', checkpoint_sha256=before,
                  baseline_sha256=digest(a.baseline), rollouts=state['num_rollout'],
                  optimizer_updates=state['num_train_step'], changed_policy_tensors=changed,
                  policy_tensors=len(state['policy']), finite_tensors=count[0],
                  source_checkpoint=str(a.checkpoint), candidate_promoted=False,
                  effective_config=cfg)
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'effective_config'}), flush=True)


if __name__ == '__main__':
    main()
