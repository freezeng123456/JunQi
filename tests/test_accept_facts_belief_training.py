"""The GPU gate cannot report success by running on CPU or ignoring drift."""
import json

import pytest
import torch

from scripts.accept_facts_belief_training import assert_exact, main


def test_gpu_gate_fails_with_json_when_cuda_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    output = tmp_path / 'acceptance.json'
    status = main(['--policy', str(tmp_path / 'not-opened.pt'), '--dataset', str(tmp_path / 'data'),
                   '--output', str(output)])
    report = json.loads(output.read_text())
    assert status == 1
    assert report['status'] == 'failed'
    assert report['failed_phase'] == 'cuda_preflight'
    assert 'CPU fallback' in report['error']


def test_exact_resume_check_rejects_tiny_tensor_and_scalar_drift():
    original = {'state': {0: {'exp_avg': torch.tensor([0.25])}}, 'param_groups': [{'lr': 1e-4}]}
    identical = {'state': {0: {'exp_avg': torch.tensor([0.25])}}, 'param_groups': [{'lr': 1e-4}]}
    assert assert_exact(original, identical) == 1
    identical['state'][0]['exp_avg'][0] += 1e-7
    with pytest.raises(AssertionError):
        assert_exact(original, identical)
    identical['state'][0]['exp_avg'][0] = 0.25
    identical['param_groups'][0]['lr'] = 2e-4
    with pytest.raises(AssertionError, match='scalar'):
        assert_exact(original, identical)
