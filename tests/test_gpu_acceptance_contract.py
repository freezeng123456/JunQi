"""The acceptance command must fail closed on hosts without real CUDA."""
import json

import pytest
import torch

from scripts import accept_facts_gpu as acceptance
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig


def test_missing_cuda_is_failure_with_machine_readable_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "audit.json"
    result = acceptance.main(["--checkpoint", str(tmp_path / "frozen.pt"), "--output", str(output)])
    audit = json.loads(output.read_text())
    assert result == 1
    assert audit["status"] == "failed"
    assert audit["failed_phase"] == "cuda_preflight"
    assert "CPU fallback/skip is forbidden" in audit["error"]
    assert "inference" not in audit


def test_frozen_baseline_guard_rejects_global_or_probability_inputs():
    for globals_ in (True, False):
        policy = JunqiNet(JunqiNetConfig(embed_dim=16, depth=1, n_head=2, cnn_channels=8,
                                         identity_encoding="legacy_probabilities", use_global_summary=globals_))
        with pytest.raises(ValueError, match="frozen facts baseline"):
            acceptance._validate_facts_policy(policy)


def test_frozen_baseline_guard_rejects_nonfinite_parameters():
    policy = JunqiNet(JunqiNetConfig(embed_dim=16, depth=1, n_head=2, cnn_channels=8,
                                     identity_encoding="known_only", use_global_summary=False))
    assert acceptance._validate_facts_policy(policy)["observation_channels"] == 317
    with torch.no_grad():
        next(policy.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        acceptance._validate_facts_policy(policy)


@pytest.mark.parametrize("output_name", ["frozen.pt", "frozen.json"])
def test_source_checkpoint_cannot_be_overwritten(tmp_path, output_name):
    source = tmp_path / ("frozen.pt" if output_name == "frozen.pt" else "frozen-smoke-runtime.pt")
    source.write_bytes(b"frozen checkpoint")
    with pytest.raises(SystemExit) as failure:
        acceptance.main(["--checkpoint", str(source), "--output", str(tmp_path / output_name)])
    assert failure.value.code == 2
    assert source.read_bytes() == b"frozen checkpoint"
