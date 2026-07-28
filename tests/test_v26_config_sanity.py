"""Sanity check: v26 JunqiNet config builds and does a forward pass.

This catches any config-field typos BEFORE we launch a 4-hour training
run. Runs on CPU (no GPU needed) to avoid competing for T4 if it's busy.
"""
from __future__ import annotations

import pytest
import torch

from junqi_core.observation import OBS_CHANNELS


def test_v26_net_builds_on_cpu():
    """Load v26 yaml, build JunqiNet, run one forward pass."""
    from scripts.train import TrainConfig, _dict_to_dataclass
    import yaml
    from pathlib import Path

    # Resolve cfg path relative to this test file so the test is portable
    # (the original hard-coded /data/home/freezeng/... only worked on the
    # author's dev box).
    cfg_path = (Path(__file__).resolve().parent.parent
                / "exps" / "beat_random_v26_net_scaleup" / "cfg.yaml")
    if not cfg_path.exists():
        pytest.skip(f"v26 cfg not found at {cfg_path}")
    with open(cfg_path) as f:
        d = yaml.safe_load(f)
    cfg = _dict_to_dataclass(TrainConfig, d)

    # Should parse cleanly
    assert cfg.net.depth == 6
    assert cfg.net.embed_dim == 192
    assert cfg.net.n_head == 8
    assert cfg.net.cnn_channels == 96
    assert cfg.total_rollouts == 1000

    # Build the actual net and count params
    from junqi_rl.networks.junqi_net import JunqiNet
    net = JunqiNet(cfg.net)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"\n  v26 JunqiNet params: {n_params:,} ({n_params/1e6:.2f}M)")
    # Expected ~3.07M from the memory estimate (v26-A after v26-B OOMed)
    assert 2_500_000 < n_params < 4_000_000, (
        f"Unexpected v26 param count: {n_params}"
    )

    # Forward pass with batch=4 (small enough for CPU)
    B = 4
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17)
    obs_gl = torch.randn(B, 28)
    legal_mask = torch.ones(B, 16641, dtype=torch.bool)
    legal_mask[:, 0] = False   # at least one illegal
    net.eval()
    with torch.no_grad():
        out = net(obs_sp, obs_gl, legal_mask)
    # Shape sanity
    assert out["log_probs"].shape == (B, 16641)
    assert out["value"].shape[0] == B
    print(f"  v26 forward OK: log_probs {out['log_probs'].shape}, "
          f"value {out['value'].shape}")
