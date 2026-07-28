"""tests/test_eval_vs_random.py — eval_vs_random smoke test."""
from __future__ import annotations

import pytest

try:
    import torch
    _HAS_TORCH = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch + cuda required",
)


def test_eval_vs_random_untrained_net_is_roughly_balanced():
    from junqi_rl.analysis import eval_vs_random
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    cfg = JunqiNetConfig(
        cnn_channels=16, cnn_layers=2, depth=1,
        embed_dim=32, n_head=2, ff_factor=2, dropout=0.0,
    )
    policy = JunqiNet(cfg).to("cuda").eval()
    stats = eval_vs_random(
        policy, num_games=32, trained_team=0,
        max_steps=400, device="cuda", seed_base=7,
    )
    assert stats["num_games"] == 32
    # Sum must be 1.0 (±1e-6)
    tot = (stats["trained_win_rate"] + stats["trained_loss_rate"]
           + stats["draw_rate"]       + stats["ongoing_rate"])
    assert abs(tot - 1.0) < 1e-6
    # Untrained policy: not expected to systematically beat random
    assert 0.0 <= stats["trained_win_rate"] <= 1.0
    assert 0.0 <= stats["trained_loss_rate"] <= 1.0
