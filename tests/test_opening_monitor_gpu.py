"""Real CUDA probe must preserve training world, pool, weights and RNG."""
import copy
import numpy as np
import pytest
import torch

pytest.importorskip("junqi_cuda")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from junqi_rl.gpu_rollout import GpuRollout
from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig
from junqi_rl.training.game_progress import capture_rng, restore_rng
from junqi_rl.training.opening_monitor import OpeningMonitor, OpeningMonitorConfig


def test_cuda_opening_probe_is_observational_only(tmp_path):
    torch.manual_seed(77)
    policy = JunqiNet(JunqiNetConfig(embed_dim=32, depth=1, n_head=4,
        cnn_channels=16, use_global_summary=False, identity_encoding="known_only")).cuda().train()
    training = GpuRollout(4, include_global=False, identity_encoding="known_only")
    training.reset(seed_base=910)
    before = copy.deepcopy(training.state_dict())
    pool = training.setup_pool_snapshot().copy()
    weights = {k: v.clone() for k, v in policy.state_dict().items()}
    rng = np.random.default_rng(6)
    saved = capture_rng(rng)
    probe = OpeningMonitor(OpeningMonitorConfig(enabled=True, boards_per_bank=4,
        batch_size=8, min_source_samples=4), tmp_path)
    result = probe.check(policy, completed_games=0, rollout=0)
    after_rng = capture_rng(rng)
    assert result["status"] == "healthy"
    assert result["fixed"]["decisions"] == 16
    assert policy.training
    for key, value in weights.items():
        torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)
    for key, value in before["native"]["arrays"].items():
        np.testing.assert_array_equal(value, training.state_dict()["native"]["arrays"][key], err_msg=key)
    np.testing.assert_array_equal(pool, training.setup_pool_snapshot())
    assert torch.equal(saved["torch_cpu"], after_rng["torch_cpu"])
    assert all(torch.equal(a, b) for a, b in zip(saved["torch_cuda"], after_rng["torch_cuda"]))
    assert saved["python"] == after_rng["python"]
    np.testing.assert_array_equal(saved["numpy"][1], after_rng["numpy"][1])
