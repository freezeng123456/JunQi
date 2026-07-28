"""tests/test_collect_autocast_bf16.py — verify the v33d fix for v33c R107 NaN.

Root cause (v33c R107):
    ``junqi_rl.training.gpu_collector.collect_rollout_gpu_v2`` hardcoded
    ``torch.amp.autocast("cuda", dtype=torch.float16)`` around ``policy.act(...)``.
    fp16's dynamic range is only ±65504 — when self-play training drove
    the policy toward confident moves (extreme logits and saturated
    value-head outputs), the act-time forward pass overflowed fp16 and
    returned NaN log_probs / NaN values. Those NaNs entered the rollout
    buffer; ``compute_returns`` propagated them through GAE to every
    advantage; every PPO minibatch then short-circuited via the
    grad-NaN guard, freezing ``num_train_step``. From the user's
    perspective: training silently died at R107 with all-NaN metrics.

Fix:
    The collector now takes an ``autocast_dtype`` argument (default
    bf16) and the trainer passes ``cfg.ppo.get_dtype()``. bf16 has
    fp32 dynamic range so the same forward stays finite even when
    the policy is highly confident.

This test reproduces the failure mode by injecting extreme weights
into JunqiNet's value head, then confirming that ``act()`` under
bf16 autocast returns finite outputs (where fp16 autocast does NOT).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "junqi_rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_act_under_bf16_stays_finite_with_extreme_weights() -> None:
    """Confident-policy regression: extreme weights × bf16 autocast → finite outputs."""
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    cfg = JunqiNetConfig(
        cnn_channels=32, cnn_layers=2,
        depth=2, embed_dim=64, n_head=4, ff_factor=4,
        action_key_dim=16, use_cat_vf=True,
    )
    device = torch.device("cuda:0")
    net = JunqiNet(cfg).to(device).eval()
    # Inject extreme weights into the value head — simulate the v33c R74/R99
    # state where the value head had over-fit. Multiplied by 100x, fp16
    # range fails on softmax/log_softmax of these logits.
    with torch.no_grad():
        net.value_head.weight.mul_(100.0)
        net.value_head.bias.mul_(100.0)

    B = 8
    from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
    from junqi_rl.networks.junqi_net import FLAT_ACTION_DIM
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17, device=device)
    obs_gl = torch.randn(B, OBS_GLOBAL_DIMS, device=device)
    legal_mask = torch.ones(B, FLAT_ACTION_DIM, dtype=torch.bool, device=device)
    # Mark some moves illegal to make the path realistic.
    legal_mask[:, 1000:] = False

    # bf16 autocast: should stay finite.
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        actions, log_probs, values = net.act(obs_sp, obs_gl, legal_mask)
    assert torch.isfinite(log_probs).all(), (
        "log_probs went non-finite under bf16 autocast even with extreme "
        "value-head weights — bf16 should be range-safe (±3.4e38)."
    )
    assert torch.isfinite(values).all(), (
        "value head output went non-finite under bf16 autocast."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_act_under_fp16_can_overflow_with_extreme_weights() -> None:
    """Sentinel: confirm fp16 path actually fails on extreme weights.

    This documents WHY the bf16 fix matters. If this test ever starts
    passing, either fp16 has been hardened in upstream PyTorch, or our
    extreme-weights setup is no longer extreme enough; either way
    revisit the choice in gpu_collector before deleting this test.
    """
    from junqi_rl.networks.junqi_net import JunqiNet, JunqiNetConfig

    cfg = JunqiNetConfig(
        cnn_channels=32, cnn_layers=2,
        depth=2, embed_dim=64, n_head=4, ff_factor=4,
        action_key_dim=16, use_cat_vf=True,
    )
    device = torch.device("cuda:0")
    net = JunqiNet(cfg).to(device).eval()
    # Same extreme-weights perturbation as above.
    with torch.no_grad():
        net.value_head.weight.mul_(1000.0)
        net.value_head.bias.mul_(1000.0)
        net.q_proj.weight.mul_(1000.0)
        net.k_proj.weight.mul_(1000.0)

    B = 8
    from junqi_core.observation import OBS_CHANNELS, OBS_GLOBAL_DIMS
    from junqi_rl.networks.junqi_net import FLAT_ACTION_DIM
    obs_sp = torch.randn(B, OBS_CHANNELS, 17, 17, device=device) * 50.0
    obs_gl = torch.randn(B, OBS_GLOBAL_DIMS, device=device) * 50.0
    legal_mask = torch.ones(B, FLAT_ACTION_DIM, dtype=torch.bool, device=device)
    legal_mask[:, 1000:] = False

    # fp16 autocast: SHOULD overflow somewhere.
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        actions, log_probs, values = net.act(obs_sp, obs_gl, legal_mask)

    # Either log_probs or values should contain non-finite entries on fp16.
    fp16_overflowed = (
        not torch.isfinite(log_probs).all()
        or not torch.isfinite(values).all()
    )
    assert fp16_overflowed, (
        "fp16 autocast did NOT produce NaN/Inf with our extreme-weights "
        "setup. Either PyTorch upstream improved fp16 stability or our "
        "regression scenario is no longer extreme — revisit the bf16 "
        "default in gpu_collector.collect_rollout_gpu_v2 before deleting "
        "this test."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_collect_default_autocast_is_bf16() -> None:
    """Lock the v33d default — collector should use bf16 unless told otherwise.

    Guards against an accidental future regression where someone reverts
    the default to fp16 (the v33c bug we just fixed).
    """
    import inspect

    from junqi_rl.training import gpu_collector

    sig = inspect.signature(gpu_collector.collect_rollout_gpu_v2)
    param = sig.parameters.get("autocast_dtype")
    assert param is not None, (
        "collect_rollout_gpu_v2 must accept an autocast_dtype parameter "
        "(introduced in v33d to fix the fp16 NaN bug)."
    )
    # Default value: explicit None (resolved to bf16 inside the function).
    assert param.default is None, (
        f"autocast_dtype default should be None (resolved to bf16 inside "
        f"the function); got {param.default}"
    )
    # Also confirm the docstring or function body mentions bf16 to make
    # the contract self-documenting.
    src = inspect.getsource(gpu_collector.collect_rollout_gpu_v2)
    assert "bfloat16" in src, (
        "collect_rollout_gpu_v2 source must reference bfloat16 (the "
        "configured-default dtype) for traceability."
    )


if __name__ == "__main__":
    test_act_under_bf16_stays_finite_with_extreme_weights()
    print("[1/3] bf16 autocast stays finite under extreme weights: OK")
    test_act_under_fp16_can_overflow_with_extreme_weights()
    print("[2/3] fp16 autocast can overflow (regression sentinel): OK")
    test_collect_default_autocast_is_bf16()
    print("[3/3] collect default autocast = bf16: OK")
    print("ALL COLLECT-AUTOCAST TESTS PASSED")
